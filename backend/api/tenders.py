"""Tender upload, review, and deterministic matching endpoints."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.database import SessionLocal, get_db
from database.models import (
    CanonicalRequirement,
    ProductFact as ProductFactRow,
    ProductSpecification,
    Tender,
    TenderMatchResult,
)
from logging_config.logger import get_logger
from src.tenders.agents import RequirementExtractionAgent, SectionDetectionAgent
from src.tenders.fact_extraction import ProductFactExtractionAgent, ProductFactStore
from src.tenders.llm_judge import LLMJudge
from src.tenders.matcher import ProductMatcher
from src.tenders.pdf_extraction import TenderPDFExtractor
from src.tenders.schemas import ProductFact, Requirement, RequirementStatus

logger = get_logger(__name__)

router = APIRouter(prefix="/api/tenders", tags=["tenders"])


class RequirementPatch(BaseModel):
    product_group: Optional[str] = None
    requirement_type: Optional[str] = None
    attribute: Optional[str] = None
    operator: Optional[str] = None
    value: Optional[Any] = None
    unit: Optional[str] = None
    confidence: Optional[float] = None
    needs_review: Optional[bool] = None
    status: Optional[str] = None
    points: Optional[float] = None
    rationale: Optional[str] = None


def _dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return json.loads(model.json())


def _requirement_from_row(row: CanonicalRequirement) -> Requirement:
    return Requirement(
        id=row.id,
        tender_id=row.tender_id,
        product_group=row.product_group,
        requirement_type=row.requirement_type,
        attribute=row.attribute,
        operator=row.operator,
        value=row.value,
        unit=row.unit,
        original_text=row.original_text,
        source_page=row.source_page,
        source_chunk_id=row.source_chunk_id,
        confidence=row.confidence or 0.0,
        needs_review=bool(row.needs_review),
        status=row.status,
        points=row.points,
        rationale=row.rationale,
    )


def _row_from_requirement(req: Requirement) -> CanonicalRequirement:
    return CanonicalRequirement(**_dump(req))


@router.post("/upload")
def upload_tender(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload tender PDF and start generic extraction in the background."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    upload_dir = Path("./data/tenders")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_content = file.file.read()
    safe_name = Path(file.filename).name
    file_path = upload_dir / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_name}"
    with open(file_path, "wb") as handle:
        handle.write(file_content)

    try:
        tender = Tender(
            id=str(uuid4()),
            filename=safe_name,
            status="processing",
            tender_metadata={
                "file_path": str(file_path),
                "message": "PDF wurde hochgeladen. Extraktion läuft.",
            },
        )
        db.add(tender)
        db.commit()
        db.refresh(tender)
        background_tasks.add_task(_extract_tender_background, tender.id, str(file_path))
        logger.info("Tender %s uploaded; extraction started in background", tender.id)
        return {
            "id": tender.id,
            "filename": tender.filename,
            "status": tender.status,
            "metadata": tender.tender_metadata,
        }
    except Exception as exc:
        db.rollback()
        logger.error("Tender upload failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


def _extract_tender_background(tender_id: str, file_path: str) -> None:
    """Run slow PDF and LLM extraction outside the upload request lifecycle."""
    db = SessionLocal()
    try:
        tender = db.query(Tender).filter(Tender.id == tender_id).first()
        if not tender:
            logger.warning("Tender %s disappeared before extraction started", tender_id)
            return

        extractor = TenderPDFExtractor()
        section_agent = SectionDetectionAgent()
        requirement_agent = RequirementExtractionAgent()

        _update_tender_progress(
            db,
            tender,
            phase="pdf_extracting",
            message="PDF wird gelesen.",
        )
        extracted = extractor.extract(file_path)
        chunks = extracted["chunks"]
        _update_tender_progress(
            db,
            tender,
            phase="sections_detecting",
            message="PDF gelesen. Abschnitte werden erkannt.",
            **(extracted.get("metadata") or {}),
            chunk_count=len(chunks),
        )
        sections = section_agent.detect(chunks)
        _update_tender_progress(
            db,
            tender,
            phase="llm_extracting",
            message="Abschnitte erkannt. Anforderungen werden extrahiert.",
            section_count=len(sections),
            chunks_total=len(chunks),
            chunks_processed=0,
            llm_calls=0,
            llm_skipped=0,
            skipped_chunks=0,
            requirement_count=0,
        )
        requirements = requirement_agent.extract(
            tender.id,
            chunks,
            sections,
            progress_callback=lambda progress: _update_tender_progress(db, tender, **progress),
        )

        db.query(CanonicalRequirement).filter(CanonicalRequirement.tender_id == tender.id).delete()
        for req in requirements:
            db.add(_row_from_requirement(req))

        tender.raw_text = extracted["raw_text"]
        tender.status = "review_ready"
        tender.tender_metadata = {
            **(tender.tender_metadata or {}),
            **(extracted.get("metadata") or {}),
            "file_path": file_path,
            "chunk_count": len(chunks),
            "section_count": len(sections),
            "requirement_count": len(requirements),
            "needs_review_count": sum(1 for req in requirements if req.needs_review),
            "message": "Extraktion abgeschlossen. Review ist bereit.",
        }
        db.commit()
        logger.info("Tender %s extracted: chunks=%s requirements=%s", tender.id, len(chunks), len(requirements))
    except Exception as exc:
        db.rollback()
        tender = db.query(Tender).filter(Tender.id == tender_id).first()
        if tender:
            tender.status = "error"
            tender.tender_metadata = {**(tender.tender_metadata or {}), "error": str(exc)}
            db.commit()
        logger.error("Tender extraction failed for %s: %s", tender_id, exc, exc_info=True)
    finally:
        db.close()


def _update_tender_progress(db: Session, tender: Tender, **metadata: Any) -> None:
    tender.status = "processing"
    tender.tender_metadata = {
        **(tender.tender_metadata or {}),
        **metadata,
    }
    db.commit()


@router.get("/{tender_id}")
def get_tender(tender_id: str, db: Session = Depends(get_db)):
    tender = db.query(Tender).filter(Tender.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")
    return {
        "id": tender.id,
        "filename": tender.filename,
        "uploaded_at": tender.uploaded_at.isoformat() if tender.uploaded_at else None,
        "status": tender.status,
        "metadata": tender.tender_metadata,
    }


@router.get("/{tender_id}/requirements")
def list_requirements(tender_id: str, db: Session = Depends(get_db)):
    rows = db.query(CanonicalRequirement).filter(CanonicalRequirement.tender_id == tender_id).all()
    return [_dump(_requirement_from_row(row)) for row in rows]


@router.patch("/requirements/{requirement_id}")
def patch_requirement(requirement_id: str, patch: RequirementPatch, db: Session = Depends(get_db)):
    row = db.query(CanonicalRequirement).filter(CanonicalRequirement.id == requirement_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Requirement not found")
    data = patch.model_dump(exclude_unset=True) if hasattr(patch, "model_dump") else patch.dict(exclude_unset=True)
    for key, value in data.items():
        setattr(row, key, value)
    if row.status == "pending":
        row.status = RequirementStatus.EDITED.value
    db.commit()
    db.refresh(row)
    return _dump(_requirement_from_row(row))


@router.post("/requirements/{requirement_id}/approve")
def approve_requirement(requirement_id: str, db: Session = Depends(get_db)):
    return _set_requirement_status(requirement_id, RequirementStatus.APPROVED.value, db)


@router.post("/requirements/{requirement_id}/reject")
def reject_requirement(requirement_id: str, db: Session = Depends(get_db)):
    return _set_requirement_status(requirement_id, RequirementStatus.REJECTED.value, db)


def _set_requirement_status(requirement_id: str, status: str, db: Session):
    row = db.query(CanonicalRequirement).filter(CanonicalRequirement.id == requirement_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Requirement not found")
    row.status = status
    db.commit()
    db.refresh(row)
    return _dump(_requirement_from_row(row))


@router.post("/{tender_id}/requirements/approve-high-confidence")
def approve_high_confidence(tender_id: str, db: Session = Depends(get_db)):
    rows = db.query(CanonicalRequirement).filter(
        CanonicalRequirement.tender_id == tender_id,
        CanonicalRequirement.confidence >= 0.85,
        CanonicalRequirement.needs_review.is_(False),
        CanonicalRequirement.status == "pending",
    ).all()
    for row in rows:
        row.status = "approved"
    db.commit()
    return {"approved": len(rows)}


@router.post("/{tender_id}/match")
def match_tender(tender_id: str, db: Session = Depends(get_db)):
    tender = db.query(Tender).filter(Tender.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    req_rows = db.query(CanonicalRequirement).filter(CanonicalRequirement.tender_id == tender_id).all()
    requirements = [_requirement_from_row(row) for row in req_rows]
    products = db.query(ProductSpecification).all()

    fact_store = ProductFactStore()
    fact_agent = ProductFactExtractionAgent()
    matcher = ProductMatcher(llm_judge=LLMJudge.from_env())

    db.query(TenderMatchResult).filter(TenderMatchResult.tender_id == tender_id).delete()
    results = []
    for product in products:
        facts = _load_or_build_facts(product, requirements, fact_store, fact_agent, db)
        result = matcher.match_product(tender.id, str(product.id), product.product_name, requirements, facts)
        result_data = _dump(result)
        db.add(
            TenderMatchResult(
                id=result.id,
                tender_id=tender.id,
                product_id=str(product.id),
                model=product.product_name,
                eligibility=result.eligibility.value,
                score=result.score,
                max_score=result.max_score,
                must_passed=result.must_passed,
                must_failed=result.must_failed,
                unknown_count=result.unknown_count,
                requirement_results=[_dump(item) for item in result.requirement_results],
            )
        )
        results.append(result_data)

    tender.status = "matching_completed"
    db.commit()
    return sorted(results, key=lambda item: (item["eligibility"] != "eligible", -item["score"], item["unknown_count"]))


def _load_or_build_facts(
    product: ProductSpecification,
    requirements: List[Requirement],
    fact_store: ProductFactStore,
    fact_agent: ProductFactExtractionAgent,
    db: Session,
) -> List[ProductFact]:
    rows = db.query(ProductFactRow).filter(ProductFactRow.product_id == str(product.id)).all()
    if rows:
        facts = [
            ProductFact(
                id=row.id,
                product_id=row.product_id,
                model=row.model,
                attribute=row.attribute,
                value=row.value,
                unit=row.unit,
                source_text=row.source_text,
                source_document=row.source_document,
                source_page=row.source_page,
                confidence=row.confidence or 0.0,
            )
            for row in rows
        ]
    else:
        facts = fact_store.facts_for_product(product)
        for fact in facts:
            db.add(ProductFactRow(**_dump(fact)))
        db.flush()

    extracted = fact_agent.extract_missing_facts(product, requirements, facts, db)
    for fact in extracted:
        db.add(ProductFactRow(**_dump(fact)))
    db.flush()
    return facts + extracted


@router.get("/{tender_id}/matches")
def list_matches(tender_id: str, db: Session = Depends(get_db)):
    rows = db.query(TenderMatchResult).filter(TenderMatchResult.tender_id == tender_id).all()
    return [
        {
            "id": row.id,
            "tender_id": row.tender_id,
            "product_id": row.product_id,
            "model": row.model,
            "eligibility": row.eligibility,
            "score": row.score,
            "max_score": row.max_score,
            "must_passed": row.must_passed,
            "must_failed": row.must_failed,
            "unknown_count": row.unknown_count,
        }
        for row in rows
    ]


@router.get("/{tender_id}/matches/{product_id}")
def get_match_detail(tender_id: str, product_id: str, db: Session = Depends(get_db)):
    row = db.query(TenderMatchResult).filter(
        TenderMatchResult.tender_id == tender_id,
        TenderMatchResult.product_id == product_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Match not found")
    return {
        "id": row.id,
        "tender_id": row.tender_id,
        "product_id": row.product_id,
        "model": row.model,
        "eligibility": row.eligibility,
        "score": row.score,
        "max_score": row.max_score,
        "must_passed": row.must_passed,
        "must_failed": row.must_failed,
        "unknown_count": row.unknown_count,
        "requirement_results": row.requirement_results or [],
    }
