"""Product filtering and matching endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel
from database.database import get_db
from database.models import ProductSpecification, RequirementSpecification
from src.matching.product_matcher import ProductMatcher
from logging_config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/products", tags=["products"])
matcher = ProductMatcher()


class ProductResponse(BaseModel):
    """Response model for product information."""
    product_name: str
    brand: Optional[str] = None
    display_brightness_nits: Optional[int] = None
    screen_to_body_ratio: Optional[float] = None
    weight_kg: Optional[float] = None
    ram_max_gb: Optional[int] = None
    storage_max_tb: Optional[float] = None
    battery_wh: Optional[float] = None
    display_size_inch: Optional[float] = None
    display_resolution: Optional[str] = None
    document_id: int
    product_spec_id: int


class ProductMatchResponse(BaseModel):
    """Response model for product matches."""
    product_name: str
    brand: Optional[str] = None
    display_brightness_nits: Optional[int] = None
    screen_to_body_ratio: Optional[float] = None
    weight_kg: Optional[float] = None
    ram_max_gb: Optional[int] = None
    storage_max_tb: Optional[float] = None
    battery_wh: Optional[float] = None
    display_size_inch: Optional[float] = None
    display_resolution: Optional[str] = None
    match_score: float
    document_id: int
    product_spec_id: int


class RequirementResponse(BaseModel):
    """Response model for requirement information."""
    id: int
    document_id: int
    min_display_brightness_nits: Optional[int] = None
    min_screen_to_body_ratio: Optional[float] = None
    max_weight_kg: Optional[float] = None
    min_ram_gb: Optional[int] = None
    min_storage_tb: Optional[float] = None
    min_battery_wh: Optional[float] = None
    display_size_inch_min: Optional[float] = None
    display_size_inch_max: Optional[float] = None
    additional_requirements: Optional[dict] = None
    extracted_at: Optional[str] = None


@router.get("/filter", response_model=List[ProductResponse])
def filter_products(
    min_brightness_nits: Optional[int] = Query(None, description="Minimale Helligkeit in nits"),
    min_screen_to_body_ratio: Optional[float] = Query(None, description="Minimale Screen-to-Body Ratio in %"),
    max_weight_kg: Optional[float] = Query(None, description="Maximales Gewicht in kg"),
    min_ram_gb: Optional[int] = Query(None, description="Minimaler RAM in GB"),
    min_battery_wh: Optional[float] = Query(None, description="Minimale Akkukapazität in Wh"),
    min_storage_tb: Optional[float] = Query(None, description="Minimaler Speicher in TB"),
    display_size_inch_min: Optional[float] = Query(None, description="Minimale Display-Größe in Zoll"),
    display_size_inch_max: Optional[float] = Query(None, description="Maximale Display-Größe in Zoll"),
    db: Session = Depends(get_db)
):
    """Filtert Produkte nach technischen Spezifikationen."""
    query = db.query(ProductSpecification)
    
    if min_brightness_nits:
        query = query.filter(ProductSpecification.display_brightness_nits >= min_brightness_nits)
    
    if min_screen_to_body_ratio:
        query = query.filter(ProductSpecification.screen_to_body_ratio >= min_screen_to_body_ratio)
    
    if max_weight_kg:
        query = query.filter(ProductSpecification.weight_kg <= max_weight_kg)
    
    if min_ram_gb:
        query = query.filter(ProductSpecification.ram_max_gb >= min_ram_gb)
    
    if min_storage_tb:
        query = query.filter(ProductSpecification.storage_max_tb >= min_storage_tb)
    
    if min_battery_wh:
        query = query.filter(ProductSpecification.battery_wh >= min_battery_wh)
    
    if display_size_inch_min:
        query = query.filter(ProductSpecification.display_size_inch >= display_size_inch_min)
    
    if display_size_inch_max:
        query = query.filter(ProductSpecification.display_size_inch <= display_size_inch_max)
    
    results = query.all()
    
    logger.info(f"Filtered {len(results)} products with filters: "
                f"min_brightness={min_brightness_nits}, "
                f"min_ratio={min_screen_to_body_ratio}, "
                f"max_weight={max_weight_kg}")
    
    return [
        ProductResponse(
            product_name=p.product_name,
            brand=p.brand,
            display_brightness_nits=p.display_brightness_nits,
            screen_to_body_ratio=p.screen_to_body_ratio,
            weight_kg=p.weight_kg,
            ram_max_gb=p.ram_max_gb,
            storage_max_tb=p.storage_max_tb,
            battery_wh=p.battery_wh,
            display_size_inch=p.display_size_inch,
            display_resolution=p.display_resolution,
            document_id=p.document_id,
            product_spec_id=p.id
        )
        for p in results
    ]


@router.post("/match-requirement/{requirement_id}", response_model=List[ProductMatchResponse])
def match_requirement_to_products(
    requirement_id: int,
    db: Session = Depends(get_db)
):
    """Findet alle Produkte, die eine bestimmte Anforderung erfüllen."""
    matches = matcher.match_requirements_to_products(requirement_id, db)
    
    if not matches:
        return []
    
    return [
        ProductMatchResponse(
            product_name=m["product_name"],
            brand=m["brand"],
            display_brightness_nits=m["display_brightness_nits"],
            screen_to_body_ratio=m["screen_to_body_ratio"],
            weight_kg=m["weight_kg"],
            ram_max_gb=m["ram_max_gb"],
            storage_max_tb=m["storage_max_tb"],
            battery_wh=m["battery_wh"],
            display_size_inch=m.get("display_size_inch"),
            display_resolution=m.get("display_resolution"),
            match_score=m["match_score"],
            document_id=m["document_id"],
            product_spec_id=m["product_spec_id"]
        )
        for m in matches
    ]


@router.get("/requirements", response_model=List[RequirementResponse])
def list_requirements(db: Session = Depends(get_db)):
    """Listet alle gespeicherten Anforderungen."""
    requirements = db.query(RequirementSpecification).all()
    
    return [
        RequirementResponse(
            id=r.id,
            document_id=r.document_id,
            min_display_brightness_nits=r.min_display_brightness_nits,
            min_screen_to_body_ratio=r.min_screen_to_body_ratio,
            max_weight_kg=r.max_weight_kg,
            min_ram_gb=r.min_ram_gb,
            min_storage_tb=r.min_storage_tb,
            min_battery_wh=r.min_battery_wh,
            display_size_inch_min=r.display_size_inch_min,
            display_size_inch_max=r.display_size_inch_max,
            additional_requirements=r.additional_requirements,
            extracted_at=r.extracted_at.isoformat() if r.extracted_at else None
        )
        for r in requirements
    ]


@router.get("/requirements/{requirement_id}", response_model=RequirementResponse)
def get_requirement(
    requirement_id: int,
    db: Session = Depends(get_db)
):
    """Holt eine spezifische Anforderung."""
    requirement = db.query(RequirementSpecification).filter(
        RequirementSpecification.id == requirement_id
    ).first()
    
    if not requirement:
        raise HTTPException(
            status_code=404,
            detail=f"Requirement {requirement_id} not found"
        )
    
    return RequirementResponse(
        id=requirement.id,
        document_id=requirement.document_id,
        min_display_brightness_nits=requirement.min_display_brightness_nits,
        min_screen_to_body_ratio=requirement.min_screen_to_body_ratio,
        max_weight_kg=requirement.max_weight_kg,
        min_ram_gb=requirement.min_ram_gb,
        min_storage_tb=requirement.min_storage_tb,
        min_battery_wh=requirement.min_battery_wh,
        display_size_inch_min=requirement.display_size_inch_min,
        display_size_inch_max=requirement.display_size_inch_max,
        additional_requirements=requirement.additional_requirements,
        extracted_at=requirement.extracted_at.isoformat() if requirement.extracted_at else None
    )


@router.get("/list", response_model=List[ProductResponse])
def list_all_products(db: Session = Depends(get_db)):
    """Listet alle gespeicherten Produkte."""
    products = db.query(ProductSpecification).all()
    
    return [
        ProductResponse(
            product_name=p.product_name,
            brand=p.brand,
            display_brightness_nits=p.display_brightness_nits,
            screen_to_body_ratio=p.screen_to_body_ratio,
            weight_kg=p.weight_kg,
            ram_max_gb=p.ram_max_gb,
            storage_max_tb=p.storage_max_tb,
            battery_wh=p.battery_wh,
            display_size_inch=p.display_size_inch,
            display_resolution=p.display_resolution,
            document_id=p.document_id,
            product_spec_id=p.id
        )
        for p in products
    ]
