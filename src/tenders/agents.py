"""Tender section and requirement extraction agents."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from logging_config.logger import get_logger
from src.llm import create_chat_llm
from src.tenders.pdf_extraction import TenderChunk
from src.tenders.schemas import Operator, Requirement, RequirementType
from taxonomy.laptop_attributes import normalize_attribute

logger = get_logger(__name__)


class SectionDetectionAgent:
    """Classify tender chunks with generic keyword signals."""

    def detect(self, chunks: Iterable[TenderChunk]) -> List[Dict[str, Any]]:
        return [self._detect_chunk(chunk) for chunk in chunks]

    def _detect_chunk(self, chunk: TenderChunk) -> Dict[str, Any]:
        text = f"{chunk.section_heading or ''}\n{chunk.text}".lower()
        section_type = "unknown"
        confidence = 0.45

        signals = [
            ("technical_requirements", ["mindestausstattung", "mindestanforderung", "technische", "prozessor", "ram", "display"]),
            ("scoring", ["bewertung", "punkte", "zuschlagskriterium", "score"]),
            ("service", ["service", "support", "garantie", "vor-ort", "reparatur"]),
            ("contract", ["vertrag", "lieferbedingung", "laufzeit", "kündigung"]),
            ("pricing", ["preis", "rabatt", "kosten", "angebotssumme"]),
            ("admin", ["formblatt", "vergabestelle", "frist", "unterschrift", "lieferanschrift", "rechnungsanschrift", "liefertermin", "auftragsbestätigung", "bestellnummer", "rechnungsstellung", "lieferbedingungen"]),
        ]
        for candidate, words in signals:
            hits = sum(1 for word in words if word in text)
            if hits:
                section_type = candidate
                confidence = min(0.95, 0.55 + hits * 0.12)
                break

        return {
            "chunk_id": chunk.chunk_id,
            "section_type": section_type,
            "product_group": self._product_group(chunk),
            "confidence": confidence,
        }

    def _product_group(self, chunk: TenderChunk) -> Optional[str]:
        source = chunk.section_heading or chunk.text[:200]
        match = re.search(r"(notebook|laptop|workstation|tablet|desktop)[^\n]{0,80}", source, re.I)
        return match.group(0).strip() if match else None


class RequirementExtractionAgent:
    """Extract canonical requirements. Uses LLM when configured, regex fallback otherwise."""

    DEVICE_KEYWORDS = [
        "notebook", "laptop", "desktop", "pc", "workstation", "basisgerät",
        "prozessor", "processor", "cpu", "vpro", "ram", "arbeitsspeicher", "ddr",
        "sodimm", "ssd", "nvme", "pcie", "speicher", "storage", "grafik",
        "graphics", "display", "bildschirm", "monitor", "wuxga", "hdmi", "usb",
        "thunderbolt", "rj45", "ethernet", "wi-fi", "wifi", "wlan", "bluetooth",
        "wake on lan", "pxe", "miracast", "touchpad", "clickpad", "tastatur",
        "keyboard", "mikrofon", "webcam", "kamera", "sensor", "kensington",
        "bios", "akku", "battery", "netzteil", "usb type-c", "windows", "tpm",
        "datenträger", "garantie",
    ]

    NON_DEVICE_PATTERNS = [
        r"\b(preis|preise|preisposition|positionsmenge|menge|mengen|rabatt|kosten|angebotssumme)\b",
        r"\b(lieferfrist|liefertermin|vertragslaufzeit|kündigung|vergabestelle|formblatt|unterschrift)\b",
        r"\b(zuschlag|bewertung|punkte|score|wertungsmatrix)\b",
        r"\b(ansprechpartner|rechnung|zahlungsbedingung|rechnungstellung)\b",
        r"\b(dienen ausschließlich der berechnung des preises|nichts mit den tatsächlich benötigten mengen)\b",
        r"\b(lieferanschrift|rechnungsanschrift|auftragsbestätigung|bestellnummer|lieferbedingung)\b",
        r"\b(übernahme\s+(liefer|rechnung))\b",
    ]

    def __init__(self) -> None:
        self.max_llm_calls = int(os.getenv("TENDER_MAX_LLM_CALLS", "30"))
        self.llm = create_chat_llm(
            temperature=0,
            request_timeout=int(os.getenv("TENDER_LLM_TIMEOUT", "45")),
            max_retries=int(os.getenv("TENDER_LLM_RETRIES", "1")),
        )

    def extract(
        self,
        tender_id: str,
        chunks: Iterable[TenderChunk],
        sections: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Requirement]:
        chunks = list(chunks)
        section_by_chunk = {section["chunk_id"]: section for section in sections}
        requirements: List[Requirement] = []
        llm_calls = 0
        llm_skipped = 0
        skipped_chunks = 0
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            section = section_by_chunk.get(chunk.chunk_id, {})
            if section.get("section_type") in {"admin", "pricing"}:
                skipped_chunks += 1
                continue
            try:
                chunk_requirements = self._extract_chunk(
                    tender_id,
                    chunk,
                    section,
                    can_use_llm=llm_calls < self.max_llm_calls,
                )
                requirements.extend(chunk_requirements)
                if self._last_extraction_used_llm:
                    llm_calls += 1
                else:
                    llm_skipped += 1
            except Exception as exc:
                logger.warning("Requirement extraction failed for chunk %s: %s", chunk.chunk_id, exc)
            if progress_callback and (index == total or index % 5 == 0):
                progress_callback(
                    {
                        "phase": "llm_extracting",
                        "chunks_processed": index,
                        "chunks_total": total,
                        "llm_calls": llm_calls,
                        "llm_max_calls": self.max_llm_calls,
                        "llm_skipped": llm_skipped,
                        "skipped_chunks": skipped_chunks,
                        "requirement_count": len(requirements),
                    }
                )
        return [r for r in requirements if not self._is_non_device_requirement(r)]

    def _is_non_device_requirement(self, req: Requirement) -> bool:
        if self._is_non_device_text(req.original_text):
            return True
        if req.attribute.startswith("unknown.") and not self._looks_like_device_config(req.original_text):
            return True
        return False

    def _extract_chunk(
        self,
        tender_id: str,
        chunk: TenderChunk,
        section: Dict[str, Any],
        can_use_llm: bool = True,
    ) -> List[Requirement]:
        self._last_extraction_used_llm = False
        regex_requirements = self._regex_extract(tender_id, chunk, section)
        if not can_use_llm or not self.llm or not self._should_use_llm(chunk, section, regex_requirements):
            return regex_requirements

        llm_requirements = []
        try:
            llm_requirements = self._llm_extract(tender_id, chunk, section)
        except Exception as exc:
            logger.warning("LLM requirement extraction failed for chunk %s: %s", chunk.chunk_id, exc)
        if llm_requirements:
            self._last_extraction_used_llm = True
            return self._merge_requirements(llm_requirements, regex_requirements)
        return regex_requirements

    def _should_use_llm(
        self,
        chunk: TenderChunk,
        section: Dict[str, Any],
        regex_requirements: List[Requirement],
    ) -> bool:
        text = f"{chunk.section_heading or ''}\n{chunk.text}"
        if len(text.strip()) < 30:
            return False
        if not self._looks_like_device_config(text):
            return False
        if self._is_non_device_text(text):
            return False
        if section.get("section_type") in {"admin", "pricing"}:
            return False
        if section.get("section_type") == "technical_requirements":
            return True
        if regex_requirements:
            return False
        keyword_hits = self._keyword_hits(text)
        if keyword_hits < 2:
            return False
        return True

    def _keyword_hits(self, text: str) -> int:
        lower = text.lower()
        requirement_words = [
            "mindestens", "mind.", "min.", "maximal", "max.", "muss", "erforderlich",
            "zwingend", "ausschluss", "optional",
        ]
        keywords = requirement_words + self.DEVICE_KEYWORDS
        return sum(1 for keyword in keywords if keyword in lower)

    def _looks_like_device_config(self, text: str) -> bool:
        lower = text.lower()
        return any(keyword in lower for keyword in self.DEVICE_KEYWORDS)

    def _is_non_device_text(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pattern, lower) for pattern in self.NON_DEVICE_PATTERNS)

    def _merge_requirements(
        self,
        primary: List[Requirement],
        fallback: List[Requirement],
    ) -> List[Requirement]:
        seen = {(req.attribute, req.original_text.strip().lower()) for req in primary}
        merged = list(primary)
        for req in fallback:
            key = (req.attribute, req.original_text.strip().lower())
            if key not in seen:
                merged.append(req)
                seen.add(key)
        return merged

    def _llm_extract(self, tender_id: str, chunk: TenderChunk, section: Dict[str, Any]) -> List[Requirement]:
        prompt = f"""Du extrahierst Anforderungen aus Ausschreibungen für Laptop/PC-Hardware.
Gib ausschließlich valides JSON zurück.
Nutze die vorgegebene Attribut-Taxonomie. Wenn kein Attribut passt, verwende unknown.<kurzer_name>.
Extrahiere nur Anforderungen, die im Text stehen oder eindeutig daraus folgen. Erfinde keine Anforderungen.
Setze needs_review=true bei confidence < 0.75, unknown.*, Mehrdeutigkeit, impliziter Interpretation oder unklarem Operator.
Extrahiere ausschließlich Geräte-Konfigurationsmerkmale, die gegen Produktdatenblätter prüfbar sind: CPU, RAM, Speicher, Grafik, Display, Anschlüsse, Netzwerk/Funk, Eingabe, Kamera, Sensoren, Sicherheit/BIOS, Akku, Netzteil, Betriebssystem, produktbezogene Garantie/Datenträger.
Ignoriere Mengen, Preispositionen, Vergabehinweise, Liefer-/Vertragsbedingungen, Bewertungspunkte, Servicekonzepte und Kopf-/Fußzeilen.

Section-Kontext: {json.dumps(section, ensure_ascii=False)}
Chunk-ID: {chunk.chunk_id}
Seite: {chunk.page_number}
Text:
{chunk.text[:6000]}

JSON-Schema:
{{"requirements":[{{"product_group":"string|null","requirement_type":"must|should|optional|scored|service|contract|unknown","attribute":"string","operator":">=|<=|=|contains|exists|one_of|compatible_with","value":"any","unit":"string|null","original_text":"string","source_page":1,"confidence":0.0,"needs_review":true,"points":null,"rationale":"string|null"}}]}}"""
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        data = self._parse_json(content)
        requirements = []
        for item in data.get("requirements", []):
            operator = self._normalize_operator(item.get("operator"))
            if operator is None:
                logger.warning(
                    "Skipping LLM requirement with unsupported operator %r for chunk %s",
                    item.get("operator"),
                    chunk.chunk_id,
                )
                continue
            item["operator"] = operator
            item["tender_id"] = tender_id
            item["source_chunk_id"] = chunk.chunk_id
            if item.get("attribute", "").startswith("unknown."):
                item["needs_review"] = True
            try:
                req = Requirement(**item)
                if self._is_non_device_text(req.original_text):
                    logger.info("Skipping non-device LLM requirement: %s", req.original_text[:80])
                    continue
                if req.attribute.startswith("unknown.") and not self._looks_like_device_config(req.original_text):
                    logger.info("Skipping unknown non-device requirement: %s", req.original_text[:80])
                    continue
                requirements.append(req)
            except Exception as exc:
                logger.warning("Skipping invalid LLM requirement for chunk %s: %s", chunk.chunk_id, exc)
        return requirements

    def _normalize_operator(self, value: Any) -> Optional[str]:
        if value in {operator.value for operator in Operator}:
            return value
        aliases = {
            "=>": Operator.GTE.value,
            "gte": Operator.GTE.value,
            "min": Operator.GTE.value,
            "=<": Operator.LTE.value,
            "lte": Operator.LTE.value,
            "max": Operator.LTE.value,
            "==": Operator.EQ.value,
            "eq": Operator.EQ.value,
            "includes": Operator.CONTAINS.value,
            "contains_any": Operator.CONTAINS.value,
            "present": Operator.EXISTS.value,
            "in": Operator.ONE_OF.value,
            "compatible": Operator.COMPATIBLE_WITH.value,
        }
        return aliases.get(str(value).strip().lower())

    def _regex_extract(self, tender_id: str, chunk: TenderChunk, section: Dict[str, Any]) -> List[Requirement]:
        requirements: List[Requirement] = []
        candidates = self._candidate_sentences(chunk.text)
        for sentence in candidates:
            parsed = self._parse_sentence(sentence)
            if not parsed:
                continue
            attr_guess = normalize_attribute(sentence)
            confidence = 0.68 if attr_guess.needs_review else 0.82
            needs_review = attr_guess.needs_review or confidence < 0.75
            requirements.append(
                Requirement(
                    tender_id=tender_id,
                    product_group=section.get("product_group"),
                    requirement_type=self._requirement_type(sentence, section.get("section_type")),
                    attribute=attr_guess.attribute,
                    operator=parsed["operator"],
                    value=parsed["value"],
                    unit=parsed.get("unit"),
                    original_text=sentence,
                    source_page=chunk.page_number,
                    source_chunk_id=chunk.chunk_id,
                    confidence=confidence,
                    needs_review=needs_review,
                    rationale="Regelbasiert aus Signalworten und Wert/Einheit extrahiert.",
                )
            )
        return requirements

    def _candidate_sentences(self, text: str) -> List[str]:
        lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
        sentences: List[str] = []
        for line in lines:
            if self._is_non_device_text(line) or not self._looks_like_device_config(line):
                continue
            if len(line) > 240:
                sentences.extend(
                    part.strip()
                    for part in re.split(r"(?<=[.;])\s+", line)
                    if part.strip() and self._looks_like_device_config(part) and not self._is_non_device_text(part)
                )
            else:
                sentences.append(line)
        keywords = r"(mindestens|mind\.?|maximal|max\.?|muss|erforderlich|zwingend|optional|garantie|datenträger|ram|arbeitsspeicher|ddr|sodimm|ssd|nvme|display|bildschirm|prozessor|cpu|vpro|wifi|wi-fi|wlan|bluetooth|hdmi|usb|thunderbolt|rj45|ethernet|akku|battery|webcam|kamera|tastatur|touchpad|sensor|bios|tpm|windows|netzteil)"
        return [line for line in sentences if re.search(keywords, line, re.I)]

    def _parse_sentence(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        number = re.search(r"(\d+(?:[,.]\d+)?)\s*(gb|tb|kg|g|wh|nits|cd/m²|cd/m2|zoll|inch|\"|jahre|monate|mp|%)?", lower)
        if re.search(r"(muss vorhanden|erforderlich|zwingend|vorhanden sein)", lower) and not number:
            return {"operator": Operator.EXISTS, "value": True, "unit": None}
        if not number:
            return {"operator": Operator.CONTAINS, "value": text, "unit": None} if re.search(r"(muss|erforderlich|zwingend)", lower) else None

        value = float(number.group(1).replace(",", "."))
        if value.is_integer():
            value = int(value)
        unit = number.group(2)
        if unit == '"':
            unit = "inch"

        if re.search(r"(maximal|max\.|höchstens|bis zu)", lower):
            operator = Operator.LTE
        elif re.search(r"(mindestens|mind\.|min\.|ab)", lower):
            operator = Operator.GTE
        else:
            operator = Operator.EQ if re.search(r"(muss|zwingend|erforderlich)", lower) else Operator.CONTAINS
        return {"operator": operator, "value": value, "unit": unit}

    def _requirement_type(self, text: str, section_type: Optional[str]) -> RequirementType:
        lower = text.lower()
        if section_type == "service":
            return RequirementType.SERVICE
        if section_type == "contract":
            return RequirementType.CONTRACT
        if "punkte" in lower or "bewertet" in lower:
            return RequirementType.SCORED
        if "optional" in lower or "kann" in lower:
            return RequirementType.OPTIONAL
        if re.search(r"(muss|zwingend|erforderlich|mindestens|mind\.|maximal|max\.|ausschluss)", lower):
            return RequirementType.MUST
        return RequirementType.UNKNOWN

    def _parse_json(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                raise
            return json.loads(match.group(0))
