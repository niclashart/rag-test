"""ProductFact extraction from existing ProductSpecification records."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
import re

from src.tenders.schemas import ProductFact, Requirement

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from database.models import Chunk, ProductSpecification


class ProductFactStore:
    """Build canonical facts from current product specs."""

    FIELD_MAP = {
        "display_brightness_nits": ("display.brightness", "nits"),
        "screen_to_body_ratio": ("display.aspect_ratio", "%"),
        "weight_kg": ("unknown.weight", "kg"),
        "ram_max_gb": ("memory.capacity", "GB"),
        "storage_max_tb": ("storage.capacity", "TB"),
        "battery_wh": ("battery.capacity", "Wh"),
        "display_size_inch": ("display.size", "inch"),
        "display_resolution": ("display.resolution", None),
    }

    RAW_PATHS = {
        "cpu.model": ["prozessor", "processor", "cpu"],
        "memory.type": ["ram", "memory"],
        "memory.slots": ["ram", "memory"],
        "storage.interface": ["speicher", "storage"],
        "ports.thunderbolt4": ["anschlüsse", "ports"],
        "ports.usb_a": ["anschlüsse", "ports"],
        "ports.usb_c": ["anschlüsse", "ports"],
        "ports.hdmi": ["anschlüsse", "ports"],
        "ports.rj45_native": ["anschlüsse", "ports"],
        "network.wifi": ["konnektivität", "connectivity"],
        "network.bluetooth": ["konnektivität", "connectivity"],
        "webcam.resolution": ["kamera", "camera"],
        "warranty.years": ["garantie", "warranty"],
    }

    def facts_for_product(self, product: "ProductSpecification") -> List[ProductFact]:
        facts: List[ProductFact] = []
        product_id = str(product.id)
        model = product.product_name
        source_document = str(product.document_id)

        for field, (attribute, unit) in self.FIELD_MAP.items():
            value = getattr(product, field, None)
            if value is not None:
                facts.append(self._fact(product_id, model, attribute, value, unit, field, source_document, 0.9))

        raw_specs = product.raw_specs or {}
        for attribute, keys in self.RAW_PATHS.items():
            value = self._find_value(raw_specs, keys)
            if value:
                facts.append(self._fact(product_id, model, attribute, value, None, str(value), source_document, 0.72))

        return facts

    def facts_for_products(self, products: Iterable["ProductSpecification"]) -> Dict[str, List[ProductFact]]:
        return {str(product.id): self.facts_for_product(product) for product in products}

    def _fact(
        self,
        product_id: str,
        model: str,
        attribute: str,
        value: Any,
        unit: str | None,
        source_text: str,
        source_document: str,
        confidence: float,
    ) -> ProductFact:
        return ProductFact(
            product_id=product_id,
            model=model,
            attribute=attribute,
            value=value,
            unit=unit,
            source_text=source_text,
            source_document=source_document,
            confidence=confidence,
        )

    def _find_value(self, data: Dict[str, Any], keys: List[str]) -> Any:
        if not isinstance(data, dict):
            return None
        for key, value in data.items():
            lower = key.lower()
            if any(candidate in lower for candidate in keys):
                return value
            if isinstance(value, dict):
                nested = self._find_value(value, keys)
                if nested:
                    return nested
        return None


class ProductFactExtractionAgent:
    """Targeted ProductFact extraction from existing product document chunks."""

    ATTRIBUTE_KEYWORDS = {
        "cpu.model": ["processor", "prozessor", "cpu", "intel", "amd", "ryzen", "core"],
        "cpu.vpro": ["vpro", "vpro", "intel vpro"],
        "memory.capacity": ["ram", "memory", "arbeitsspeicher", "gb"],
        "memory.type": ["ddr", "lpddr", "ram", "memory"],
        "memory.slots": ["slot", "sodimm", "so-dimm"],
        "storage.capacity": ["ssd", "storage", "massenspeicher", "speicher", "tb", "gb"],
        "storage.interface": ["nvme", "pcie", "sata", "ssd"],
        "display.size": ["display", "screen", "bildschirm", "inch", "zoll"],
        "display.resolution": ["resolution", "auflösung", "wuxga", "fhd", "1920", "2560"],
        "display.brightness": ["brightness", "helligkeit", "nits", "cd/m"],
        "display.color_coverage.srgb": ["srgb", "color", "farbe"],
        "display.anti_glare": ["anti-glare", "antiglare", "entspiegelt"],
        "display.touch": ["touch", "touchscreen"],
        "ports.thunderbolt4": ["thunderbolt", "tb4"],
        "ports.usb_a": ["usb-a", "usb a", "type-a"],
        "ports.usb_c": ["usb-c", "usb c", "type-c"],
        "ports.hdmi": ["hdmi"],
        "ports.rj45_native": ["rj45", "ethernet", "lan"],
        "network.wifi": ["wifi", "wi-fi", "wlan"],
        "network.bluetooth": ["bluetooth"],
        "network.wake_on_lan": ["wake on lan", "wol"],
        "network.pxe": ["pxe"],
        "webcam.resolution": ["webcam", "camera", "kamera", "mp"],
        "battery.capacity": ["battery", "akku", "wh"],
        "battery.runtime": ["runtime", "laufzeit", "hours", "stunden"],
        "warranty.years": ["warranty", "garantie", "years", "jahre"],
        "unknown.weight": ["weight", "gewicht", "kg", "lbs"],
    }

    def extract_missing_facts(
        self,
        product: "ProductSpecification",
        requirements: List[Requirement],
        existing_facts: List[ProductFact],
        db: "Session",
    ) -> List[ProductFact]:
        existing_attributes = {fact.attribute for fact in existing_facts}
        missing_attributes = sorted({
            req.attribute for req in requirements
            if req.attribute not in existing_attributes and not req.attribute.startswith("unknown.")
        })
        extracted: List[ProductFact] = []
        for attribute in missing_attributes:
            chunk = self._best_chunk(product.document_id, attribute, db)
            if not chunk:
                continue
            fact = self._extract_fact_from_text(product, attribute, chunk)
            if fact:
                extracted.append(fact)
        return extracted

    def _best_chunk(self, document_id: int, attribute: str, db: "Session") -> Optional["Chunk"]:
        from database.models import Chunk

        keywords = self.ATTRIBUTE_KEYWORDS.get(attribute, attribute.split("."))
        chunks = db.query(Chunk).filter(Chunk.document_id == document_id).all()
        scored = []
        for chunk in chunks:
            text = chunk.text.lower()
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            if score:
                scored.append((score, len(chunk.text), chunk))
        if not scored:
            return None
        return sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0][2]

    def _extract_fact_from_text(self, product: "ProductSpecification", attribute: str, chunk: "Chunk") -> Optional[ProductFact]:
        text = chunk.text
        value = self._value_for_attribute(attribute, text)
        if value is None:
            return None
        unit = self._unit_for_attribute(attribute, text)
        return ProductFact(
            product_id=str(product.id),
            model=product.product_name,
            attribute=attribute,
            value=value,
            unit=unit,
            source_text=text[:1200],
            source_document=str(product.document_id),
            source_page=chunk.page_number,
            confidence=0.68,
        )

    def _value_for_attribute(self, attribute: str, text: str):
        lower = text.lower()
        if attribute in {"display.anti_glare", "display.touch", "ports.thunderbolt4", "ports.hdmi", "ports.rj45_native", "network.wake_on_lan", "network.pxe", "cpu.vpro"}:
            return True
        if attribute == "memory.capacity":
            return self._max_number_before_unit(lower, "gb")
        if attribute == "storage.capacity":
            tb = self._max_number_before_unit(lower, "tb")
            if tb is not None:
                return tb
            return self._max_number_before_unit(lower, "gb")
        if attribute == "display.size":
            return self._first_number_near(lower, ["inch", "zoll", '"'])
        if attribute == "display.brightness":
            return self._first_number_near(lower, ["nits", "cd/m"])
        if attribute == "battery.capacity":
            return self._first_number_near(lower, ["wh"])
        if attribute == "warranty.years":
            return self._first_number_near(lower, ["year", "years", "jahr", "jahre"])
        if attribute == "unknown.weight":
            return self._first_number_near(lower, ["kg", "lbs"])
        if attribute == "network.wifi":
            match = re.search(r"wi-?fi\s*\d+[e]?", text, re.I)
            return match.group(0) if match else text[:160]
        if attribute == "network.bluetooth":
            match = re.search(r"bluetooth\s*\d+(?:\.\d+)?", text, re.I)
            return match.group(0) if match else text[:160]
        if attribute == "display.resolution":
            match = re.search(r"\d{3,4}\s*x\s*\d{3,4}", text, re.I)
            return match.group(0) if match else text[:160]
        return text[:240]

    def _unit_for_attribute(self, attribute: str, text: str) -> Optional[str]:
        lower = text.lower()
        if attribute == "memory.capacity":
            return "GB"
        if attribute == "storage.capacity":
            return "TB" if "tb" in lower else "GB"
        if attribute == "display.size":
            return "inch"
        if attribute == "display.brightness":
            return "nits"
        if attribute == "battery.capacity":
            return "Wh"
        if attribute == "warranty.years":
            return "years"
        if attribute == "unknown.weight":
            return "kg" if "kg" in lower else "lbs"
        return None

    def _max_number_before_unit(self, text: str, unit: str) -> Optional[float]:
        values = [float(match.replace(",", ".")) for match in re.findall(rf"(\d+(?:[,.]\d+)?)\s*{unit}\b", text)]
        if not values:
            return None
        value = max(values)
        return int(value) if value.is_integer() else value

    def _first_number_near(self, text: str, units: List[str]) -> Optional[float]:
        for unit in units:
            match = re.search(rf"(\d+(?:[,.]\d+)?)\s*{re.escape(unit)}", text)
            if match:
                value = float(match.group(1).replace(",", "."))
                return int(value) if value.is_integer() else value
        return None
