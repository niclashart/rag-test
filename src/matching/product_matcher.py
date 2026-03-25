"""Product matcher for matching products with requirements."""
from sqlalchemy.orm import Session
from database.models import ProductSpecification, RequirementSpecification
from typing import List, Dict
from logging_config.logger import get_logger

logger = get_logger(__name__)


class ProductMatcher:
    """Matcht Produkte mit Anforderungen."""
    
    def match_requirements_to_products(
        self,
        requirement_id: int,
        db: Session
    ) -> List[Dict]:
        """Findet alle Produkte, die eine bestimmte Anforderung erfüllen."""
        
        requirement = db.query(RequirementSpecification).filter(
            RequirementSpecification.id == requirement_id
        ).first()
        
        if not requirement:
            logger.warning(f"Requirement {requirement_id} not found")
            return []
        
        logger.info(f"Matching products for requirement {requirement_id}")
        logger.info(f"Requirement details: min_brightness={requirement.min_display_brightness_nits}, "
                   f"min_ratio={requirement.min_screen_to_body_ratio}, "
                   f"max_weight={requirement.max_weight_kg}, "
                   f"min_ram={requirement.min_ram_gb}, "
                   f"min_storage={requirement.min_storage_tb}, "
                   f"min_battery={requirement.min_battery_wh}")
        
        # Starte mit allen Produkten
        query = db.query(ProductSpecification)
        
        # Filter nach Display-Helligkeit (NULL-Werte werden ausgeschlossen)
        if requirement.min_display_brightness_nits:
            query = query.filter(
                ProductSpecification.display_brightness_nits.isnot(None),
                ProductSpecification.display_brightness_nits >= requirement.min_display_brightness_nits
            )
            logger.debug(f"Filter: min_brightness >= {requirement.min_display_brightness_nits}")
        
        # Filter nach Screen-to-Body Ratio (NULL-Werte werden ausgeschlossen)
        if requirement.min_screen_to_body_ratio:
            query = query.filter(
                ProductSpecification.screen_to_body_ratio.isnot(None),
                ProductSpecification.screen_to_body_ratio >= requirement.min_screen_to_body_ratio
            )
            logger.debug(f"Filter: min_screen_to_body_ratio >= {requirement.min_screen_to_body_ratio}")
        
        # Filter nach Gewicht (NULL-Werte werden ausgeschlossen)
        if requirement.max_weight_kg:
            query = query.filter(
                ProductSpecification.weight_kg.isnot(None),
                ProductSpecification.weight_kg <= requirement.max_weight_kg
            )
            logger.debug(f"Filter: max_weight_kg <= {requirement.max_weight_kg}")
        
        # Filter nach RAM (NULL-Werte werden ausgeschlossen)
        if requirement.min_ram_gb:
            query = query.filter(
                ProductSpecification.ram_max_gb.isnot(None),
                ProductSpecification.ram_max_gb >= requirement.min_ram_gb
            )
            logger.debug(f"Filter: min_ram_gb >= {requirement.min_ram_gb}")
        
        # Filter nach Speicher (NULL-Werte werden ausgeschlossen)
        if requirement.min_storage_tb:
            query = query.filter(
                ProductSpecification.storage_max_tb.isnot(None),
                ProductSpecification.storage_max_tb >= requirement.min_storage_tb
            )
            logger.debug(f"Filter: min_storage_tb >= {requirement.min_storage_tb}")
        
        # Filter nach Akku (NULL-Werte werden ausgeschlossen)
        if requirement.min_battery_wh:
            query = query.filter(
                ProductSpecification.battery_wh.isnot(None),
                ProductSpecification.battery_wh >= requirement.min_battery_wh
            )
            logger.debug(f"Filter: min_battery_wh >= {requirement.min_battery_wh}")
        
        # Filter nach Display-Größe (NULL-Werte werden ausgeschlossen)
        if requirement.display_size_inch_min:
            query = query.filter(
                ProductSpecification.display_size_inch.isnot(None),
                ProductSpecification.display_size_inch >= requirement.display_size_inch_min
            )
            logger.debug(f"Filter: display_size_inch >= {requirement.display_size_inch_min}")
        if requirement.display_size_inch_max:
            query = query.filter(
                ProductSpecification.display_size_inch.isnot(None),
                ProductSpecification.display_size_inch <= requirement.display_size_inch_max
            )
            logger.debug(f"Filter: display_size_inch <= {requirement.display_size_inch_max}")
        
        matching_products = query.all()
        logger.info(f"Found {len(matching_products)} matching products after filtering")
        
        # Prüfe ob überhaupt Anforderungen vorhanden sind
        has_any_requirement = any([
            requirement.min_display_brightness_nits,
            requirement.min_screen_to_body_ratio,
            requirement.max_weight_kg,
            requirement.min_ram_gb,
            requirement.min_storage_tb,
            requirement.min_battery_wh,
            requirement.display_size_inch_min,
            requirement.display_size_inch_max,
            requirement.additional_requirements
        ])
        
        if not has_any_requirement:
            logger.warning(f"No requirements found for requirement {requirement_id} - cannot match products")
            return []  # Keine Anforderungen = keine Matches
        
        # Berechne Match-Score für jedes Produkt
        results = []
        for product in matching_products:
            score = self._calculate_match_score(product, requirement)
            results.append({
                "product": product,
                "match_score": score,
                "product_name": product.product_name,
                "brand": product.brand,
                "display_brightness_nits": product.display_brightness_nits,
                "screen_to_body_ratio": product.screen_to_body_ratio,
                "weight_kg": product.weight_kg,
                "ram_max_gb": product.ram_max_gb,
                "storage_max_tb": product.storage_max_tb,
                "battery_wh": product.battery_wh,
                "display_size_inch": product.display_size_inch,
                "display_resolution": product.display_resolution,
                "document_id": product.document_id,
                "product_spec_id": product.id
            })
        
        # Sortiere nach Match-Score (höchster zuerst)
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return results
    
    def _calculate_match_score(self, product: ProductSpecification, requirement: RequirementSpecification) -> float:
        """Berechnet einen Match-Score zwischen 0 und 1 basierend auf erfüllten Anforderungen."""
        
        # Zähle alle Anforderungen und wie viele erfüllt sind
        total_requirements = 0
        fulfilled_requirements = 0
        bonus_points = 0.0
        
        # Prüfe Display-Helligkeit
        if requirement.min_display_brightness_nits:
            total_requirements += 1
            if product.display_brightness_nits is not None:
                if product.display_brightness_nits >= requirement.min_display_brightness_nits:
                    fulfilled_requirements += 1
                    # Bonus für deutlich bessere Helligkeit
                    if product.display_brightness_nits > requirement.min_display_brightness_nits * 1.2:
                        bonus_points += 0.05
        
        # Prüfe Screen-to-Body Ratio
        if requirement.min_screen_to_body_ratio:
            total_requirements += 1
            if product.screen_to_body_ratio is not None:
                if product.screen_to_body_ratio >= requirement.min_screen_to_body_ratio:
                    fulfilled_requirements += 1
                    # Bonus für besseres Ratio
                    if product.screen_to_body_ratio > requirement.min_screen_to_body_ratio * 1.1:
                        bonus_points += 0.03
        
        # Prüfe Gewicht
        if requirement.max_weight_kg:
            total_requirements += 1
            if product.weight_kg is not None:
                if product.weight_kg <= requirement.max_weight_kg:
                    fulfilled_requirements += 1
                    # Bonus für deutlich leichteres Gerät
                    if product.weight_kg < requirement.max_weight_kg * 0.9:
                        bonus_points += 0.05
        
        # Prüfe RAM
        if requirement.min_ram_gb:
            total_requirements += 1
            if product.ram_max_gb is not None:
                if product.ram_max_gb >= requirement.min_ram_gb:
                    fulfilled_requirements += 1
                    # Bonus für mehr RAM
                    if product.ram_max_gb > requirement.min_ram_gb * 1.5:
                        bonus_points += 0.03
        
        # Prüfe Speicher
        if requirement.min_storage_tb:
            total_requirements += 1
            if product.storage_max_tb is not None:
                if product.storage_max_tb >= requirement.min_storage_tb:
                    fulfilled_requirements += 1
                    # Bonus für mehr Speicher
                    if product.storage_max_tb > requirement.min_storage_tb * 1.5:
                        bonus_points += 0.03
        
        # Prüfe Akku
        if requirement.min_battery_wh:
            total_requirements += 1
            if product.battery_wh is not None:
                if product.battery_wh >= requirement.min_battery_wh:
                    fulfilled_requirements += 1
                    # Bonus für größeren Akku
                    if product.battery_wh > requirement.min_battery_wh * 1.2:
                        bonus_points += 0.03
        
        # Prüfe Display-Größe (min)
        if requirement.display_size_inch_min:
            total_requirements += 1
            if product.display_size_inch is not None:
                if product.display_size_inch >= requirement.display_size_inch_min:
                    fulfilled_requirements += 1
        
        # Prüfe Display-Größe (max)
        if requirement.display_size_inch_max:
            total_requirements += 1
            if product.display_size_inch is not None:
                if product.display_size_inch <= requirement.display_size_inch_max:
                    fulfilled_requirements += 1
        
        # Prüfe zusätzliche Anforderungen aus additional_requirements
        additional_matches = self._match_additional_requirements(
            product, requirement.additional_requirements if requirement.additional_requirements else {}
        )
        total_requirements += additional_matches["total"]
        fulfilled_requirements += additional_matches["fulfilled"]
        
        # Berechne Basis-Score: Anteil erfüllter Anforderungen
        if total_requirements == 0:
            # Keine Anforderungen definiert - das ist ein Problem!
            logger.warning(f"No requirements found for requirement {requirement.id} - cannot calculate match score")
            return 0.0  # Score 0 wenn keine Anforderungen vorhanden sind
        
        base_score = fulfilled_requirements / total_requirements
        
        # Füge Bonus-Punkte hinzu (maximal bis 1.0)
        final_score = min(base_score + bonus_points, 1.0)
        
        logger.debug(f"Match score: {fulfilled_requirements}/{total_requirements} fulfilled "
                    f"(structured: {fulfilled_requirements - additional_matches['fulfilled']}, "
                    f"additional: {additional_matches['fulfilled']}/{additional_matches['total']}), "
                    f"base={base_score:.2f}, bonus={bonus_points:.2f}, final={final_score:.2f}")
        
        return final_score
    
    def _match_additional_requirements(self, product: ProductSpecification, additional_reqs: dict) -> dict:
        """Prüft zusätzliche Anforderungen gegen Produktspezifikationen."""
        if not additional_reqs:
            return {"total": 0, "fulfilled": 0}
        
        total = 0
        fulfilled = 0
        
        # Hole alle Produktspezifikationen aus raw_specs
        product_specs = product.raw_specs if product.raw_specs else {}
        
        # Iteriere über alle zusätzlichen Anforderungen
        for req_key, req_value in additional_reqs.items():
            if req_value is None:
                continue
            
            total += 1
            
            # Versuche die entsprechende Spezifikation im Produkt zu finden
            matched = self._check_requirement_match(req_key, req_value, product_specs, product)
            
            if matched:
                fulfilled += 1
                logger.debug(f"Additional requirement matched: {req_key} = {req_value}")
            else:
                logger.debug(f"Additional requirement NOT matched: {req_key} = {req_value}")
        
        return {"total": total, "fulfilled": fulfilled}
    
    def _check_requirement_match(self, req_key: str, req_value: any, product_specs: dict, product: ProductSpecification) -> bool:
        """Prüft ob eine einzelne Anforderung vom Produkt erfüllt wird."""
        
        # Normalisiere Schlüsselnamen (verschiedene Schreibweisen)
        req_key_lower = req_key.lower()
        
        # Suche nach passenden Feldern in den Produktspezifikationen
        # Prüfe sowohl in product_specs (raw_specs) als auch in den strukturierten Feldern
        
        # 1. Direkter Match im product_specs
        if req_key in product_specs:
            return self._compare_values(req_value, product_specs[req_key])
        
        # 2. Suche nach ähnlichen Schlüsseln (case-insensitive)
        for spec_key, spec_value in product_specs.items():
            if req_key_lower in spec_key.lower() or spec_key.lower() in req_key_lower:
                if self._compare_values(req_value, spec_value):
                    return True
        
        # 3. Prüfe verschachtelte Strukturen (z.B. "prozessor.modell" oder verschachtelte Objekte)
        if isinstance(product_specs, dict):
            # Suche in verschachtelten Objekten
            for key, value in product_specs.items():
                if isinstance(value, dict):
                    # Prüfe ob req_key in diesem verschachtelten Objekt ist
                    if req_key_lower in key.lower():
                        # Suche in den verschachtelten Werten
                        for sub_key, sub_value in value.items():
                            if self._compare_values(req_value, sub_value):
                                return True
                    # Prüfe auch direkt in verschachtelten Objekten
                    if req_key in value:
                        if self._compare_values(req_value, value[req_key]):
                            return True
        
        # 4. Spezielle Mappings für häufige Anforderungen
        # Prozessor-Anforderungen
        if any(x in req_key_lower for x in ["prozessor", "processor", "cpu"]):
            processor_specs = self._get_nested_value(product_specs, ["prozessor", "processor", "cpu"])
            if processor_specs:
                processor_text = str(processor_specs).lower()
                req_text = str(req_value).lower()
                # Prüfe ob Prozessor-Anforderung im Produkt enthalten ist
                if req_text in processor_text or any(word in processor_text for word in req_text.split() if len(word) > 3):
                    return True
        
        # WiFi-Anforderungen
        if any(x in req_key_lower for x in ["wifi", "wi-fi", "wlan"]):
            wifi_specs = self._get_nested_value(product_specs, ["konnektivität", "connectivity", "wifi", "wi-fi"])
            if wifi_specs:
                wifi_text = str(wifi_specs).lower()
                req_text = str(req_value).lower()
                if req_text in wifi_text:
                    return True
        
        # Bluetooth-Anforderungen
        if "bluetooth" in req_key_lower:
            bt_specs = self._get_nested_value(product_specs, ["konnektivität", "connectivity", "bluetooth"])
            if bt_specs:
                bt_text = str(bt_specs).lower()
                req_text = str(req_value).lower()
                if req_text in bt_text:
                    return True
        
        # Port-Anforderungen
        if any(x in req_key_lower for x in ["port", "anschluss", "hdmi", "thunderbolt", "usb"]):
            ports_specs = self._get_nested_value(product_specs, ["anschlüsse", "ports", "anschluss"])
            if ports_specs:
                if isinstance(ports_specs, dict):
                    for port_key, port_value in ports_specs.items():
                        if req_key_lower in port_key.lower() or port_key.lower() in req_key_lower:
                            if self._compare_values(req_value, port_value):
                                return True
                else:
                    ports_text = str(ports_specs).lower()
                    req_text = str(req_value).lower()
                    if req_text in ports_text:
                        return True
        
        return False
    
    def _get_nested_value(self, data: dict, keys: list) -> any:
        """Holt einen Wert aus verschachtelten Strukturen."""
        if not isinstance(data, dict):
            return None
        
        for key in keys:
            # Suche case-insensitive
            for k, v in data.items():
                if k.lower() == key.lower():
                    return v
        return None
    
    def _compare_values(self, requirement: any, product_value: any) -> bool:
        """Vergleicht einen Anforderungswert mit einem Produktwert."""
        if product_value is None:
            return False
        
        # String-Vergleich (case-insensitive)
        if isinstance(requirement, str) and isinstance(product_value, str):
            req_lower = requirement.lower()
            prod_lower = product_value.lower()
            # Exakter Match oder Teilstring-Match
            return req_lower == prod_lower or req_lower in prod_lower or prod_lower in req_lower
        
        # Numerischer Vergleich
        if isinstance(requirement, (int, float)) and isinstance(product_value, (int, float)):
            # Für Mindestanforderungen: Produkt >= Anforderung
            return product_value >= requirement
        
        # Boolean-Vergleich
        if isinstance(requirement, bool) and isinstance(product_value, bool):
            return requirement == product_value
        
        # Listen-Vergleich
        if isinstance(requirement, list) and isinstance(product_value, list):
            return any(self._compare_values(req_item, prod_item) for req_item in requirement for prod_item in product_value)
        
        # Fallback: String-Konvertierung und Vergleich
        req_str = str(requirement).lower()
        prod_str = str(product_value).lower()
        return req_str in prod_str or prod_str in req_str
