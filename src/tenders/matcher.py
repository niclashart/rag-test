"""Deterministic product matcher for canonical requirements."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.tenders.schemas import (
    Eligibility,
    MatchResult,
    MatchStatus,
    Operator,
    ProductFact,
    Requirement,
    RequirementMatch,
    RequirementStatus,
    RequirementType,
)


class UnitNormalizer:
    """Normalize numeric values into canonical units."""

    _ALIASES = {
        "gb": "GB",
        "gbyte": "GB",
        "tb": "TB",
        "inch": "inch",
        "zoll": "inch",
        '"': "inch",
        "wh": "Wh",
        "nits": "nits",
        "nit": "nits",
        "cd/m²": "nits",
        "cd/m2": "nits",
        "jahre": "years",
        "jahr": "years",
        "years": "years",
        "year": "years",
        "monate": "months",
        "months": "months",
        "kg": "kg",
        "g": "g",
    }

    @classmethod
    def normalize(cls, value: Any, unit: Optional[str]) -> Tuple[Any, Optional[str]]:
        if value is None or unit is None:
            return value, unit
        unit_norm = cls._ALIASES.get(str(unit).strip().lower(), unit)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value, unit_norm

        if unit_norm == "TB":
            return numeric * 1000, "GB"
        if unit_norm == "g":
            return numeric / 1000, "kg"
        if unit_norm == "months":
            return numeric / 12, "years"
        return numeric, unit_norm


class ProductMatcher:
    """Match approved canonical requirements against product facts."""

    def __init__(self, llm_judge: Any = None) -> None:
        self.llm_judge = llm_judge

    def match_product(
        self,
        tender_id: str,
        product_id: str,
        model: str,
        requirements: Iterable[Requirement],
        facts: Iterable[ProductFact],
    ) -> MatchResult:
        approved = [
            req for req in requirements
            if req.status in {RequirementStatus.APPROVED, RequirementStatus.EDITED}
        ]
        fact_index = self._index_facts(facts)
        results: List[RequirementMatch] = []

        must_passed = 0
        must_failed = 0
        unknown_count = 0
        score = 0.0
        max_score = 0.0

        for req in approved:
            fact = self._best_fact(req, fact_index)
            match = self._match_requirement(req, fact)
            results.append(match)

            if match.status == MatchStatus.UNKNOWN:
                unknown_count += 1

            if req.requirement_type == RequirementType.MUST:
                if match.status == MatchStatus.FULFILLED:
                    must_passed += 1
                elif match.status == MatchStatus.NOT_FULFILLED:
                    must_failed += 1
            elif req.requirement_type in {RequirementType.SCORED, RequirementType.OPTIONAL, RequirementType.SHOULD}:
                points = float(req.points or 1.0)
                max_score += points
                if match.status == MatchStatus.FULFILLED:
                    score += points

        if must_failed:
            eligibility = Eligibility.NOT_ELIGIBLE
        elif unknown_count and any(req.requirement_type == RequirementType.MUST for req in approved):
            must_unknown = any(
                req.requirement_type == RequirementType.MUST and res.status == MatchStatus.UNKNOWN
                for req, res in zip(approved, results)
            )
            eligibility = Eligibility.UNKNOWN if must_unknown else Eligibility.ELIGIBLE
        else:
            eligibility = Eligibility.ELIGIBLE

        return MatchResult(
            tender_id=tender_id,
            product_id=product_id,
            model=model,
            eligibility=eligibility,
            score=score,
            max_score=max_score,
            must_passed=must_passed,
            must_failed=must_failed,
            unknown_count=unknown_count,
            requirement_results=results,
        )

    def _index_facts(self, facts: Iterable[ProductFact]) -> Dict[str, List[ProductFact]]:
        index: Dict[str, List[ProductFact]] = {}
        for fact in facts:
            index.setdefault(fact.attribute, []).append(fact)
        return index

    def _best_fact(self, req: Requirement, fact_index: Dict[str, List[ProductFact]]) -> Optional[ProductFact]:
        candidates = fact_index.get(req.attribute, [])
        if not candidates:
            return None
        return sorted(candidates, key=lambda fact: fact.confidence, reverse=True)[0]

    def _match_requirement(self, req: Requirement, fact: Optional[ProductFact]) -> RequirementMatch:
        if fact is None:
            return self._result(req, None, MatchStatus.UNKNOWN, "Kein ProductFact für dieses Attribut vorhanden.")

        status, reason = self._compare(req, fact)
        if status == MatchStatus.UNKNOWN and self.llm_judge is not None:
            try:
                status, reason = self.llm_judge.judge(req, fact)
            except Exception as exc:
                reason = f"{reason} LLM-Judge fehlgeschlagen: {exc}"
        return self._result(req, fact, status, reason)

    def _result(
        self,
        req: Requirement,
        fact: Optional[ProductFact],
        status: MatchStatus,
        reason: str,
    ) -> RequirementMatch:
        return RequirementMatch(
            requirement_id=req.id,
            attribute=req.attribute,
            requirement_text=req.original_text,
            required_value=req.value,
            product_value=fact.value if fact else None,
            status=status,
            reason=reason,
            tender_source_page=req.source_page,
            product_source_text=fact.source_text if fact else None,
            product_source_document=fact.source_document if fact else None,
        )

    def _compare(self, req: Requirement, fact: ProductFact) -> Tuple[MatchStatus, str]:
        required, req_unit = UnitNormalizer.normalize(req.value, req.unit)
        actual, fact_unit = UnitNormalizer.normalize(fact.value, fact.unit)

        if req_unit and fact_unit and req_unit != fact_unit:
            return MatchStatus.UNKNOWN, f"Einheiten nicht vergleichbar: Anforderung {req_unit}, Produkt {fact_unit}."

        if req.operator in {Operator.GTE, Operator.LTE, Operator.EQ}:
            return self._compare_ordered(req.operator, required, actual, req_unit)
        if req.operator == Operator.CONTAINS:
            return self._contains(required, actual)
        if req.operator == Operator.EXISTS:
            exists = actual not in (None, False, "", [])
            expected = bool(required)
            status = MatchStatus.FULFILLED if exists == expected else MatchStatus.NOT_FULFILLED
            return status, f"Existenzprüfung: Produktwert ist {actual!r}."
        if req.operator == Operator.ONE_OF:
            values = required if isinstance(required, list) else [required]
            for value in values:
                status, _ = self._contains(value, actual)
                if status == MatchStatus.FULFILLED:
                    return status, f"Produktwert {actual!r} passt zu erlaubtem Wert {value!r}."
            return MatchStatus.NOT_FULFILLED, f"Produktwert {actual!r} ist nicht in {values!r}."
        if req.operator == Operator.COMPATIBLE_WITH:
            return self._compatible(required, actual)
        return MatchStatus.UNKNOWN, f"Operator {req.operator} nicht entscheidbar."

    def _compare_ordered(self, operator: Operator, required: Any, actual: Any, unit: Optional[str]) -> Tuple[MatchStatus, str]:
        req_num = self._to_number(required)
        actual_num = self._to_number(actual)
        unit_text = f" {unit}" if unit else ""

        if req_num is None or actual_num is None:
            return self._compatible(required, actual)

        if operator == Operator.GTE:
            ok = actual_num >= req_num
            symbol = ">="
        elif operator == Operator.LTE:
            ok = actual_num <= req_num
            symbol = "<="
        else:
            ok = actual_num == req_num
            symbol = "="

        status = MatchStatus.FULFILLED if ok else MatchStatus.NOT_FULFILLED
        return status, f"Produktwert {actual_num:g}{unit_text}; gefordert {symbol} {req_num:g}{unit_text}."

    def _contains(self, required: Any, actual: Any) -> Tuple[MatchStatus, str]:
        if actual is None:
            return MatchStatus.UNKNOWN, "Produktwert fehlt."
        req_text = self._normalize_text(required)
        actual_text = self._normalize_text(actual)
        if req_text in actual_text:
            return MatchStatus.FULFILLED, f"Produktwert enthält {required!r}."
        ratio = SequenceMatcher(None, req_text, actual_text).ratio()
        if ratio >= 0.82:
            return MatchStatus.FULFILLED, f"Produktwert ist fuzzy-kompatibel mit {required!r} (Ähnlichkeit {ratio:.2f})."
        return MatchStatus.NOT_FULFILLED, f"Produktwert {actual!r} enthält {required!r} nicht."

    def _compatible(self, required: Any, actual: Any) -> Tuple[MatchStatus, str]:
        req_text = self._normalize_text(required)
        actual_text = self._normalize_text(actual)
        if not actual_text:
            return MatchStatus.UNKNOWN, "Produktwert fehlt."
        if req_text == actual_text or req_text in actual_text or actual_text in req_text:
            return MatchStatus.FULFILLED, f"Textwerte sind kompatibel: {actual!r}."
        if self._wifi_rank(actual_text) and self._wifi_rank(req_text):
            ok = self._wifi_rank(actual_text) >= self._wifi_rank(req_text)
            status = MatchStatus.FULFILLED if ok else MatchStatus.NOT_FULFILLED
            return status, f"Wi-Fi-Version Produkt {actual!r}, gefordert {required!r}."
        ratio = SequenceMatcher(None, req_text, actual_text).ratio()
        if ratio >= 0.86:
            return MatchStatus.FULFILLED, f"Fuzzy-kompatibel (Ähnlichkeit {ratio:.2f})."
        return MatchStatus.UNKNOWN, f"Textvergleich unklar: Produkt {actual!r}, gefordert {required!r}."

    def _to_number(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"\d+(?:[,.]\d+)?", str(value))
        if not match:
            return None
        return float(match.group(0).replace(",", "."))

    def _normalize_text(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value).lower()).strip()

    def _wifi_rank(self, value: str) -> Optional[int]:
        match = re.search(r"wi-?fi\s*(\d+)", value)
        return int(match.group(1)) if match else None
