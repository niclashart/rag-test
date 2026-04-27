"""Optional LLM judge for unresolved text compatibility cases."""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from src.llm import create_chat_llm, has_llm_credentials
from src.tenders.schemas import MatchStatus, ProductFact, Requirement


class LLMJudge:
    """Judge only cases the deterministic matcher marks unknown."""

    def __init__(self) -> None:
        self.llm = create_chat_llm(
            temperature=0,
            request_timeout=60,
            max_retries=2,
        )

    @classmethod
    def from_env(cls) -> Optional["LLMJudge"]:
        return cls() if has_llm_credentials() else None

    def judge(self, requirement: Requirement, fact: ProductFact) -> Tuple[MatchStatus, str]:
        prompt = f"""Du bist LLM-Judge fuer unklare Laptop-Ausschreibungs-Matches.
Deterministische numerische Muss-Kriterien sind bereits geprueft. Entscheide nur anhand der Quellen.
Gib ausschliesslich valides JSON zurueck.

Regeln:
- Keine fehlenden Fakten erfinden.
- fulfilled nur, wenn Produktquelle die Anforderung nachvollziehbar erfuellt.
- not_fulfilled, wenn Produktquelle klar widerspricht.
- unknown, wenn Quelle nicht reicht.

Anforderung:
Attribut: {requirement.attribute}
Operator: {requirement.operator}
Wert: {requirement.value} {requirement.unit or ""}
Text: {requirement.original_text}

Produktfakt:
Wert: {fact.value} {fact.unit or ""}
Quelle: {fact.source_text}

JSON:
{{"status":"fulfilled|not_fulfilled|unknown","reason":"kurze nachvollziehbare Begruendung"}}"""
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        data = self._parse_json(content)
        status = data.get("status", "unknown")
        if status not in {"fulfilled", "not_fulfilled", "unknown"}:
            status = "unknown"
        reason = data.get("reason") or "LLM-Judge ohne Begruendung."
        return MatchStatus(status), f"LLM-Judge: {reason}"

    def _parse_json(self, content: str) -> dict:
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
