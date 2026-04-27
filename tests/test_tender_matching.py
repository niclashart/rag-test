import unittest
from types import SimpleNamespace

from src.tenders.agents import RequirementExtractionAgent
from src.tenders.pdf_extraction import TenderChunk
from src.tenders.fact_extraction import ProductFactExtractionAgent
from src.tenders.matcher import ProductMatcher, UnitNormalizer
from src.tenders.schemas import (
    Eligibility,
    MatchStatus,
    Operator,
    ProductFact,
    Requirement,
    RequirementStatus,
    RequirementType,
)


class TenderMatchingTests(unittest.TestCase):
    def test_unit_normalization(self):
        self.assertEqual(UnitNormalizer.normalize(1, "TB"), (1000.0, "GB"))
        self.assertEqual(UnitNormalizer.normalize(1400, "g"), (1.4, "kg"))
        self.assertEqual(UnitNormalizer.normalize(24, "months"), (2.0, "years"))
        self.assertEqual(UnitNormalizer.normalize(300, "cd/m²"), (300.0, "nits"))

    def test_must_failure_makes_product_not_eligible(self):
        requirement = Requirement(
            tender_id="t1",
            requirement_type=RequirementType.MUST,
            attribute="memory.capacity",
            operator=Operator.GTE,
            value=16,
            unit="GB",
            original_text="mindestens 16 GB RAM",
            confidence=0.9,
            needs_review=False,
            status=RequirementStatus.APPROVED,
        )
        fact = ProductFact(
            product_id="p1",
            model="Model A",
            attribute="memory.capacity",
            value=8,
            unit="GB",
            source_text="8 GB memory",
            source_document="doc",
            confidence=0.9,
        )

        result = ProductMatcher().match_product("t1", "p1", "Model A", [requirement], [fact])

        self.assertEqual(result.eligibility, Eligibility.NOT_ELIGIBLE)
        self.assertEqual(result.must_failed, 1)
        self.assertEqual(result.requirement_results[0].status, MatchStatus.NOT_FULFILLED)

    def test_unknown_must_keeps_product_unknown(self):
        requirement = Requirement(
            tender_id="t1",
            requirement_type=RequirementType.MUST,
            attribute="ports.thunderbolt4",
            operator=Operator.EXISTS,
            value=True,
            original_text="Thunderbolt 4 muss vorhanden sein",
            confidence=0.9,
            needs_review=False,
            status=RequirementStatus.APPROVED,
        )

        result = ProductMatcher().match_product("t1", "p1", "Model A", [requirement], [])

        self.assertEqual(result.eligibility, Eligibility.UNKNOWN)
        self.assertEqual(result.unknown_count, 1)

    def test_optional_scored_score(self):
        requirements = [
            Requirement(
                tender_id="t1",
                requirement_type=RequirementType.SCORED,
                attribute="network.wifi",
                operator=Operator.COMPATIBLE_WITH,
                value="Wi-Fi 6",
                original_text="Wi-Fi 6 wird gewertet",
                confidence=0.9,
                needs_review=False,
                status=RequirementStatus.APPROVED,
                points=5,
            ),
            Requirement(
                tender_id="t1",
                requirement_type=RequirementType.OPTIONAL,
                attribute="display.brightness",
                operator=Operator.GTE,
                value=400,
                unit="nits",
                original_text="optional mindestens 400 nits",
                confidence=0.9,
                needs_review=False,
                status=RequirementStatus.APPROVED,
            ),
        ]
        facts = [
            ProductFact(
                product_id="p1",
                model="Model A",
                attribute="network.wifi",
                value="Wi-Fi 7",
                source_text="Wi-Fi 7",
                source_document="doc",
                confidence=0.9,
            ),
            ProductFact(
                product_id="p1",
                model="Model A",
                attribute="display.brightness",
                value=300,
                unit="nits",
                source_text="300 nits",
                source_document="doc",
                confidence=0.9,
            ),
        ]

        result = ProductMatcher().match_product("t1", "p1", "Model A", requirements, facts)

        self.assertEqual(result.eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(result.score, 5)
        self.assertEqual(result.max_score, 6)

    def test_rejected_requirement_ignored(self):
        requirement = Requirement(
            tender_id="t1",
            requirement_type=RequirementType.MUST,
            attribute="memory.capacity",
            operator=Operator.GTE,
            value=64,
            unit="GB",
            original_text="mindestens 64 GB RAM",
            status=RequirementStatus.REJECTED,
        )

        result = ProductMatcher().match_product("t1", "p1", "Model A", [requirement], [])

        self.assertEqual(result.eligibility, Eligibility.ELIGIBLE)
        self.assertEqual(result.must_failed, 0)

    def test_targeted_fact_extraction_from_chunk(self):
        agent = ProductFactExtractionAgent()
        product = SimpleNamespace(id=7, product_name="Model A", document_id=3)
        chunk = SimpleNamespace(
            text="Display specifications: brightness 400 nits, anti-glare.",
            page_number=5,
        )

        fact = agent._extract_fact_from_text(product, "display.brightness", chunk)

        self.assertIsNotNone(fact)
        self.assertEqual(fact.attribute, "display.brightness")
        self.assertEqual(fact.value, 400)
        self.assertEqual(fact.unit, "nits")
        self.assertEqual(fact.source_page, 5)

    def test_invalid_llm_operator_falls_back_to_regex(self):
        agent = RequirementExtractionAgent()
        agent.llm = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(
                content='{"requirements":[{"requirement_type":"must","attribute":"memory.capacity","operator":"!=","value":8,"unit":"GB","original_text":"nicht 8 GB RAM","source_page":1,"confidence":0.9,"needs_review":true}]}'
            )
        )
        chunk = TenderChunk(
            chunk_id="chunk-1",
            text="Mindestens 16 GB RAM erforderlich. SSD, Display, Prozessor, USB und Wi-Fi sind zu prüfen.",
            page_number=1,
        )

        requirements = agent._extract_chunk("t1", chunk, {"section_type": "technical_requirements"})

        self.assertEqual(len(requirements), 1)
        self.assertEqual(requirements[0].operator, Operator.GTE)
        self.assertEqual(requirements[0].value, 16)

    def test_requirement_extraction_ignores_price_quantity_notice(self):
        agent = RequirementExtractionAgent()
        agent.llm = None
        chunk = TenderChunk(
            chunk_id="chunk-1",
            text=(
                "HINWEIS: die angegebenen Mengen dienen ausschließlich der Berechnung des Preises.\n"
                "- 16GB DDR5\n"
                "- SSD NVMe mind. PCIe Gen4x4 500 GB\n"
                "- 2x Thunderbolt 4\n"
            ),
            page_number=1,
        )

        requirements = agent._extract_chunk("t1", chunk, {"section_type": "technical_requirements"})

        original_texts = [req.original_text for req in requirements]
        self.assertFalse(any("Berechnung des Preises" in text for text in original_texts))
        self.assertTrue(any(req.attribute == "memory.capacity" for req in requirements))
        self.assertTrue(any(req.attribute == "storage.capacity" for req in requirements))
        self.assertTrue(any(req.attribute == "ports.thunderbolt4" for req in requirements))


if __name__ == "__main__":
    unittest.main()
