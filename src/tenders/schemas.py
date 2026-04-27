"""Canonical tender matching schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


def new_id() -> str:
    return str(uuid4())


class RequirementType(str, Enum):
    MUST = "must"
    SHOULD = "should"
    OPTIONAL = "optional"
    SCORED = "scored"
    SERVICE = "service"
    CONTRACT = "contract"
    UNKNOWN = "unknown"


class RequirementStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


class Operator(str, Enum):
    GTE = ">="
    LTE = "<="
    EQ = "="
    CONTAINS = "contains"
    EXISTS = "exists"
    ONE_OF = "one_of"
    COMPATIBLE_WITH = "compatible_with"


class Eligibility(str, Enum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    UNKNOWN = "unknown"


class MatchStatus(str, Enum):
    FULFILLED = "fulfilled"
    NOT_FULFILLED = "not_fulfilled"
    UNKNOWN = "unknown"


class Tender(BaseModel):
    id: str = Field(default_factory=new_id)
    filename: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "uploaded"
    raw_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Requirement(BaseModel):
    id: str = Field(default_factory=new_id)
    tender_id: str
    product_group: Optional[str] = None
    requirement_type: RequirementType = RequirementType.UNKNOWN
    attribute: str
    operator: Operator
    value: Any
    unit: Optional[str] = None
    original_text: str
    source_page: Optional[int] = None
    source_chunk_id: Optional[str] = None
    confidence: float = 0.0
    needs_review: bool = True
    status: RequirementStatus = RequirementStatus.PENDING
    points: Optional[float] = None
    rationale: Optional[str] = None

    @validator("confidence")
    def confidence_between_zero_and_one(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return value


class ProductFact(BaseModel):
    id: str = Field(default_factory=new_id)
    product_id: str
    model: str
    attribute: str
    value: Any
    unit: Optional[str] = None
    source_text: str
    source_document: str
    source_page: Optional[int] = None
    confidence: float = 0.0


class RequirementMatch(BaseModel):
    requirement_id: str
    attribute: str
    requirement_text: str
    required_value: Any
    product_value: Any = None
    status: MatchStatus
    reason: str
    tender_source_page: Optional[int] = None
    product_source_text: Optional[str] = None
    product_source_document: Optional[str] = None


class MatchResult(BaseModel):
    id: str = Field(default_factory=new_id)
    tender_id: str
    product_id: str
    model: str
    eligibility: Eligibility
    score: float
    max_score: float
    must_passed: int
    must_failed: int
    unknown_count: int
    requirement_results: List[RequirementMatch]

