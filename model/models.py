from pydantic import BaseModel, RootModel, Field
from typing import List, Union, Optional, Dict
from enum import Enum


# ---- Existing models (backward compatible) ----

class Metadata(BaseModel):
    Summary: List[str]
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]  # Can be "Not Available"
    SentimentTone: str

class ChangeFormat(BaseModel):
    Page: str
    Changes: Union[str, List[str]]

class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass

class PromptType(str, Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    DOCUMENT_COMPARISON = "document_comparison"
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"
    KEY_TERM_EXTRACTION = "key_term_extraction"


# ---- Phase 1C: Structured Key-Term Extraction Models ----

class PartyInfo(BaseModel):
    """A party involved in the contract."""
    name: str
    role: str = Field(description="Role in the contract: buyer, seller, contractor, client, etc.")

class PaymentTerm(BaseModel):
    """A payment obligation in the contract."""
    amount: str = Field(description="Dollar/currency amount")
    currency: str = Field(default="USD", description="Currency code")
    frequency: str = Field(description="monthly, one-time, milestone-based, etc.")
    due_date: str = Field(default="Not specified", description="When payment is due")

class Obligation(BaseModel):
    """An obligation imposed by the contract."""
    party: str = Field(description="Who is obligated")
    description: str = Field(description="What they must do")
    deadline: str = Field(default="Not specified", description="By when")
    consequence_of_breach: str = Field(default="Not specified", description="What happens if breached")

class NonCompete(BaseModel):
    """Non-compete clause details."""
    exists: bool = False
    duration: str = Field(default="N/A", description="Time period of restriction")
    geographic_scope: str = Field(default="N/A", description="Geographic area covered")

class RiskFlag(BaseModel):
    """A flagged risk in the contract."""
    clause: str = Field(description="The problematic clause text")
    risk_level: str = Field(description="low, medium, high, or critical")
    plain_english: str = Field(description="What this means in simple business terms")
    recommendation: str = Field(description="What the user should do about it")

class ContractKeyTerms(BaseModel):
    """
    Structured extraction of actionable contract terms.
    This is the primary output of the key-term extraction pipeline.
    """
    parties: List[PartyInfo] = Field(default_factory=list)
    effective_date: str = Field(default="Not found")
    expiration_date: str = Field(default="Not found")
    auto_renewal: bool = False
    payment_terms: List[PaymentTerm] = Field(default_factory=list)
    obligations: List[Obligation] = Field(default_factory=list)
    termination_clauses: List[str] = Field(default_factory=list)
    liability_cap: str = Field(default="Not specified")
    confidentiality_period: str = Field(default="Not specified")
    governing_law: str = Field(default="Not found")
    non_compete: Optional[NonCompete] = None
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
