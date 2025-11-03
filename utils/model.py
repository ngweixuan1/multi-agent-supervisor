
from typing import List, Dict, Any, Optional, Literal, Annotated, TypedDict
from pydantic import BaseModel, Field, validator
import os

# Part 1
class FinancialFields(BaseModel):
    """Structured schema for extracted financial metrics."""
    corporate_income_tax_2024_billion: float | None = Field(None, description="Corporate income tax for FY2024 in billions")
    corporate_income_tax_yoy_percent: float | None = Field(None, description="Year-over-year % change for corporate income tax")
    total_topups_2024_billion: float | None = Field(None, description="Total top-ups in FY2024 in billions")
    operating_revenue_taxes_list: List[str] = Field(default_factory=list, description="List of operating revenue tax names")
    latest_actual_fiscal_position_billion: float | None = Field(None, description="Latest actual fiscal position in billions")

# Part 2

class ConfigModel(BaseModel):
    extracted_text_path: str = Field(..., description="Path to extracted JSON text file")
    target_pages_part_2: List[int] = Field(..., description="Pages to process for normalization + summarization")
    output_dir: Optional[str] = Field("outputs", description="Directory for saving outputs")

    class Config:
        extra = "ignore" 

    @validator("extracted_text_path")
    def validate_path(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Extracted text JSON not found at {v}")
        return v

class ExtractedTextModel(BaseModel):
    metadata: Dict[str, Any]
    elements: List[Dict[str, Any]]


class Part2AnswerSchema(BaseModel):
    original_text: str = Field(..., description="The extracted original sentence in the text containing the date")
    normalized_date: str = Field(..., description="The normalized date")
    status: str = Field(..., description= "<Expired|Ongoing|Upcoming> compared to 2024-01-01")



# Part 3
class RevenueOutput(BaseModel):
    revenue_streams: str


class ExpenditureOutput(BaseModel):
    expenditure_streams: str


class FinalAnswer(BaseModel):
    direct_answer: str


class Router(TypedDict):
    next: Annotated[
        Literal["revenue_node", "expenditure_node", "FINISH"],
        "Next worker or FINISH.",
    ]
    reasoning: Annotated[str, "Explain your routing reasoning."]


class BudgetState(TypedDict, total=False):
    query: str
    revenue: Optional[str]
    expenditure: Optional[str]
    cur_reasoning: Optional[str]
    final_output: Optional[FinalAnswer]
    loop_count: int
    last_node: Optional[str]

class Part3ConfigModel(BaseModel):
    extracted_text_path: str
    max_loop: int = Field(default=5)
    model_name: str = Field(default="gemini-2.5-flash")