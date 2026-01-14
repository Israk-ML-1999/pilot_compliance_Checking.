from pydantic import BaseModel
from typing import Optional, List, Any

class Violation(BaseModel):
    rule_reference: str
    description: str

class ScheduleSummary(BaseModel):
    key_points: List[str]

class ComplianceReport(BaseModel):
    schedule_summary: Any 
    violations: List[Violation]
    email_report: Optional[str] = None
    answer: Optional[str] = None # For pure text questions