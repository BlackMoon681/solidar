from pydantic import BaseModel


class ProjectInput(BaseModel):
    # Structural
    current_progress: float
    remaining_ratio:  float = None
    budget_gap:       float
    nb_marches:       int

    # Execution (binary 0/1)
    delay:          int = 0
    schedule_slip:  int = 0
    pacing:         int = 0
    pressure:       int = 0

    # Budget (binary 0/1)
    cost_overruns:    int = 0
    budget_revisions: int = 0
    funding_risk:     int = 0
    margin_pressure:  int = 0

    # Operational (binary 0/1)
    errors:            int = 0
    contractor_issues: int = 0
    resource_shortage: int = 0
    coordination:      int = 0

    # External (binary 0/1)
    client_changes: int = 0
    supplier_delays: int = 0
    regulatory:     int = 0
    external_risk:  int = 0
    dependency:     int = 0

    # Governance (binary 0/1)
    reporting:     int = 0
    decision_delay: int = 0
    risk_tracking: int = 0
    escalation:    int = 0


class ChatTurn(BaseModel):
    role: str          # "user" or "bot"
    content: str


class Question(BaseModel):
    question: str
    history: list[ChatTurn] = []
