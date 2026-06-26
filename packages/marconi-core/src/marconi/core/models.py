from __future__ import annotations

from pydantic import BaseModel


class ValidationIssue(BaseModel):
    block_id: str | None = None
    field: str | None = None
    message: str
