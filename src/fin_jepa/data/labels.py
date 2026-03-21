"""
Distress event label database.

Workstream: Build distress event label database (5 outcome types).

Outcome types
-------------
1. stock_decline      — significant stock price decline (e.g. >30% within 12m)
2. earnings_restate   — earnings restatement filed with SEC
3. audit_qualification — going-concern or adverse audit opinion
4. sec_enforcement    — SEC enforcement action (AAERs, litigation releases)
5. bankruptcy         — Chapter 7 or Chapter 11 filing

TODO:
  - Pull restatement data from Audit Analytics or SEC EDGAR
  - Pull going-concern opinions from Audit Analytics
  - Pull SEC AAERs from EDGAR enforcement page
  - Pull bankruptcy filings from BankruptcyData.com or PACER
  - Compute stock_decline labels from forward return series
  - Merge all labels keyed by (cik, period_end)
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd


class OutcomeType(StrEnum):
    STOCK_DECLINE = "stock_decline"
    EARNINGS_RESTATE = "earnings_restate"
    AUDIT_QUALIFICATION = "audit_qualification"
    SEC_ENFORCEMENT = "sec_enforcement"
    BANKRUPTCY = "bankruptcy"


ALL_OUTCOMES: list[str] = list(OutcomeType)


def build_label_database(
    processed_dir: Path,
    horizon_days: int = 365,
    decline_threshold: float = -0.30,
) -> pd.DataFrame:
    """Construct the full distress label table.

    Returns a DataFrame with columns:
        [cik, period_end, stock_decline, earnings_restate,
         audit_qualification, sec_enforcement, bankruptcy]
    All label columns are binary int (0/1).
    """
    raise NotImplementedError("Implement label database construction.")
