from __future__ import annotations

from agent.services.report_store import ReportStore


def cleanup_old_reports(report_store: ReportStore, days: int = 30) -> int:
    return report_store.cleanup_old(days=days)
