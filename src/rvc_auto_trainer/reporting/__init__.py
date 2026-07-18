"""Offline report generation."""

from .html_report import (
    HTMLReportGenerator,
    ReportData,
    discover_report_data,
    generate_html_report,
    make_relative_media_path,
)

__all__ = [
    "HTMLReportGenerator",
    "ReportData",
    "discover_report_data",
    "generate_html_report",
    "make_relative_media_path",
]
