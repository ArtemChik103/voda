"""Service and Web-GIS package for KosmoHackathon 2026."""

from src.service.api import app
from src.service.report import (
    generate_analytical_report,
    render_html_report,
    assess_hydrological_risk,
    calculate_landcover_breakdown,
)

__all__ = [
    "app",
    "generate_analytical_report",
    "render_html_report",
    "assess_hydrological_risk",
    "calculate_landcover_breakdown",
]
