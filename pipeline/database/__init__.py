"""
ASI-Forge Database Module

Manages experimental data and results. Contains the Summarizer agent.

Based on ASI-Arch by Liu et al. (2025) - "AlphaGo Moment for Model Architecture Discovery"
"""

from .mongo_database import create_client
from .interface import program_sample, update
from .element import DataElement