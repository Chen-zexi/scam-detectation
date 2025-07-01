"""
Scam Detection Evaluation Pipeline

A flexible pipeline for evaluating LLMs on scam detection tasks.
Works with any dataset that has a 'label' column (1=scam, 0=legitimate).
Supports various content types: emails, texts, conversations, messages, etc.
"""

from .evaluator import ScamDetectionEvaluator, PhishingEvaluator
from .annotation_pipeline import LLMAnnotationPipeline
from .data_loader import DatasetLoader
from .prompt_generator import PromptGenerator
from .metrics_calculator import MetricsCalculator
from .results_saver import ResultsSaver
from .api_provider import LLM
from .api_call import make_api_call

__version__ = "2.0.0"
__author__ = "Your Name"

__all__ = [
    'ScamDetectionEvaluator',
    'PhishingEvaluator',  # Backward compatibility
    'LLMAnnotationPipeline',
    'DatasetLoader', 
    'PromptGenerator',
    'MetricsCalculator',
    'ResultsSaver',
    'LLM',
    'make_api_call'
] 