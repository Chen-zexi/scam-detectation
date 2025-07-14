"""
Scam Detection Evaluation Pipeline

A flexible pipeline for evaluating LLMs on scam detection tasks.
Works with any dataset that has a 'label' column (1=scam, 0=legitimate).
Supports various content types: emails, texts, conversations, messages, etc.
"""

# Re-export main classes for backward compatibility
from .evaluate import ScamDetectionEvaluator, PhishingEvaluator
from .annotate import LLMAnnotationPipeline
from .synthesize import TranscriptGenerator
from .llm_core import LLM

__version__ = "2.0.0"
__author__ = "Your Name"

__all__ = ['ScamDetectionEvaluator', 'PhishingEvaluator', 
           'LLMAnnotationPipeline', 'TranscriptGenerator', 'LLM'] 