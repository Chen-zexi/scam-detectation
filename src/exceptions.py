"""
Custom exceptions for the Scam Detection system.

This module provides specific exception classes for different error scenarios,
enabling better error handling and more informative error messages.
"""


class ScamDetectionBaseError(Exception):
    """Base exception class for all scam detection errors."""
    pass


# Configuration Errors
class ConfigurationError(ScamDetectionBaseError):
    """Raised when there are configuration issues."""
    pass


class MissingAPIKeyError(ConfigurationError):
    """Raised when required API keys are not set."""
    pass


class InvalidProviderError(ConfigurationError):
    """Raised when an unsupported provider is specified."""
    pass


# Data Errors
class DataError(ScamDetectionBaseError):
    """Base class for data-related errors."""
    pass


class DatasetNotFoundError(DataError):
    """Raised when a dataset file cannot be found."""
    pass


class InvalidDatasetError(DataError):
    """Raised when dataset format is invalid or missing required columns."""
    pass


class InsufficientDataError(DataError):
    """Raised when there's not enough data for the requested operation."""
    pass


# Processing Errors
class ProcessingError(ScamDetectionBaseError):
    """Base class for processing-related errors."""
    pass


class ModelInitializationError(ProcessingError):
    """Raised when LLM model initialization fails."""
    pass


class APICallError(ProcessingError):
    """Raised when API calls fail."""
    pass


class ResponseParsingError(ProcessingError):
    """Raised when LLM response parsing fails."""
    pass


# Synthesis Errors
class SynthesisError(ScamDetectionBaseError):
    """Base class for synthesis-related errors."""
    pass


class UnknownSynthesisTypeError(SynthesisError):
    """Raised when an unknown synthesis type is requested."""
    pass


class InvalidCategoryError(SynthesisError):
    """Raised when an invalid category is specified for a synthesis type."""
    pass


class SchemaValidationError(SynthesisError):
    """Raised when schema validation fails."""
    pass


# Database Errors
class DatabaseError(ScamDetectionBaseError):
    """Base class for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DatabaseOperationError(DatabaseError):
    """Raised when database operations fail."""
    pass


# Checkpoint Errors
class CheckpointError(ScamDetectionBaseError):
    """Base class for checkpoint-related errors."""
    pass


class CheckpointLoadError(CheckpointError):
    """Raised when checkpoint loading fails."""
    pass


class CheckpointSaveError(CheckpointError):
    """Raised when checkpoint saving fails."""
    pass


class CheckpointCompatibilityError(CheckpointError):
    """Raised when checkpoint is incompatible with current configuration."""
    pass