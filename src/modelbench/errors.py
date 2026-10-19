"""Exception hierarchy. Every error ModelBench raises deliberately derives from ModelBenchError."""


class ModelBenchError(Exception):
    """Base class for all ModelBench errors."""


class MetricError(ModelBenchError, ValueError):
    """Metric inputs are empty, mismatched, non-numeric, or otherwise unusable."""
