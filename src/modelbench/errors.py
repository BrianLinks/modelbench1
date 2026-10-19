"""Exception hierarchy. Every error ModelBench raises deliberately derives from ModelBenchError.

Errors that signal bad caller input also derive from ValueError so generic handlers keep working.
"""


class ModelBenchError(Exception):
    """Base class for all ModelBench errors."""


class MetricError(ModelBenchError, ValueError):
    """Metric inputs are empty, mismatched, non-numeric, or otherwise unusable."""


class IdentifierError(ModelBenchError, ValueError):
    """A dataset, experiment, or run identifier is malformed."""


class SeedError(ModelBenchError, ValueError):
    """A random seed is out of range or not an integer."""


class DatasetError(ModelBenchError, ValueError):
    """A dataset file or table is unreadable, malformed, or cannot be split as requested."""


class ConfigError(ModelBenchError, ValueError):
    """An experiment configuration is invalid."""
