"""Exceptions raised by the FLOWCAT adapter."""

from __future__ import annotations


class FlowcatAdapterError(RuntimeError):
    """Base class for adapter failures."""


class MissingFlowcatOutputError(FlowcatAdapterError, FileNotFoundError):
    """A required FLOWCAT artifact is absent."""


class FlowcatSchemaError(FlowcatAdapterError, ValueError):
    """A FLOWCAT artifact does not match the supported schema."""


class FlowcatGeometryError(FlowcatAdapterError, ValueError):
    """Image geometry or a coordinate convention is incompatible."""


class UnsafeArtifactError(FlowcatAdapterError, PermissionError):
    """Loading was refused because the artifact can execute Python code."""


class MissingOptionalDependencyError(FlowcatAdapterError, ImportError):
    """An explicitly requested optional reader is not installed."""
