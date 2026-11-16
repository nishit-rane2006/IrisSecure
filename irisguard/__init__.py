"""IrisGuard adaptive template protection and poisoning detection."""

from .afsdd import AFSDD, AFSDDConfig
from .template_manager import (
    IrisGuardPolicy,
    QualityAwarePolicy,
    StaticTemplatePolicy,
    TemplateManager,
    UnprotectedAdaptivePolicy,
)

__all__ = [
    "AFSDD",
    "AFSDDConfig",
    "TemplateManager",
    "StaticTemplatePolicy",
    "UnprotectedAdaptivePolicy",
    "QualityAwarePolicy",
    "IrisGuardPolicy",
]
