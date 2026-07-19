"""Isolated v3 label-aware evaluation and aggregation contracts.

This namespace intentionally has no scorer and does not import the frozen
ViTTrace runner/evaluator modules.
"""

from .registry import ArmRegistry, ArmSpec, ContrastSpec, validate_arm_registry

__all__ = [
    "ArmRegistry",
    "ArmSpec",
    "ContrastSpec",
    "validate_arm_registry",
]
