"""Optional PyTorch/PyG SAGE-AVO and baseline architectures."""

__all__ = ["ALL_VARIANTS", "LEARNED_VARIANTS", "SAGEAVO", "build_sage_avo_variant"]

try:
    from .sage_avo import SAGEAVO as SAGEAVO
    from .variants import (
        ALL_VARIANTS as ALL_VARIANTS,
        LEARNED_VARIANTS as LEARNED_VARIANTS,
        build_sage_avo_variant as build_sage_avo_variant,
    )
except ImportError:
    # The scientific core remains importable without optional ML dependencies.
    __all__ = []
