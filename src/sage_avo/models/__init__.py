"""Optional PyTorch/PyG SAGE-AVO and baseline architectures."""

__all__: list[str] = []

try:
    from .sage_avo import SAGEAVO
    from .variants import ALL_VARIANTS, LEARNED_VARIANTS, build_sage_avo_variant

    __all__.extend(["ALL_VARIANTS", "LEARNED_VARIANTS", "SAGEAVO", "build_sage_avo_variant"])
except ImportError:
    # The scientific core remains importable without optional ML dependencies.
    pass
