"""Independent mature-astrocyte DAPI + GFAP-only analysis.

This package is deliberately not imported by the eGFP analysis path.  The
controller may call :func:`analyze_dapi_gfap_only` only after channel routing
has established that eGFP is absent, both DAPI and GFAP are available, and the
input does not declare a neonatal age profile.
"""

from .gfap_only_analysis import (
    GFAPOnlyConfig,
    GFAPOnlyResult,
    analyze_dapi_gfap_only,
)
from .gfap_nucleus_ownership import (
    ExclusiveNucleusProjection,
    GFAPAssociationResult,
    GFAPNucleusOwnershipConfig,
    GFAPNucleusOwnershipResult,
    LinkedNucleusInventory,
    link_slice_instances_3d,
    project_exclusive_nucleus_owners,
    resolve_gfap_nucleus_owners,
    select_gfap_associated_owners,
)

__all__ = [
    "GFAPOnlyConfig",
    "GFAPOnlyResult",
    "analyze_dapi_gfap_only",
    "ExclusiveNucleusProjection",
    "GFAPAssociationResult",
    "GFAPNucleusOwnershipConfig",
    "GFAPNucleusOwnershipResult",
    "LinkedNucleusInventory",
    "link_slice_instances_3d",
    "project_exclusive_nucleus_owners",
    "resolve_gfap_nucleus_owners",
    "select_gfap_associated_owners",
]
