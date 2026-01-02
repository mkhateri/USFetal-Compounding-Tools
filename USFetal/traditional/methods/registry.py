# methods/registry.py

"""
Compounding Method Registry

Collects all fusion methods and exposes them as a single dictionary
for use in the main USFetal pipeline.
"""

from methods.arithmetic import METHOD_REGISTRY as ARITHMETIC_REGISTRY
from methods.pca import METHOD_REGISTRY as PCA_REGISTRY
from methods.ica import METHOD_REGISTRY as ICA_REGISTRY
from methods.dog import METHOD_REGISTRY as DOG_REGISTRY
from methods.variational import METHOD_REGISTRY as VARIATIONAL_REGISTRY



# Merge all registries into one global method registry
METHOD_REGISTRY = {
    **ARITHMETIC_REGISTRY,
    **PCA_REGISTRY,
    **ICA_REGISTRY,
    **DOG_REGISTRY,
    **VARIATIONAL_REGISTRY,
}
