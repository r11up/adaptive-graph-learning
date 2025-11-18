"""QAGTA — Quantum-Assisted Adaptive Graph Construction and Temporal Pattern Analysis.

A hybrid quantum-classical pipeline for representation learning on
multivariate temporal data:

    time series -> quantum latent encoding -> adaptive edge learning
    -> dynamic graph -> graph attention propagation -> anomaly scoring

The package is organised by subsystem:

- :mod:`qagta.data`      -- loading, normalisation and windowing of temporal data
- :mod:`qagta.quantum`   -- variational quantum encoder and quantum similarity metrics
- :mod:`qagta.graph`     -- adaptive edge learning and dynamic graph construction
- :mod:`qagta.models`    -- graph attention / message-passing networks and fusion
- :mod:`qagta.training`  -- hybrid optimisation and evaluation utilities
- :mod:`qagta.pipeline`  -- end-to-end orchestration
"""

from qagta.config import PipelineConfig
from qagta.pipeline import QuantumAdaptiveGraphPipeline

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "QuantumAdaptiveGraphPipeline",
    "__version__",
]
