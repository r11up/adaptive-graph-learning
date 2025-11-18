"""Training subsystem: hybrid optimisation, parameter-shift rule, evaluation."""

from qagta.training.evaluate import EvaluationResult, comparison_table, evaluate_embeddings
from qagta.training.parameter_shift import (
    apply_parameter_shift_step,
    expectation_jacobian,
    quantum_weight_gradient,
)
from qagta.training.trainer import train_graph_stage, train_quantum_encoder

__all__ = [
    "train_quantum_encoder",
    "train_graph_stage",
    "expectation_jacobian",
    "quantum_weight_gradient",
    "apply_parameter_shift_step",
    "evaluate_embeddings",
    "comparison_table",
    "EvaluationResult",
]
