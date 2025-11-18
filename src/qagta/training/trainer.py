"""Hybrid quantum-classical training loops.

Training proceeds in two stages:

1. **Encoder pre-training** — the quantum encoder is optimised as an
   autoencoder with a reconstruction objective on normal-only data.
2. **Joint graph stage** — the dynamic graph constructor, graph encoder and
   decision module are optimised on a latent self-reconstruction objective.
   Optionally the quantum circuit weights are co-adapted in the same loop
   (hybrid optimisation): classical parameters update via backpropagation,
   quantum weights via either autograd through the native simulator or the
   parameter-shift rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from qagta.graph.constructor import DynamicGraphConstructor
from qagta.models.decision import DecisionModule
from qagta.quantum.encoder import QuantumEncoder
from qagta.training.parameter_shift import apply_parameter_shift_step


@dataclass
class TrainingHistory:
    encoder_loss: list[float] = field(default_factory=list)
    graph_loss: list[float] = field(default_factory=list)


def train_quantum_encoder(
    encoder: QuantumEncoder,
    x_train: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.01,
    log_every: int = 5,
    verbose: bool = True,
) -> list[float]:
    """Stage 1: reconstruction pre-training of the quantum autoencoder."""
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    encoder.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, _ = encoder(x_train)
        loss = criterion(recon, x_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if verbose and (epoch + 1) % log_every == 0:
            print(f"[encoder] epoch {epoch + 1}/{epochs}  loss={loss:.4f}")
    return losses


def train_graph_stage(
    encoder: QuantumEncoder,
    constructor: DynamicGraphConstructor,
    graph_model: nn.Module,
    decision: DecisionModule,
    x_train: torch.Tensor,
    epochs: int = 40,
    lr: float = 0.005,
    joint_quantum: bool = False,
    quantum_gradient: str = "autograd",
    quantum_lr: float = 0.01,
    use_fidelity: bool = True,
    latent_mask: float = 0.5,
    log_every: int = 10,
    verbose: bool = True,
) -> list[float]:
    """Stage 2: train topology construction and graph propagation.

    The objective is *contextual* latent reconstruction: a fraction
    ``latent_mask`` of each node's own latent features is masked out before
    fusion, so the target can only be recovered from what the graph stage
    propagates in from neighbouring nodes. This is what forces the learned
    topology to be informative — with the node's own latent left intact the
    objective is satisfiable by an identity mapping and the graph carries no
    signal. It also produces the anomaly cue the detector relies on: nodes
    that their neighbourhood cannot explain land in a distinct region of the
    representation space.

    With ``joint_quantum=True`` the graph objective also updates the
    quantum circuit: via plain autograd (``quantum_gradient='autograd'``)
    or via the parameter-shift rule (``quantum_gradient='parameter_shift'``),
    in which case the latent gradient produced by backpropagation through
    the classical stages is chained with shifted circuit evaluations.
    """
    classical_params = (
        list(constructor.parameters())
        + list(graph_model.parameters())
        + list(decision.parameters())
    )
    if joint_quantum and quantum_gradient == "autograd":
        classical_params += list(encoder.parameters())
    optimizer = torch.optim.Adam(classical_params, lr=lr)
    criterion = nn.MSELoss()

    losses = []
    constructor.train()
    graph_model.train()
    decision.train()
    encoder.train() if joint_quantum else encoder.eval()

    for epoch in range(epochs):
        optimizer.zero_grad()

        use_shift = joint_quantum and quantum_gradient == "parameter_shift"
        if joint_quantum and not use_shift:
            latent, states = encoder.encode(x_train, return_state=True)
        else:
            with torch.no_grad():
                latent, states = encoder.encode(x_train, return_state=True)
        if use_shift:
            # Leaf latent so autograd hands us dL/dz for the chain rule.
            latent = latent.detach().requires_grad_(True)

        target = latent.detach()
        graph = constructor(latent, states if use_fidelity else None)
        propagated = graph_model(graph.x, graph.edge_index, graph.edge_attr)

        # Mask part of each node's own latent so the reconstruction target
        # has to be recovered from propagated neighbourhood information.
        if latent_mask > 0.0:
            keep = (torch.rand_like(latent) >= latent_mask).float()
            fusion_latent = latent * keep
        else:
            fusion_latent = latent

        fused = decision(propagated, fusion_latent)
        loss = criterion(decision.reconstruct(fused), target)
        loss.backward()
        optimizer.step()

        if use_shift and latent.grad is not None:
            angles = encoder.encode_angles(x_train).detach()
            apply_parameter_shift_step(
                encoder.circuit, angles, latent.grad, lr=quantum_lr
            )

        losses.append(float(loss.detach()))
        if verbose and (epoch + 1) % log_every == 0:
            n_edges = graph.edge_index.shape[1]
            print(
                f"[graph] epoch {epoch + 1}/{epochs}  loss={loss:.4f}  edges={n_edges}"
            )
    return losses
