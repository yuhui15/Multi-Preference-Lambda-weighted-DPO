# 🧠 Multi-Preference Lambda-weighted DPO (λ-DPO)

**λ-DPO** is a novel extension of Direct Preference Optimization (DPO) designed to support **multi-objective alignment** through **listwise human feedback** and **lambda-weighted aggregation**.

## 🚀 Key Features

- **Multi-Preference Optimization**: Incorporates multiple human preference dimensions (e.g., helpfulness, harmlessness, conciseness), each associated with its own listwise preference distribution.
- **Listwise Extension**: Generalizes the DPO loss from pairwise to listwise comparisons, capturing richer ranking information across candidate responses.
- **Lambda-weighted Aggregation**: Introduces a controllable or sampled vector **λ ∈ Δᵐ** (the m-dimensional simplex) to combine individual preference losses.
- **Robust Generalization**: By sampling λ vectors during training, the model learns to generalize across various user preference configurations, enabling dynamic alignment at inference time.

## 🧮 Loss Formulation

Let:
- `x` be an input prompt,
- `{y₁, ..., y_N}` be a set of candidate outputs,
- `π_θ(y|x)` be the current policy,
- `π_ref(y|x)` be a fixed reference policy,
- and `p_i^{*(k)}` be the listwise human preference distribution for the k-th objective.

Each component loss is defined as:

```math
L^{(k)}(θ) = E_{(x, {y_i}) ∼ D} \left[ 
  - \sum_{i=1}^{N} p_i^{*(k)} \cdot \log \left(
    \frac{\left( \frac{π_θ(y_i|x)}{π_{ref}(y_i|x)} \right)^β}
         {\sum_{j=1}^{N} \left( \frac{π_θ(y_j|x)}{π_{ref}(y_j|x)} \right)^β}
  \right) \right]
```

Aggregating all $m$ preferences with a sampled weight vector $\lambda \in \Delta^m$
gives the final training objective:

```math
L(θ) = \mathbb{E}_{\boldsymbol{\lambda} \sim P(\lambda)} \left[\sum_{k=1}^m
\lambda_k \, L^{(k)}(θ)\right]
```

Here $P(\lambda)$ can be any distribution over the simplex (e.g. Dirichlet),
allowing the model to learn from diverse preference trade-offs.

In practice the repository computes $L^{(k)}(θ)$ for each preference dimension
and samples a new $\lambda$ at every training step. The weighted sum directs the
model to outputs that satisfy different user priorities.

## 🔧 Running Tests

Install the Python dependencies listed in `requirements.txt` and then execute
the following commands inside the `lambda_dpo` directory to verify the
installation:

```bash
cd lambda_dpo
make test
```

For complete usage instructions, please refer to `lambda_dpo/README.md`.
