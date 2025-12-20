# Scope and Future Work

This repository is a **designed falsifier**: a deliberately austere testbed for probing whether **functional identity** is *locally recoverable* after controlled perturbation when recovery is constrained by increasingly informative local structure (0th / 1st / 2nd order constraints).

The README reports a decisive negative result **within this specific system and protocol**: matching local geometric proxies (spectra, Jacobians, Hessian–vector products) does **not** reliably return the system to its pre-perturbation functional behavior in a *single recovery step*.

This document clarifies what is (and is not) claimed, and lays out the next experiments that would either **strengthen** the result into a broader statement or **kill it** cleanly.

---

## What This Repository Is

- A transparent, reproducible loop that couples:
  1) a state (e.g., matrix/parameters),
  2) a fixed correspondence/measurement map producing observables,
  3) an update rule (a learner) that modifies the state.

- A hard test of a specific assumption used implicitly across interpretability, robustness, and model-editing practice:

> “If we match the right local geometric information, we can locally reconstruct (or recover) the original function.”

Here, that assumption is challenged under tightly controlled degrees of freedom.

---

## What This Repository Is Not

### Not a general theorem about all neural networks
The evidence here is a **counterexample inside a constrained testbed**, not a universal proof about transformers, MLPs, or large-scale foundation models.

### Not a quantum implementation (yet)
The current code path is **strictly classical**. References to the Autodidactic Universe and SQNT are **lineage and motivation**, and a roadmap toward *nonlocal/topological invariants*—not a claim that quantum superposition or adaptive quantum topology is currently executed in the experiments.

---

## Scope Boundaries and Decision Boundaries

### The “local” notion in this repo
The primary decision boundary is **1-step recovery**. This is intentionally strict: it isolates whether the *local signal* is even pointing in the right direction, before allowing multi-step effects to smear the conclusion.

### Where the result currently holds
Within the implemented system and protocol:
- the perturbation (“damage”) regime,
- the observable family,
- the constraint families (0th/1st/2nd order),
- and the recovery update rule,

the measured recovery signal (CI) is near-zero, despite constraints being satisfiable.

### What is an overreach
Any claim of the form:
- “mechanistic interpretability is dead,”
- “alignment is impossible,”
- “locality is universally broken,”

is **not warranted** by this repo alone. Those are *hypotheses* that require additional experiments (below).

---

## Working Hypotheses Motivated by the Negative Result

These are the strongest, testable hypotheses suggested by the observed behavior.

### H1 — Geometry/Function Orthogonality (within the protocol)
Local geometric proxies can be **optimized** while functional identity remains unrecovered.

Operational signature:
- constraint error ↓ while CI ≈ 0.

### H2 — Functional Aliasing (non-injectivity of local signatures)
Many distinct parameter states share similar local derivative footprints (e.g., similar Jacobians/HVPs) while implementing different global behavior.

Operational signature:
- multiple recovered states satisfy the same constraint threshold but exhibit divergent function metrics.

### H3 — Locality Gap grows with dimension/complexity (speculative)
As model dimensionality increases, the set of “geometric look-alikes” may grow faster than the set of function-preserving states.

Operational signature:
- as architecture size increases, constraint satisfaction remains easy while CI remains flat (or worsens).

These hypotheses are **not conclusions**. They are a research agenda.

---

## Immediate Expansions That Preserve the “Austere Falsifier” Standard

The goal is to expand scope without allowing degrees of freedom to rescue the result.

### 1) Multi-step recovery curve (k steps, fixed protocol)
Run recovery for k ∈ {1, 2, 5, 10, 25, 50} steps, with **no tuning**.

Report:
- CI(k),
- loss trajectories (pre → post → recover_k),
- constraint error trajectories.

Decision rule:
- If CI(k) rises meaningfully toward 1.0 under the same strict constraints and fixed protocol, the “1-step wall” becomes a statement about *step-size/locality*, not about recoverability in principle.

### 2) Constraint-vs-function ablation plot
For each constraint family, log per-run:
- constraint distance (e.g., spectral/Jacobian/HVP mismatch),
- functional loss (or task loss),
- CI.

Decision rule:
- If constraint distance correlates strongly with CI across seeds/perturbations, then the negative result is likely due to a particular mismatch in how constraints were formed or applied.
- If correlation remains weak, the “geometry ≠ function” claim is strengthened.

### 3) Minimal architecture generalization (no scale jump)
Repeat the same falsifier protocol on a small set of transparent architectures:
- cyclic RNN (baseline),
- small MLP,
- small GRU.

Hold constant:
- the perturbation regime,
- CI definition,
- the “no tuning” policy,
- evaluation artifacts.

Decision rule:
- If the failure persists across architectures, it begins to look like a **class** of counterexamples rather than a single-model quirk.

---

## “Kill Switch” Experiments

These are experiments that, if successful, would *invalidate* the strongest reading of the current result.

### K1 — Recoverability under a strictly defined longer-step local recovery
If CI(k) reliably becomes large (e.g., CI(k) ≥ 0.8) for modest k without changing the recovery objective or constraints, then the main story is not “irrecoverable,” but “not 1-step recoverable.”

### K2 — A constraint family that is demonstrably function-informative
If a new constraint family (still local in some clearly defined sense) correlates tightly with function recovery across perturbations and architectures, then “local proxies are insufficient” becomes “these particular proxies were insufficient.”

### K3 — Explicit demonstration that the correspondence map is losing identity information
If the mapping from state → observables is shown to be non-injective in practice (many-to-one), then failure could be an information bottleneck rather than a geometry/function gap.

(If K3 holds, the next work is: repair/replace the map, then rerun the falsifier.)

---

## SQNT Roadmap (Why mention it at all?)

SQNT is relevant as a **motivation for nonlocal/topological invariants**: it suggests that functional stability may be encoded in **global structure** rather than local metric neighborhoods.

A conservative, testable trajectory is:

1) Establish the classical locality boundary (this repo).
2) Identify a *candidate global invariant* that is stable under the perturbation regime.
3) Implement topology-plastic or topology-aware updates that preserve that invariant.
4) Re-run CI tests to see whether global invariants enable recovery where local geometry fails.

This keeps the SQNT connection scientific: **it becomes an experimental fork**, not a rhetorical flourish.

---

## Minimal Terminology

- **Functional identity**: the behavior of the system as measured on a fixed evaluation distribution (or fixed task).
- **Geometric basin**: a region in parameter/state space defined by local metric structure (e.g., curvature, derivatives, spectra).
- **Functional basin**: the set of states that implement the same functional identity within tolerance.
- **Continuation Interest (CI)**: the normalized recovery progress metric defined in the README.
- **Locality gap** (hypothesis): measurable divergence between “near in local geometry” and “near in behavior.”

---

## Deliverables That Would Upgrade This Repo → Preprint

1.	CI(k) recovery curves for k ∈ {1, 2, 5, 10, 25, 50} under a fixed, no-tuning protocol, with saved artifacts (loss, CI, constraint error).
2.	Constraint–function coupling analysis: scatter/correlation plots of constraint distance vs CI across seeds/perturbations (demonstrate coupling or decoupling explicitly).
3.	Minimal architecture generalization under identical evaluation: cyclic RNN → small MLP → small GRU (same perturbation regime, same CI, same constraints).
4.	Ablation proving “constraint satisfiable ≠ function recovered”: show constraints decrease while CI stays low (or vice versa), with logs sufficient to reproduce.
5.	Scope & universality paragraph (preprint-ready) stating, in one place:
	•	what was tested (system + perturbation + constraints + recovery rule),
	•	what was falsified (local proxy sufficiency under this protocol),
	•	what remains hypothesis (scaling behavior; applicability to other architectures).

---

## Contact

Christopher Altman — x@christopheraltman.com
