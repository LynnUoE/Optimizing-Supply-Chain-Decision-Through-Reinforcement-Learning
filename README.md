# Optimizing Supply Chain Decision Through Reinforcement Learning

A hybrid optimization framework for **Supply Chain Inventory Management (SCIM)** that combines **Deep Reinforcement Learning (DRL)** with **Multi-Stage Stochastic Programming (MSP)**. The proposed method — **DRLBD** (Deep Reinforcement Learning with Batch Decision) — uses DRL for long-term production planning and MSP for short-term logistics optimization, achieving superior cost efficiency and robustness compared to standalone approaches.

Built upon the [SCIMAI-Gym](https://github.com/frenkowski/SCIMAI-Gym) environment by [Stranieri et al. (2024)](https://doi.org/10.1016/j.ijpe.2023.109099).

## Overview

Supply chain inventory management requires sequential decisions under uncertainty — how much to produce, where to ship, and how to balance costs against fluctuating demand. Traditional methods face fundamental trade-offs: Multi-Stage Stochastic Programming (MSP) provides strong theoretical guarantees but suffers from scalability issues due to exponential scenario tree growth, while Deep Reinforcement Learning (DRL) offers adaptability but struggles to incorporate domain-specific constraints in a structured way.

**DRLBD bridges this gap** by decomposing the problem into two complementary layers:

1. **DRL (PPO)** learns long-term production quantities based on inventory levels and demand history
2. **MSP (Gurobi)** optimizes short-term logistics decisions (shipping, vehicle allocation, inventory updates) over a bounded scenario tree

The MSP logistics cost feeds back into the DRL reward signal, enabling the agent to internalize how production decisions affect downstream distribution efficiency.

## Problem Formulation

We model a **single-item, two-echelon divergent supply chain** under periodic review, consisting of a central factory and *J* distribution warehouses. At each timestep *t*, the agent decides:

- **Production quantity** *x_t*: how many batches to produce at the factory
- **Distribution amounts** *z_{j,t}*: how many batches to ship to each warehouse *j*

The objective is to **minimize total cost**, which includes production, variable/fixed logistics, storage, and backorder (penalty) costs. Demand is seasonal and stochastic, modeled with sinusoidal patterns plus random noise.

### Evaluation Metrics

| Metric | Scenario | Definition |
|---|---|---|
| **Opt-gap** | Basic | `(Cost_alg − Cost_MSP) / Cost_MSP × 100%` |
| **EVPI-gap** | Advanced | `(Cost_alg − Cost_PI) / Cost_PI × 100%` |

## Methodology

### Deep Reinforcement Learning

The SCIM problem is formulated as a Markov Decision Process (MDP):

- **State**: Inventory levels at all locations + recent demand history
- **Action**: Production quantity + shipment amounts to each warehouse
- **Reward**: Negative of total cost (including MSP-optimized logistics cost)

We primarily use **Proximal Policy Optimization (PPO)** for its stable convergence via clipped surrogate objectives, with **A3C** as a secondary baseline.

### Multi-Stage Stochastic Programming

MSP optimizes short-term logistics by modeling demand uncertainty through a scenario tree. It determines optimal shipping quantities, vehicle assignments, and inventory updates subject to capacity, balance, and vehicle constraints. The MSP horizon is kept short (e.g., 4 steps) to maintain computational tractability.

### Hybrid Integration (DRLBD)

```
┌─────────────────────────────────────────────────┐
│                  DRLBD Loop                      │
│                                                  │
│   DRL Agent (PPO)                                │
│   ├─ Observes: inventory levels, demand history  │
│   ├─ Decides: production quantity x_t            │
│   └─ Receives: reward = −(total cost from MSP)   │
│          │                                       │
│          ▼                                       │
│   MSP Module (Gurobi)                            │
│   ├─ Input: x_t + current state + scenario tree  │
│   ├─ Optimizes: shipping z_t, vehicles y_t       │
│   └─ Output: logistics cost → DRL reward         │
└─────────────────────────────────────────────────┘
```

## Experiments

### Basic Scenario

- **Setting**: *J* = 2 warehouses, *T* = 7 days, Bernoulli noise ε ~ B(0.5)
- **Benchmark**: Exact MSP solution (scenario tree tractable with Gurobi)

| Algorithm | Opt-gap (%) |
|---|---|
| **DRLBD** | **8.25 (±2.4)** |
| PPO | 26.91 (±10.2) |
| A3C | 30.43 (±12.3) |
| (s, Q) | 66.87 (±31.5) |
| EVP | 121.58 (±63.2) |
| PI | −6.52 (±3.1) |

### Advanced Scenario

- **Setting**: *J* = 5 or 10 warehouses, *T* = 12 months, negative binomial noise
- **Benchmark**: Perfect Information (PI) lower bound

| Algorithm | EVPI-gap (%) J=5 | EVPI-gap (%) J=10 |
|---|---|---|
| **DRLBD** | **102.46 (±23)** | **111.37 (±20)** |
| PPO | 137.52 (±35) | 146.18 (±28) |
| A3C | 141.23 (±33) | 150.40 (±29) |
| (s, Q) | 151.90 (±36) | 194.05 (±44) |
| EVP | 555.67 (±166) | 584.92 (±130) |

### Cumulative Costs (Basic Scenario, 250 Episodes)

| Algorithm | Average Cost |
|---|---|
| **DRLBD** | **145 (±5)** |
| PPO | 180 (±13) |
| (s, Q) | 199 (±24) |
| MSP | 134 (±5) |
| EVP | 402 (±222) |
| PI | 89 (±12) |

DRLBD consistently achieves the lowest cost and tightest variance among all learning-based methods, approaching the theoretical MSP optimum.

### Training Efficiency (J = 10)

| Algorithm | Training Time (min) | Inference Time (ms) |
|---|---|---|
| DRLBD | 7.15 | 33 |
| PPO | 7.10 | 24 |
| A3C | 7.85 | 29 |
| (s, Q) | 6.40 | 13 |

## Project Structure

```
.
├── SCIMAI-Gym-main/              # Core codebase (Jupyter Notebooks)
│   ├── *.ipynb                   # Experiment notebooks
│   ├── Paper_Results/            # Pre-trained checkpoints & result plots
│   └── Supplementary_Material/
├── gurobi.log                    # Gurobi solver log
├── .idea/                        # IDE configuration
└── .gitattributes
```

## Environment Parameters

| Parameter | Basic | Advanced (J=5 / J=10) |
|---|---|---|
| Distribution warehouses (*J*) | 2 | 5 / 10 |
| Time horizon (*T*) | 7 days | 12 months |
| Demand amplitude (*D*) | 5 | 2 |
| Demand period (*P*) | 5 | 6 |
| Noise distribution | Bernoulli(0.5) | NegBin(r=3, p=0.7) |
| Max production | — | 13 / 25 |
| Factory capacity | — | 18 / 36 |
| Warehouse capacity | — | 6 |
| Backorder cost | — | 10 |

## Requirements

- Python 3.7+
- [OpenAI Gym](https://github.com/openai/gym) 0.19.0
- [Ray / RLlib](https://github.com/ray-project/ray) 1.5.2 (PPO, A3C implementations)
- [Ax](https://github.com/facebook/Ax) 0.2.1 (Bayesian Optimization for (s, Q) policy)
- [Gurobi](https://www.gurobi.com/) (MSP solver — requires license)
- [Matplotlib](https://github.com/matplotlib/matplotlib) 3.4.3

### Installation

```bash
pip install gym==0.19.0 ray[rllib]==1.5.2 ax-platform==0.2.1 matplotlib==3.4.3
```

> **Note**: Gurobi requires a separate installation and license. Free academic licenses are available at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/LynnUoE/Optimizing-Supply-Chain-Decision-Through-Reinforcement-Learning.git
   cd Optimizing-Supply-Chain-Decision-Through-Reinforcement-Learning
   ```

2. **Open the experiment notebook** in `SCIMAI-Gym-main/` and run sections in order:
   - **Environment Setup** — Install dependencies
   - **Reinforcement Learning Classes** — Define the supply chain Gym environment
   - **Global Parameters** — Set seed, episode count (250 recommended), output directory
   - **Supply Chain Environment Initialization** — Instantiate and verify configuration

3. **Train baselines**
   - Run **Oracle / PI** sections for upper/lower bound benchmarks
   - Run **(s, Q)-Policy** sections with Ax for Bayesian Optimization baseline
   - Run **MSP / EVP** sections for stochastic programming baselines

4. **Train DRL agents**
   - Configure PPO and A3C hyperparameters (network architecture, learning rate, batch size)
   - Run **RL Train Agents** to train via Ray Tune with ASHA scheduling

5. **Train DRLBD**
   - Run the hybrid DRLBD training which integrates PPO decisions with MSP logistics optimization

6. **Evaluate**
   - Run **Final Results** to compare opt-gap / EVPI-gap and cumulative costs across all methods

## Algorithms Compared

| Method | Type | Description |
|---|---|---|
| **DRLBD** | Hybrid (proposed) | PPO for production + MSP for logistics; cost feedback loop |
| PPO | DRL | Proximal Policy Optimization — clipped surrogate objective |
| A3C | DRL | Asynchronous Advantage Actor-Critic — parallel workers |
| (s, Q) | Classical | Reorder when stock < *s*, order quantity *Q*; tuned via BO |
| EVP | Deterministic | Expected Value Problem — assumes mean demand |
| MSP | Stochastic Programming | Exact multi-stage optimization (basic scenario only) |
| PI | Oracle | Perfect Information — knows future demand (lower bound) |

## Key Findings

- **DRLBD achieves the lowest optimality gap** (8.25%) in the basic scenario, compared to 26.91% for standalone PPO and 30.43% for A3C.
- **DRLBD scales better** than all baselines as supply chain complexity increases (J = 5 → 10), with EVPI-gap growing only modestly.
- **Classical methods degrade sharply**: (s, Q) and EVP policies show dramatic performance loss under high uncertainty and larger networks.
- **Training is efficient**: all DRL methods train in ~7 minutes, with millisecond-level inference times suitable for real-time decision-making.

## References

This project builds on the SCIMAI-Gym framework. Key references:

```bibtex
@article{Stranieri2024,
  title   = {Combining deep reinforcement learning and multi-stage stochastic
             programming to address the supply chain inventory management problem},
  journal = {International Journal of Production Economics},
  volume  = {268},
  pages   = {109099},
  year    = {2024},
  author  = {Stranieri, Francesco and Fadda, Edoardo and Stella, Fabio},
  doi     = {10.1016/j.ijpe.2023.109099}
}

@misc{Stranieri2022,
  title     = {Comparing Deep Reinforcement Learning Algorithms in
               Two-Echelon Supply Chains},
  author    = {Stranieri, Francesco and Stella, Fabio},
  year      = {2022},
  publisher = {arXiv},
  doi       = {10.48550/ARXIV.2204.09603}
}
```

## License

This project incorporates the [SCIMAI-Gym](https://github.com/frenkowski/SCIMAI-Gym) library, released under the [MIT License](https://github.com/frenkowski/SCIMAI-Gym/blob/main/LICENSE).
