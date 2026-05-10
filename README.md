# CASR: Cache-Based Adaptive Scheduler for Serverless Computing

## Implementation and Evaluation Study



![Python](https://img.shields.io/badge/Python-3.11-blue)




![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)




![License](https://img.shields.io/badge/License-MIT-green)




![Status](https://img.shields.io/badge/Status-Complete-brightgreen)



---

## Overview

This repository presents a complete from-scratch
implementation of the CASR algorithm published in:

> Chen et al., "CASR: Cache-Based Adaptive Scheduler
> for Serverless Runtime", Future Generation Computer
> Systems, 2025.

### What Makes This Project Unique
- Implemented entirely from scratch in Python
- Evaluated on real Microsoft Azure Functions 2019 dataset
- Novel K=4 queue granularity experiment extending the paper
- Complete Jupyter notebook with live interactive demos

---

## Key Results

### CASR vs 5 Baseline Algorithms (K=3, 200 Episodes)

| Workload | CASR Cold% | FaaSCache Cold% | Hist WMT | CASR WMT |
|----------|-----------|-----------------|----------|----------|
| Common | 88.972% | 99.999% | 24.970s | 0.000s |
| Significant | 96.152% | 100.000% | 25.705s | 0.000s |
| Random | 85.070% | 99.999% | 12.282s | 0.000s |

### Novel K=4 Experiment Results

| Workload | K=3 Cold% | K=4 Cold% | Improvement |
|----------|-----------|-----------|-------------|
| Common | 88.972% | 88.811% | 0.161% better |
| Significant | 96.152% | 90.252% | 5.900% better |
| Random | 85.070% | 81.655% | 3.415% better |

### Key Findings
- CASR achieves **zero wasted memory time** across all workloads
- CASR outperforms FaaSCache by **3.8% to 14.9%** in cold start rate
- K=4 queues improve cold start by up to **5.9%** over original K=3
- Both K=3 and K=4 maintain **0.000s WMT** consistently

---

## Project Structure

```
CASR_Project/
├── config.py          ← All hyperparameters and settings
├── simulator.py       ← Azure dataset loader and simulator
├── scache.py          ← W-TinyLFU S-Cache implementation
├── environment.py     ← Reinforcement learning environment
├── ppo_agent.py       ← PPO agent with Actor-Critic networks
├── baselines.py       ← 5 baseline algorithm implementations
├── train.py           ← PPO training loop
├── evaluate.py        ← Complete evaluation framework
├── results/           ← K=3 results and graphs
├── results_k4/        ← K=4 experiment results and graphs
├── trained_model/     ← K=3 trained PPO model
├── trained_model_k4/  ← K=4 trained PPO model
└── CASR_Implementation_Study.ipynb
```

---

## Algorithm Overview

### CASR has Two Core Components

**Component 1: S-Cache**

Organizes containers into K queues by cold start overhead:

| Queue | Range | Function Type | Azure Distribution |
|-------|-------|---------------|-------------------|
| 0 | 0 to 1 second | Lightweight HTTP | 9.4% |
| 1 | 1 to 60 seconds | Medium APIs | 85.3% |
| 2 | 60+ seconds | Heavy ML | 5.0% |

Each queue uses W-TinyLFU caching:
- Window cache for new containers
- Main cache for frequently used containers
- Automatic eviction of least useful containers

**Component 2: PPO Reinforcement Learning**

AI agent that dynamically scales queue capacities:
- State: 21-dimensional vector (7 metrics per queue)
- Action: 27 possible scaling combinations (3^3)
- Reward: Weighted combination of cold starts and WMT
- Decision frequency: Every 10,000 function invocations

**Key Innovation:**

Available Containers = Total in Memory - Currently Running

This accounts for busy containers that cannot serve
new requests, leading to smarter eviction decisions!

---

## Novel Contribution: K=4 Experiment

Original paper used K=3 queues. We extended this
by splitting the dominant Queue 1 (85.3% of calls):

| Queue | Range | Type |
|-------|-------|------|
| 0 | 0 to 1 second | Lightweight |
| 1 | 1 to 30 seconds | Medium Light |
| 2 | 30 to 60 seconds | Medium Heavy |
| 3 | 60+ seconds | Heavy ML |

**Finding:** K=4 improves cold start rate by up to
5.900% while maintaining zero wasted memory time!

---

## Dataset

**Microsoft Azure Functions 2019**
- 14 days of real production cloud data
- 1,332,032 function calls per day
- Real anonymized function traces
- Download: https://github.com/Azure/AzurePublicDataset

---

## Installation

### Requirements
- Python 3.11
- Windows / Linux / Mac

### Setup Steps

**Step 1: Clone repository**
```
git clone https://github.com/Krishn4nmol/CASR_Project.git
cd CASR_Project
```

**Step 2: Create virtual environment**
```
python -m venv casr_env
casr_env\Scripts\activate
```

**Step 3: Install packages**
```
pip install -r requirements.txt
```

**Step 4: Download Azure dataset**

Download from Azure Public Dataset repository.
Place CSV files in data/ folder.

---

## How to Run

### Train PPO Agent

python train.py full

Training takes approximately 5 minutes for 200 episodes.

### Evaluate All Algorithms

python evaluate.py

Evaluation takes approximately 2 hours with cooling breaks.

### View Interactive Results

jupyter notebook

Open CASR_Implementation_Study.ipynb

---

## Training Results

### PPO Convergence (K=3)

Total episodes:   200
Best reward:      -0.0447
Training time:    ~5 minutes
WMT during training: 0.000s always

### PPO Convergence (K=4)

Total episodes:   200
Best reward:      -0.1067
Training time:    ~5 minutes
State dimensions: 28 (4 x 7)
Action space:     81 (3^4)

---

## Branches

| Branch | Description | Contents |
|--------|-------------|----------|
| main | K=3 implementation | Core code + K=3 results |
| k4-experiment | K=4 extension | K=4 results + Jupyter notebook |

---

## Baseline Algorithms

| Algorithm | Strategy | WMT | Cold Start |
|-----------|----------|-----|------------|
| S-Cache | Pure W-TinyLFU caching | 0.000s | ~88% |
| LCS | Least Cold Start eviction | ~7s | ~87% |
| FaaSCache | Fixed window keep-alive | 0.000s | ~100% |
| Hist | Historical frequency | ~25s | ~61% |
| Fixed | Fixed 10 min keep-alive | ~1-9s | ~51% |

---

## Why CASR Wins

Hist achieves lowest cold starts (61%)
BUT wastes 25 seconds per invocation!

Fixed achieves 51% cold start rate
BUT wastes 1-9 seconds per invocation!

CASR achieves:
✅ Zero wasted memory time
✅ Beats FaaSCache by 3.8-14.9%
✅ Sustainable for production use
✅ Automatically adapts to workload

---

## Author

**Anmol Krishna**
Student Researcher
GitHub: [Krishn4nmol](https://github.com/Krishn4nmol)

---

## Reference

Chen et al., "CASR: Cache-Based Adaptive Scheduler
for Serverless Runtime", Future Generation Computer
Systems, 2025.

---

## License

MIT License - Free to use and modify for research purposes.