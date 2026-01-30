# Inclusive Healthcare Simulation

Imagine policy-makers could anticipate the impact of inclusive legal policies using a simulation tool.
Imagine they could explore how policies affect the most disadvantaged groups of people in specific contexts, such as people experiencing homelessness (PEH) in healthcare. 

This repository contains the first step towards this goal: an agent-based simulation framework for policy design in inequity contexts. We define a **multi-agent reinforcement learning** (MARL) environment where agents behave to restore their capabilities under the constraints of a given policy, and examine how well are they able to thrive in different scenarios. We specifically track their opportunities (capabilities), and see how these are deprived, restored or even expanded at different instants of time. 

These are the learnt strategies and outcomes with POLICY ON and POLICY OFF for N = 4 PEH agents, M= 8 social worker agents and S = 7x7 environment size: 

### Side-by-Side Comparison
![Comparison](output/run_20260130-120717/policy_evolution_comparison.gif)

Building upon [Aguilera et al. (2024)](https://arxiv.org/abs/2503.18389) — *Agent-based Modeling meets the Capability Approach for Human Development: Simulating Homelessness Policy-making*. arXiv:2503.18389 [cs.AI] and [Aguilera et al. (2025)](https://arxiv.org/abs/2503.18389) — *Barriers to Healthcare: Agent-Based Modeling to Mitigate Inequity*. arXiv:2507.23644 [cs.AI] 

# Optimal Results 
Using Q-learning and PBRS, this is the optimal strategy of the environment:
![Results](output/results.png)

To obtain these results, execute from root directory: python -m learning.qpbrs

## Features 
- Heterogenous agent behaviour: profile-dependent reward shaping
- Binary state transitions: environmental and legal constraints
- Capability expansion/restoration tracking: bodily health and affiliation

## Next Steps 
- Scale the number of agents
- Visualize it in OSM space
- Expand the action and state space for a holistic modelling and evaluation

## Installation
```bash
pip install -r requirements.txt
python -m learning.synthetic_data  # generates synthetic population and calibrated parameters based on data
python -m learning.qpbrs # generates pbrs learning and policy evaluation results for a particular N and S
