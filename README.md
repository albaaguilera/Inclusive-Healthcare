# Inclusive Healthcare Simulation

Imagine policy-makers could anticipate the impact of legal policies using a simulation tool.
Imagine they could explore how policies affect the most disadvantaged groups of people and provide the inclusive support they need in time. 

This repository contains the first step towards this goal: an agent-based simulation framework for policy design in inequity contexts. We define a **multi-agent reinforcement learning** (MARL) environment where agents behave to restore their capabilities under the constraints of a given policy, and examine how well are they able to thrive in different scenarios. We specifically track their opportunities (capabilities), and see how these are deprived, restored or even expanded at different instants of time. 

This is a random run of the environment for a multi-agent (SA): 
<table>
<tr>
<td align="center"><strong>Single-Agent Behaviour</strong><br><img src="output/environment.png" width="600"/></td>
</tr>
</table>

Building upon [Aguilera et al. (2024)](https://arxiv.org/abs/2503.18389) — *Agent-based Modeling meets the Capability Approach for Human Development: Simulating Homelessness Policy-making*. arXiv:2503.18389 [cs.AI] and [Aguilera et al. (2025)](https://arxiv.org/abs/2503.18389) — *Barriers to Healthcare: Agent-Based Modeling to Mitigate Inequity*. arXiv:2507.23644 [cs.AI] 

Created using Python's Gymnasium and PettingZoo libraries.

# Current Results 
![Results](output/results.png)

To obtain these results, execute from root directory: python -m learning.qlearningMAexplore

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
