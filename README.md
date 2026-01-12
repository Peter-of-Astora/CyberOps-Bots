# CyberOps-Bots
A hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs).
# Introduction
The CyberOps-Bots framework is a hierarchical multi-agent system designed to enhance cloud network resilience against dynamic and evolving cyber threats. It synergizes **Large Language Models** (LLMs) for high-level tactical planning and **Multi-Agent Reinforcement Learning** (MARL) for localized defense execution, enabling adaptive, interpretable, and human-in-the-loop (HITL) cyber defense operations.

The framework operates through three coordinated layers:

1.  **Env Layer**: Simulates a dynamic adversarial cloud environment where attackers perform lateral movement across virtual subnets. We use YawningTitan to implement that.
2.  **LLM Layer**: Serves as the global decision-maker, integrating four modules:
    1.  **Perception**: Converts structured network states into natural language descriptions.
    2.  **Planning**: Employs the ReAct paradigm for multi-step reasoning and tactical planning.
    3.  **Memory**: Tracks attack chains via Short-Term Memory (STM) and Long-Term Memory (LTM) for proactive defense.
    4.  **Action/Tool Integration**: Dispatches tactical commands to lower-layer RL agents.
3.  **RL Layer**: Comprises pre-trained, functionally heterogeneous RL agents (e.g., Fortify, Recover, Purge, Block) that execute atomic defense actions (e.g., patching, isolating, resetting nodes) within localized subnets.

# Requirements
+ Python = 3.9
+ See `requirements.txt`

# Usage
## LLM
+ Method 1: Modify the URL and API KEY in `query.py` to invoke the LLM API.
+ Method 2: Modify `query.py` to deploy a local LLM for more efficient inference.

## Context
Define the network scenarios in `prompts/subnet_context.txt`, with information for each network segment written line by line.

## Tools
Configure the lower-level RL agents, along with their corresponding tool descriptions and parameter formats, in `prompts/tool_list.py`.

## Heterogeneous Separated Pre-training
1.  Design the sub-observation functions for lower-level RL agents in `net_env.py` (implemented as `iso_observe()`, `recover_observe()`, etc., in the uploaded code).
2.  Design the sub-action execution transfer function `action_transfer()` for lower-level RL agents in `net_env.py`.
3.  Design the reward functions for lower-level RL agents in `net_env.py`.
4.  Design the termination condition and reset functions for lower-level RL agents in `net_env.py` (implemented as `iso_reset()` and `judge_terminate()` in the uploaded code).
5.  Specifically configure the training workflow and scenario settings for each lower-level RL agent in `sub_env.py`, including attack strategies, number of attackers, network scale, node configurations, and training algorithms.
6.  Use `train_sub_agent` and `test_sub_agent` in `sub_env.py` to train and test the RL agents. The pre-trained agent parameters will be saved in `rl_models`.

## Run
Use the `WorkFlowRunner` class in `workflow_runner.py` to execute the workflow.

## Human Instructions Integration
Modify `prompts/user_prompt.txt` or `prompts/sys_prompts.txt` to inject human instructions or additional prior knowledge.

## Baselines
+ Configs: See YAML files in `configs`
+ Run: Use code in `tests`
