# CyberOps-Bots
A hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs).
# Introduction
The CyberOps-Bots framework is a hierarchical multi-agent system designed to enhance cloud network resilience against dynamic and evolving cyber threats. It synergizes **Large Language Models** (LLMs) for high-level tactical planning and **Multi-Agent Reinforcement Learning** (MARL) for localized defense execution, enabling adaptive, interpretable, and human-in-the-loop (HITL) cyber defense operations.

The framework operates through three coordinated layers:

1. **Env Layer**: Simulates a dynamic adversarial cloud environment where attackers perform lateral movement across virtual subnets. 	We use YawningTitan to implement that.
2. **LLM Layer**: Serves as the global decision-maker, integrating four modules:
    1. **Perception**: Converts structured network states into natural language descriptions.
    2. **Planning**: Employs the ReAct paradigm for multi-step reasoning and tactical planning.
    3. **Memory**: Tracks attack chains via Short-Term Memory (STM) and Long-Term Memory (LTM) for proactive defense.
    4. **Action/Tool Integration**: Dispatches tactical commands to lower-layer RL agents.
3. **RL Layer**: Comprises pre-trained, functionally heterogeneous RL agents (e.g., Fortify, Recover, Purge, Block) that execute atomic defense actions (e.g., patching, isolating, resetting nodes) within localized subnets.

# Requirements
+ Python = 3.9
+ See `requirements.txt`

# Usage
## LLM
+ Method 1: Modify the URL and API KEY in `query.py` to invoke the LLM API.
+ Method 2: Modify `query.py` to deploy a local LLM for more efficient inference.

## Context
<font style="color:rgb(55, 65, 81);">Define the network scenarios in </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">prompts/subnet_context.txt</font>**`<font style="color:rgb(55, 65, 81);">, with information for each network segment written line by line.</font>

## Tools
<font style="color:rgb(55, 65, 81);">Configure the lower-level RL agents, along with their corresponding tool descriptions and parameter formats, in </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">prompts/tool_list.py</font>**`<font style="color:rgb(55, 65, 81);">.</font>

## Heterogeneous Separated Pre-training
1. <font style="color:rgb(55, 65, 81);">Design the sub-observation functions for lower-level RL agents in </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">net_env.py</font>**`<font style="color:rgb(55, 65, 81);"> (implemented as </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">iso_observe()</font>**`<font style="color:rgb(55, 65, 81);">, </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">recover_observe()</font>**`<font style="color:rgb(55, 65, 81);">, etc., in the uploaded code).</font>
2. <font style="color:rgb(55, 65, 81);">Design the sub-action execution transfer function</font><font style="color:rgb(55, 65, 81);"> </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">action_transfer()</font>**`<font style="color:rgb(55, 65, 81);"> </font><font style="color:rgb(55, 65, 81);">for lower-level RL agents in</font><font style="color:rgb(55, 65, 81);"> </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">net_env.py</font>**`<font style="color:rgb(55, 65, 81);">.</font>
3. <font style="color:rgb(55, 65, 81);">Design the reward functions for lower-level RL agents in</font><font style="color:rgb(55, 65, 81);"> </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">net_env.py</font>**`<font style="color:rgb(55, 65, 81);">.</font>
4. <font style="color:rgb(55, 65, 81);">Design the termination condition and reset functions for lower-level RL agents in</font><font style="color:rgb(55, 65, 81);"> </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">net_env.py</font>**`<font style="color:rgb(55, 65, 81);"> </font><font style="color:rgb(55, 65, 81);">(implemented as</font><font style="color:rgb(55, 65, 81);"> </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">iso_reset()</font>**`<font style="color:rgb(55, 65, 81);"> </font><font style="color:rgb(55, 65, 81);">and</font><font style="color:rgb(55, 65, 81);"> </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">judge_terminate()</font>**`<font style="color:rgb(55, 65, 81);"> </font><font style="color:rgb(55, 65, 81);">in the uploaded code).</font>
5. <font style="color:rgb(55, 65, 81);">Specifically configure the training workflow and scenario settings for each lower-level RL agent in</font><font style="color:rgb(55, 65, 81);"> </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">sub_env.py</font>**`<font style="color:rgb(55, 65, 81);">, including attack strategies, number of attackers, network scale, node configurations, and training algorithms.</font>
6. <font style="color:rgb(55, 65, 81);">Use </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">train_sub_agent</font>**`<font style="color:rgb(55, 65, 81);"> and </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">test_sub_agent</font>**`<font style="color:rgb(55, 65, 81);"> in </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">sub_env.py</font>**`<font style="color:rgb(55, 65, 81);"> to train and test the RL agents. The pre-trained agent parameters will be saved in </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">rl_models</font>**`<font style="color:rgb(55, 65, 81);">.</font>

## Run
<font style="color:rgb(55, 65, 81);">Use the </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">WorkFlowRunner</font>**`<font style="color:rgb(55, 65, 81);"> class in </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">workflow_runner.py</font>**`<font style="color:rgb(55, 65, 81);"> to execute the workflow.</font>

## Human Instructions Integration
<font style="color:rgb(55, 65, 81);">Modify </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">prompts/user_prompt.txt</font>**`<font style="color:rgb(55, 65, 81);"> or </font>`**<font style="color:rgb(17, 24, 39);background-color:rgb(236, 236, 236);">prompts/sys_prompts.txt</font>**`<font style="color:rgb(55, 65, 81);"> to inject human instructions or additional prior knowledge.</font>

## Baselines
+ Configs: See YAML files in `configs`
+ Run: Use code in `tests`
