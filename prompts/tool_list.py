import json

assign_agents_tool = {
    "Name": "assign_agent", 
    "Description": "This tool enables the LLM to allocate different types and quantities of RL defense agents (reset, isolate, patch, recover) to various subnets. The input parameters must be provided in JSON format. The JSON object must contain exactly four keys: 'reset', 'isolate', 'patch', and 'recover'. The value for each key must be a list of integers representing the quantity of that specific agent type to be deployed to each corresponding subnet. You can assign 4 agents in total. For example, when there are 3 subnets, to assign 1 reset agent to subnet_0, 1 reset agent to subnet_1; and 2 recover agents to subnet_2: {'reset': [1, 1, 0], 'isolate': [0, 0, 0], 'patch': [0, 0, 0], 'recover': [0, 0, 2]}", 
    "Parameters": {"reset": 'allocation list', "isolate": 'allocation list', "patch": 'allocation list', "recover": 'allocation list'}
}

execute_action_tool = {
    "Name": "execute_action", 
    "Description": "This tool allows the LLM to execute a specific defensive action (reset, isolate, patch, or recover) on a particular node within a specified subnet. The input parameters must be provided in JSON format containing three required keys: 'subnet_index' (integer indicating the target subnet), 'node_index' (integer identifying the specific node within that subnet), and 'action_name' (string specifying the action to perform: 'reset', 'isolate', 'patch', or 'recover'). For example, to isolate the node_20 in subnet_9: {'subnet_index': 9, 'node_index': 20, 'action_name': 'isolate'}", 
    "Parameters": {"subnet_index": 'index of the subnet', "node_index": 'index of the node', "action_name": "name of the action"}
}

tool_data = [assign_agents_tool, execute_action_tool]
tools = json.dumps(tool_data, indent=4)

if __name__ == "__main__":
    print(tools)
