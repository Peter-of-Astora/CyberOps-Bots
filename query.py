import json
import re
import requests

def llm_query(prompt, last_message=None, system_prompt=None):
    """
    带记忆的对话函数
    :param prompt: 当前用户输入
    :param last_message: 上一次的对话内容（格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]）
    :param system_prompt: 系统提示词，用于设置AI的行为和角色
    :return: AI回复内容,以及这一次的对话历史（用于传入下一次对话）
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"  # Modify the URL
    
    # 构造消息列表
    messages = []
    
    # 添加系统提示词（如果提供且不在已有历史中）
    if system_prompt:
        # 检查上次对话历史中是否已包含系统提示
        system_exists = False
        if last_message:
            for msg in last_message:
                if msg.get("role") == "system":
                    system_exists = True
                    break
        
        # 如果不存在系统提示，则添加
        if not system_exists:
            messages.append({"role": "system", "content": system_prompt})
    
    # 添加上次对话历史（如果存在）
    if last_message:
        messages.extend(last_message)  # 添加上次对话历史
    
    # 添加当前用户输入
    messages.append({"role": "user", "content": prompt})  # 添加当前输入
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": messages,   # 传入完整上下文
        "enable_thinking": False,   # 使用推理模型时，是否使用推理
        "thinking_budget": 512,  # 限制思考的token数 
    }
    headers = {
        "Authorization": "Bearer Your API Key",  # Use Your Key Here
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    ai_response = response_data['choices'][0]['message']['content']
    
    # 返回AI回复和更新后的对话历史（仅保留最近一轮）
    return ai_response, [messages[-1], {"role": "assistant", "content": ai_response}]

def get_prompt(observation_tuple, subnet_node_num, value_node_num):
    # 接收观察信息
    depths, depth_speeds, iso_nums, compromised_nums, concentrations = observation_tuple
    # 得到子网数量
    subnet_num = len(compromised_nums)
    # 读取系统提示词
    sys_prompt = ""
    with open('/root/work/project/prompts/sys_prompt.txt', 'r', encoding='utf-8') as f:
        for line in f:   
            sys_prompt += line
    # 读取子网上下文
    subnet_context = ""
    with open('/root/work/project/prompts/subnet_context.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        subnet_context = ''.join(lines[ :subnet_num]).strip()
    # 读取工具列表
    from prompts import tool_list
    tools = tool_list.tools
    # 计算重要节点索引
    value_node_indexes = [subnet_node_num - i - 1 for i in range(value_node_num)]
    # 在系统提示词中加入所需信息
    sys_prompt = sys_prompt % (subnet_num, subnet_node_num, value_node_indexes, subnet_context, tools)
    # 读取用户提示词
    user_prompt = ""
    with open('/root/work/project/prompts/user_prompt.txt', 'r', encoding='utf-8') as f:
        for line in f:
            user_prompt += line
    # 在用户提示词中加入observation内容
    for i in range(subnet_num):
        user_prompt += "For Subnet {}, Critical Distance: {}, Penetration Speed: {}, Compromised Nodes: {}, Isolated Nodes: {}, Attack Path Concentration: {}.\n".format(i, depths[i], depth_speeds[i], compromised_nums[i], iso_nums[i], concentrations[i])
    
    return sys_prompt, user_prompt
    
def parse_response(response):
    # 尝试匹配 ```json 和 ``` 之间的内容
    pattern = r'```json\s*(.*?)\s*```'
    match_json = re.search(pattern, response, re.DOTALL)
    if match_json:
        # 提取代码块内的内容
        response = match_json.group(1)
    else:
        pass
    
    # 解析JSON字符串
    data = json.loads(response) 

    # 提取Action名称和参数
    action_name = data['Action']['Name']
    action_parameters = data['Action']['Parameter']
    
    # 提取thought
    thought = data.get('Thought')
    
    return action_name, action_parameters, thought
    

if __name__ == "__main__":
    # # 第一次调用（无历史）
    # response1, last_message = llm_query("My name is Peter.")
    # print("AI:", response1)

    # # 第二次调用（传入上一次的对话历史）
    # response2, last_message = llm_query("What's my name?", last_message, 'You have to answer in Chinese')
    # print("AI:", response2)
    
    sys_prompt, user_prompt = get_prompt(([1, 5], [2, 1], [3, 1], [6, 2], [1, 1]), 30, 2)
    
    response, last_message = llm_query(user_prompt, system_prompt=sys_prompt)
    print(response)
    
    # response, last_message = llm_query(user_prompt, last_message=last_message, system_prompt=sys_prompt)
    # print(response)
    
    action_name, action_parameters, thought = parse_response(response)
    print(type(action_parameters))
    
