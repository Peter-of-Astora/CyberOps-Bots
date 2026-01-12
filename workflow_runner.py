from net_env import NetEnv
from sub_env import SubTrainEnv
from stable_baselines3 import PPO, DQN, A2C # type: ignore
import numpy as np
import attack
import query
from datetime import datetime
from tqdm import tqdm
import time

class WorkFlowRunner():
    def __init__(self, subnet_num, subnet_node_num, entry_node_num, value_node_num, agent_num, total_step, 
                attack_mode='recon', attack_intensity=2, algorithm='Ours'):  # 修改agent_num时，注意同步修改tool_list中的描述
        # 接收相关参数
        self.subnet_num = subnet_num
        self.subnet_node_num = subnet_node_num
        self.entry_node_num = entry_node_num
        self.value_node_num = value_node_num
        self.agent_num = agent_num
        self.total_step = total_step
        self.attack_mode = attack_mode
        self.attack_intensity = attack_intensity
        # 实例化环境
        self.env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode=attack_mode, attack_intensity=attack_intensity, algorithm=algorithm)
        self.env.reset()
        # 实例化子环境类（只用于实例化子智能体）
        self.sub_env = SubTrainEnv(1, 30, 2, 2, 'patch')
        # 加载子智能体
        self.isolate_agent = DQN.load('/root/work/project/rl_models/isolate_agent', env=self.sub_env)
        self.patch_agent = DQN.load('/root/work/project/rl_models/patch_agent', env=self.sub_env)
        self.recover_agent = DQN.load('/root/work/project/rl_models/recover_agent', env=self.sub_env)
        self.reset_agent = DQN.load('/root/work/project/rl_models/reset_agent', env=self.sub_env)
        # 短期记忆：上一次的对话记录
        self.short_term_memory = None
        # 长期记忆：保存最深攻击链与对应的深度（可选）
        self.long_term_memory = None
        # 状态空间指标变量，以属性保存方便计算差值
        self.depths = []  # 渗透深度：关键节点到攻陷节点的距离，注意是越小渗透越深
        self.depth_speeds = []  # 渗透速度
        self.iso_nums = []  # 隔离节点
        self.compromised_nums = []  # 攻陷节点
        self.concentrations = []  # 攻击链集中度
        # 渗透深度初始化
        interfaces = self.env.interface_list
        for interface in interfaces:
            # 计算各子网渗透深度
            deep_node_index, depth = attack.calculate_penetrate_depth(interface)
            self.depths.append(depth)
        # 记录LLM相关日志
        self.log_file = "/root/work/project/log_run/llm_log.txt"
        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d-%H%M")
        with open(self.log_file, 'w') as f:
            f.write("LLM Log - {} - Initialized\n".format(formatted_time))
        
    def get_obs_matrix(self, attack_history):
        '''
        获取将状态空间转换为自然语言的指标数据\n
        attack_history: 各子网攻击历史
        '''
        # 获取各个子网的NetworkInterface
        interfaces = self.env.interface_list
        # 存储各指标
        depths = []  # 渗透深度
        depth_speeds = []  # 渗透速度
        iso_nums = []  # 隔离节点
        compromised_nums = []  # 攻陷节点
        concentrations = []  # 攻击链集中度
        # 再增加体系性，分不同的体系设计指标，再参考论文，分不同的方面来写，更有条理（也不用太多）
        # 只依赖这些数值与指标直接进行上层决策，没有体现大模型的作用，其实可以用规则来替代，大模型适合理解抽象/宏观的东西，直接用这些指标去掉了太多信息
        # 攻防场景（资产、威胁等）更重要，更需要去设计提示词
        # 遍历各子网计算
        for interface in interfaces:
            # 计算各子网渗透深度
            deep_node_index, depth = attack.calculate_penetrate_depth(interface)
            depths.append(depth)
            # 计算节点状态指标
            n_nodes = interface.get_total_num_nodes()
            obs = interface.get_current_observation()
            isolated_state = obs[n_nodes**2: n_nodes**2 + n_nodes]
            compromised_state = obs[n_nodes**2 + n_nodes: n_nodes**2 + 2 * n_nodes]
            isolated_node_num = np.count_nonzero(isolated_state == 1)
            compromised_node_num = np.count_nonzero(compromised_state == 1)
            iso_nums.append(isolated_node_num)
            compromised_nums.append(compromised_node_num)
        # 计算子网渗透速度
        if len(self.depths) != 0:
            for i in range(self.subnet_num):
                if self.depths[i] == -1 and depths[i] != -1:  # 如果本来不存在路径，即原来深度为-1，不能直接通过相减得到速度
                    speed = 4 - depths[i]
                else:
                    speed = self.depths[i] - depths[i]  # 注意是谁减去谁
                depth_speeds.append(speed)
        # 计算各子网攻击链集中度
        concentrations = attack.calculate_attack_concentration(attack_history)
        # 计算完后存储到对应属性中
        self.depths = depths
        self.depth_speeds = depth_speeds
        self.iso_nums = iso_nums
        self.compromised_nums = compromised_node_num
        self.concentrations = concentrations
        
        return depths, depth_speeds, iso_nums, compromised_nums, concentrations
    
    def assign_agents(self, parameters):
        '''
        执行assign_agents动作的函数\n
        参数是JSON格式的各类型agent的分配列表
        '''
        # 参数验证
        # Validate parameters type and structure
        if not isinstance(parameters, dict):
            raise TypeError("Parameter must be a dictionary")

        # Check required keys exist
        required_keys = ['reset', 'isolate', 'patch', 'recover']
        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Missing required key: '{key}'")
        
        # 验证参数存在后，再提取参数
        reset_list = parameters['reset']
        iso_list = parameters['isolate']
        patch_list = parameters['patch']
        recover_list = parameters['recover']
    
        # 1. 检查所有参数都是列表
        if not all(isinstance(param, list) for param in [reset_list, iso_list, patch_list, recover_list]):
            raise TypeError("The allocation list has to be a List.")
        
        # 2. 检查所有列表长度都为10
        if not all(len(param) == self.subnet_num for param in [reset_list, iso_list, patch_list, recover_list]):
            raise ValueError(f"The length of allocation list has to be equal to the number of subnets. Here we have {self.subnet_num} subnets.")
        
        # 3. 检查所有分配值都是非负整数
        all_allocations = reset_list + iso_list + patch_list + recover_list
        if not all(isinstance(x, int) and x >= 0 for x in all_allocations):
            raise ValueError("The assigned agent number has to be non-negative integers.")
        
        # 4. 检查总代理数
        total_agents = sum(all_allocations)
        if total_agents < self.agent_num:
            raise ValueError(f"You have to assign {self.agent_num} agents in total, but you assigned {total_agents}.")
        
        # 遍历各个子网分配RL智能体
        agent_count = 0
        for subnet_index in range(self.subnet_num):
            # 各智能体在该子网分配数量
            reset_allocated = reset_list[subnet_index]
            iso_allocated = iso_list[subnet_index]
            patch_allocated = patch_list[subnet_index]
            recover_allocated = recover_list[subnet_index]
            # 按数量循环执行智能体
            for i in range(reset_allocated):
                obs = self.env.reset_observe(subnet_index)
                action, _states = self.reset_agent.predict(obs, deterministic=True)
                action = self.env.action_transfer('reset', action, subnet_index)
                agent_count += 1
                if agent_count == self.agent_num:  # 最后一个执行的智能体要调用step方法
                    observation, reward, terminated, truncated, info = self.env.step(action)
                else:
                    self.env.defend(action)
            for i in range(iso_allocated):
                obs, _ = self.env.iso_observe(subnet_index)
                action, _states = self.isolate_agent.predict(obs, deterministic=True)
                action = self.env.action_transfer('isolate', action, subnet_index)
                agent_count += 1
                if agent_count == self.agent_num:
                    observation, reward, terminated, truncated, info = self.env.step(action)
                else:
                    self.env.defend(action)
            for i in range(patch_allocated):
                obs, _ = self.env.patch_observe(subnet_index)
                action, _states = self.patch_agent.predict(obs, deterministic=True)
                action = self.env.action_transfer('patch', action, subnet_index)
                agent_count += 1
                if agent_count == self.agent_num:
                    observation, reward, terminated, truncated, info = self.env.step(action)
                else:
                    self.env.defend(action)
            for i in range(recover_allocated):
                obs, _ = self.env.recover_observe(subnet_index)
                action, _states = self.recover_agent.predict(obs, deterministic=True)
                action = self.env.action_transfer('recover', action, subnet_index)
                agent_count += 1
                if agent_count == self.agent_num:
                    observation, reward, terminated, truncated, info = self.env.step(action)
                else:
                    self.env.defend(action)
        
        return terminated
            
    def execute_action(self, parameters):
        # 参数验证
        # Validate parameters type and structure
        if not isinstance(parameters, dict):
            raise TypeError("Parameters must be a dictionary")

        # Check required keys exist
        required_keys = ['subnet_index', 'node_index', 'action_name']
        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Missing required parameter: '{key}'")

        # Extract parameters
        subnet_index = parameters['subnet_index']
        node_index = parameters['node_index']
        action_name = parameters['action_name']

        # Validate subnet_index - non-negative integer within range
        if not isinstance(subnet_index, int):
            raise TypeError("subnet_index must be an integer")
        if subnet_index < 0:
            raise ValueError("subnet_index must be non-negative")
        if subnet_index >= self.subnet_num:
            raise ValueError(f"subnet_index must be less than {self.subnet_num}")

        # Validate node_index - non-negative integer within range
        if not isinstance(node_index, int):
            raise TypeError("node_index must be an integer")
        if node_index < 0:
            raise ValueError("node_index must be non-negative")
        if node_index >= self.subnet_node_num:
            raise ValueError(f"node_index must be less than {self.subnet_node_num}")

        # Validate action_name - must be one of the allowed actions
        valid_actions = ['reset', 'isolate', 'patch', 'recover']
        if not isinstance(action_name, str):
            raise TypeError("action_name must be a string")
        if action_name not in valid_actions:
            raise ValueError(f"action_name must be one of: {', '.join(valid_actions)}")
        
        # 再获取参数（其实是重复的，但是好看）
        subnet_index = parameters['subnet_index']
        node_index = parameters['node_index']
        action_name = parameters['action_name']
        # 获取所需变量
        interface = self.env.interface_list[subnet_index]
        node = self.env.nodes_list[subnet_index][str(node_index)]
        # 执行动作
        if action_name == 'reset':
            interface.make_node_safe(node)
            # 记录环境日志
            with open(self.env.log_file, 'a') as f:
                f.write("Defender Action: Reset Node {} in subnet {}\n".format(str(node_index), subnet_index))
        elif action_name == 'isolate':
            interface.isolate_node(node)
            # 记录环境日志
            with open(self.env.log_file, 'a') as f:
                f.write("Defender Action: Isolate Node {} in subnet {}\n".format(str(node_index), subnet_index))
        elif action_name == 'patch':
            node._vulnerability = max(node._vulnerability - 0.5, 0.01)
            # 记录环境日志
            with open(self.env.log_file, 'a') as f:
                f.write("Defender Action: Patch Node {} in subnet {}\n".format(str(node_index), subnet_index))
        elif action_name == 'recover':
            interface.reconnect_node(node)
            # 记录环境日志
            with open(self.env.log_file, 'a') as f:
                f.write("Defender Action: Reconnect Node {} in subnet {}\n".format(str(node_index), subnet_index))
    
    def run(self):
        # 初始化env，attack_history
        self.env.reset()
        attack_history = self.env.attack_history
        # 循环与环境交互
        for i in tqdm(range(self.total_step), unit="iter", ncols=80):
            terminated = False  # 终止判断
            flag = 0 # 判断LLM是否分配RL的标志值
            fail_count = 0  # 出错次数，不超过5，否则直接跳过回合
            fail_notice = ''  # 提示LLM不要再犯错误，放到user_prompt前面
            # 循环直到LLM决定分配RL agents，或失败太多次
            while flag == 0 and fail_count < 5:
                # 观察
                observation_tuple = self.get_obs_matrix(attack_history)
                # 获取提示
                sys_prompt, user_prompt = query.get_prompt(observation_tuple, self.subnet_node_num, self.value_node_num)
                # 如果已出错过，则将对应的出错提示放到user_prompt
                if fail_count != 0:
                    user_prompt = 'Last Error: ' + fail_notice + user_prompt
                # 如果渗透深度危险，则启动保存的长期记忆
                for subnet_index in range(self.subnet_num):
                    if self.depths[subnet_index] <= 2 and self.depths[subnet_index] != -1:
                        enter_path, predict_path = attack.provoke_long_memory(self.env.interface_list[subnet_index])
                        user_prompt = user_prompt + f"Warning: Subnet {subnet_index} in danger! Existed attack chain: {enter_path}. It is predicted that attack chain {predict_path} is likely to be conducted. It is suggested that isolate node {predict_path[0]} to secure important nodes (Use execute_action tool).\n"
                # 记录User日志
                with open(self.log_file, 'a') as f:
                    f.write('User:\n{}\n'.format(user_prompt))
                # 访问LLM
                if self.short_term_memory == None:  # 尚未保存短期记忆，直接对话
                    response, self.short_term_memory = query.llm_query(user_prompt, system_prompt=sys_prompt)
                else:  # 已保存短期记忆，带记忆对话
                    response, self.short_term_memory = query.llm_query(user_prompt, last_message=self.short_term_memory, system_prompt=sys_prompt)
                # 解析LLM输出并执行
                try:  # 先检查能否正常解析
                    action_name, action_parameters, thought = query.parse_response(response)
                except Exception as e:
                    # 记录解析异常
                    with open(self.log_file, 'a') as f:
                        f.write(f"Error: {type(e).__name__}: {e}\n")
                    # 在小循环中继续尝试
                    fail_count += 1
                    fail_notice = f"Error: {type(e).__name__}: {e}\n"
                    continue
                if action_name == 'assign_agent':
                    try:  # 尝试执行，避免LLM给出异常参数
                        terminated = self.assign_agents(action_parameters)
                    except (TypeError, ValueError) as e:
                        # 记录异常
                        with open(self.log_file, 'a') as f:
                            f.write(f"Error: {type(e).__name__}: {e}\n")
                        # 在小循环中继续尝试
                        fail_count += 1
                        fail_notice = f"Error: {type(e).__name__}: {e}\n"
                        continue
                    flag = 1  # 分配RL智能体后，置标志为1
                elif action_name == 'execute_action':
                    try:  # 尝试执行，避免LLM给出异常参数
                        self.execute_action(action_parameters)
                    except (TypeError, ValueError) as e:
                        # 记录异常
                        with open(self.log_file, 'a') as f:
                            f.write(f"Error: {type(e).__name__}: {e}\n")
                        # 在小循环中继续尝试
                        fail_count += 1
                        fail_notice = f"Error: {type(e).__name__}: {e}\n"
                        continue
                else:
                    # 选择无效动作，记录异常日志，在小循环中再尝试
                    with open(self.log_file, 'a') as f:
                        f.write('Invalid Action Chosen!\n')
                    fail_count += 1
                    fail_notice = 'Invalid Action Chosen!\n'
                    continue
                # 记录日志
                with open(self.log_file, 'a') as f:
                    f.write('LLM:\n{}\n'.format(response))
            # 判断终止
            if terminated == True:
                self.env.reset()
            # 更新attack_history
            attack_history = self.env.attack_history  # 因为调用assign_agents中的step后才会攻击，所以小循环内不用更新attack_history
            # 记录step
            with open(self.log_file, 'a') as f:
                f.write("---------------------Step {}---------------------\n".format(i))
        return
    
    def close(self):
        self.env.close()
                
        
    
    
if __name__ == "__main__":    
    # region 测试get_obs_ matrix
    # runner = WorkFlowRunner(2, 10, 2, 2, 2, 100)
    # interface = runner.env.interface_list[1]
    # interface.isolate_node(runner.env.nodes_list[1]['2'])
    # print(runner.get_obs_matrix([['0', '0', '1', '4'], []]))
    # interface.attack_node(runner.env.nodes_list[1]['0'], use_vulnerability=True)
    # interface.attack_node(runner.env.nodes_list[1]['0'], use_vulnerability=True)
    # interface.attack_node(runner.env.nodes_list[1]['1'], use_vulnerability=True)
    # interface.attack_node(runner.env.nodes_list[1]['4'], use_vulnerability=True)
    # interface.attack_node(runner.env.nodes_list[1]['4'], use_vulnerability=True)
    # print(runner.get_obs_matrix([['0', '0', '1', '4'], []]))
    # endregion
    
    runner = WorkFlowRunner(5, 30, 2, 2, 4, 5, attack_intensity=8)
    runner.run()
    runner.close()
    
