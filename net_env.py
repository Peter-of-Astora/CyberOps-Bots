import gymnasium as gym # type: ignore
import numpy as np
from gymnasium import spaces # type: ignore
import random
from yawning_titan.game_modes.game_mode_db import default_game_mode
from yawning_titan.envs.generic.core.network_interface import NetworkInterface

from datetime import datetime
import csv

from stable_baselines3 import PPO, DQN, A2C # type: ignore
from tree_network import create_tree, calculate_interfaces_efficiency
from attack import get_all_attackable_nodes, shortest_path_length, calculate_penetrate_depth, dijkstra_shortest_path

from torch.utils.tensorboard import SummaryWriter

class NetEnv(gym.Env):
    def __init__(self, subnet_num, subnet_node_num, entry_node_num, value_node_num, 
                attack_mode='recon', reward_mode='default', attack_intensity=2, algorithm='DQN', 
                isolate_alpha=0.1, isolate_beta=0.2, compromised_alpha=0.5, compromised_beta=0.7, episode_gamma=0.06, 
                log_file = '/root/work/project/log_run/env_log.txt'):
        '''
        subnet_num: num of subnets\n        
        subnet_node_num: num of nodes in every subnet\n        
        entry_node_num: num of entry nodes in every subnet\n        
        value_node_num: num of high value nodes in every subnet\n        
        attack_mode: mode of attacking. Use random, recon, penetrate, impact\n        
        reward_mode: mode of reward function. Use default, isolate, patch, reset, recover\n
        '''
        super().__init__()
        # 计算各空间大小
        self.action_type_num = 4
        observation_len = subnet_num * ((subnet_node_num + 2)**2 + (subnet_node_num + 2) * 7 + 3)
        action_num = subnet_num * subnet_node_num * self.action_type_num
        # 接收相关参数
        self.subnet_num = subnet_num
        self.subnet_node_num = subnet_node_num
        self.entry_node_num = entry_node_num
        self.value_node_num = value_node_num
        self.attack_mode = attack_mode
        self.reward_mode = reward_mode
        self.attack_intensity = attack_intensity
        self.algorithm = algorithm
        self.isolated_alpha = isolate_alpha
        self.isolated_beta = isolate_beta
        self.compromised_alpha = compromised_alpha
        self.compromised_beta = compromised_beta
        self.episode_gamma = episode_gamma
        # 定义动作、状态空间
        self.action_space = spaces.Discrete(action_num)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(observation_len, ), dtype=np.float32)
        # 创建网络以及子网环境
        self.networks = [create_tree(subnet_node_num, entry_node_num, value_node_num) for i in range(subnet_num)]
        self.nodes_list = [net.get_nodes(key_by_name=True) for net in self.networks]
        game_mode = default_game_mode()
        self.interface_list = [NetworkInterface(game_mode=game_mode, network=net) for net in self.networks]
        # 各子网重要程度列表生成
        self.sign_arr = np.random.uniform(0, 1, size=subnet_num).astype(np.float32)
        indices = np.random.choice(len(self.sign_arr), size=1, replace=False)  # 选择作为一个重要程度更高的子网
        self.sign_arr[indices] = 0.9 + np.random.uniform(0, 0.1, size=1).astype(np.float32)
        # 存储奖励计算相关变量
        self.isolated_node_num = 0
        self.compromised_node_num = 0
        # episode计数器（准确地说是step计数器，但是是用来测量episode长度的，每个episode结束后会重置为0）
        self.episode_counter = 0
        # 日志文件初始化
        self.log_file = log_file
        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d-%H%M")
        with open(self.log_file, 'w') as f:
            f.write("Environment Log - {} - Initialized\n".format(formatted_time))
        # tensorboard writer初始化，以及一些要可视化的数据保存
        self.writer = SummaryWriter('/root/work/project/log/env_log/{}_{}x{}with{}e{}v{}att_{}_{}'.format(algorithm, self.subnet_num, self.subnet_node_num, 
                                                                                        self.entry_node_num, self.value_node_num, self.attack_intensity, self.attack_mode, 
                                                                                        formatted_time))
        self.writer_counter = 0
        self.episode_lengths = []
        self.healthy_ratio_list = []
        self.efficiency_list = []
        self.reward_list = []
        self.network_average_vulnerabilities = []
        self.defender_loss = 0  # 计算网络可用性时的权重系数
        # 用于计算子智能体reward的一些中间变量
        self.reset_compromised_node_bool = 0
        self.patch_node_vul = 0
        self.recover_iso_node_bool = 0
        self.iso_observation_last = []  # 上一次的iso观察
        self.patch_terminated = False
        self.best_reset_node_index = 0  # 最佳reset节点索引，执行reset_observe时计算
        # 保存各个子网的攻击历史，用于状态空间自然语言描述统计
        self.attack_history = []
        for i in range(self.subnet_num):  # 每个子网的攻击历史为一个列表，都保存到attack_history中
            self.attack_history.append([])
        # 单独保存的可视化数据，并初始化数据文件
        self.healthy_rate_epi_length_data_path = "/root/work/project/data/{}/{}_healthy_rate_epi_length_{}nets_{}a_{}.csv".format(self.algorithm, self.algorithm, self.subnet_num, self.attack_intensity, self.attack_mode)
        self.net_effi_epi_length_data_path = "/root/work/project/data/{}/{}_net_effi_epi_length_{}nets_{}a_{}.csv".format(self.algorithm, self.algorithm, self.subnet_num, self.attack_intensity, self.attack_mode)
        with open(self.healthy_rate_epi_length_data_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Healthy_ratio', 'Episode_length'])  # 写入表头
        with open(self.net_effi_epi_length_data_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Network_availiability', 'Episode_length'])  # 写入表头
        
        
    def step(self, action):
        # 解析动作
        subnet_index = action // (self.subnet_node_num * self.action_type_num)
        node_index = (action % (self.subnet_node_num * self.action_type_num)) // self.action_type_num
        act_index = action % self.action_type_num
        
        subnet_node_list = self.nodes_list[subnet_index]
        subnet_interface = self.interface_list[subnet_index]
        node = subnet_node_list[str(node_index)]  # 按名称取Node类
        
        # 执行动作
        self.defend(action)
        
        # 按攻击规模进行多次攻击
        if self.attack_intensity > 0:
            for i in range(self.attack_intensity):
                # 攻击者攻击
                if self.attack_mode == 'random':
                    attack_result = self.random_attack()
                elif self.attack_mode == 'recon':
                    attack_result = self.recon_attack()
                elif self.attack_mode == 'penetrate':
                    attack_result = self.penetrate_attack()
                elif self.attack_mode == 'impact':
                    target = 0  # impact攻击者默认攻击最后一个关键节点
                    # impact攻击者每次有些微概率（1%）变更攻击目标（但是依然是关键节点之一，且切换时针对所有子网同步切换）
                    r = random.random()
                    if self.value_node_num > 1 and r > 0.99:
                        target = random.randint(0, self.value_node_num - 1)
                        with open(self.log_file, 'a') as f:
                                f.write("The impact attacker switched target to node {} of every subnet.\n".format(self.subnet_node_num - 1 - target))
                    attack_result = self.impact_attack(target=target)
        
        # 观察环境
        obs_list = [interface.get_current_observation() for interface in self.interface_list]
        observation = np.concatenate(obs_list)
        
        # 计算奖励
        if self.reward_mode == 'default':
            reward = self.default_reward_function()
        elif self.reward_mode == 'isolate':
            reward = self.iso_reward_function(action)
        elif self.reward_mode == 'patch':
            reward = self.patch_reward_function(action)
        elif self.reward_mode == 'reset':
            reward = self.reset_reward_function(action)
        elif self.reward_mode == 'recover':
            reward = self.recover_reward_function(action)
        
        # 判断终止
        terminated = False
        if self.judge_terminate(obs_list):
            terminated = True
            if self.reward_mode != 'reset':  # reset智能体不接受终止惩罚
                reward -= 10
            # 记录数据/指标
            self.episode_lengths.append(self.episode_counter)
            self.writer.add_scalar("Mean_episode_length-Episodes", sum(self.episode_lengths) / len(self.episode_lengths), len(self.episode_lengths))
            self.writer.add_scalar("Mean_episode_length-Steps", sum(self.episode_lengths) / len(self.episode_lengths), self.writer_counter)
            self.writer.add_scalar("Episode_length-Steps", self.episode_counter, self.writer_counter)  # 不平均的也要记，用于连续计算环境变化数据
            
        if self.episode_counter >= 30:
            terminated = True
            reward += 10
            with open(self.log_file, 'a') as f:
                f.write("Defenders win!\n")
            # 记录数据/指标
            self.episode_lengths.append(self.episode_counter)
            self.writer.add_scalar("Mean_episode_length-Episodes", sum(self.episode_lengths) / len(self.episode_lengths), len(self.episode_lengths))
            self.writer.add_scalar("Mean_episode_length-Steps", sum(self.episode_lengths) / len(self.episode_lengths), self.writer_counter)
            self.writer.add_scalar("Episode_length-Steps", self.episode_counter, self.writer_counter)  # 不平均的也要记，用于连续计算环境变化数据
            
        truncated = False
        info = {}
        
        # 计数器更新
        self.episode_counter += 1
        # 记录日志
        with open(self.log_file, 'a') as f:
            f.write("Reward is {}\n".format(reward))
        # 记录episode
        with open(self.log_file, 'a') as f:
            f.write("---------------------Episode {}---------------------\n".format(self.episode_counter))
        # 记录数据/指标
        # 平均网络可用性/健康率
        if self.defender_loss == 0:  # 正常记录健康率和可用性
            self.healthy_ratio_list.append(1 - (self.compromised_node_num + self.isolated_node_num) / (self.subnet_node_num * self.subnet_num))
            self.efficiency_list.append(calculate_interfaces_efficiency(self.interface_list))
        else:  # 防御者失败时，健康率和可用性都置为0
            self.healthy_ratio_list.append(0)
            self.efficiency_list.append(0)
        self.writer.add_scalar("Mean_Healthy_ratio-Steps", sum(self.healthy_ratio_list) / len(self.healthy_ratio_list), len(self.healthy_ratio_list))
        self.writer.add_scalar("Mean_Network_efficiency-Steps", sum(self.efficiency_list) / len(self.efficiency_list), len(self.efficiency_list))
        # 网络可用性/健康率-episode长度（瀑布）
        self.writer.add_scalar("Healthy_ratio-Episode Length", self.healthy_ratio_list[-1], self.episode_counter)
        self.writer.add_scalar("Network_efficiency-Episode Length", self.efficiency_list[-1], self.episode_counter)
        with open(self.healthy_rate_epi_length_data_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.healthy_ratio_list[-1], self.episode_counter])  # 写入数据
        with open(self.net_effi_epi_length_data_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.efficiency_list[-1], self.episode_counter])  # 写入数据
        # 平均奖励
        self.reward_list.append(reward)
        self.writer.add_scalar("Mean_Reward-Steps", sum(self.reward_list) / len(self.reward_list), self.writer_counter)
        self.writer.add_scalar("Reward-Steps", reward, self.writer_counter)
        # 网络脆弱程度（箱线）
        if terminated == True:  # 终止时记录最终的网络脆弱程度
            self.network_average_vulnerabilities.append(self.calculate_vulnerability())
            self.writer.add_scalar("Mean_Network_Vulnerability-Steps", sum(self.network_average_vulnerabilities) / len(self.network_average_vulnerabilities), self.writer_counter)
        # 更新tensorboard计数器
        self.writer_counter += 1
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # 重置network与nodes
        self.networks = [create_tree(self.subnet_node_num, self.entry_node_num, self.value_node_num) for i in range(self.subnet_num)]
        self.nodes_list = [net.get_nodes(key_by_name=True) for net in self.networks]
        # 重置network_interface
        game_mode = default_game_mode()
        self.interface_list = [NetworkInterface(game_mode=game_mode, network=net) for net in self.networks]
        # 重置计数器
        self.episode_counter = 0
        # 观察
        obs_list = [interface.get_current_observation() for interface in self.interface_list]
        observation = np.concatenate(obs_list)
        # 记录日志
        with open(self.log_file, 'a') as f:
            f.write("Reset!\n")
        # 重置攻击历史
        self.attack_history = []
        for i in range(self.subnet_num):
            self.attack_history.append([])
        # 重置权重系数
        self.defender_loss = 0
        info = {}
        return observation, info

    def iso_reset(self):
        '''
        专门为isolate_rl_agent提供的高效率reset,只重置网络,不重置计数器
        '''
        # 重置network与nodes
        self.networks = [create_tree(self.subnet_node_num, self.entry_node_num, self.value_node_num) for i in range(self.subnet_num)]
        self.nodes_list = [net.get_nodes(key_by_name=True) for net in self.networks]
        # 重置network_interface
        game_mode = default_game_mode()
        self.interface_list = [NetworkInterface(game_mode=game_mode, network=net) for net in self.networks]
    
    def render(self):
        ...

    def close(self):
        self.writer.close()
    
    def defend(self, action):
        # 解析动作
        subnet_index = action // (self.subnet_node_num * self.action_type_num)
        node_index = (action % (self.subnet_node_num * self.action_type_num)) // self.action_type_num
        act_index = action % self.action_type_num
        
        subnet_node_list = self.nodes_list[subnet_index]
        subnet_interface = self.interface_list[subnet_index]
        node = subnet_node_list[str(node_index)]  # 按名称取Node类
        
        # 获取子网信息
        interface = self.interface_list[subnet_index]
        n_nodes = interface.get_total_num_nodes()
        obs = interface.get_current_observation()
        isolated_state = obs[n_nodes**2: n_nodes**2 + n_nodes]
        
        # 执行动作
        if act_index == 0:
            # 执行动作前保留数据用于计算iso_reward
            self.iso_observation_last, _ = self.iso_observe(subnet_index)
            # 隔离节点
            if node_index < self.entry_node_num:
                # 如果是入口节点只能重置
                subnet_interface.make_node_safe(node)
                # 记录日志
                with open(self.log_file, 'a') as f:
                    f.write("Defender Action: Reset Node {} in subnet {}\n".format(str(node_index), subnet_index))
            else:
                # 不是入口节点可以隔离
                subnet_interface.isolate_node(node)
                # 记录日志
                with open(self.log_file, 'a') as f:
                    f.write("Defender Action: Isolate Node {} in subnet {}\n".format(str(node_index), subnet_index))
        elif act_index == 1:
            # 重连节点
            self.recover_iso_node_bool = isolated_state[node_index]  # 用于计算recover_reward
            subnet_interface.reconnect_node(node)
            # 记录日志
            with open(self.log_file, 'a') as f:
                f.write("Defender Action: Reconnect Node {} in subnet {}\n".format(str(node_index), subnet_index))
        elif act_index == 2:
            # 重置节点
            self.reset_compromised_node_bool = node.true_compromised_status # 用于计算reset_reward 
            subnet_interface.make_node_safe(node)
            # 记录日志
            with open(self.log_file, 'a') as f:
                f.write("Defender Action: Reset Node {} in subnet {}\n".format(str(node_index), subnet_index))
        else:
            # 修复漏洞
            self.patch_node_vul = node._vulnerability
            node._vulnerability = max(node._vulnerability - 0.5, 0.01)
            # 记录日志
            with open(self.log_file, 'a') as f:
                f.write("Defender Action: Patch Node {} in subnet {}\n".format(str(node_index), subnet_index))
        
    def observe(self):
        # 观察环境
        obs_list = [interface.get_current_observation() for interface in self.interface_list]
        observation = np.concatenate(obs_list)
        
        return observation
    
    def iso_observe(self, subnet_index):
        '''
        将大observation转换成isolate_agent的小obs并返回,参数为对应子网索引\n
        isolate_agent的观察空间是各节点的深度值,若没有被攻陷则设为-1
        '''
        # 确定对应子网
        interface = self.interface_list[subnet_index]
        # 获取网络信息
        n_nodes = interface.get_total_num_nodes()  # 节点数量
        obs = interface.get_current_observation()  # 状态空间
        connections = obs[: n_nodes**2]
        matrix = connections.reshape((n_nodes, n_nodes))  # 邻接矩阵
        compromised_dict = interface.get_all_node_compromised_states()  # 攻陷字典
        compromised_state = list(compromised_dict.values())  # 攻陷状态
        compromised_node_indexes = [i for i, x in enumerate(compromised_state) if x == 1]  # 攻陷节点索引
        # 初始化观察数组（没有被攻陷就都是1000，尽量使状态空间单调）
        obs = np.full(len(compromised_state), -1, dtype=np.float32)
        # 循环计算每个被攻陷节点的深度（取最小值）
        for compromised_node in compromised_node_indexes:
            depths = [shortest_path_length(matrix, compromised_node, self.subnet_node_num - 1 - i) for i in range(self.value_node_num)]  # 取相对于各个关键节点的深度
            depth = min([d for d in depths if d >= 0], default=-1)  # 取深度列表中的非负最小深度，没有则取负一
            if depth == -1:
                continue
            obs[compromised_node] = depth
        
        # 判断是否完成任务
        terminated = False
        if (obs == -1).all():
            terminated = True
        
        return obs, terminated
    
    def patch_observe(self, subnet_index):
        '''
        将大observation转换成patch_agent的小obs并返回,参数为对应子网索引\n
        patch_agent的观察空间是各节点的漏洞值的需求性(0/1)和关键性(0/1)的和
        '''
        # 确定对应子网
        interface = self.interface_list[subnet_index]
        subnet_node_list = self.nodes_list[subnet_index]
        # 初始化观察数组
        obs = np.full(self.subnet_node_num, -100, dtype=np.float32)  # 设观察空间默认值为-100
        # 计算该子网的关键路径（入口到关键节点的最短路径）
        matrix = interface.adj_matrix
        path = []
        for i in range(self.value_node_num):
            path.extend(dijkstra_shortest_path(matrix, 0, self.subnet_node_num - 1 - i))
        # 遍历每一个节点
        for node_index in range(self.subnet_node_num):
            # 循环取得漏洞值，若不为0.01，则在观察空间中置1
            if subnet_node_list[str(node_index)]._vulnerability != 0.01:
                obs[node_index] = 1
                # 判断是否为关键路径节点，若是，则再在观察空间中加10
                if node_index in path:
                    obs[node_index] += 10
        
        # 记录日志
        # with open(self.log_file, 'a') as f:
        #     f.write("Patch agent obs: {}\n".format(str(obs)))

        # 是否完成所有必要patch
        terminated = False
        if 11 not in obs:
            terminated = True
        
        # 保存用于计算reward
        self.patch_terminated = True
        
        return obs, terminated
        
    def reset_observe(self, subnet_index):
        '''
        将大observation转换成reset_agent的小obs并返回,参数为对应子网索引\n
        reset_agent的观察空间是各节点是否被攻陷
        '''
        # 确定对应子网
        interface = self.interface_list[subnet_index]
        # 获取网络信息
        compromised_dict = interface.get_all_node_compromised_states()  # 攻陷字典
        compromised_state = list(compromised_dict.values())  # 攻陷状态
        # 得到观察数组
        obs = np.array(compromised_state, dtype=np.float32)
        # 进一步简化观察数组，去掉索引靠前的所有1，只留最后一个1
        indices = np.where(obs == 1)[0] # 找到所有1的索引
        if len(indices) > 0: # 确保至少有1个1，否则[:-1]切片会出错
            obs[indices[:-1]] = 0 # 将最后一个1之前的所有1置为0
            # 计算最佳reset节点
            self.best_reset_node_index = indices[-1]
        
        return obs
    
    def recover_observe(self, subnet_index):
        '''
        将大observation转换成recover_agent的小obs并返回,参数为对应子网索引\n
        recover_agent的观察空间是各节点是否被隔离
        '''
        # 确定对应子网
        interface = self.interface_list[subnet_index]
        # 获取网络信息
        n_nodes = interface.get_total_num_nodes()  # 节点数量
        total_observation = interface.get_current_observation()  # 状态空间
        isolated_state = total_observation[n_nodes**2: n_nodes**2 + n_nodes]
        # 提取出状态空间
        obs = isolated_state[:self.subnet_node_num]
        # 判断是否没有隔离节点
        terminated = False
        if all(x == 0 for x in obs):
            terminated = True
        
        return obs, terminated
    
    def action_transfer(self, action_type, sub_action, subnet_index):
        '''
        将各个sub_agent的动作翻译成全局动作\n
        action_type: isolate, patch, reset, recover\n
        sub_action: the action of subagent\n
        subnet_index: the index of the subnet\n
        '''
        # 得到对应agent的动作索引
        if action_type == 'isolate':
            action_type_index = 0
        elif action_type == 'recover':
            action_type_index = 1
        elif action_type == 'reset':
            action_type_index = 2
        elif action_type == 'patch':
            action_type_index = 3
        # 子动作对应节点的索引
        node_index = sub_action
        # 计算全局动作
        action = subnet_index * (self.subnet_node_num * self.action_type_num) + \
            node_index * self.action_type_num + \
            action_type_index
            
        return action
        
    def random_attack(self):
        '''
        Attack in a totally random way.
        '''
        attack_subnet_index = random.randint(0, self.subnet_num - 1)  # 随机选择子网
        attack_node_index = random.randint(0, self.subnet_node_num - 1)  # 随机选择节点
        attacked_interface = self.interface_list[attack_subnet_index]
        attacked_node_list = self.nodes_list[attack_subnet_index]
        attacked_node = attacked_node_list[str(attack_node_index)]
        attack_result = attacked_interface.attack_node(attacked_node, use_vulnerability=True)
        # 记录日志
        with open(self.log_file, 'a') as f:
            f.write("Attacker Action: Attack Node {} in subnet {} with result {}\n".format(attacked_node.name, attack_subnet_index, attack_result))
        # 记录攻击历史
        self.attack_history[attack_subnet_index].append(attacked_node.name)
        
        return attack_result
        
    def recon_attack(self):
        '''
        Attack the nodes that can be attacked, and attack as much nodes as possible.
        '''
        # 获取所有可攻击的节点
        attackable_nodes = get_all_attackable_nodes(self.interface_list)
        # with open(self.log_file, 'a') as f:
        #     f.write(str(attackable_nodes) + '\n')
        
        # 随机选择要攻击的子网（子网可被攻击的节点不能为无）
        while True:
            subnet_index = random.randint(0, self.subnet_num - 1)
            if len(attackable_nodes[subnet_index]) != 0:
                break
        subnet_attackable_nodes = attackable_nodes[subnet_index]
        
        # 随机选择要攻击的节点
        random_index = random.randint(0, len(subnet_attackable_nodes) - 1)
        node_index = subnet_attackable_nodes[random_index]
        
        # 攻击
        attacked_node = self.nodes_list[subnet_index][str(node_index)]
        attack_result = self.interface_list[subnet_index].attack_node(attacked_node, use_vulnerability=True)
        # 记录日志
        with open(self.log_file, 'a') as f:
            f.write("Attacker Action: Attack Node {} in subnet {} with result {}\n".format(attacked_node.name, subnet_index, attack_result))
        # 记录攻击历史
        self.attack_history[subnet_index].append(attacked_node.name)
        
        return attack_result
        
    def penetrate_attack(self):
        '''
        Attack the nodes that can be attacked, and the ones with less protection.
        '''
        # 为防止始终攻击同一节点，以一定概率调用impact攻击，增加策略多样性、动态性
        r = random.uniform(0, 1)
        if r >= 0.6:
            attack_result = self.impact_attack()
            return attack_result
        
        # 获取所有可攻击的节点
        attackable_nodes = get_all_attackable_nodes(self.interface_list)
        
        # 要攻击的子网以及要攻击的子网节点
        subnet_index = 0
        node_index = 0
        max_vulnerability = 0
        
        # 遍历每个子网的可攻击节点，并保存漏洞值最大的
        for i in range(self.subnet_num):
            subnet_attackable_nodes = attackable_nodes[i]
            # 遍历这个子网的每一个可攻击节点，
            for j in range(len(subnet_attackable_nodes)):
                temp_node_index = subnet_attackable_nodes[j]  # 取出这个节点的索引
                vulnerability = self.nodes_list[i][str(temp_node_index)].vulnerability_score  # 节点的漏洞值
                # 打擂台
                if vulnerability > max_vulnerability:
                    subnet_index = i
                    node_index = temp_node_index
                    max_vulnerability = vulnerability
                
        # 攻击
        attacked_node = self.nodes_list[subnet_index][str(node_index)]
        attack_result = self.interface_list[subnet_index].attack_node(attacked_node, use_vulnerability=True)
        # 记录日志
        with open(self.log_file, 'a') as f:
            f.write("Attacker Action: Attack Node {} in subnet {} with result {}\n".format(attacked_node.name, subnet_index, attack_result))
        # 记录攻击历史
        self.attack_history[subnet_index].append(attacked_node.name)
        
        return attack_result
        
    def impact_attack(self, target=0):
        '''
        Attack the nodes that can be attacked, especially those closer to the high value node. \n
        
        target: the reverse index of high value node. Target=0 means the last of the nodes is the high value node.
        '''
        # 获取所有可攻击的节点
        attackable_nodes = get_all_attackable_nodes(self.interface_list)
        
        # 获取网络相关信息
        high_value_node_index = self.subnet_node_num - 1 - target  # 重要节点的索引，默认攻击最后一个重要节点(target=0)
        n_nodes = self.interface_list[0].get_total_num_nodes()
        
        subnet_index = 0
        node_index = 0
        min_distance = -1
        # 遍历每个子网，找与重要节点最近的可攻击节点
        for i in range(self.subnet_num):
            subnet_attackable_nodes = attackable_nodes[i]
            # 得到每个子网的邻接矩阵
            obs = self.interface_list[i].get_current_observation()
            connections = obs[: n_nodes**2]
            matrix = connections.reshape((n_nodes, n_nodes))
            # 遍历子网中每个可攻击节点，找出最近的
            for j in range(len(subnet_attackable_nodes)):
                temp_node_index = subnet_attackable_nodes[j]
                distance = shortest_path_length(matrix, temp_node_index, high_value_node_index)  # 计算与重要节点距离
                # 如果没有可行路径则跳过这个节点
                if distance < 0:
                    continue
                # 如果比现有保存的路径短，则使用更短的路径
                if min_distance < 0 or distance < min_distance: 
                    subnet_index = i
                    node_index = temp_node_index
                    min_distance = distance
        # 攻击
        attacked_node = self.nodes_list[subnet_index][str(node_index)]
        attack_result = self.interface_list[subnet_index].attack_node(attacked_node, use_vulnerability=True)
        # 记录日志
        with open(self.log_file, 'a') as f:
            f.write("Attacker Action: Attack Node {} in subnet {} with result {}\n".format(attacked_node.name, subnet_index, attack_result))
        # 记录攻击历史
        self.attack_history[subnet_index].append(attacked_node.name)
        
        return attack_result
        
    def default_reward_function(self):
        # 上次的异常状态节点数量
        last_isolated_node_num = self.isolated_node_num
        last_compromised_node_num = self.compromised_node_num
        # 更新异常状态节点数量
        self.update_abnormal_node_num()
        nodes_n = self.interface_list[0].get_total_num_nodes()
        # 计算reward
        reward = self.isolated_alpha * (0 - self.isolated_node_num) + \
            self.isolated_beta * (last_isolated_node_num - self.isolated_node_num) + \
            self.compromised_alpha * (0 - self.compromised_node_num) + \
            self.compromised_beta * (last_compromised_node_num - self.compromised_node_num)
        # episode_reward = self.episode_counter * self.episode_gamma
        # reward = reward + episode_reward
            
        return reward
    
    def iso_reward_function(self, action):
        # 解析动作
        subnet_index = action // (self.subnet_node_num * self.action_type_num)
        node_index = (action % (self.subnet_node_num * self.action_type_num)) // self.action_type_num
        act_index = action % self.action_type_num
        
        # 找到上次观察中所需要隔离的节点
        obs = np.array(self.iso_observation_last)
        mask = obs >= 0
        if np.any(mask):
            # 获取非负最小值的索引
            min_index = np.argmin(obs[mask])
            # 转换为原始列表中的索引
            original_indices = np.arange(len(obs))[mask]
            # 获取非负最小元素的索引，即需要隔离的节点
            isolate_node_index = original_indices[min_index]
        else:
            # 不存在攻击链
            isolate_node_index = 0
        
        # 判断动作索引与计算得到的索引是否一致，且动作为隔离，且不是入口节点（隔离入口节点没有用）
        if node_index == isolate_node_index and act_index == 0 and node_index >= self.entry_node_num:
            reward = 10
        else:
            reward = -1
        
        # 更新异常节点状态
        self.update_abnormal_node_num()
        
        # 判断是否完全任务，如果完成任务，奖励增加
        obs, terminated = self.iso_observe(subnet_index)
        if terminated:
            reward += 100
        
        return reward
    
    def patch_reward_function(self, action):
        # 解析动作
        subnet_index = action // (self.subnet_node_num * self.action_type_num)
        node_index = (action % (self.subnet_node_num * self.action_type_num)) // self.action_type_num
        act_index = action % self.action_type_num
        
        subnet_node_list = self.nodes_list[subnet_index]
        subnet_interface = self.interface_list[subnet_index]
        node = subnet_node_list[str(node_index)]  # 按名称取Node类
        # 计算该子网的关键路径
        matrix = subnet_interface.adj_matrix
        path = []
        for i in range(self.value_node_num):
            path.extend(dijkstra_shortest_path(matrix, 0, self.subnet_node_num - 1 - i))
        # 判断节点是否处于关键攻击路径上
        significance_bool = False
        if node_index in path:
            significance_bool = True
        # 动作同时具有有效性和准确性，则给reward，并且是patch操作
        if self.patch_node_vul != 0.01 and significance_bool and act_index == 3:
            reward = 10
        else:
            reward = -1
        
        # 获取观察判断是否完全任务，完成任务则奖励增加
        obs, terminated = self.patch_observe(subnet_index)
        if terminated:
            reward += 100
        
        return reward
    
    def recover_reward_function(self, action):
        # 解析动作
        subnet_index = action // (self.subnet_node_num * self.action_type_num)
        node_index = (action % (self.subnet_node_num * self.action_type_num)) // self.action_type_num
        act_index = action % self.action_type_num
        
        # 若准确恢复隔离节点，则给reward
        if self.recover_iso_node_bool == 1 and act_index == 1:
            reward = 10
        else:
            reward = -1
        
        # 更新异常节点状态
        self.update_abnormal_node_num()
        
        # 若恢复完了所有隔离节点，reward再增加
        if self.isolated_node_num == 0:
            reward += 100
        
        return reward
                
    def reset_reward_function(self, action):
        # 解析动作
        subnet_index = action // (self.subnet_node_num * self.action_type_num)
        node_index = (action % (self.subnet_node_num * self.action_type_num)) // self.action_type_num
        act_index = action % self.action_type_num
        
        # 如果是最佳reset节点，且是reset操作，给reward
        if self.best_reset_node_index == node_index and act_index == 2:
            reward = 10
        else:
            reward = -100
        
        # 更新异常节点状态
        self.update_abnormal_node_num()
        
        return reward
    
    def update_abnormal_node_num(self):
        # 先清零
        self.isolated_node_num = 0
        self.compromised_node_num = 0
        # 累加异常节点数量
        for interface in self.interface_list:
            n_nodes = interface.get_total_num_nodes()
            obs = interface.get_current_observation()
            isolated_state = obs[n_nodes**2: n_nodes**2 + n_nodes]
            compromised_state = obs[n_nodes**2 + n_nodes: n_nodes**2 + 2 * n_nodes]
            self.isolated_node_num += np.count_nonzero(isolated_state == 1)
            self.compromised_node_num += np.count_nonzero(compromised_state == 1)
        # 记录日志
        with open(self.log_file, 'a') as f:
            f.write("State: {} nodes isolated and {} nodes compromised\n".format(self.isolated_node_num, self.compromised_node_num))
        
    def calculate_vulnerability(self):
        # 保存所有的漏洞值
        vulnerability_list = []
        # 遍历各个子网保存漏洞值
        for i in range(self.subnet_num):
            nodes = self.nodes_list[i]
            for j in range(self.subnet_node_num):
                node = nodes[str(j)]
                vulnerability_list.append(node._vulnerability)
        # 求平均
        mean_vulnerability = sum(vulnerability_list) / len(vulnerability_list)
        
        return mean_vulnerability
    
    def judge_terminate(self, obs_list):
        '''
        To judge if the high value node is compromised. If so, then game is terminated.
        '''
        terminated = False
        
        for interface in self.interface_list:
            compromised_dict = interface.get_all_node_compromised_states()
            compromised_state = list(compromised_dict.values())
            for i in range(self.value_node_num):
                # 关键节点的索引是倒数几个节点，循环检查是否有被攻陷的，如果有，则终止
                if compromised_state[-1 - i] == 1:
                    terminated = True
                    with open(self.log_file, 'a') as f:
                        f.write("The node {} compromised, game terminated.\n".format(self.subnet_node_num - i - 1))
            
        return terminated
        
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env # type: ignore
    from stable_baselines3.common.env_util import make_vec_env # type: ignore
    from datetime import datetime
    import torch
    from torch.utils.tensorboard import SummaryWriter
    
    from workflow_runner import WorkFlowRunner
    
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M")
    
    env_kwargs = {
        "subnet_num": 1, 
        "subnet_node_num": 20, 
        "entry_node_num": 2, 
        "value_node_num": 2, 
        "attack_mode": 'recon'
    }
    
    vec_env = make_vec_env(
        NetEnv, 
        n_envs=16,           # 并行环境数量
        env_kwargs=env_kwargs  # 传递参数
    )
    
    env = NetEnv(1, 20, 2, 2, attack_mode='recon')
    
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                    net_arch=[256, 256])
    model = DQN(
        'MlpPolicy', 
        vec_env, 
        verbose=False, 
        policy_kwargs=policy_kwargs,
        tensorboard_log='/root/work/project/log/DQN_try_{}'.format(formatted_time),
        learning_rate=1e-4,  # 学习率
        buffer_size=1000000,  # 经验回放缓冲区大小
        batch_size=256,  # 每次梯度更新的样本数
        gamma=0.99,  # 折扣因子
        train_freq=4,  # 每4步进行一次训练
        target_update_interval=1000,  # 目标网络更新间隔(步数)
        exploration_fraction=0.1,  # 探索率衰减的时间比例
        exploration_initial_eps=1.0,  # 初始探索率
        exploration_final_eps=0.05,  # 最终探索率
        learning_starts=1000,  # 学习开始前的随机步数
        max_grad_norm=10,  # 梯度裁剪的最大值
    )
    model.learn(total_timesteps=int(10e5), progress_bar=True)
    model.save('dqn_net_try')
    
    del model
    
    model = DQN.load("dqn_net_try", env=env)
    
    obs, info = env.reset()
    
    # writer = SummaryWriter('/root/work/project/log/DQN_try_test_{}'.format(formatted_time))
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        print(rewards)
        # writer.add_scalar('Reward/episodes', rewards, i)
    
    # writer.close()
    print(env.attack_history)
    history = env.attack_history
    env.close()
