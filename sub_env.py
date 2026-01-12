import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
from stable_baselines3.common.env_util import make_vec_env # type: ignore
from stable_baselines3 import PPO, DQN, A2C # type: ignore
from datetime import datetime
import numpy as np
from net_env import NetEnv
import torch
import random

class SubTrainEnv(gym.Env):
    def __init__(self, subnet_num, subnet_node_num, entry_node_num, value_node_num, subagent_type):
        super().__init__()
        self.subagent_type = subagent_type
        # 定义NetEnv环境类
        if subagent_type == 'isolate':
            self.env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='impact', reward_mode='isolate', 
                            attack_intensity=1, algorithm='DQN')
        elif subagent_type == 'patch':
            self.env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='recon', reward_mode='patch', 
                            attack_intensity=0, algorithm='DQN')
        elif subagent_type == 'reset':
            self.env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='random', reward_mode='reset', 
                            attack_intensity=3, algorithm='DQN')
        elif subagent_type == 'recover':
            self.env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='recon', reward_mode='recover', 
                            attack_intensity=0, algorithm='DQN')
        
        # 子智能体的动作与观察空间都与单个子网节点数一致
        self.action_space = spaces.Discrete(subnet_node_num)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(subnet_node_num, ), dtype=np.float32)
        # 子智能体专用计数器，用于控制部分智能体的重置时机
        self.sub_step_counter = 0
    
    def step(self, action):
        # 将子动作翻译后执行
        total_action = self.env.action_transfer(self.subagent_type, action, 0)
        total_obs, rewards, terminated, truncated, info = self.env.step(total_action)
        # 得到子观察空间（训练时只用一个子网，所以都是0）
        if self.subagent_type == 'isolate':
            obs, terminated = self.env.iso_observe(0)  # iso_agent有任务完成判断，注意接口不同
            # 隔离智能体训练时，三步后重置场景
            if self.sub_step_counter == 3:
                terminated = True
        elif self.subagent_type == 'patch':
            obs, terminated = self.env.patch_observe(0)  # patch_agent有任务完成判断，注意接口不同
            # 补丁智能体训练时，完成必要重置或四十步后重置场景
            if self.sub_step_counter == 40:
                terminated = True
        elif self.subagent_type == 'reset':
            obs = self.env.reset_observe(0)
        elif self.subagent_type == 'recover':
            obs, terminated = self.env.recover_observe(0)
            # 恢复智能体训练时，十步后重置场景（因为隔离节点不会自动增加，所以设置得短一点）
            if self.sub_step_counter == 10:
                terminated = True
        # 计数器更新
        self.sub_step_counter += 1
        return obs, rewards, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # isolate_agent和recover_agent以及reset_agent使用自己的reset方法
        if self.subagent_type == 'isolate':
            obs, info = self.iso_reset()
        elif self.subagent_type == 'recover':
            obs, info = self.recover_reset()
        elif self.subagent_type == 'reset':
            obs, info = self.reset_reset()
        # 其他使用通用的reset方法，同时翻译成子观察空间
        elif self.subagent_type == 'patch':
            obs, info = self.env.reset()
            obs, terminated = self.env.patch_observe(0)
        # 计数器重置
        self.sub_step_counter = 0
        return obs, info
    
    def iso_reset(self):
        # 对于隔离智能体，重置时模拟攻击场景
        self.env.reset()
        # 预先攻击，攻击次数随机，模拟紧急场景
        for i in range(np.random.choice([2, 3, 4])):
            # 每次随机选择目标节点
            target = np.random.choice(self.env.value_node_num)
            self.env.impact_attack(target=target)
        # 获取观察
        obs, terminated = self.env.iso_observe(0)
        info = {}
        
        return obs, info
    
    def recover_reset(self):
        # 对于恢复智能体，重置时模拟有节点被隔离场景
        self.env.reset()
        # 训练时只用一个子网，取出其节点与interface
        nodes = self.env.nodes_list[0]
        interface = self.env.interface_list[0]
        # 预先隔离,随机选择隔离的节点数量
        for i in range(np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])):
            # 每次随机选择隔离节点
            target = np.random.choice(self.env.subnet_node_num)
            interface.isolate_node(nodes[str(target)])
        # 获取观察
        obs, _ = self.env.recover_observe(0)
        info = {}

        return obs, info
        
    def reset_reset(self):
        # 对于Reset智能体，重置时模拟有节点被攻陷场景
        self.env.reset()
        # 训练时只用一个子网，取出其节点与interface
        nodes = self.env.nodes_list[0]
        interface = self.env.interface_list[0]
        # 预先隔离,随机选择攻击的节点数量
        for i in range(np.random.choice([2, 3, 4])):
            # 执行随机攻击
            self.env.recon_attack()
        # 获取观察
        obs = self.env.reset_observe(0)
        info = {}

        return obs, info
        
    def render(self):
        ...
        
    def close(self):
        self.env.close()
        
def train_sub_agent(agent_type):
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M")
    
    env_kwargs = {
        "subnet_num": 1, 
        "subnet_node_num": 30, 
        "entry_node_num": 2, 
        "value_node_num": 2, 
        "subagent_type": agent_type
    }
    
    vec_env = make_vec_env(
        SubTrainEnv, 
        n_envs=16,           # 并行环境数量
        env_kwargs=env_kwargs  # 传递参数
    )
    
    env = SubTrainEnv(1, 30, 2, 2, agent_type)
    
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                    net_arch=[256, 256])
    model = DQN(
        'MlpPolicy', 
        vec_env, 
        verbose=False, 
        policy_kwargs=policy_kwargs, 
        tensorboard_log='/root/work/project/log/DQN_{}_{}'.format(agent_type, formatted_time),
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
    model.learn(total_timesteps=int(1e5), progress_bar=True)
    model.save('/root/work/project/rl_models/{}_agent'.format(agent_type))
    
    del model
    
    model = DQN.load('/root/work/project/rl_models/{}_agent'.format(agent_type), env=env)
    
    obs, info = env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        print(rewards)
    
    env.close()
    
    return
    
def test_sub_agent(agent_type):
    env = SubTrainEnv(1, 30, 2, 2, agent_type)
    model = DQN.load('/root/work/project/rl_models/{}_agent'.format(agent_type), env=env)
    
    obs, info = env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        print(rewards)
    
    env.close()
    
    return
    
if __name__ == "__main__":
    # train_sub_agent('recover')
    test_sub_agent('recover')
