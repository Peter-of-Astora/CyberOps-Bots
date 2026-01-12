'''This set of Code is DEPRECATED. Please go to sub_env.py'''
import gymnasium as gym # type: ignore
from stable_baselines3.common.env_util import make_vec_env # type: ignore
from stable_baselines3 import PPO, DQN, A2C # type: ignore
from datetime import datetime
import numpy as np
from net_env import NetEnv
import torch
import random


class IsoEnv(gym.Env):
    def __init__(self, subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='impact', 
                attack_intensity=2, algorithm='DQN'):
        super().__init__()
        self.env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode=attack_mode, reward_mode='isolate', 
                        attack_intensity=attack_intensity, algorithm=algorithm)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def step(self, action):
        # 预先攻击，攻击次数随机，模拟紧急场景
        for i in range(np.random.choice(4)):
            # 每次随机选择目标节点
            target = np.random.choice(self.env.value_node_num)
            self.env.impact_attack(target=target)
        obs, rewards, terminated, truncated, info = self.env.step(action)
        # 每次决策后reset，然后重新构建场景
        self.env.iso_reset()
        return obs, rewards, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return obs, info
    
    def render(self):
        ...
        
    def close(self):
        self.env.close()
        

class ReconEnv(gym.Env):
    def __init__(self, subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='impact', 
                attack_intensity=2, algorithm='DQN'):
        super().__init__()
        self.env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode=attack_mode, reward_mode='isolate', 
                        attack_intensity=attack_intensity, algorithm=algorithm)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def step(self, action):
        # 预先攻击，攻击次数随机，模拟紧急场景
        for i in range(np.random.choice(4)):
            # 每次随机选择目标节点
            target = np.random.choice(self.env.value_node_num)
            self.env.impact_attack(target=target)
        obs, rewards, terminated, truncated, info = self.env.step(action)
        # 每次决策后reset，然后重新构建场景
        self.env.iso_reset()
        return obs, rewards, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return obs, info
    
    def render(self):
        ...
        
    def close(self):
        self.env.close()

def train_isolate_agent(subnet_num=1, subnet_node_num=30, entry_node_num=2, value_node_num=2):
    '''
    训练隔离用智能体，使用专门的环境交互
    '''
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M")
    
    env_kwargs = {
        "subnet_num": subnet_num, 
        "subnet_node_num": subnet_node_num, 
        "entry_node_num": entry_node_num, 
        "value_node_num": value_node_num, 
        "attack_mode": 'impact', 
        "attack_intensity": 0,  # 已安排特定初始场景，不用再每次step都攻击，容易导致经常-100奖励（也可以考虑使用）
        "algorithm": 'DQN'
    }
    
    vec_env = make_vec_env(
        IsoEnv, 
        n_envs=16,           # 并行环境数量
        env_kwargs=env_kwargs  # 传递参数
    )
    
    env = IsoEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='impact', attack_intensity=0)
    
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                    net_arch=[256, 256])
    model = DQN(
        'MlpPolicy', 
        vec_env, 
        verbose=False, 
        policy_kwargs=policy_kwargs, 
        tensorboard_log='/root/work/project/log/DQN_iso_{}'.format(formatted_time),
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
    model.save('/root/work/project/rl_models/isolate_agent')
    
    del model
    
    model = DQN.load('/root/work/project/rl_models/isolate_agent', env=env)
    
    obs, info = env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        print(rewards)
    
    env.close()
    
    return

def test_iso_agent(subnet_num=1, subnet_node_num=30, entry_node_num=2, value_node_num=2):
    env = IsoEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='impact', attack_intensity=0)
    model = DQN.load('/root/work/project/rl_models/isolate_agent', env=env)
    obs, info = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        print(rewards)
    
    env.close()
    
    return
    

def train_patch_agent(subnet_num=1, subnet_node_num=30, entry_node_num=2, value_node_num=2):
    '''
    训练补丁智能体，环境交互不变
    '''
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M")
    
    env_kwargs = {
        "subnet_num": subnet_num, 
        "subnet_node_num": subnet_node_num, 
        "entry_node_num": entry_node_num, 
        "value_node_num": value_node_num, 
        "attack_mode": 'recon', 
        "attack_intensity": 1,  # 训练patch智能体，攻击不重要，设为1，加长episode长度
        "algorithm": 'DQN', 
        "reward_mode": 'patch'  # 将奖励函数模式更改为patch
    }
    
    vec_env = make_vec_env(
        NetEnv, 
        n_envs=16,           # 并行环境数量
        env_kwargs=env_kwargs  # 传递参数
    )
    
    env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='recon', attack_intensity=1, reward_mode='patch')  # 奖励函数模式更改为patch
    
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                    net_arch=[256, 256])
    model = DQN(
        'MlpPolicy', 
        vec_env, 
        verbose=False, 
        policy_kwargs=policy_kwargs,
        tensorboard_log='/root/work/project/log/DQN_patch_{}'.format(formatted_time),
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
    model.save('/root/work/project/rl_models/patch_agent')
    
    del model
    
    model = DQN.load('/root/work/project/rl_models/patch_agent', env=env)
    
    obs, info = env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        print(rewards)
    
    env.close()
    
    return

def test_patch_agent(subnet_num=1, subnet_node_num=30, entry_node_num=2, value_node_num=2):
    env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='recon', attack_intensity=1, reward_mode='patch')
    model = DQN.load('/root/work/project/rl_models/patch_agent', env=env)
    
    obs, info = env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        print(rewards)
    
    env.close()
    
    return

def train_reset_agent(subnet_num=1, subnet_node_num=30, entry_node_num=2, value_node_num=2):
    '''
    训练重置智能体，环境交互不变
    '''
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M")
    
    env_kwargs = {
        "subnet_num": subnet_num, 
        "subnet_node_num": subnet_node_num, 
        "entry_node_num": entry_node_num, 
        "value_node_num": value_node_num, 
        "attack_mode": 'recon', 
        "attack_intensity": 1,  # 训练reset智能体，攻击不重要，设为1，加长episode长度
        "algorithm": 'DQN', 
        "reward_mode": 'reset'  # 将奖励函数模式更改为reset
    }
    
    vec_env = make_vec_env(
        NetEnv, 
        n_envs=16,           # 并行环境数量
        env_kwargs=env_kwargs  # 传递参数
    )
    
    env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='recon', attack_intensity=1, reward_mode='reset')  # 奖励函数模式更改为patch
    
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                    net_arch=[256, 256])
    model = DQN(
        'MlpPolicy', 
        vec_env, 
        verbose=False, 
        policy_kwargs=policy_kwargs,
        tensorboard_log='/root/work/project/log/DQN_reset_{}'.format(formatted_time),
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
    model.save('/root/work/project/rl_models/reset_agent')
    
    del model
    
    model = DQN.load('/root/work/project/rl_models/reset_agent', env=env)
    
    obs, info = env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        print(rewards)
    
    env.close()
    
    return

def test_reset_agent(subnet_num=1, subnet_node_num=30, entry_node_num=2, value_node_num=2):
    env = NetEnv(subnet_num, subnet_node_num, entry_node_num, value_node_num, attack_mode='recon', attack_intensity=1, reward_mode='reset')  # 奖励函数模式更改为patch
    model = DQN.load('/root/work/project/rl_models/reset_agent', env=env)
    
    obs, info = env.reset()
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated == True:
            obs, info = env.reset()
        # print(rewards)
    
    env.close()
    
    return

if __name__ == "__main__":
    '''This set of Code is DEPRECATED. Please go to sub_env.py'''
    # train_isolate_agent()
    # train_patch_agent()
    # train_reset_agent()
    # test_iso_agent()
    # test_patch_agent()
    test_reset_agent()