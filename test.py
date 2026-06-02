# 对于一个游戏，使用bert dqn完成主动探索提升

from game import default_game, Game_with_navigator, game_state_from_game
from bert_state_action_encoder import State_Action_Encoder
import random
from collections import deque
from q_net import ChoiceTextQnet
from bert_dqn import BertDQNAgent

class TrajectoryReplayBuffer:
    def __init__(self, capacity=1000):
        """
        capacity: Buffer 能容纳的最大轨迹数量
        """
        # 使用 deque (双端队列)，当超过容量时会自动从左侧弹出老数据
        self.buffer = deque(maxlen=capacity)
    
    def push(self, trajectory):
        """
        存入一条完整的轨迹。
        trajectory 应该是一个 list，里面包含这一局游戏从头到尾的每一个 step 字典
        """
        self.buffer.append(trajectory)
        
    def sample(self):
        """
        随机抽取一条轨迹用于训练
        """
        return random.choice(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    

MAX_STEPS = 100

def play_one_episode(game: Game_with_navigator, agent: BertDQNAgent, epsilon=1.0):
    states = []
    actions = []
    rewards = []
    last_reward = 0
    _ = game.reset()
    counter = 0
    while counter < MAX_STEPS:
        if game.done:
            break
        game_state = game_state_from_game(game, need_worldmap=False, need_action_obs_pairs=False)
        states.append(game_state) # 注意这里如果bert输入不包含worldmap相关信息，那么编码时也不需要包含worldmap相关信息
        prev_moves = game.info['moves']
        admissible_commands = game.get_admissible_commands()
        if random.random() < epsilon:
            cmd = random.choice(admissible_commands)
        else:
            # Use the Q-network to choose the best action
            cmd = agent.select_action(game_state, admissible_commands, epsilon=epsilon)
        _ = game.act(cmd)
        current_moves = game.info['moves']
        counter += max(1, current_moves - prev_moves)
        actions.append(cmd)
        instant_reward = game.accumulated_score() - last_reward
        if instant_reward != 0:
            print(f"执行命令: {cmd}，获得奖励: {instant_reward}，当前总分: {game.accumulated_score()}")
        rewards.append(instant_reward)
        last_reward = game.accumulated_score()
    game.info['our_moves'] = counter
    return states, actions, rewards

def train_on_trajectory(trajectory):
    raise NotImplementedError("请在这里实现你的训练逻辑！输入是一条历史轨迹，包含 states, actions, rewards 三个列表。你需要使用这些数据来计算 loss 并更新你的 BertDQNAgent 的参数。")

def run(agent = None):
    game = default_game()
    buffer = TrajectoryReplayBuffer(capacity=500)
    # 1. 【热身阶段】让智能体用纯随机（epsilon=1.0）或者初期的网络先乱玩 50 局
    print("正在收集初始经验...")
    while len(buffer) < 50:
        trajectory = play_one_episode(game, None, epsilon=1.0) # 玩一局游戏
        buffer.push(trajectory) # 管它分高分低，全塞进去
    # 2. 【正式训练阶段】
    if agent is None:
        agent = BertDQNAgent() # 初始化你的BertDQN智能体
    else:
        print("使用传入的智能体继续训练...")
    for episode in range(1000):
        # 边玩边存
        trajectory = play_one_episode(game, agent, epsilon=0.1) # 带着探索去玩一局
        buffer.push(trajectory)
        
        # 每玩完一局，从 Buffer 里随机抽一条“历史轨迹”出来训练网络
        sampled_trajectory = buffer.sample()
        
        # 拿着这条历史轨迹，喂给你的 DRQN 计算 Loss 并更新参数
        train_on_trajectory(sampled_trajectory)
    return agent