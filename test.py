# 对于一个游戏，使用bert dqn完成主动探索提升

from game import default_game, Game_with_navigator, game_state_from_game
from bert_state_action_encoder import State_Action_Encoder
import random
from collections import deque
from q_net import ChoiceTextQnet
from bert_dqn import BertDQNAgent
from recordclass import recordclass
import torch
from tqdm import tqdm

Trajectory = recordclass('Trajectory', 'states actions rewards')

class TrajectoryReplayBuffer:
    def __init__(self, capacity=1000):
        """
        capacity: Buffer 能容纳的最大轨迹数量
        """
        # 使用 deque (双端队列)，当超过容量时会自动从左侧弹出老数据
        self.buffer = deque(maxlen=capacity)
    
    def push(self, trajectory: Trajectory):
        """
        存入一条完整的轨迹。
        trajectory 应该是一个 Trajectory 类型的对象
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
            cmd = agent.select_action(game_state, admissible_commands)
        _ = game.act(cmd)
        current_moves = game.info['moves']
        counter += max(1, current_moves - prev_moves)
        actions.append(cmd)
        instant_reward = game.accumulated_score() - last_reward
        # if instant_reward != 0:
        #     print(f"执行命令: {cmd}，获得奖励: {instant_reward}，当前总分: {game.accumulated_score()}")
        rewards.append(instant_reward)
        last_reward = game.accumulated_score()
    game.info['our_moves'] = counter
    result = Trajectory(states=states, actions=actions, rewards=rewards)
    return result

def train_on_trajectory(trajectory: Trajectory, agent: BertDQNAgent):
    if agent.optimizer is None:
        agent.init_optimizer() # 确保优化器已经初始化
    agent.optimizer.zero_grad() # 清空梯度
    trajectory_length = len(trajectory.states)
    for t in range(trajectory_length):
        state = trajectory.states[t]
        action = trajectory.actions[t]
        reward = trajectory.rewards[t]
        q_selected = agent.q_values([state], [action], target_q=False) # (1,) - 选中动作的Q值
        q_selected = q_selected.squeeze(0) # 转成标量
        if t < trajectory_length - 1:
            next_state = trajectory.states[t + 1]
            admissible_commands = next_state.get_admissible_commands()
            q_next_values = agent.q_values([next_state] * len(admissible_commands), admissible_commands, target_q=True)
            max_q_next = torch.max(q_next_values).item()
        else:
            max_q_next = 0.0
        target_q = reward + 0.99 * max_q_next
        step_loss = (q_selected - target_q) ** 2
        step_loss = step_loss / trajectory_length
        step_loss.backward() # 反向传播
    agent.optimizer.step() # 更新参数
    agent.optimizer.zero_grad() # 清空梯度
    # 更新target网络的参数（软更新）
    tau = 0.005
    with torch.no_grad():
        for target_param, q_param in zip(agent.target_q_net.parameters(), agent.q_net.parameters()):
            target_param.copy_(tau * q_param + (1.0 - tau) * target_param)
    

def run():
    buffer = TrajectoryReplayBuffer(capacity=500)
    # 1. 【热身阶段】让智能体用纯随机（epsilon=1.0）或者初期的网络先乱玩 50 局
    print("正在收集初始经验...")
    while len(buffer) < 50:
        game = default_game()
        trajectory = play_one_episode(game, None, epsilon=1.0) # 玩一局游戏
        buffer.push(trajectory) # 管它分高分低，全塞进去
    # 2. 【正式训练阶段】
    agent = BertDQNAgent() # 初始化你的BertDQN智能体
    print("初始化智能体并开始正式训练...")
    for episode in tqdm(range(1000), desc="训练进度"):
        # 边玩边存
        game = default_game()
        trajectory = play_one_episode(game, agent, epsilon=0.1) # 带着探索去玩一局
        buffer.push(trajectory)
        print(f'{episode+1}: 总奖励: {sum(trajectory.rewards)}, 轨迹长度: {len(trajectory.states)}')
        
        # 每玩完一局，从 Buffer 里随机抽一条“历史轨迹”出来训练网络
        sampled_trajectory = buffer.sample()
        
        # 拿着这条历史轨迹，喂给你的 DRQN 计算 Loss 并更新参数
        train_on_trajectory(sampled_trajectory, agent)
    agent.q_net.save_checkpoint() # 训练结束后保存模型
    return agent