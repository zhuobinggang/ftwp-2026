from q_net import ChoiceTextQnet
from bert_state_action_encoder import State_Action_Encoder
import random
from pydash.arrays import chunk
import torch
from bert_utils import DEVICE

BATCH_SIZE = 16

class BertDQNAgent:
    def __init__(self, q_net = None):
        self.encoder = State_Action_Encoder()
        print("使用外部传入的Q网络") if q_net is not None else print("初始化Q网络")
        self.q_net = q_net if q_net is not None else ChoiceTextQnet()
        self.q_net.to(DEVICE)
        self.target_q_net = ChoiceTextQnet()
        self.target_q_net.to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict()) # 初始化时让目标网络和主网络的权重相同
        self.target_q_net.eval() # 目标网络只用于推理，不参与训练，所以设置为 eval 模式
        for param in self.target_q_net.parameters():
            param.requires_grad = False
        self.optimizer = None

    def init_optimizer(self):
        # 只把 q_network (MLP) 的参数传给优化器
        self.optimizer = torch.optim.AdamW(
            self.q_net.parameters(), 
            lr=1e-3,               # MLP 打分头可以用大一点的学习率
            weight_decay=0.01      # 适当的权重衰减防止过拟合
        )

    def q_values(self, game_states, admissible_commands, target_q = False):
        assert len(game_states) == len(admissible_commands), f"game_states和admissible_commands的长度应该相等！但得到的长度分别是{len(game_states)}和{len(admissible_commands)}"
        q_net = self.target_q_net if target_q else self.q_net
        q_values = []
        for blob in chunk(list(zip(game_states, admissible_commands)), BATCH_SIZE):
            batch_game_states = [x[0] for x in blob]
            batch_commands = [x[1] for x in blob]
            # Encode the game state with each admissible command
            batch_cls_embeddings = self.encoder.encode(batch_game_states, batch_commands)
            assert batch_cls_embeddings.shape[0] == len(batch_commands) and batch_cls_embeddings.shape[1] == 768, f"编码得到的特征维度不对！应该是(batch_size, 768)，但得到的是{batch_cls_embeddings.shape}"
            # Get Q-values for each command
            batch_q_values = q_net.forward(batch_cls_embeddings)
            assert batch_q_values.shape[0] == len(batch_commands) and batch_q_values.shape[1] == 1, f"Q网络输出的维度不对！应该是(batch_size, 1)，但得到的是{batch_q_values.shape}"
            q_values.append(batch_q_values)
        q_values = torch.cat(q_values, dim=0) # (num_commands, 1)
        assert q_values.shape[0] == len(admissible_commands) and q_values.shape[1] == 1, f"最终得到的Q值维度不对！应该是(num_commands, 1)，但得到的是{q_values.shape}"
        return q_values.squeeze(1) # (num_commands,)


    def select_action(self, game_state, admissible_commands, need_more=False):
        game_states = [game_state] * len(admissible_commands)
        with torch.no_grad():
            q_values = self.q_values(game_states, admissible_commands, target_q=False) # (num_commands,)
        best_index = torch.argmax(q_values).item()
        if need_more:
            return admissible_commands[best_index], {'q_values': q_values.cpu().numpy(), 'best_index': best_index}
        return admissible_commands[best_index]

def test_bert_dqn_agent():
    from game import default_game, game_state_from_game
    agent = BertDQNAgent()
    game = default_game()
    _ = game.reset()
    walkthrough = game.clean_walkthrough()
    states = []
    actions = []
    rewards = []
    last_reward = 0
    for cmd in walkthrough:
        print('命令：', cmd)
        if cmd == 'prepare meal':
            break
        states.append(game_state_from_game(game, need_worldmap=False)) # 注意这里如果bert输入不包含worldmap相关信息，那么编码时也不需要包含worldmap相关信息
        actions.append(cmd)
        _ = game.act(cmd)
        rewards.append(game.accumulated_score() - last_reward)
        last_reward = game.accumulated_score()
    state = states[-2]
    admissible_commands = state.get_admissible_commands()
    best_command, info = agent.select_action(state, admissible_commands, need_more=True)
    return info