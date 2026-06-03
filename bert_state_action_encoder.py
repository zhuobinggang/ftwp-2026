from bert_utils import init_bert_ours, default_tokenizer, DEVICE
from game import default_game
from game import Game_with_navigator, Game_state_clean, game_state_from_game
from bert_utils import default_tokenizer, special_tokens_dict, EMPTY_RECIPE, EMPTY_INVENTORY
import torch

MAX_TOKEN_SIZE = 342

def actions_history(action_obs_pairs,seperator=', '): # 注意这里有个空格
    return seperator.join([action for action, obs in action_obs_pairs])

# @Sample: final_tokens
# 
# <s> History: go east, examine cookbook, drop red potato, drop white onion, cook red onion with stove, take knife, chop red onion with knife, drop knife </s>
# Reward obtained: 2 </s>
# Recipe: Ingredients: red onion Directions: chop the red onion fry the red onion prepare meal </s>
# Inventory: 1 a chopped fried red onion </s>
# Action: have fun </s>
# 
def bert_tokenize_for_dqn(game_state: Game_state_clean, action: str, need_padding=False):
    toker = default_tokenizer()
    CLS, SEP, PAD = special_tokens_dict().cls, special_tokens_dict().sep, special_tokens_dict().pad
    # Reward obtained: n <SEP>\n
    reward_obtained_text = f'Reward obtained: {game_state.accumulated_score()} {SEP}\n'
    reward_obtained_tokens = toker.encode(reward_obtained_text, add_special_tokens=False)
    # Recipe: xxx <SEP>\n
    recip_text = game_state.recipe_clean().strip()
    if recip_text == '':
        recip_text = EMPTY_RECIPE
    recip_text = f"Recipe: {recip_text} {SEP}\n"
    recip_tokens = toker.encode(recip_text, add_special_tokens=False)
    # Inventory: xxx <SEP>\n
    inventory_text = game_state.inventory_clean().strip()
    if inventory_text == '':
        inventory_item_count = 0
        inventory_text = EMPTY_INVENTORY
    else:
        inventory_item_count = 1 + inventory_text.count(',')
    inventory_text = f'Inventory: {inventory_item_count} {inventory_text} {SEP}\n'
    inventory_tokens = toker.encode(inventory_text, add_special_tokens=False)
    # Action: xxx <SEP>
    text_b = f"Action: {action} {SEP}"
    text_b_tokens = toker.encode(text_b, add_special_tokens=False)

    # History: act0, act1, ... actn <SEP>\n
    history_text = actions_history(game_state.action_obs_pairs) # act0, act1, ... actn
    history_tokens = toker.encode(history_text, add_special_tokens=False) # NOTE: can be truncated
    neccessary_tokens_length = len(reward_obtained_tokens) + len(recip_tokens) + len(inventory_tokens) + len(text_b_tokens)
    history_length_limit = MAX_TOKEN_SIZE - neccessary_tokens_length - 8
    if len(history_tokens) > history_length_limit:
        history_tokens = history_tokens[-history_length_limit:]
        print('注意：history tokens被截断了！')
    history_prefix_tokens = toker.encode('History: ', add_special_tokens=False) # 3
    history_suffix_tokens = toker.encode(f' {SEP}\n', add_special_tokens=False) # 3
    history_tokens = history_prefix_tokens + history_tokens + history_suffix_tokens

    # final tokens
    final_tokens_prefix = toker.encode(f'{CLS} ', add_special_tokens=False) # 2
    final_tokens = final_tokens_prefix + history_tokens + reward_obtained_tokens + recip_tokens + inventory_tokens + text_b_tokens

    if need_padding:
        # 填充到最大长度
        pad_tokens = toker.encode(f'{PAD}', add_special_tokens=False) * (MAX_TOKEN_SIZE - len(final_tokens))
        final_tokens = final_tokens + pad_tokens

    return final_tokens, text_b_tokens

def test_truncate_history():
    # bert = init_bert_ours()
    toker = default_tokenizer()
    game = default_game()
    _ = game.reset()
    walkthrough = game.clean_walkthrough()
    for cmd in walkthrough:
        print('命令：', cmd)
        if cmd == 'prepare meal':
            break
        _ = game.act(cmd)
    game.action_obs_pairs = game.action_obs_pairs * 200 # 测试超长历史
    final_tokens, text_b_tokens = bert_tokenize_for_dqn(game, 'prepare meal')
    print(toker.decode(final_tokens))
    assert len(final_tokens) == MAX_TOKEN_SIZE, f"final tokens长度超过限制了！长度为{len(final_tokens)}"

# @Sample: final_tokens
# 
# <s> History: go east </s>
# Reward obtained: 0 </s>
# Recipe: missing </s>
# Inventory: 3 a white onion, a raw red potato, a red onion </s>
# Action: prepare meal </s>
#
def test_empty_cookbook():
    # bert = init_bert_ours()
    toker = default_tokenizer()
    game = default_game()
    _ = game.reset()
    walkthrough = game.clean_walkthrough()
    for cmd in walkthrough:
        print('命令：', cmd)
        if cmd == 'examine cookbook':
            break
        _ = game.act(cmd)
    final_tokens, text_b_tokens = bert_tokenize_for_dqn(game, 'prepare meal')
    print(toker.decode(final_tokens))



class State_Action_Encoder:
    def __init__(self):
        self.toker = default_tokenizer()
        self.bert = init_bert_ours()
        self.bert.to(DEVICE)
        self.bert.eval() # 注意这里bert只用来编码，不进行训练，所以设置为eval模式
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def encode(self, game_states: list[Game_state_clean], actions: list[str]):
        token_ids_list = []
        for game_state, action in zip(game_states, actions):
            token_ids, _ = bert_tokenize_for_dqn(game_state, action, need_padding=True)
            token_ids_list.append(token_ids)
        prompt_ids = torch.tensor(token_ids_list) # (batch, length)
        with torch.no_grad():
            hidden_states = self.bert(input_ids=prompt_ids.to(DEVICE), output_hidden_states=True).hidden_states
        cls_token_index = 0
        out = hidden_states[-1][:, cls_token_index] # (batch, 768)
        assert out.shape[0] == len(game_states) and out.shape[1] == 768, f"bert编码得到的特征维度不对！应该是(batch, 768)，但得到的是{out.shape}"
        return out
    

def test_state_action_encoder():
    encoder = State_Action_Encoder()
    game = default_game()
    _ = game.reset()
    walkthrough = game.clean_walkthrough()
    states = []
    actions = []
    rewards = []
    last_reward = 0
    for cmd in walkthrough:
        states.append(game_state_from_game(game, need_worldmap=False)) # 注意这里如果bert输入不包含worldmap相关信息，那么编码时也不需要包含worldmap相关信息
        print('命令：', cmd)
        if cmd == 'prepare meal':
            break
        actions.append(cmd)
        _ = game.act(cmd)
        rewards.append(game.accumulated_score() - last_reward)
        last_reward = game.accumulated_score()
    states = states[:-1] # 最后一个状态是执行prepare meal之后的状态，不用来编码
    logits = encoder.encode(states, actions)
    print('编码得到的logits shape:', logits.shape)


def test_load_same_bert():
    bert1 = init_bert_ours()
    bert2 = init_bert_ours()
    toker = default_tokenizer()
    text = "Hello, world!"
    tokens1 = toker.encode(text, add_special_tokens=False)
    tokens2 = toker.encode(text, add_special_tokens=False)
    hidden_states1 = bert1(input_ids=torch.tensor([tokens1]), output_hidden_states=True).hidden_states[-1] # 这里不需要放到GPU上，因为我们只是想测试两份bert的输出是否一样，不需要计算梯度
    hidden_states2 = bert2(input_ids=torch.tensor([tokens2]), output_hidden_states=True).hidden_states[-1]
    assert torch.allclose(hidden_states1, hidden_states2), "两份bert的输出不一样！"
