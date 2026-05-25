# 2026.5.21 精简化，只使用navigator
from dataset_create import read_csv_dataset, get_cv_games
import re
import pandas as pd
from tqdm import tqdm
from common_new import logging, beutiful_print_command_and_probs, get_time_str
from bert_utils import default_tokenizer, special_tokens_dict, EMPTY_RECIPE, EMPTY_INVENTORY
from bert_utils import BertInput, command_indexs_tokenized, init_bert_ours, DEVICE, NextCommandResult
logger = logging.getLogger('agent_navigator')
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch import nn, optim
from functools import lru_cache
import numpy as np
from recordclass import recordclass
from pydash.arrays import chunk
import random
import os
import common_new as common
from game import Game_with_navigator, Game_state_clean, Game_state, test_game

LEARNING_RATE = 1e-5
CSV_SUFFIX = '_with_navigator'
SAVE_DIR = common.CHECKPOINT_DIR
assert os.path.exists(SAVE_DIR), f"Save dir {SAVE_DIR} does not exist!"

TRAIN_SPLIT = 'train'
# PART_VALID_SPLIT = 'partial_valid'
VALID_SPLIT = 'valid'
TEST_SPLIT = 'test'
MAX_TEST_STEP = 100
MAX_TOKEN_SIZE = 342
NEGATIVE_SAMPLE_SIZE = 5

BATCH_SIZE = 8

DANGER_FILTER_ON = False # NOTE: 对于navigator only模型，danger filter不需要打开

GAME_INIT_FUNC = Game_with_navigator

BEST_MODELS = [-1, -1, -1]

def get_writer():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    writer.global_step = 0
    return writer

def bert_tokenize_prompt_cut_theirs(game: Game_with_navigator, action: str):
    toker = default_tokenizer()
    CLS, SEP = special_tokens_dict().cls, special_tokens_dict().sep
    text = f'{CLS} '
    # before_history_text += f"Room: {game_state.room} {SEP} "
    inventory_text = game.inventory_clean().strip()
    if inventory_text == '':
        inventory_item_count = 0
        inventory_text = EMPTY_INVENTORY
    else:
        inventory_item_count = 1 + inventory_text.count(',')
    text += f'{inventory_item_count} {inventory_text} '
    recip_text = game.recipe_clean().strip()
    if recip_text == '':
        recip_text = EMPTY_RECIPE
    text += f"{recip_text} {game.description_clean()} " # NOTE: 2025.5.11 space is important!
    text_b = f"{SEP} {action} {SEP}"
    tokens = toker.encode(text, add_special_tokens=False) # list of numbers
    text_b_tokens = toker.encode(text_b, add_special_tokens=False)
    if len(tokens) + len(text_b_tokens) > MAX_TOKEN_SIZE:
        tokens = tokens[:MAX_TOKEN_SIZE - len(text_b_tokens)]
    return tokens, text_b_tokens

# NOTE: 使用CLS token作为解码token
def to_bert_input_theirs(game: Game_with_navigator, action: str, positive = True, need_padding = True):
    a_tokens, b_tokens = bert_tokenize_prompt_cut_theirs(game, action) # (length)
    prompt_ids = a_tokens + b_tokens
    attention_mask = [1] * len(prompt_ids)
    pad_size = 0
    if need_padding and len(prompt_ids) < MAX_TOKEN_SIZE:
        pad_size = MAX_TOKEN_SIZE - len(prompt_ids)
        prompt_ids += [default_tokenizer().pad_token_id] * pad_size
        attention_mask += [0] * pad_size
    if need_padding:
        assert len(prompt_ids) == MAX_TOKEN_SIZE, f"prompt_ids length {len(prompt_ids)} != {MAX_TOKEN_SIZE}"
    labels = [-100] * MAX_TOKEN_SIZE if need_padding else [-100] * len(prompt_ids)
    action_idx = 1 if positive else 0
    labels[0] = command_indexs_tokenized()[action_idx] # 2026 5.24 BUG
    # prepare token_type_ids
    token_type_ids = [0] * (len(a_tokens) + 1) + [1] * (len(b_tokens) - 1) # 需要注意的是，b_tokens的第一个token是SEP，所以需要-1
    if need_padding:
        token_type_ids += [0] * pad_size
    return BertInput(
        input_ids = prompt_ids,
        attention_mask = attention_mask,
        labels = labels,
        token_type_ids = token_type_ids
    )

# 只用于从csv的行转换成game_state
def row_to_game_state(row):
    game_state = Game_state_clean()
    game_state.room = row['room']
    game_state.action_obs_pairs = row['action_obs_pairs'] # NOTE
    game_state.recipe_good = row['recipe']
    game_state.inventory_good = row['inventory']
    game_state.available_commands_good = row['admissible_commands']
    #if common.COMMAND_LIST_SHUFFLE: # NOTE: 2026.5.24 数据集生成的时候已经打乱过了，不需要再打乱
    #    random.shuffle(game_state.available_commands_good)
    game_state.description_good = row['description']
    return game_state

def test_row_to_game_state():
    df = read_csv_dataset(inputpath = 'good_dataset', split = 'fake_test', suffix = '_with_navigator')
    row = df.iloc[0]
    game_state = row_to_game_state(row)
    print(game_state)

@lru_cache(maxsize=1)
def dataloader_get(split = 'train'):
    csv = read_csv_dataset(split = split, suffix=CSV_SUFFIX)
    csv = csv.sample(frac=1)
    bert_inputs = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        state = row_to_game_state(row) # NOTE: 2025.5.5 打乱以提高模型的泛化能力
        # assert state.get_admissible_commands() == row['admissible_commands'] # CHECKED 26.5.24
        negative_commands = [command for command in state.get_admissible_commands() if command != row['action']]
        negative_commands = negative_commands[:NEGATIVE_SAMPLE_SIZE]
        for command in negative_commands:
            bert_input = to_bert_input_theirs(state, command, positive=False, need_padding=True)
            bert_inputs.append(bert_input)
        bert_input = to_bert_input_theirs(state, row['action'], positive=True, need_padding=True)
        bert_inputs.append(bert_input)
    all_input_ids = torch.tensor([bert_input.input_ids for bert_input in bert_inputs], dtype=torch.long)
    all_attention_mask = torch.tensor([bert_input.attention_mask for bert_input in bert_inputs], dtype=torch.long)
    all_label_ids = torch.tensor([bert_input.labels for bert_input in bert_inputs], dtype=torch.long)
    # NOTE: only for their model
    all_token_type_ids = torch.tensor([bert_input.token_type_ids for bert_input in bert_inputs], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_attention_mask, all_label_ids, all_token_type_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    logger.warning(f'Dataloader for split {split} created with {len(bert_inputs)} samples, batch size {BATCH_SIZE}, total batches {len(train_dataloader)}.')
    return train_dataloader

@torch.no_grad()
def batch_predict(bert, batch_bert_input):
    input_ids = torch.tensor([bert_input.input_ids for bert_input in batch_bert_input], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor([bert_input.attention_mask for bert_input in batch_bert_input], dtype=torch.long).to(DEVICE)
    # NOTE: 2025.5.11 RoBERTa don't use token_type_ids! Error happens if use it!
    # token_type_ids = torch.tensor([bert_input.token_type_ids for bert_input in batch_bert_input], dtype=torch.long).to(DEVICE)
    outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # 🔍 临时添加这行，看看是不是这里少了一个：
    if len(batch_bert_input) != logits.size(0):
        raise ValueError(f"警告：输入了 {len(batch_bert_input)} 个样本，但模型返回了 {logits.size(0)} 个 Logits！")
    cls_token_index = 0
    logits = logits[:, cls_token_index] # (batch_size, 30522)
    command_length = 2 # 0 or 1
    command_indexs = command_indexs_tokenized()[:command_length]
    command_logits = logits[:, command_indexs] # (batch_size, 2)
    # BUG: 2026.5.22 等等，这里如果直接用softmax的话，如果指令数量大于批次大小，就会出问题
    # return command_logits.softmax(dim=1)[:, 1].tolist() # probabilities of positive class
    return command_logits[:, 1].tolist() # logits of positive class 修正与2026.5.22

# NOTE: 2026.5.22 删除参数中的commands，因为可以直接从game_state中获取，减少出错的可能性
def get_next_command_batch(bert, game_state: Game_state):
    # 🌟 如果 bert 是被 Accelerator 包装过的，提取出最底层的原始 torch Module
    if hasattr(bert, 'module'):
        unwrapped_bert = bert.module
    else:
        unwrapped_bert = bert
    # 对于每一个action，计算它的概率
    bert_inputs = []
    for command in game_state.get_admissible_commands():
        bert_input = to_bert_input_theirs(game_state, command, positive=True, need_padding=True)
        bert_inputs.append(bert_input)
    command_logits = []
    for batch_bert_input in chunk(bert_inputs, BATCH_SIZE):
        command_logits += batch_predict(unwrapped_bert, batch_bert_input)
    command_index = np.argmax(command_logits)
    max_prob_command = game_state.get_admissible_commands()[command_index]
    # beutiful_print_command_and_probs game_state.get_admissible_commands(), command_logits)
    result = NextCommandResult(command_index, max_prob_command, command_logits)
    return result

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.bert = init_bert_ours()
        self.bert = None
        self.prefix = 'roberta_navigator'
        self.valid_score = -1
    def init_bert(self):
        if not self.bert:
            self.bert = init_bert_ours()
    def predict(self, game_state:Game_state): # @return: action
        result = get_next_command_batch(self.bert, game_state)
        return result.command
    def save_checkpoint(self, base_path = 'checkpoints', epoch = -1, valid_score = -1):
        # path = f'{base_path}/{self.prefix}_epoch_{epoch}.pth'
        path = f'{base_path}/{self.prefix}_best.pth'
        torch.save({
            'epoch': epoch,
            'state': self.state_dict(),
            'valid_score': valid_score,
        }, path)
    def load_checkpoint(self, path):
        self.init_bert() # NOTE: 需要先初始化然后加载
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['state'])
        self.valid_score = checkpoint.get('valid_score', -1)
        self.stop_epoch = checkpoint.get('epoch', -1)


# vvvvv with UCB1 vvvvv


Room = recordclass('Room', 'name east west north south')

# 只考虑房间名，库存状况和recipe的检查状况
def game_state_to_ucb1_key(game_state: Game_state):
    recipe_got = True if game_state.recipe_clean() else False
    return f'Room: {game_state.room}\nInventory: {game_state.inventory_clean()}\nRecipe: {recipe_got}'

def ucb1(action_cnt, total_cnt):
    if action_cnt == 0:
        return 5
    else:
        return np.sqrt(2*np.log(total_cnt)/action_cnt) # 如果total_cnt=10, action_cnt=1，值大概为2.15，总之都不会比2.5大的感觉。

def maxmin_norm(p): # 返回0-1之间的值
    return (p - p.min())/(p.max() - p.min())

def choose_action_ubc1(logits, action_visited_count, alpha=1):
    """
    :param logits: vector with logits for actions
    :param sacnt: vector with counts for each visit of the action
    :returns: action number
    """
    total_visits = sum(action_visited_count) # 该状态下所有行动的执行次数
    uscore = [alpha*ucb1(v, total_visits) for v in action_visited_count] # 加权分
    ssc = maxmin_norm(logits) + torch.tensor(uscore).cuda() # 如果所有指令都没有访问过，那么所有指令的分数都是5，很公平。如果指令被访问过，那么logits几乎不影响它的分数 -> 意味着模型会优先选择没有访问过的指令
    return ssc.argmax(), ssc.softmax(dim=0)


class Model_ucb1(Model):
    def __init__(self):
        super().__init__()
        self.prefix = 'Model_ucb1'
        self.reset_state_action_count()
    def incresase_state_action_count(self, key, action):
        self.state_action_count[key][action] += 1
    def get_state_action_count(self, key, action):
        if key not in self.state_action_count:
            self.state_action_count[key] = {}
        if action not in self.state_action_count[key]:
            self.state_action_count[key][action] = 0
        return self.state_action_count[key][action]
    def reset_state_action_count(self, room_name = ''):
        logger.debug(f'\n\n\n === \n\n重新开始，清空地图信息, 房间名: {room_name}')
        self.state_action_count = {} # 记录每个状态下的动作选择次数
    def reset_state_action_count_if_need(self, game_state: Game_state):
        action_obs_pairs = game_state.clean_action_obs_pairs()
        if len(action_obs_pairs) == 0:
            self.reset_state_action_count(game_state.room)
        self.current_room = game_state.room # 总是要更新当前房间，但是在更新之前需要先更新世界地图（如果有必要）
    def calculated_state_action_count(self, game_state: Game_state, actions):
        # NOTE: 使用move_action_mask来促进模型探索新的房间
        state_key = game_state_to_ucb1_key(game_state)
        state_action_executed_count = [self.get_state_action_count(state_key, action) for action in actions]
        # 通过mask来屏蔽掉已经知道的房间
        state_action_executed_count_mask = [0] * len(actions)
        all_direction_known = True
        room_object = game_state.worldMap[game_state.room]
        direction_count = 0
        for idx, (action, executed_count) in enumerate(zip(actions, state_action_executed_count)):
            if action.startswith('go '):
                direction_count += 1
                direction = action.replace('go ', '')
                # dbg(f'Room {game_state.room} Dcirection {direction} exist, executed {executed_count} times.')
                if direction not in room_object: # 未知房间
                    all_direction_known = False
                    # dbg(f'Room {game_state.room} Dcirection {direction} unknown.')
                elif executed_count == 0: # 知道房间存在，但是没有真正执行过
                    state_action_executed_count_mask[idx] = 1
                    # dbg(f'Room {game_state.room} Dcirection {direction} known but never executed.')
                else: # 知道房间存在，并且执行过
                    # dbg(f'Room {getattr(room_object, direction)} known and already visited {executed_count} times.')
                    pass
        if all_direction_known: # 清空所有的已知房间的值
            state_action_executed_count_mask = [0] * len(actions)
            if direction_count > 0:
                # logger.debug(f'All {direction_count} directions are known, resetting the mask.')
                pass
        masked_state_action_executed_count = [a + b for a, b in zip(state_action_executed_count, state_action_executed_count_mask)]
        return masked_state_action_executed_count
    def danger_action_mask(self, game_state: Game_state, actions):
        raise NotImplementedError('Should not be used in navigator only model!')
        danger_action_mask = [0] * len(actions)
        # NOTE: 2025.5.16 使用bert来判断危险指令
        recip_text = game_state.recipe_clean().strip()
        if recip_text == '':
            return danger_action_mask
        for idx, action in enumerate(actions):
            prefix = action.split()[0]
            if prefix in MAYBE_DANGER_COMMAND_PREFIX:
                if use_bert_to_identify_danger_command(recip_text, action):
                    # logger.debug(f'Action {action} is dangerous. I will mark it as executed.')
                    danger_action_mask[idx] = 1
        return danger_action_mask
    def predict(self, game_state:Game_state):
        # NOTE: 更新世界地图(根据上一步动作的结果)，只要发生移动必须对链接进行更新
        all_actions = game_state.get_admissible_commands()
        self.reset_state_action_count_if_need(game_state)
        masked_state_action_executed_count = self.calculated_state_action_count(game_state, all_actions)
        if DANGER_FILTER_ON: # NOTE: 2025.5.16 使用bert来判断危险指令
            danger_action_mask = self.danger_action_mask(game_state, all_actions)
            masked_state_action_executed_count = [a + b for a, b in zip(masked_state_action_executed_count, danger_action_mask)]
        # NOTE: 获取logits并使用ucb1算法选择动作
        result = get_next_command_batch(self.bert, game_state) # 说明在这里面，all actions变了
        logits = result.logits # (actions_length)
        # BUG reported
        assert len(logits) == len(all_actions), f"Logits length {len(logits)} does not match actions length {len(all_actions)}"
        logits = torch.tensor(logits).to(DEVICE)
        best_action_idx, action_prob = choose_action_ubc1(logits, masked_state_action_executed_count)
        best_action = all_actions[best_action_idx]
        if DANGER_FILTER_ON: # DEBUG
            model_choice_index = logits.argmax().item()
            if danger_action_mask[model_choice_index] == 1:
                logger.warning(f'Action {all_actions[model_choice_index]} is dangerous. I marked it as executed. Final action: {best_action}')
        state_key = game_state_to_ucb1_key(game_state)
        self.incresase_state_action_count(state_key, best_action)
        if False:
            logger.debug(f'Recipe: {game_state.recipe_clean()}\n')
            logger.debug(f'Description: {game_state.description_clean()}\n')
            logger.debug(f'Inventory: {game_state.inventory_clean()}\n')
            logger.debug(f'Inventory: {game_state.clean_action_obs_pairs()[-5:]}\n')
            beutiful_print_command_and_probs(actions, action_prob, log_func=logger.debug)
            logger.debug(f'Action: {best_action}\n\n')
        return best_action
    
def train(model, split = 'train', log_name = '', writer = None):
    train_dataloader = dataloader_get(split=split)
    # training
    from accelerate import Accelerator
    accelerator = Accelerator()
    # model.cuda()
    model.train()
    logger.debug('Model train on.')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # 从1e-3到2e-5
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        if batch_idx % 1000 == 0:
            logger.warning(f'Training iteration {batch_idx}')
        input_ids, input_mask, label_ids, token_type_ids = batch
        # NOTE: 2025.5.11 RoBERTa don't use token_type_ids! Error happens if use it!
        outputs = model.bert(input_ids=input_ids.to(DEVICE), 
                   attention_mask=input_mask.to(DEVICE), 
                   # token_type_ids=token_type_ids.to(DEVICE),
                   labels=label_ids.to(DEVICE))
        loss = outputs.loss
        accelerator.backward(loss)
        writer.add_scalar(f'Loss/train_{log_name}', loss.item(), writer.global_step)
        writer.global_step += 1
        optimizer.step()
        optimizer.zero_grad()

def get_model(checkpoint_path = None, init_func = Model_ucb1):
    model = init_func()
    model.prefix = 'roberta_theirs'
    model.init_bert()
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    model.cuda()
    return model

def valid_all(model: Model, split = 'partial_valid', game_init_func = None):
    if game_init_func is None:
        game_init_func = GAME_INIT_FUNC
        assert GAME_INIT_FUNC == Game_with_navigator
    game_paths = get_cv_games(split=split)
    score = 0
    max_score = 0
    steps = []
    logger.debug(f'Validating {split} games, total {len(game_paths)}')
    for game_path in tqdm(game_paths, desc=f"Validating {split} games"):
        game = game_init_func(game_path)
        result = test_game(game, model, max_step=MAX_TEST_STEP)
        score += result.score
        max_score += result.max_score
        steps.append(result.step)
        # dbg(f'Valid results,  {result.score} / {result.max_score}, steps {result.step}, game {game_path}')
    average_step = np.mean(steps)
    norm_score = score / max_score
    print(f'Validation on {split} norm_score: {norm_score}')
    return norm_score, average_step

def train_repeat(testing = False):
    global BEST_MODELS, MAX_TEST_STEP, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT
    if testing:
        MAX_TEST_STEP = 20
        TRAIN_SPLIT = 'fake_train_100'
        VALID_SPLIT = 'fake_valid_10'
        TEST_SPLIT = 'fake_test_10'
        repeat = 1
        epoch = 5
    else:
        repeat = 1
        epoch = 5
    logger.warning(f'Training with {repeat} repeats, {epoch} epochs each, {MAX_TEST_STEP} max test steps. Train split: {TRAIN_SPLIT}, valid split: {VALID_SPLIT}, test split: {TEST_SPLIT}.')
    INIC_FUNC = Model_ucb1
    ucb1_on = 'with UCB1' if INIC_FUNC == Model_ucb1 else 'w/o UCB1'
    for rp in range(repeat):
        model = get_model(init_func = INIC_FUNC)
        model.prefix = f'roberta_navigator_{rp}'
        if testing:
            model.prefix += '_testing'
        max_score = 0
        writer = get_writer()
        for i in range(epoch):
            train(model, split=TRAIN_SPLIT, log_name=f'{rp}', writer=writer)
            score, avg_step = valid_all(model, split=VALID_SPLIT, game_init_func=GAME_INIT_FUNC)
            logger.error(f'{get_time_str()} Full valid score ({rp}) {ucb1_on}: {score}, average step {avg_step}')
            logger.warning(f'{get_time_str()} Full valid score ({rp}) {ucb1_on}: {score}, average step {avg_step}')
            # get_writer().add_scalar(f'Score/valid_rp{rp}', score, i)
            if score > max_score:
                max_score = score
                BEST_MODELS[rp] = i
                logger.error(f'Best model ({rp}) at epoch {i}, score {max_score}, average step {avg_step}. BEST_MODELS: {BEST_MODELS}')
                # get_writer().add_scalar(f'Score/best_valid_rp{rp}', score, i)
                # 补上测试分数
                # score, avg_step = valid_all(model, split=TEST_SPLIT, game_init_func=GAME_INIT_FUNC)
                # logger.error(f'Full test score ({rp}) {ucb1_on}: {score}, average step {avg_step}')
                # print(f'Full test score ({rp}) {ucb1_on}: {score}, average step {avg_step}')
                model.save_checkpoint(base_path = SAVE_DIR, epoch=i, valid_score=score)