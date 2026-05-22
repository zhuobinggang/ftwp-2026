# from transformers import BertForMaskedLM, AutoTokenizer

from functools import lru_cache
import torch
from game import Game_handle_worldmap
import logging
from recordclass import recordclass
import logging
from typing import List

logger = logging.getLogger('bert_utils')
dbg = logger.debug

NextCommandResult = recordclass('NextCommandResult', 'index command logits distribution')
SpecialTokens = recordclass('SpecialTokens', 'cls sep pad mask unk')
BertInput = recordclass('BertInput', 'input_ids attention_mask labels token_type_ids')

BERT_BASE_UNCASED_MODEL_ID = 'bert-base-uncased'
ROBERTA_BASE_UNCASED_MODEL_ID = 'FacebookAI/roberta-base'
HISTORY_WINDOW = 20
EMPTY_INVENTORY = 'empty'
EMPTY_RECIPE = 'missing'
MAX_TOKEN_SIZE = 480
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# LOG_PROMPT = True # bert_tokenize_prompt_cut will print the prompt

@lru_cache(maxsize=1)
def default_tokenizer():
    from transformers import RobertaTokenizer
    # return AutoTokenizer.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)
    return RobertaTokenizer.from_pretrained(ROBERTA_BASE_UNCASED_MODEL_ID)

@lru_cache(maxsize=1)
def special_tokens_dict():
    toker = default_tokenizer()
    return SpecialTokens(
        cls = toker.cls_token,
        sep = toker.sep_token,
        pad = toker.pad_token,
        mask = toker.mask_token,
        unk = toker.unk_token
    )

def init_bert_ours():
    from transformers import RobertaForMaskedLM
    # return BertForMaskedLM.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)
    return RobertaForMaskedLM.from_pretrained(ROBERTA_BASE_UNCASED_MODEL_ID)

def bert_prompt_from_game_state(game: Game_handle_worldmap, need_action_history = True, seperater = None):
    if not seperater:
        seperater = special_tokens_dict().sep
    x = ''
    x += f"Room: {game.room} {seperater}"
    inventory_text = game.inventory_clean().strip()
    if inventory_text == '':
        inventory_text = EMPTY_INVENTORY
        inventory_item_count = 0
    else:
        inventory_item_count = 1 + inventory_text.count(',')
    x += f"Inventory: {inventory_text}. Total {inventory_item_count} items. {seperater}"
    recip_text = game.recipe_clean().strip()
    if recip_text == '':
        recip_text = EMPTY_RECIPE
    x += f"Recipe: {recip_text} {seperater}"
    if need_action_history:
        action_history_text = game.action_history(history_window=HISTORY_WINDOW)
        x += f"Action history: {action_history_text} {seperater}" 
    available_commands_text = game.available_commands_text()
    x += f'Available actions:\n{available_commands_text}'
    return x

def bert_tokenize_prompt_cut(game: Game_handle_worldmap, need_action_history = True):
    toker = default_tokenizer()
    SEP, CLS = special_tokens_dict().sep, special_tokens_dict().cls
    before_history_text = f'{CLS} '
    before_history_text += f"Room: {game.room} {SEP} "
    inventory_text = game.inventory_clean().strip()
    if inventory_text == '':
        inventory_text = EMPTY_INVENTORY
        inventory_item_count = 0
    else:
        inventory_item_count = 1 + inventory_text.count(',')
    before_history_text += f"Inventory: {inventory_text}. Total {inventory_item_count} items. {SEP} "
    recip_text = game.recipe_clean().strip()
    if recip_text == '':
        recip_text = EMPTY_RECIPE
    before_history_text += f"Recipe: {recip_text} {SEP} "
    before_history_text += 'Action history: '
    before_history_tokens = toker.encode(before_history_text, add_special_tokens=False) # list of numbers
    history_tokens = []
    if need_action_history:
        history_text = game.action_history(history_window=HISTORY_WINDOW)
        history_text += f" {SEP} "
        history_tokens = toker.encode(history_text, add_special_tokens=False) # list of numbers
    after_history_text = f'Available actions:\n{game.available_commands_text()} {SEP} '
    after_history_tokens = toker.encode(after_history_text, add_special_tokens=False) # list of numbers
    # trim history tokens
    if len(before_history_tokens) + len(history_tokens) + len(after_history_tokens) > MAX_TOKEN_SIZE:
        length_limit = MAX_TOKEN_SIZE - len(before_history_tokens) - len(after_history_tokens)
        start = max(0, len(history_tokens) - length_limit)
        history_tokens = history_tokens[start:]
        # dbg(f'prompt tokens length > {MAX_TOKEN_SIZE}, trim history tokens:')
        # dbg(toker.decode(before_history_tokens + history_tokens + after_history_tokens))
    return before_history_tokens + history_tokens + after_history_tokens


@lru_cache(maxsize=None)
def command_indexs_tokenized(command_length = 100):
    tokenizer = default_tokenizer()
    command_index_string = ' '.join([str(item) for item in list(range(command_length))])
    results =  tokenizer.encode(command_index_string, add_special_tokens = False)
    assert len(results) == command_length, f"command_indexs_tokenized: {len(results)} != {command_length}"
    return results


def tokenize_game_state(game: Game_handle_worldmap):
    return bert_tokenize_prompt_cut(game)

# @parameter: x是包含[MASK]标记的prompt
def get_mask_logits_simple(bert, game: Game_handle_worldmap):
    prompt_ids = tokenize_game_state(game)
    prompt_ids = torch.tensor(prompt_ids).unsqueeze(0) # (1, length)
    with torch.no_grad():
        logits = bert(input_ids=prompt_ids.to(DEVICE)).logits # (batch_size, seq_length, vocab_size)
    mask_token_index = (prompt_ids == default_tokenizer().mask_token_id)[0].nonzero(as_tuple=True)[0] # TODO: 检查
    return logits[0, mask_token_index] # (30522)

# @parameter: x是包含[MASK]标记的prompt
def get_cls_logits_simple(bert, game: Game_handle_worldmap):
    prompt_ids = tokenize_game_state(game) # [[token_length]]
    prompt_ids = torch.tensor(prompt_ids).unsqueeze(0) # (1, length)
    with torch.no_grad():
        logits = bert(input_ids=prompt_ids.to(DEVICE)).logits
    cls_token_index = 0
    return logits[0, cls_token_index] # (30522)


def action_select_loss(bert, game: Game_handle_worldmap, action_idx: int):
    prompt_ids = tokenize_game_state(game)
    prompt_ids = torch.tensor(prompt_ids).unsqueeze(0) # (1, length)
    label_code_in_tokenizer = command_indexs_tokenized()[action_idx]
    labels = torch.empty(prompt_ids.shape, dtype=torch.long).fill_(-100)
    batch_idx, cls_idx = 0, 0
    labels[batch_idx, cls_idx] = label_code_in_tokenizer
    outputs = bert(input_ids=prompt_ids.to(DEVICE), labels=labels.to(DEVICE))
    return outputs.loss

# vvvvv BATCHED VERSION vvvvv
# NOTE: 使用CLS token作为解码token
def to_bert_input(game: Game_handle_worldmap, action_idx: int, need_padding = True):
    prompt_ids = bert_tokenize_prompt_cut(game) # (length)
    attention_mask = [1] * len(prompt_ids)
    if need_padding and len(prompt_ids) < MAX_TOKEN_SIZE:
        pad_size = MAX_TOKEN_SIZE - len(prompt_ids)
        prompt_ids += [default_tokenizer().pad_token_id] * pad_size
        attention_mask += [0] * pad_size
    if need_padding:
        assert len(prompt_ids) == MAX_TOKEN_SIZE, f"prompt_ids length {len(prompt_ids)} != {MAX_TOKEN_SIZE}"
    # prompt_ids = torch.tensor(prompt_ids).unsqueeze(0) # (1, length)
    labels = [-100] * MAX_TOKEN_SIZE if need_padding else [-100] * len(prompt_ids)
    labels[0] = command_indexs_tokenized()[action_idx]
    return BertInput(
        input_ids = prompt_ids,
        attention_mask = attention_mask,
        labels = labels
    )


def to_bert_inputs(games:List[Game_handle_worldmap], action_idxs: List[int]):
    return [to_bert_input(game, action_idx) for game, action_idx in zip(games, action_idxs)]

def action_select_loss_batched(bert, games:List[Game_handle_worldmap], action_idxs: List[int]):
    bert_input = to_bert_inputs(games, action_idxs)
    outputs = bert(input_ids=bert_input.input_ids.to(DEVICE), 
                   attention_mask=bert_input.attention_mask.to(DEVICE), 
                   labels=bert_input.labels.to(DEVICE))
    return outputs.loss
# ^^^^^ BATCHED VERSION ^^^^^

# 用于critic
# @parameter: x是包含[MASK]标记的prompt
def get_cls_output(model, x):
    tokenizer = default_tokenizer()
    inputs = tokenizer(x, return_tensors="pt")
    out = model(**inputs.to(DEVICE), output_hidden_states=True) # 23 layers tuple
    last_layer = out.hidden_states[-1] # (1, 52, 768)
    cls_out = last_layer[:, 0] # (1, 768)
    return cls_out


def get_command_logits_simple(model, game: Game_handle_worldmap, commands, from_cls = True):
    if len(commands) == 0:
        logger.error(f'没有可用的命令了！game state:\n{bert_prompt_from_game_state(game)}')
        return None
    if not from_cls:
        mask_logits = get_mask_logits_simple(model, game) # (1, 50368)
    else:
        mask_logits = get_cls_logits_simple(model, game) # (1, 50368)
    command_length = len(commands)
    command_indexs = command_indexs_tokenized()[:command_length]
    command_logits = mask_logits[command_indexs] # (command_length)
    return command_logits # (command_length)

def get_command_distribution_simple(model, game: Game_handle_worldmap, commands):
    command_logits = get_command_logits_simple(model, game, commands)
    # print(command_logits) # NOTE: TESTING
    command_logits[command_logits < 0] = 0 # 出现负数无法用于建构distribution，会报错，因此直接设置为0即可
    import torch
    dist = torch.distributions.categorical.Categorical(probs = command_logits)
    return dist


# 拥有探索性
def get_next_command_by_distribution(model, game: Game_handle_worldmap, commands):
    dist = get_command_distribution_simple(model, game, commands)
    command_index = dist.sample().item()
    command = commands[command_index]
    results = NextCommandResult(command_index, command, distribution=dist)
    return results


# 贪婪
def get_next_command(model, game: Game_handle_worldmap, commands):
    command_logits = get_command_logits_simple(model, game, commands) # (command_length)
    command_index = command_logits.argmax().item()
    command = commands[command_index]
    results = NextCommandResult(command_index, command, command_logits)
    return results