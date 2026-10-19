# 获取train set中1000个游戏，其中900个作为训练，100个作为验证
# 动态生成训练数据
import common_new as common
import os
import random
import pandas as pd
from functools import lru_cache
BASE_PATH = common.GAME_BASE_PATH
from game import Game_handle_worldmap
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import nn, optim
import numpy as np
from recordclass import recordclass
import copy


def get_clean_clean_walkthrough(game_path):
    game = Game_handle_worldmap(game_path)
    game.reset()
    clean_walkthrough = game.clean_walkthrough()
    clean_clean_walkthrough = []
    for cmd in clean_walkthrough:
        if game.done:
            break
        admissible_commands = game.get_admissible_commands()
        if cmd not in admissible_commands: # 可能存在admissible_commands中没有的指令，比如反复open door之类的，直接跳过，只要保证game.done就行
            pass
        else: # 如果指令在admissible_commands中，则正常执行
            clean_clean_walkthrough.append(cmd)
            game.act(cmd)
    if not game.done:
        raise ValueError(f'Game {game_path} walkthrough cannot finish the game, current admissible_commands {admissible_commands}\ncurrent command {cmd}')
    assert game.done
    return clean_clean_walkthrough

@lru_cache(maxsize=None)
def all_game_paths(folder_path):
    return common.all_paths_with_suffix(folder_path, '.z8')

def get_cv_games(path = BASE_PATH, split = 'fake_test'):
    return all_game_paths(f'{path}/{split}')

def extract_walkthrough_dataset(train_game_paths):
    gamesteps = []
    for game_path in tqdm(train_game_paths):
        clean_walkthrough = get_clean_clean_walkthrough(game_path)
        cmd_index = 0
        game = Game_handle_worldmap(game_path)
        game.reset()
        assert game.action_obs_pairs == []
        while cmd_index < len(clean_walkthrough):
            cmd = clean_walkthrough[cmd_index]
            if game.done:
                break
            admissible_commands = game.get_admissible_commands().copy() # NOTE: 这里必须copy，因为后面可能会修改admissible_commands
            random.shuffle(admissible_commands) # NOTE：确保这是唯一一次shuffle
            game_step = {
                'game_path': os.path.split(game_path)[-1],
                'room': game.room,
                'step': game.info['moves'], # NOTE: 这里的step是游戏中的步数
                'action': cmd, # NOTE: 会被导航后下一个动作覆盖
                'action_obs_pairs': game.action_obs_pairs.copy(), # BUG: 必须copy，不然会覆盖
                'recipe': game.recipe, # 在Game_handle_recipe中经过清理
                'inventory': game.inventory_clean(),
                'admissible_commands': admissible_commands,
                'description': game.description_clean(),
                'won': game.info['won'],
                'lost': game.info['lost'],
                'score': game.info['score'],
                # 'entities': game.info['entities'],
                'max_score': game.info['max_score'],
            }
            gamesteps.append(game_step)
            game.act(cmd) # 执行的可能是导航后的下一个指令
            cmd_index += 1
        assert game.done
    return pd.DataFrame(gamesteps)

def get_trains_and_tests():
    cache_dir = "temp"
    train_cache_path = os.path.join(cache_dir, "trains_900.feather")
    test_cache_path = os.path.join(cache_dir, "tests_100.feather")

    # 1. 检查本地缓存是否存在
    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print(f"Loading cached datasets from {train_cache_path} and {test_cache_path}...")
        df_trains = pd.read_feather(train_cache_path)
        df_tests = pd.read_feather(test_cache_path)
        return df_trains, df_tests

    # 2. 缓存缺失时重新生成
    print("Cache not found. Extracting walkthrough datasets (this may take 3-5 mins)...")
    train_games = get_cv_games(split='train')
    trains = train_games[:900]
    tests = train_games[900:1000]

    df_trains = extract_walkthrough_dataset(trains)
    df_tests = extract_walkthrough_dataset(tests)

    # 3. 自动创建目录并保存
    os.makedirs(cache_dir, exist_ok=True)
    df_trains.reset_index(drop=True).to_feather(train_cache_path)
    df_tests.reset_index(drop=True).to_feather(test_cache_path)
    print(f"Datasets cached to {train_cache_path} and {test_cache_path}.")

    return df_trains, df_tests

    return df_trains, df_tests

class TextClassificationDataset(Dataset):
    def __init__(self, prompt_label_dics, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts_a = [item['prompt_a'] for item in prompt_label_dics]
        self.prompts_b = [item['prompt_b'] for item in prompt_label_dics]
        self.labels = [item['label'] for item in prompt_label_dics]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # AutoTokenizer 会根据底层是 BERT 还是 RoBERTa 自动处理特殊 token 和 token_type_ids
        encoding = self.tokenizer(
            self.prompts_a[idx],
            self.prompts_b[idx],
            truncation='only_first',
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

        # BERT 会有 token_type_ids，RoBERTa 则不会；有就带上，没有就忽略
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        return item

class Game_state_clean():
    def __init__(self):
        self.recipe_good = ''
        self.inventory_good = ''
        self.description_good = ''
        self.available_commands_good = []
        self.accumulated_score_good = 0
    def recipe_clean(self):
        return self.recipe_good
    def inventory_clean(self):
        return self.inventory_good
    def description_clean(self):
        return self.description_good
    def filtered_available_commands(self):
        return self.available_commands_good.copy()
    def get_admissible_commands(self):
        return self.available_commands_good.copy()
    def accumulated_score(self):
        return self.accumulated_score_good


# 只用于从csv的行转换成game_state
def row_to_game_state(row):
    game_state = Game_state_clean()
    game_state.room = row['room']
    game_state.action_obs_pairs = row['action_obs_pairs'] # NOTE
    game_state.recipe_good = row['recipe']
    game_state.inventory_good = row['inventory']
    game_state.available_commands_good = row['admissible_commands']
    game_state.description_good = row['description']
    return game_state

def get_prompt_label_dic(state: Game_state_clean, command: str, label: int):
    prompt_a = ''
    # before_history_text += f"Room: {game_state.room} {SEP} "
    inventory_text = state.inventory_clean().strip()
    if inventory_text == '':
        inventory_item_count = 0
        inventory_text = 'empty'
    else:
        inventory_item_count = 1 + inventory_text.count(',')
    prompt_a += f'{inventory_item_count} {inventory_text} '
    recip_text = state.recipe_clean().strip()
    if recip_text == '':
        recip_text = 'missing'
    prompt_a += f"{recip_text} {state.description_clean()}" # NOTE: 2025.5.11 space is important!
    prompt_b = f"next command: {command}"
    return {'prompt_a': prompt_a, 'prompt_b': prompt_b, 'label': label}

def prompt_label_dic_generate(df_datas, negative_size = 0): # 实时生成negative samples，返回dataloader
    csv = df_datas
    csv = csv.sample(frac=1) # 打乱
    prompt_label_dics = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        state = row_to_game_state(row) # NOTE: 2025.5.5 打乱以提高模型的泛化能力
        all_actions = state.get_admissible_commands()
        if negative_size > 0:
            all_negative_commands = [command for command in all_actions if command != row['action']]
            negative_commands = random.sample(all_negative_commands, min(negative_size, len(all_negative_commands)))
            for command in negative_commands:
                prompt_label_dics.append(get_prompt_label_dic(state, command, 0)) # negative sample
        prompt_label_dics.append(get_prompt_label_dic(state, row['action'], 1)) # positive sample
    return prompt_label_dics

def predict(model, tokenizer, prompt_a, prompt_b, max_length=480):
    model.eval()
    encoding = tokenizer(
        prompt_a,
        prompt_b,
        truncation='only_first',
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # 动态将 tokenizer 返回的所有 tensor 推送到指定设备
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        # 通过 **inputs 解包，BERT 会接收 token_type_ids，RoBERTa 忽略它
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        score = probs[0, 1].item()

    return score

def train(df_datas, model_name='FacebookAI/roberta-base', epochs=3, batch_size=16, lr=2e-5, max_length=480, negative_size=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    prompt_label_dics = prompt_label_dic_generate(df_datas, negative_size=negative_size)
    dataset = TextClassificationDataset(prompt_label_dics, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            # 将 batch 内所有张量推送到 GPU/CPU 并通过 **batch 传入
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.4f}")
    return model, tokenizer


def run_test(model, tokenizer, tests):
    model.eval()
    if model is None or tokenizer is None:
        print("模型未训练，请先运行 run_train()")
        return
    y_true = []
    y_pred = []
    # tests_flat = [x for game in tests for x in game]  # 展开每个游戏的所有数据
    for i, test_case in enumerate(tqdm(tests, desc="Testing")):
        prompt_a = test_case['prompt_a']
        prompt_b = test_case['prompt_b']
        label = test_case['label']
        # cmd = test_case['cmd']
        score = predict(model, tokenizer, prompt_a, prompt_b)
        predicted_label = 1 if score >= 0.5 else 0
        y_true.append(label)
        y_pred.append(predicted_label)
    return y_true, y_pred


# ==========================================
# 4. 保存与加载
# ==========================================
def save_model(model, tokenizer, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"模型和 Tokenizer 已成功保存至: {save_directory}")

def load_model(save_directory, target_device=None):
    if target_device is None:
        target_device = device
    
    # 无论是从 BERT 还是 RoBERTa 保存的目录，Auto 类都会读取 config.json 自动恢复原架构
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    model.to(target_device)
    model.eval()
    return model, tokenizer

### 数据集准备就绪，准备训练，准备一个简单的模型
def example_train_test():
    df_trains, df_tests = get_trains_and_tests()
    # data_trains = prompt_label_dic_generate(df_trains, negative_size=5)
    data_tests = prompt_label_dic_generate(df_tests, negative_size=5)
    print('数据集生成完成')
    model, tokenizer = train(df_trains, negative_size=5)
    y_true, y_pred = run_test(model, tokenizer, data_tests)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    save_model(model, tokenizer, save_directory="./temp/model_900_games")

print('参考example_train_test()进行训练和测试（非正式运行）')


def predict_by_game_state(model, tokenizer, game_state: Game_state_clean):
    # 将game_state转换为复数prompt_a和prompt_b，然后取分数最高的那个动作作为最终动作
    prompt_dic = get_prompt_label_dic(game_state, 'dummy_command', 0) # NOTE: 这里的label是无关紧要的
    prompt_dics = []
    for command in game_state.get_admissible_commands():
        temp_dic = prompt_dic.copy()
        temp_dic['prompt_b'] = f"next command: {command}"
        temp_dic['cmd'] = command
        prompt_dics.append(temp_dic)
    results = []
    for dic in prompt_dics:
        score = predict(model, tokenizer, dic['prompt_a'], dic['prompt_b'])
        results.append({'cmd': dic['cmd'], 'score': score})
    return results

class Model():
    def __init__(self):
        self.bert = None
        self.tokenizer = None
        self.valid_score = -1
    def init_bert(self, model_folder = './temp/model_900_games'):
        model, tokenizer = load_model(model_folder)
        self.bert = model
        self.tokenizer = tokenizer
        self.valid_score = -1
    def predict(self, game_state: Game_state_clean): # @return: action
        cmd_scores = predict_by_game_state(self.bert, self.tokenizer, game_state)
        best_action = max(cmd_scores, key=lambda x: x['score'])['cmd']
        return best_action

def game_state_to_ucb1_key(game_state: Game_state_clean):
    recipe_got = True if game_state.recipe_clean() else False
    return f'Room: {game_state.room}\nInventory: {game_state.inventory_clean()}\nRecipe: {recipe_got}'

def ucb1(action_cnt, total_cnt):
    if action_cnt == 0:
        return 5
    else:
        return np.sqrt(2*np.log(total_cnt)/action_cnt) # 如果total_cnt=10, action_cnt=1，值大概为2.15，总之都不会比2.5大的感觉。

def maxmin_norm(p): # 返回0-1之间的值
    return (p - p.min())/(p.max() - p.min())

def choose_action_ubc1(cmd_scores, action_visited_count, alpha=1):
    """
    :param logits: vector with logits for actions
    :param sacnt: vector with counts for each visit of the action
    :returns: action number
    """
    logits = torch.tensor([x['score'] for x in cmd_scores]).cuda()
    total_visits = sum(action_visited_count) # 该状态下所有行动的执行次数
    uscore = [alpha*ucb1(v, total_visits) for v in action_visited_count] # 加权分
    ssc = maxmin_norm(logits) + torch.tensor(uscore).cuda() # 如果所有指令都没有访问过，那么所有指令的分数都是5，很公平。如果指令被访问过，那么logits几乎不影响它的分数 -> 意味着模型会优先选择没有访问过的指令
    return ssc.argmax(), ssc.softmax(dim=0)

DANGER_FILTER_ON = False
PRINT_CMD_SCORE = False
class Model_ucb1(Model):
    def __init__(self):
        super().__init__()
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
        self.state_action_count = {} # 记录每个状态下的动作选择次数
    def reset_state_action_count_if_need(self, game_state: Game_state_clean):
        action_obs_pairs = game_state.action_obs_pairs
        if len(action_obs_pairs) == 0:
            self.reset_state_action_count(game_state.room)
        self.current_room = game_state.room # 总是要更新当前房间，但是在更新之前需要先更新世界地图（如果有必要）
    def calculated_state_action_count(self, game_state: Game_state_clean, actions):
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
    def danger_action_mask(self, game_state: Game_state_clean, actions):
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
    def predict(self, game_state: Game_state_clean):
        # NOTE: 更新世界地图(根据上一步动作的结果)，只要发生移动必须对链接进行更新
        all_actions = game_state.get_admissible_commands() # NOTE: 必须确保get_admissible_commands返回的动作顺序和之前的动作顺序一致
        if len(all_actions) == 0:
            return ''
        self.reset_state_action_count_if_need(game_state)
        masked_state_action_executed_count = self.calculated_state_action_count(game_state, all_actions)
        if DANGER_FILTER_ON: # NOTE: 2025.5.16 使用bert来判断危险指令
            danger_action_mask = self.danger_action_mask(game_state, all_actions)
            masked_state_action_executed_count = [a + b for a, b in zip(masked_state_action_executed_count, danger_action_mask)]
        # NOTE: 获取logits并使用ucb1算法选择动作
        cmd_scores = predict_by_game_state(self.bert, self.tokenizer, game_state)
        # TODO: 打印
        if PRINT_CMD_SCORE:
            print_cmd_scores(cmd_scores, game_state)
        best_action_idx, action_prob = choose_action_ubc1(cmd_scores, masked_state_action_executed_count)
        best_action = all_actions[best_action_idx]
        state_key = game_state_to_ucb1_key(game_state)
        self.incresase_state_action_count(state_key, best_action)
        assert all_actions == game_state.get_admissible_commands()
        return best_action

def print_cmd_scores(cmd_scores, game_state: Game_state_clean):
    print(f"Room: {game_state.room}\nInventory: {game_state.inventory_clean()}\nRecipe: {game_state.recipe_clean()}")
    for cmd_score in cmd_scores:
        print(f"Command: {cmd_score['cmd']}, Score: {cmd_score['score']:.4f}")


class Game_state_clean_with_worldmap(Game_state_clean):
    def __init__(self):
        super().__init__()
        self.worldMap = {}
        self.itemMap = {}

def game_state_from_game(game: Game_handle_worldmap, need_worldmap = True, need_action_obs_pairs = True):
    game_state = Game_state_clean_with_worldmap()
    game_state.recipe_good = game.recipe_clean()
    game_state.inventory_good = game.inventory_clean()
    game_state.description_good = game.description_clean()
    game_state.available_commands_good = game.get_admissible_commands().copy()
    if need_worldmap:
        game_state.worldMap = copy.deepcopy(game.worldMap)
        game_state.itemMap = copy.deepcopy(game.itemMap)
    game_state.room = game.room
    if need_action_obs_pairs:
        game_state.action_obs_pairs = game.action_obs_pairs.copy()
    return game_state

TestResult = recordclass('TestResult', 'step score max_score info')
def test_game(game: Game_handle_worldmap, model: Model, max_step = 100, need_print = False):
    # import game_for_llm
    # max_step = 50 # 2025.8.28 实验用，实验结束后删除
    # dbg('Testing: Model eval on, model cuda on.')
    model.bert.eval()
    model.bert.to(device)
    obs, info = game.reset()
    counter = 0
    final_action = ''
    while counter < max_step:
        action = model.predict(game_state_from_game(game))
        if action == '':
            print(f'No admissible commands available, stopping the game.')
            final_action = 'No admissible commands available'
            info['lost'] = True
            break
        prev_moves = game.info['moves']
        obs, reward, done, info = game.act(action)
        if need_print:
            print(f'{action} => {obs}')
        current_moves = info['moves']
        counter += max(1, current_moves - prev_moves) # 考虑到可能的多步行动（比如高级命令）
        final_action = action
        if done:
            break
    # result = (counter, info['score'], info['max_score'], info)
    print(f'Game done: {info["score"]} / {info["max_score"]}, steps {counter}, won: {info["won"]}, lost: {info["lost"]}, path: {game.game_path}')
    if info['lost']:
        print(f'Game lost: final action: {final_action}')
    result = TestResult(counter, info['score'], info['max_score'], info)
    return result

def example_run_test():
    model = Model_ucb1()
    model.init_bert()
    train_games = get_cv_games(split='train')
    tests = train_games[900:1000]
    steps = []
    scores = []
    max_scores = []
    for game_path in tests:
        game = Game_handle_worldmap(game_path)
        result = test_game(game, model, max_step=100, need_print=False)
        steps.append(result.step)
        scores.append(result.score)
        max_scores.append(result.max_score)
    print(f'Average steps: {sum(steps) / len(steps)}')
    print(f'Norm score: {sum(scores) / sum(max_scores)}')