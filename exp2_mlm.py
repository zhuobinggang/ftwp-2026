# 使用MLM模型替代ModelForSequenceClassification模型
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
# 替换为 AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import nn, optim
import numpy as np
from recordclass import recordclass
import copy

# 定义用于 Cloze 分类的目标词（注意 RoBERTa 中单词前带空格）
POS_WORD = " yes"
NEG_WORD = " no"

def get_clean_clean_walkthrough(game_path):
    game = Game_handle_worldmap(game_path)
    game.reset()
    clean_walkthrough = game.clean_walkthrough()
    clean_clean_walkthrough = []
    for cmd in clean_walkthrough:
        if game.done:
            break
        admissible_commands = game.get_admissible_commands()
        if cmd not in admissible_commands:
            pass
        else:
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
            admissible_commands = game.get_admissible_commands().copy()
            random.shuffle(admissible_commands)
            game_step = {
                'game_path': os.path.split(game_path)[-1],
                'room': game.room,
                'step': game.info['moves'],
                'action': cmd,
                'action_obs_pairs': game.action_obs_pairs.copy(),
                'recipe': game.recipe,
                'inventory': game.inventory_clean(),
                'admissible_commands': admissible_commands,
                'description': game.description_clean(),
                'won': game.info['won'],
                'lost': game.info['lost'],
                'score': game.info['score'],
                'max_score': game.info['max_score'],
            }
            gamesteps.append(game_step)
            game.act(cmd)
            cmd_index += 1
        assert game.done
    return pd.DataFrame(gamesteps)

def get_trains_and_tests():
    cache_dir = "temp"
    train_cache_path = os.path.join(cache_dir, "trains_900.feather")
    test_cache_path = os.path.join(cache_dir, "tests_100.feather")

    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print(f"Loading cached datasets from {train_cache_path} and {test_cache_path}...")
        df_trains = pd.read_feather(train_cache_path)
        df_tests = pd.read_feather(test_cache_path)
        return df_trains, df_tests

    print("Cache not found. Extracting walkthrough datasets (this may take 3-5 mins)...")
    train_games = get_cv_games(split='train')
    trains = train_games[:900]
    tests = train_games[900:1000]

    df_trains = extract_walkthrough_dataset(trains)
    df_tests = extract_walkthrough_dataset(tests)

    os.makedirs(cache_dir, exist_ok=True)
    df_trains.reset_index(drop=True).to_feather(train_cache_path)
    df_tests.reset_index(drop=True).to_feather(test_cache_path)
    print(f"Datasets cached to {train_cache_path} and {test_cache_path}.")

    return df_trains, df_tests

class TextClassificationDataset(Dataset):
    def __init__(self, prompt_label_dics, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts_a = [item['prompt_a'] for item in prompt_label_dics]
        self.prompts_b = [item['prompt_b'] for item in prompt_label_dics]
        self.labels = [item['label'] for item in prompt_label_dics]

        # 预先获取目标 token 的 ID
        self.pos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(POS_WORD)[0])
        self.neg_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(NEG_WORD)[0])
        self.mask_token_id = tokenizer.mask_token_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.prompts_a[idx],
            self.prompts_b[idx],
            truncation='only_first',
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 构造 MLM Labels：除 mask 位置外全部设为 -100
        labels = torch.full_like(input_ids, -100)
        
        # 找到输入中的 <mask> 位置
        mask_indices = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_indices) > 0:
            target_mask_idx = mask_indices[0]
            target_token_id = self.pos_token_id if self.labels[idx] == 1 else self.neg_token_id
            labels[target_mask_idx] = target_token_id

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

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

def row_to_game_state(row):
    game_state = Game_state_clean()
    game_state.room = row['room']
    game_state.action_obs_pairs = row['action_obs_pairs']
    game_state.recipe_good = row['recipe']
    game_state.inventory_good = row['inventory']
    game_state.available_commands_good = row['admissible_commands']
    game_state.description_good = row['description']
    return game_state

def get_prompt_label_dic(state: Game_state_clean, command: str, label: int, mask_token: str = "<mask>"):
    inventory_text = state.inventory_clean().strip()
    if inventory_text == '':
        inventory_item_count = 0
        inventory_text = 'empty'
    else:
        inventory_item_count = 1 + inventory_text.count(',')
    
    prompt_a = f"{inventory_item_count} {inventory_text} "
    recip_text = state.recipe_clean().strip()
    if recip_text == '':
        recip_text = 'missing'
    prompt_a += f"{recip_text} {state.description_clean()}"
    
    # Cloze 提示模板：在动作前放置 <mask> 引导预测
    prompt_b = f"{mask_token} {command}"
    return {'prompt_a': prompt_a, 'prompt_b': prompt_b, 'label': label}

def prompt_label_dic_generate(df_datas, negative_size = 0, mask_token = "<mask>"):
    csv = df_datas.sample(frac=1)
    prompt_label_dics = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        state = row_to_game_state(row)
        all_actions = state.get_admissible_commands()
        if negative_size > 0:
            all_negative_commands = [c for c in all_actions if c != row['action']]
            negative_commands = random.sample(all_negative_commands, min(negative_size, len(all_negative_commands)))
            for command in negative_commands:
                prompt_label_dics.append(get_prompt_label_dic(state, command, 0, mask_token=mask_token))
        prompt_label_dics.append(get_prompt_label_dic(state, row['action'], 1, mask_token=mask_token))
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
    
    inputs = {k: v.to(device) for k, v in encoding.items()}
    mask_token_id = tokenizer.mask_token_id
    pos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(POS_WORD)[0])
    neg_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(NEG_WORD)[0])

    with torch.no_grad():
        outputs = model(**inputs)
        # 获取 logits [batch_size, seq_len, vocab_size]
        logits = outputs.logits
        
        # 定位 <mask> token 的位置
        mask_indices = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)
        if len(mask_indices[1]) == 0:
            # 容错降级
            mask_idx = 0
        else:
            mask_idx = mask_indices[1][0]

        # 提取 mask 位置在 pos/neg 目标词上的 logits
        token_logits = logits[0, mask_idx, [neg_token_id, pos_token_id]]
        probs = torch.softmax(token_logits, dim=-1)
        score = probs[1].item() # 正类（yes）概率

    return score

def get_time_str():
    from datetime import datetime
    dt = datetime.now()
    return dt.strftime('%Y%m%d_%H%M%S_%f')

def get_writer(base_log_dir="runs"):
    from tensorboardX import SummaryWriter
    log_dir = os.path.join(base_log_dir, f'run_{get_time_str()}')
    writer = SummaryWriter(log_dir=log_dir)
    writer.global_step = 0
    return writer

def train(df_datas, model_name='FacebookAI/roberta-base', epochs=3, batch_size=16, lr=2e-5, max_length=480, negative_size=5, log_interval=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 使用 AutoModelForMaskedLM 代替分类模型
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    mask_token = tokenizer.mask_token
    prompt_label_dics = prompt_label_dic_generate(df_datas, negative_size=negative_size, mask_token=mask_token)
    dataset = TextClassificationDataset(prompt_label_dics, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    writer = get_writer()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val

            if writer.global_step % log_interval == 0:
                writer.add_scalar('Train/Batch_Loss', loss_val, writer.global_step)
            
            writer.global_step += 1

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.4f}")
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch + 1)

    writer.close()
    return model, tokenizer

def run_test(model, tokenizer, tests):
    model.eval()
    if model is None or tokenizer is None:
        print("模型未训练，请先运行 run_train()")
        return
    y_true = []
    y_pred = []
    for i, test_case in enumerate(tqdm(tests, desc="Testing")):
        prompt_a = test_case['prompt_a']
        prompt_b = test_case['prompt_b']
        label = test_case['label']
        score = predict(model, tokenizer, prompt_a, prompt_b)
        predicted_label = 1 if score >= 0.5 else 0
        y_true.append(label)
        y_pred.append(predicted_label)
    return y_true, y_pred

def save_model(model, tokenizer, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"模型和 Tokenizer 已成功保存至: {save_directory}")

def load_model(save_directory, target_device=None):
    if target_device is None:
        target_device = device
    
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    # 加载 MaskedLM 模型
    model = AutoModelForMaskedLM.from_pretrained(save_directory)
    model.to(target_device)
    model.eval()
    return model, tokenizer

def example_train_test():
    df_trains, df_tests = get_trains_and_tests()
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    data_tests = prompt_label_dic_generate(df_tests, negative_size=5, mask_token=tokenizer.mask_token)
    print('数据集生成完成')
    model, tokenizer = train(df_trains, negative_size=5)
    y_true, y_pred = run_test(model, tokenizer, data_tests)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    save_model(model, tokenizer, save_directory="./temp/model_900_games")

def predict_by_game_state(model, tokenizer, game_state: Game_state_clean):
    mask_token = tokenizer.mask_token
    prompt_dic = get_prompt_label_dic(game_state, 'dummy_command', 0, mask_token=mask_token)
    prompt_dics = []
    for command in game_state.get_admissible_commands():
        temp_dic = prompt_dic.copy()
        temp_dic['prompt_b'] = f"{mask_token} {command}"
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
    def predict(self, game_state: Game_state_clean):
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
        return np.sqrt(2*np.log(total_cnt)/action_cnt)

def maxmin_norm(p):
    return (p - p.min())/(p.max() - p.min())

def choose_action_ubc1(cmd_scores, action_visited_count, alpha=1):
    logits = torch.tensor([x['score'] for x in cmd_scores]).cuda()
    total_visits = sum(action_visited_count)
    uscore = [alpha*ucb1(v, total_visits) for v in action_visited_count]
    ssc = maxmin_norm(logits) + torch.tensor(uscore).cuda()
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
        self.state_action_count = {}
    def reset_state_action_count_if_need(self, game_state: Game_state_clean):
        action_obs_pairs = game_state.action_obs_pairs
        if len(action_obs_pairs) == 0:
            self.reset_state_action_count(game_state.room)
        self.current_room = game_state.room
    def calculated_state_action_count(self, game_state: Game_state_clean, actions):
        state_key = game_state_to_ucb1_key(game_state)
        state_action_executed_count = [self.get_state_action_count(state_key, action) for action in actions]
        state_action_executed_count_mask = [0] * len(actions)
        all_direction_known = True
        room_object = game_state.worldMap[game_state.room]
        direction_count = 0
        for idx, (action, executed_count) in enumerate(zip(actions, state_action_executed_count)):
            if action.startswith('go '):
                direction_count += 1
                direction = action.replace('go ', '')
                if direction not in room_object:
                    all_direction_known = False
                elif executed_count == 0:
                    state_action_executed_count_mask[idx] = 1
                else:
                    pass
        if all_direction_known:
            state_action_executed_count_mask = [0] * len(actions)
        masked_state_action_executed_count = [a + b for a, b in zip(state_action_executed_count, state_action_executed_count_mask)]
        return masked_state_action_executed_count
    def danger_action_mask(self, game_state: Game_state_clean, actions):
        raise NotImplementedError('Should not be used in navigator only model!')
    def predict(self, game_state: Game_state_clean):
        all_actions = game_state.get_admissible_commands()
        if len(all_actions) == 0:
            return ''
        self.reset_state_action_count_if_need(game_state)
        masked_state_action_executed_count = self.calculated_state_action_count(game_state, all_actions)
        if DANGER_FILTER_ON:
            danger_action_mask = self.danger_action_mask(game_state, all_actions)
            masked_state_action_executed_count = [a + b for a, b in zip(masked_state_action_executed_count, danger_action_mask)]
        cmd_scores = predict_by_game_state(self.bert, self.tokenizer, game_state)
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
        counter += max(1, current_moves - prev_moves)
        final_action = action
        if done:
            break
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