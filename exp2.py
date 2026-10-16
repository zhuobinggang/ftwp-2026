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
from game import Game_state_clean
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import nn, optim



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
    train_cache_path = os.path.join(cache_dir, "trains_900.pkl")
    test_cache_path = os.path.join(cache_dir, "tests_100.pkl")

    # 1. 检查本地缓存是否存在
    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print(f"Loading cached datasets from {train_cache_path} and {test_cache_path}...")
        df_trains = pd.read_pickle(train_cache_path)
        df_tests = pd.read_pickle(test_cache_path)
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
    df_trains.to_pickle(train_cache_path)
    df_tests.to_pickle(test_cache_path)
    print(f"Datasets cached to {train_cache_path} and {test_cache_path}.")

    return df_trains, df_tests

# TODO: 待完成
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
        if negative_size > 0:
            all_negative_commands = [command for command in state.get_admissible_commands() if command != row['action']]
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

def train(df_datas, model_name='FacebookAI/roberta-base', epochs=3, batch_size=4, lr=2e-5, max_length=480):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    prompt_label_dics = prompt_label_dic_generate(df_datas, negative_size=5)
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
def main():
    df_trains, df_tests = get_trains_and_tests()
    data_trains = prompt_label_dic_generate(df_trains, negative_size=5)
    data_tests = prompt_label_dic_generate(df_tests, negative_size=5)
    model, tokenizer = train(df_trains)
    y_true, y_pred = run_test(model, tokenizer, data_tests)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    save_model(model, tokenizer, save_directory="./temp/model_900_games")