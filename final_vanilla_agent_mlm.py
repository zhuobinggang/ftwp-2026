# 使用MLM模型替代ModelForSequenceClassification模型
import final_vanilla_agent
from final_vanilla_agent import *
from transformers import AutoModelForMaskedLM

# 定义用于 Cloze 分类的目标词（注意 RoBERTa 中单词前带空格）
POS_WORD = " yes"
NEG_WORD = " no"

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
        
        # 找到输入中的 <mask>/[MASK] 位置
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
    
    # Cloze 提示模板：在动作前放置 mask token 引导预测
    prompt_b = f"{mask_token} {command}"
    return {'prompt_a': prompt_a, 'prompt_b': prompt_b, 'label': label}

def prompt_label_dic_generate(df_datas, negative_size=0, mask_token="<mask>"):
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
        logits = outputs.logits
        
        # 定位 <mask>/[MASK] token 的位置
        mask_indices = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)
        if len(mask_indices[1]) == 0:
            mask_idx = 0
        else:
            mask_idx = mask_indices[1][0]

        # 提取 mask 位置在 pos/neg 目标词上的 logits
        token_logits = logits[0, mask_idx, [neg_token_id, pos_token_id]]
        probs = torch.softmax(token_logits, dim=-1)
        score = probs[1].item()

    return score

def train(df_datas, model_name='FacebookAI/roberta-base', epochs=3, batch_size=16, lr=2e-5, max_length=480, negative_size=5, log_interval=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

def load_model(save_directory, target_device=None):
    if target_device is None:
        target_device = device
    
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = AutoModelForMaskedLM.from_pretrained(save_directory)
    model.to(target_device)
    model.eval()
    return model, tokenizer

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

# =========================================================================
# 命名空间同步：确保 final_vanilla_agent 中复用的 Model / Model_ucb1 / test_game
# 内部调用的 load_model 和 predict_by_game_state 准确路由到当前 MLM 实现
# =========================================================================
final_vanilla_agent.POS_WORD = POS_WORD
final_vanilla_agent.NEG_WORD = NEG_WORD
final_vanilla_agent.TextClassificationDataset = TextClassificationDataset
final_vanilla_agent.get_prompt_label_dic = get_prompt_label_dic
final_vanilla_agent.prompt_label_dic_generate = prompt_label_dic_generate
final_vanilla_agent.predict = predict
final_vanilla_agent.train = train
final_vanilla_agent.run_test = run_test
final_vanilla_agent.load_model = load_model
final_vanilla_agent.predict_by_game_state = predict_by_game_state
final_vanilla_agent.example_train_test = example_train_test