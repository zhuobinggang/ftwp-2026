from agent_navigator import *
from game import test_game
import common_new as common
from dataset_create import read_csv_dataset

m = get_model('checkpoints/ftwp_our_navigator_only/roberta_navigator_0_best.pth', Model_ucb1)
m.train()
optimizer = optim.AdamW(m.parameters(), lr=LEARNING_RATE) # 从1e-3到2e-5

game = Game_with_navigator(common.GAME_BASE_PATH + '/fake_test_10/tw-cooking-recipe2+take2+cook+go12-EKP9iMylTo8yTxWj.ulx')

csv = read_csv_dataset(inputpath = 'good_dataset', split = 'fake_valid_10', suffix = '_with_navigator')
row = csv.iloc[6]

best_action = row['action']
fake_action = row['admissible_commands'][-1] # 提高最后一个action被选中的概率，方便测试

# 获取各个action的logits

state = row_to_game_state(row)
# bert_input = to_bert_input_theirs(state, best_action, fake_action, need_padding = True)
next_command = get_next_command_batch(m.bert, state)
print(state.get_admissible_commands())
print(next_command.logits)
print(f"action_selected: {state.get_admissible_commands()[next_command.command_index]}")
print(f"action_selected: {next_command.command}")

# 微调
bi = to_bert_input_theirs(state, fake_action, positive=True, need_padding = True)
input_ids = torch.tensor(bi.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
attention_mask = torch.tensor(bi.attention_mask, dtype=torch.long).unsqueeze(0).to(DEVICE)
label_ids = torch.tensor(bi.labels, dtype=torch.long).unsqueeze(0).to(DEVICE)
loss = m.bert(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids).loss
print(f"loss: {loss.item()}")

# backward
loss.backward()
optimizer.step()
optimizer.zero_grad()

print("微调完成！")

# 查看微调后结果
next_command = get_next_command_batch(m.bert, state)
print(state.get_admissible_commands())
print(next_command.logits)
print(f"action_selected: {state.get_admissible_commands()[next_command.index]}")
print(f"action_selected: {next_command.command}")
