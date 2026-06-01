from agent_cta import *
from dataset_create import *

model = get_model('/home/zhuobinggang/research/ftwp-2026/checkpoints/cta_vanilla/roberta_navigator_20260529_150834_265935_best.pth')

# TODO: 未完成
def valid_by_csv(model: Model_ucb1, split = 'fake_test_10'):
    csv = read_csv_dataset(inputpath = 'good_dataset/without_navigator', split = split, suffix = '')
    for index, row in csv.iterrows():
        state = row_to_game_state(row)
        target = row['action']
        admissible_commands = row['admissible_commands']
        bert_inputs = []
        for cmd in admissible_commands:
            bert_input = to_bert_input_theirs(state, cmd, positive=False, need_padding=True)
            bert_inputs.append(bert_input)
        command_logits = []
        for batch_bert_input in chunk(bert_inputs, BATCH_SIZE):
            command_logits += batch_predict(model.bert, batch_bert_input)
        command_index = np.argmax(command_logits)
        max_prob_command = admissible_commands[command_index]