from agent_navigator import *
print(f'TEMP_SAVE_DIR: {TEMP_SAVE_DIR}')
train_repeat(testing=False)
test_all_checkpoints(testing=False)