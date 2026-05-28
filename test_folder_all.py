# 1. 读取folder内所有checkpoint
# 2. 对每个checkpoint进行评估，记录结果
# 3. 输出评估结果

import common_new as common

checkpoint_paths = common.all_paths_with_suffix(common.CHECKPOINT_DIR, '.pth')

from agent_navigator import get_model, valid_all

for i, checkpoint_path in enumerate(checkpoint_paths):
    print(f'Evaluating {i}/{len(checkpoint_paths)} checkpoint: {checkpoint_path}')
    model = get_model(checkpoint_path)
    print(f'Stop epoch: {model.stop_epoch}, valid score: {model.valid_score}')
    norm_score, average_step = valid_all(model, split = 'test')
    print(f'Checkpoint: {checkpoint_path}, norm_score: {norm_score}, average_step: {average_step}')

