# 1. 读取folder内所有checkpoint
# 2. 对每个checkpoint进行评估，记录结果
# 3. 输出评估结果

import logging

import common_new as common

checkpoint_paths = common.all_paths_with_suffix(common.CHECKPOINT_DIR, '.pth')

from agent_navigator import get_model, valid_all

logger = logging.getLogger('agent_navigator_test')

for i, checkpoint_path in enumerate(checkpoint_paths):
    msg = f'Evaluating {i}/{len(checkpoint_paths)} checkpoint: {checkpoint_path}'
    print(msg)
    logger.warning(msg)
    model = get_model(checkpoint_path)
    msg = f'Stop epoch: {model.stop_epoch}, valid score: {model.valid_score}'
    print(msg) 
    logger.warning(msg)
    norm_score, average_step = valid_all(model, split = 'test')
    msg = f'Checkpoint: {checkpoint_path}, norm_score: {norm_score}, average_step: {average_step}'
    print(msg)
    logger.warning(msg)

