# 2026.7.16 实验idea: 首先全图探索，再开始具体行动
# 是否需要对training walkthrough重写？可以将examine cookbook之前的部分全部包揽
# 不重写也可以
# 游戏使用game_cmd_no_filter，checkpoints使用cta_nav_cmdraw
# 

from game_cmd_no_filter import Game_with_navigator_no_cmd_filter
import game
import agent_cta
import common_new
from common_new import logging
import dataset_create
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)

def test_game(ga: Game_with_navigator_no_cmd_filter, 
              model = game.Fake_model(), 
              max_step = 100, 
              need_print = False):
    # import game_for_llm
    # max_step = 50 # 2025.8.28 实验用，实验结束后删除
    # dbg('Testing: Model eval on, model cuda on.')
    if model.training:
        model.eval()
        logger.debug('Model eval on.')
    if not next(model.parameters()).is_cuda:
        model.cuda()
        logger.debug('Model cuda on.')
    obs, info = ga.reset()
    # TODO: 在这里插入全地图探索
    counter = 0
    final_action = ''
    while counter < max_step:
        action = model.predict(game.game_state_from_game(ga))
        prev_moves = ga.info['moves']
        obs, reward, done, info = ga.act(action)
        if need_print:
            print(f'{action} => {obs}')
        current_moves = info['moves']
        counter += max(1, current_moves - prev_moves) # 考虑到可能的多步行动（比如高级命令）
        final_action = action
        if done:
            break
    # result = (counter, info['score'], info['max_score'], info)
    logger.debug(f'Game done: {info["score"]} / {info["max_score"]}, steps {counter}, won: {info["won"]}, lost: {info["lost"]}, path: {ga.game_path}')
    if info['lost']:
        logger.warning(f'Game lost: final action: {final_action}')
    result = game.TestResult(counter, info['score'], info['max_score'], info)
    return result

def valid_all(model: agent_cta.Model_ucb1, split = 'fake_test_10', game_init_func = Game_with_navigator_no_cmd_filter):
    game_paths = dataset_create.get_cv_games(split=split)
    score = 0
    max_score = 0
    steps = []
    logger.debug(f'Validating {split} games, total {len(game_paths)}')
    for game_path in tqdm(game_paths, desc=f"Validating {split} games"):
        game = game_init_func(game_path)
        result = test_game(game, model, max_step=100)
        score += result.score
        max_score += result.max_score
        steps.append(result.step)
        # dbg(f'Valid results,  {result.score} / {result.max_score}, steps {result.step}, game {game_path}')
    average_step = np.mean(steps)
    norm_score = score / max_score
    print(f'Validation on {split} norm_score: {norm_score}')
    return norm_score, average_step

# TODO: 重写test函数
def test_all_checkpoints(checkpoint_dir = "./checkpoints/cta_nav_cmdraw", test_split = 'test'):
    checkpoint_paths = common_new.all_paths_with_suffix(checkpoint_dir, '.pth')
    for i, checkpoint_path in enumerate(checkpoint_paths):
        msg = f'Evaluating {i}/{len(checkpoint_paths)} checkpoint: {checkpoint_path}'
        print(msg)
        logger.warning(msg)
        model = agent_cta.get_model(checkpoint_path)
        msg = f'Stop epoch: {model.stop_epoch}, valid score: {model.valid_score}'
        print(msg) 
        logger.warning(msg)
        norm_score, average_step = valid_all(model, split = test_split)
        msg = f'Checkpoint: {checkpoint_path}, test score: {norm_score}, average_step: {average_step}'
        print(msg)
        logger.error(msg)