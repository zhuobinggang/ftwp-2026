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
    train_games = get_cv_games(split = 'train')
    trains = train_games[:900]
    tests = train_games[900:1000]
    return extract_walkthrough_dataset(trains), extract_walkthrough_dataset(tests)
