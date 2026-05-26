# 读取game文件，然后录入csv文件

import os
import re
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from game import Game_with_navigator
import common_new as common
from common_new import COMMAND_LIST_SHUFFLE
import logging
logger = logging.getLogger('dataset_create_taku')
dbg = logger.debug
import random

BASE_PATH = common.GAME_BASE_PATH
TRAIN_PATH = f'{BASE_PATH}/train'
TEST_PATH = f'{BASE_PATH}/test'
VALID_PATH = f'{BASE_PATH}/valid'

GAME_INIT_FUNC = Game_with_navigator

@lru_cache(maxsize=None)
def all_game_paths(test_path = TEST_PATH):
    import os
    results = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(test_path):
        if filename.endswith('.z8'):  # 只处理 .z8 文件
            file_path = os.path.join(test_path, filename)
            results.append(file_path)
    return results

def get_cv_games(path = BASE_PATH, split = 'fake_test'):
    return all_game_paths(f'{path}/{split}')


# 2.`get_clean_clean_walkthrough`函数会先跑一次游戏，然后确保游戏能正常结束。
# 3. 同上函数，我们会先将`take apple from fridge`替换成`take apple`
# 4. 同上函数，如果是admissable actions中不存在的指令，我们也跳过
def get_clean_clean_walkthrough(game_path):
    assert GAME_INIT_FUNC == Game_with_navigator
    game = GAME_INIT_FUNC(game_path)
    game.reset()
    clean_walkthrough = game.clean_walkthrough()
    clean_clean_walkthrough = []
    for cmd in clean_walkthrough:
        if game.done:
            break
        # take apple from fridge -> take apple
        if cmd.startswith('take'):
            cmd = re.sub(r'\sfrom.*$', '', cmd)
        admissible_commands = game.get_admissible_commands()
        if cmd not in admissible_commands: # 可能存在admissible_commands中没有的指令，比如eat meal是
            if cmd == 'eat meal':
                logger.debug(f'Eat meal not in admissible_commands, I will add it to make sure the game can done.')
                clean_clean_walkthrough.append(cmd)
                game.act(cmd)
            else:
                # print(f'Command {cmd} not in {admissible_commands}, I will skip it.')
                # logger.warning(f'Command {cmd} not in {admissible_commands}, I will skip it.') # NOTE：大量存在，主要是open指令会被反复执行
                pass
        else: # 如果指令在admissible_commands中，则正常执行
            clean_clean_walkthrough.append(cmd)
            game.act(cmd)
    assert game.done
    return clean_clean_walkthrough

# TODO: 2026.5.19 需要测试
def extract_walkthrough_dataset_with_navigator(split = 'fake_test', test_game_path = ''):
    assert GAME_INIT_FUNC == Game_with_navigator, 'We need to use Game_with_navigator to generate the dataset.'
    if test_game_path:
        train_games = [test_game_path]
    else:
        train_games = get_cv_games(split = split)
    gamesteps = []
    for game_path in tqdm(train_games):
        clean_walkthrough = get_clean_clean_walkthrough(game_path)
        if test_game_path:
            print(f'clean walkthrough: {clean_walkthrough}')
        cmd_index = 0
        game = GAME_INIT_FUNC(game_path)
        game.reset()
        assert game.action_obs_pairs == [], f'At the beginning of the game, action_obs_pairs should be empty, but got {game.action_obs_pairs}'
        while cmd_index < len(clean_walkthrough):
            cmd = clean_walkthrough[cmd_index]
            if game.done:
                break
            admissible_commands = game.get_admissible_commands().copy() # NOTE: 这里必须copy，因为后面可能会修改admissible_commands
            if cmd == 'eat meal': # 修正admissible_commands
                # logger.debug(f'Eat meal not in admissible_commands, I will add it to make sure the game can done.')
                admissible_commands.append(cmd)
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
            need_no_execute = False
            need_no_append_gamesteps = False
            if cmd.startswith('go'): # 导航到物品
                next_non_go_index = cmd_index + 1
                while next_non_go_index < len(clean_walkthrough) and clean_walkthrough[next_non_go_index].startswith('go'):
                    next_non_go_index += 1
                non_go_command = clean_walkthrough[next_non_go_index]
                entity = None
                if non_go_command.startswith('take'):
                    entity = non_go_command.split()[1]
                elif non_go_command.startswith('cook'):
                    entity = non_go_command.split()[-1]
                if entity and entity in game.itemMap: # 可以导航到物品
                    need_no_execute = True # 导航在这个代码快中执行
                    prev_room = game.room
                    navigate_action = f'navigate to {entity}'
                    clean_navigate_commands = []
                    if navigate_action not in admissible_commands: # 可能出现循环导航，可以直接省略
                        if game.room == game.itemMap[entity]['room']: # BUG: 本来应该进这里的却没有
                            logger.warning(f'循环导航，省略当前指令，不需要置入数据集')
                            need_no_append_gamesteps = True
                        else:
                            raise ValueError(f'Command {navigate_action} not in {admissible_commands}, current room {game.room}, game path {game_path}, itemMap {game.itemMap}')
                    else:
                        logger.warning(f'导航指令生成并使用：{navigate_action}')
                        print(f'导航指令生成并使用：{navigate_action}')
                        game_step['action'] = navigate_action # 覆盖动作
                        clean_navigate_commands = game.navigate_to_item(entity)
                        for command in clean_navigate_commands:
                            game.act(command)
                    cmd = non_go_command
                    cmd_index = next_non_go_index
            if not need_no_append_gamesteps:
                gamesteps.append(game_step)
            if not need_no_execute:
                game.act(cmd) # 执行的可能是导航后的下一个指令
                cmd_index += 1
        assert game.done
    return pd.DataFrame(gamesteps)


def create_csv_dataset(outputpath = 'good_dataset', suffix = '_with_navigator', testing = False):
    dataset_names = ['fake_train_100','fake_valid_10', 'fake_test_10'] if testing else ['train', 'valid', 'test']
    for split in dataset_names:
        df = extract_walkthrough_dataset_with_navigator(split)
        df.to_csv(os.path.join(outputpath,
                        f'walkthrough_{split}{suffix}.csv'), index=False)
        

def read_csv_dataset(inputpath = 'good_dataset', split = 'fake_test', suffix = '_with_navigator'):
    path = os.path.join(inputpath,
                        f'walkthrough_{split}{suffix}.csv')
    print(path)
    df= pd.read_csv(path)
    df['action_obs_pairs'] = df['action_obs_pairs'].apply(eval)
    df['admissible_commands'] = df['admissible_commands'].apply(eval)
    df['recipe'] = df['recipe'].fillna('')
    df['inventory'] = df['inventory'].fillna('')
    return df


def run():
    random.seed(2026)
    create_csv_dataset(testing = True)
    create_csv_dataset(testing = False)