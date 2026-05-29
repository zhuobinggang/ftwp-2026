# 看看随机行动100步的agent能得到什么样的表现，作为一个baseline
# 再看看加上ucb1的结果
import common_new as common
import os
# from game_cmd_gen import Game_command_generate
from game import Game_with_navigator
import random
from tqdm import tqdm

MAX_STEPS = 100

def random_act(game: Game_with_navigator):
    _ = game.reset()
    counter = 0
    for i in range(MAX_STEPS):
        admissible_commands = game.get_admissible_commands()
        cmd = random.choice(admissible_commands)
        prev_moves = game.info['moves']
        # print(f'Randomly choose command: {cmd}')
        obs, reward, done, info = game.act(cmd)
        current_moves = game.info['moves']
        ACT, OBS = game.action_obs_pairs[-1]
        # print(f'{ACT} -> {OBS}')
        if reward != 0:
            # print(f'Got reward: {reward}')
            pass
        counter += max(1, current_moves - prev_moves)
        # rint(f'Observation: {obs}, Reward: {reward}, Done: {done}, Info: {info}')
        if done:
            # print('Game ended')
            # print(f'Last command: {cmd}')
            break
    game.info['our_moves'] = counter
    return game.info

def run():
    all_valid_game_paths = common.all_paths_with_suffix(os.path.join(common.GAME_BASE_PATH, 'valid'), '.z8')
    moves = []
    scores = []
    max_scores = []
    for game_path in tqdm(all_valid_game_paths, desc="Running random agent"):
        game = Game_with_navigator(game_path)
        info = random_act(game)
        moves.append(info['our_moves'])
        scores.append(info['score'])
        max_scores.append(info['max_score'])
    print(f"Norm score: {sum(scores)/sum(max_scores)}\nAverage moves: {sum(moves)/len(moves)}")

def run_test():
    all_valid_game_paths = common.all_paths_with_suffix(os.path.join(common.GAME_BASE_PATH, 'valid'), '.z8')
    game = Game_with_navigator(all_valid_game_paths[0])
    info = random_act(game)
    print(f"{info['score']} / {info['max_score']} in {info['our_moves']} moves")
    return info
