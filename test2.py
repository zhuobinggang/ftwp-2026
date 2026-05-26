# 检查game转换成的prompt和row转换成的promt是否一致
from agent_navigator import *

game = Game_with_navigator(common.GAME_BASE_PATH + '/fake_valid_10/tw-cooking-recipe2+cook+cut+go12-d1aLHkNpfEYmSRXj.ulx')
csv = read_csv_dataset(inputpath = 'good_dataset', split = 'fake_valid_10')
csv_row = csv.iloc[7]
print(f'prev command: {csv_row["action_obs_pairs"][-1][0]}')
print(f'next command: {csv_row["action"]}')
row_state = row_to_game_state(csv_row)

_ = game.reset()
walkthrough = game.clean_walkthrough()
for command in walkthrough:
    if command == csv_row['action']:
        break
    if command in game.get_admissible_commands():
        _ = game.act(command)

# assert game.inventory_clean() == row_state.inventory_clean() # NOTE: 顺序不一样……
assert game.description_clean() == row_state.description_clean()
assert game.room == row_state.room
assert game.recipe_clean() == row_state.recipe_clean()
assert game.clean_action_obs_pairs() == row_state.clean_action_obs_pairs()
assert common.compare_lists_ignore_order(game.get_admissible_commands(), row_state.get_admissible_commands())
assert game.get_admissible_commands() == game.get_admissible_commands(), f'确保同一个状态下多次调用get_admissible_commands结果一致，避免因为随机打乱导致的测试不稳定'
