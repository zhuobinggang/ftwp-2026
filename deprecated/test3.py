from dataset_create import get_cv_games
import game_cmd_gen
import game
from common_new import compare_lists_ignore_order
import common_new as common

# game_path =  '/home/zhuobinggang/research/datasets/ftwp/games/fake_valid_10/tw-cooking-recipe2+cook+cut+open+go12-Oxbqh6QbtggnFovy.ulx'
# game_path =  '/home/zhuobinggang/research/datasets/ftwp/games/fake_valid_10/tw-cooking-recipe2+cook+cut+open+drop+go12-ob72HGxah9L2f8PN.ulx'


START_WORDS = ['cook ', 'chop ', 'dice ', 'slice ']

def check_command_cool(game1, game2, walkthrough):
    for step, command in enumerate(walkthrough):
        # print(f'step {step}, command: {command}')
        admissible_commands1 = game1.get_admissible_commands()
        admissible_commands2 = game2.get_admissible_commands()
        if not compare_lists_ignore_order(admissible_commands1, admissible_commands2):
            common.print_list_differences(admissible_commands1, admissible_commands2)
            set1 = set(admissible_commands1)
            set2 = set(admissible_commands2)
            only_in_set1 = set1 - set2
            all_cook_commands = True
            for cmd in only_in_set1:
                if not any(cmd.startswith(start_word) for start_word in START_WORDS):
                    all_cook_commands = False
                    break
            if not all_cook_commands:
                raise ValueError(f'Admissible commands do not match for step {step}, command {command}, game1 admissible commands: {admissible_commands1}, game2 admissible commands: {admissible_commands2}')
        _ = game1.act(command)
        _ = game2.act(command)


# 对所有valid_10的游戏进行测试
def run():
    game_paths = get_cv_games(path= '/home/zhuobinggang/research/datasets/ftwp/games', split = 'fake_train_100')
    for game_path in game_paths:
        game1 = game_cmd_gen.Game_command_generate(game_path)
        game2 = game.Game_with_navigator(game_path)
        _ = game1.reset()
        _ = game2.reset()
        walkthrough = game1.clean_walkthrough()
        # print('walkthrough:', walkthrough)
        check_command_cool(game1, game2, walkthrough)