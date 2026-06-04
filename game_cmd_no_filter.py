# 不对指令进行额外的过滤
from game import Game_with_navigator
import common_new as common
    
class Game_with_navigator_no_cmd_filter(Game_with_navigator):
    def filter_cook_commands(self, cmds):
        return cmds
    def filter_take_commands(self, cmds):
        return cmds
    def try_add_take_commands(self, cmds):# 在库存满了的时候不能take，但是我们希望能继续生成用于负反馈
        return cmds
    def filter_prepare_meal_command(self, cmds):
        return cmds
    def filter_examine_cookbook_command(self, cmds):
        return cmds

    
def default_game():
    return Game_with_navigator_no_cmd_filter(f'{common.GAME_BASE_PATH}/valid/tw-cooking-recipe1+cook+cut+drop+go6-M2qEFeOXcol3H1ql.ulx')

