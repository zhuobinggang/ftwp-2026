from common_new import *
import common_new as common
from agent_cta import *

assert common.GAME_WITH_NAVIGATOR
assert common.GAME_RAW_COMMANDS
assert GAME_INIT_FUNC == Game_with_navigator_no_cmd_filter

testing = True if common.args.test else False
print(f'Testing mode: {testing}')

train_repeat(testing=testing)
test_all_checkpoints(testing=testing)