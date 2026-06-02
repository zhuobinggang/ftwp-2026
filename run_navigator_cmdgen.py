from common_new import *
import common_new as common
from agent_cta import *

assert common.GAME_WITH_NAVIGATOR
assert common.GAME_CMD_GENERATE
assert GAME_INIT_FUNC == Game_command_generate_nav
assert TEMP_SAVE_DIR == "./checkpoints/cta_nav_cmdgen"

testing = True if common.args.test else False
print(f'Testing mode: {testing}')

train_repeat(testing=testing)
test_all_checkpoints(testing=testing)