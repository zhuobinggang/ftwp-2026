from common_new import *
import common_new as common
from agent_cta import *

assert common.GAME_WITH_NAVIGATOR
assert GAME_INIT_FUNC == Game_with_navigator

testing = True if common.args.test else False
print(f'Testing mode: {testing}')

train_repeat(testing=testing)
test_all_checkpoints(testing=testing)