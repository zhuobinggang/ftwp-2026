from agent_cta import *

assert common.GAME_WITH_NAVIGATOR
assert GAME_INIT_FUNC == Game_with_navigator

train_repeat(testing=False)
test_all_checkpoints(testing=False)