# vanilla CogniTextworldAgent

from agent_cta import *

assert not common.GAME_WITH_NAVIGATOR
assert GAME_INIT_FUNC == Game_handle_worldmap

def run():
    train_repeat(testing = True)