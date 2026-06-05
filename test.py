from game_cmd_no_filter import default_game
from game import test_game
game = default_game()
from agent_cta import get_model

model = get_model('/home/zhuobinggang/research/ftwp-2026/checkpoints/cta_nav/roberta_navigator_20260601_125837_512488_best.pth')

test_game(game, model, need_print=True)
