from agent_navigator import *
from game import test_game
import common_new as common

m = get_model('checkpoints/ftwp_our_navigator_only/roberta_navigator_0_epoch_1.pth', Model_ucb1)
game = Game_with_navigator(common.GAME_BASE_PATH + '/fake_test_10/tw-cooking-recipe2+take2+cook+go12-EKP9iMylTo8yTxWj.ulx')

