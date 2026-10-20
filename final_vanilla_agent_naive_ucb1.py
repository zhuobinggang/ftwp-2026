# Naive UCB1: 只要执行过的指令就不再执行

from final_vanilla_agent import *

DANGER_FILTER_ON = False
PRINT_CMD_SCORE = False
class Model_ucb1_naive(Model):
    def __init__(self):
        super().__init__()
        self.reset_state_action_count()
    def incresase_state_action_count(self, key, action):
        self.state_action_count[key][action] += 1
    def get_state_action_count(self, key, action):
        if key not in self.state_action_count:
            self.state_action_count[key] = {}
        if action not in self.state_action_count[key]:
            self.state_action_count[key][action] = 0
        return self.state_action_count[key][action]
    def reset_state_action_count(self, room_name = ''):
        self.state_action_count = {} # 记录每个状态下的动作选择次数
    def reset_state_action_count_if_need(self, game_state: Game_state_clean):
        action_obs_pairs = game_state.action_obs_pairs
        if len(action_obs_pairs) == 0:
            self.reset_state_action_count(game_state.room)
        self.current_room = game_state.room # 总是要更新当前房间，但是在更新之前需要先更新世界地图（如果有必要）
    def predict(self, game_state: Game_state_clean):
        # NOTE: 更新世界地图(根据上一步动作的结果)，只要发生移动必须对链接进行更新
        all_actions = game_state.get_admissible_commands() # NOTE: 必须确保get_admissible_commands返回的动作顺序和之前的动作顺序一致
        if len(all_actions) == 0:
            return ''
        self.reset_state_action_count_if_need(game_state)
        cmd_scores = predict_by_game_state(self.bert, self.tokenizer, game_state)
        # TODO: 打印
        if PRINT_CMD_SCORE:
            print_cmd_scores(cmd_scores, game_state)
        # 如果是执行过的指令就直接排除
        state_key = game_state_to_ucb1_key(game_state)
        blocked_action_count = 0
        for idx, action in enumerate(all_actions):
            executed_count = self.get_state_action_count(state_key, action)
            if executed_count > 0:
                cmd_scores[idx] = -float('inf')
                blocked_action_count += 1
        if blocked_action_count == len(all_actions):
            print('已经执行过所有的动作，无法选择新的动作，直接退出')
            return ''
        best_action_idx, action_prob = -1, 0.0
        for idx, score in enumerate(cmd_scores):
            if score > action_prob:
                best_action_idx = idx
                action_prob = score
        assert best_action_idx >= 0, f'No valid action found for game state: {game_state}'
        best_action = all_actions[best_action_idx]
        self.incresase_state_action_count(state_key, best_action)
        assert all_actions == game_state.get_admissible_commands()
        return best_action


def example_run_test():
    model = Model_ucb1_naive()
    model.init_bert(model_folder="./temp/model_4440_games_small_lr")
    tests = get_cv_games(split='test')
    steps = []
    scores = []
    max_scores = []
    for game_path in tests:
        game = Game_handle_worldmap(game_path)
        result = test_game(game, model, max_step=100, need_print=False)
        steps.append(result.step)
        scores.append(result.score)
        max_scores.append(result.max_score)
    print(f'Average steps: {sum(steps) / len(steps)}')
    print(f'Norm score: {sum(scores) / sum(max_scores)}')