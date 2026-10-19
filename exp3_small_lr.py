"""
[:900]训练集合在完整测试集上的结果:
Average steps: 27.955252918287936
Norm score: 0.35487731897067626
"""

from exp3 import *

def example_train_test():
    df_trains, df_tests = get_trains_and_tests()
    model, tokenizer = train(df_trains, negative_size=5, lr=1e-5)
    data_tests = prompt_label_dic_generate(df_tests, negative_size=5)
    y_true, y_pred = run_test(model, tokenizer, data_tests)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    save_model(model, tokenizer, save_directory="./temp/model_4440_games_small_lr")


def example_run_test():
    model = Model_ucb1()
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


if __name__ == "__main__":
    example_train_test()
    example_run_test()