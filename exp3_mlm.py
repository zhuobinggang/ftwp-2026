"""
[:900]训练集合在完整测试集上的结果:
Average steps: 27.955252918287936
Norm score: 0.35487731897067626
"""

from exp2_mlm import *

def get_trains_and_tests():
    cache_dir = "temp"
    train_cache_path = os.path.join(cache_dir, "trains_all.feather")
    test_cache_path = os.path.join(cache_dir, "tests_all.feather")

    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print(f"Loading cached datasets from {train_cache_path} and {test_cache_path}...")
        df_trains = pd.read_feather(train_cache_path)
        df_tests = pd.read_feather(test_cache_path)
        return df_trains, df_tests

    print("Cache not found. Extracting walkthrough datasets (this may take 3-5 mins)...")
    trains = get_cv_games(split='train')
    tests = get_cv_games(split='test')

    df_trains = extract_walkthrough_dataset(trains)
    df_tests = extract_walkthrough_dataset(tests)

    # 3. 自动创建目录并保存
    os.makedirs(cache_dir, exist_ok=True)
    
    # Feather 要求索引必须是默认的连续整数 RangeIndex
    # reset_index 确保即使提取的数据包含非默认索引也不会抛错
    df_trains.reset_index(drop=True).to_feather(train_cache_path)
    df_tests.reset_index(drop=True).to_feather(test_cache_path)
    print(f"Datasets cached to {train_cache_path} and {test_cache_path}.")

    return df_trains, df_tests

def example_train_test():
    df_trains, df_tests = get_trains_and_tests()
    model, tokenizer = train(df_trains, negative_size=5)
    data_tests = prompt_label_dic_generate(df_tests, negative_size=5)
    y_true, y_pred = run_test(model, tokenizer, data_tests)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    save_model(model, tokenizer, save_directory="./temp/model_4440_games_mlm")


def example_run_test():
    model = Model_ucb1()
    model.init_bert(model_folder="./temp/model_4440_games_mlm")
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