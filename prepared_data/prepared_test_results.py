import pandas as pd


def results_for_drop_data(test_org, test_with_id):
    home = 0.42
    draw = 0.27
    away = 0.31
    df_all = test_org.merge(test_with_id.drop_duplicates(), on=['id'],
                            how='left', indicator=True)

    test_drop = test_org[df_all['_merge'] == 'left_only']

    print(f'Complementing the result for {test_drop.shape[0]}  (home:1,draw:0,away:0)')
    results_drop = pd.DataFrame({'id': test_drop['id'].values, 'home': home, 'draw': draw, 'away': away})

    return results_drop


def result_predict_prob_to_dataFrame(results_test, test):
    return pd.DataFrame(data=results_test, columns=['away', 'draw', 'home'], index=test.index).reset_index()


def append_test_result_dropna_result(results_test, results_dropna):
    return results_test.append(results_dropna, ignore_index=True)


def check_results_contain_all_test_id(results, test_org):
    print(results['id'].sort_values(by='id ') == test_org['id'].sort_values(by='id '))
