import pandas as pd


def results_for_drop_data(test_org, test):
    home = 0.42
    draw = 0.27
    away = 0.31

    test_drop = not_duplicate_elements_in_dataframes(test_org, test)

    results_drop = pd.DataFrame({'id': test_drop.index, 'home': home, 'draw': draw, 'away': away})

    return results_drop


def completed_test_result_and_save(test_org, result_learn, name_result):
    test_result_drop = results_for_drop_data(test_org, result_learn)
    test_result_all = append_test_results(result_learn, test_result_drop)
    test_result_all.to_csv('results/' + name_result, index=False)
    return test_result_all


def result_predict_prob_to_dataFrame(results_test, test):
    return pd.DataFrame(data=results_test, columns=['away', 'draw', 'home'], index=test.index).reset_index()


def append_test_results(results_test, results_dropna):
    return results_test.append(results_dropna, ignore_index=True)


def check_results_contain_all_test_id(results, test_org):
    print(results['id'].sort_values(by='id ') == test_org['id'].sort_values(by='id '))


def not_duplicate_elements_in_dataframes(test_org, test):
    test = test.reset_index()
    df_all = test_org.merge(test.drop_duplicates(), on=['id'],
                            how='left', indicator=True)

    test_not_duplicate = test_org[df_all['_merge'] == 'left_only']

    test_not_duplicate = test_not_duplicate.set_index('id')

    print(f'Number of not duplicate elements  {test_not_duplicate.shape[0]}')

    return test_not_duplicate
