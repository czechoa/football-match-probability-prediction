import pandas as pd

from prepared_data.completed_missing_data import completed_missing_data, convert_historical_date_to_date_difference, \
    remove_described_col_and_set_index_id


def results_for_drop_data(test_org, test_with_id):
    home = 0.42
    draw =  0.27
    away = 0.31
    df_all = test_org.merge(test_with_id.drop_duplicates(), on=['id'],
                            how='left', indicator=True)

    test_drop = test_org[df_all['_merge'] == 'left_only']

    print(f'Complementing the result for {test_drop.shape[0]}  (home:1,draw:0,away:0)')
    results_drop = pd.DataFrame({'id': test_drop['id'].values, 'home': home, 'draw': draw, 'away': away})

    return results_drop



