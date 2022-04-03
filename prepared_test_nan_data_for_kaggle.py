import pandas as pd

from prepared_data.completed_missing_data import completed_missing_data, convert_historical_date_to_date_difference, \
    remove_described_col


def return_results_for_drop_data(test_org, test_with_id):
    df_all = test_org.merge(test_with_id.drop_duplicates(), on=['id'],
                            how='left', indicator=True)

    test_drop = test_org[df_all['_merge'] == 'left_only']

    print(f'Complementing the result for {test_drop.shape[0]}  (home:1,draw:0,away:0)')
    results_drop = pd.DataFrame({'id': test_drop['id'].values, 'home': 1, 'draw': 0, 'away': 0})
    return results_drop


# %%


columns_without_coach_col = [column for column in test_drop.columns if 'coach' not in column]
test_drop = test_drop[columns_without_coach_col]

prosense_nan_value = pd.DataFrame(test_drop.isna().sum() / test_org.shape[0], columns=['Presence of nan value'])
prosense_nan_value = prosense_nan_value[prosense_nan_value['Presence of nan value'] > 0].sort_values(
    by='Presence of nan value',
    ascending=True)

# procent 0.00696 don;t have 1 history columns
# %%
from prepared_data.completed_missing_data import completed_missing_data_for_test_drop
import pandas as pd
test_org = pd.read_csv('data/test.csv')
test = completed_missing_data_for_test_drop(test_org)
