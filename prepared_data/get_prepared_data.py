from prepared_data.adding_new_features import adding_new_features
from prepared_data.completed_missing_data import completed_missing_data, convert_historical_date_to_date_difference, \
    map_target, remove_described_col_and_set_index_id


def get_prepared_data(data_org, number_of_history_matches =8):
    data = completed_missing_data(data_org, number_of_history_matches)
    data = convert_historical_date_to_date_difference(data)
    data = map_target(data)
    data = adding_new_features(data, number_of_history_matches)
    data = remove_described_col_and_set_index_id(data)
    return data


