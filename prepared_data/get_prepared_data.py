from prepared_data.adding_new_features import adding_new_features
from prepared_data.completed_missing_data import completed_missing_data, convert_historical_date_to_date_difference, \
    map_target, remove_described_col_and_set_index_id


def get_prepared_data(data_org, number_of_history_matches=8, league_mean_ratting_all=None, teams_mean_ratting_all=None, map_target=True ):
    data = completed_data_convert_type(data_org, number_of_history_matches,map_target)

    data, league_mean_ratting_all, teams_mean_ratting_all = adding_new_features(data, number_of_history_matches,
                                                                                league_mean_ratting_all,
                                                                                teams_mean_ratting_all)

    data = remove_described_col_and_set_index_id(data)
    return data, league_mean_ratting_all, teams_mean_ratting_all


def completed_data_convert_type(data_org, number_of_history_matches, map_target):
    data = completed_missing_data(data_org, number_of_history_matches)
    data = convert_historical_date_to_date_difference(data)
    if map_target:
        data = map_target(data)

    return data
