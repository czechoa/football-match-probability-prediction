import numpy as np
from pandas.api.types import is_numeric_dtype


def completed_missing_data(data, number_of_history_matches):
    data_org = data.copy()

    # check_prosense_nan_values(data, data_org)

    # print('remove history columns',list(range(number_of_history_matches+1,11)))

    data = remove_history_col(data, number_of_history_matches)
    # check_prosense_nan_values(data, data_org)

    # print('remove object with more that  col 50 % empty...')
    data = data[data.isnull().sum(axis=1) < data.shape[0] / 2]
    # check_prosense_nan_values(data, data_org)

    # print('date to datetime type and remove nan date....')
    data = date_col_to_datetime_type(data)
    # check_prosense_nan_values(data, data_org)

    # print('filna coach_id and change to is new coach')
    # data = remove_coach_cols(data)
    data = fillna_with_zero_coach_cols(data)
    data = change_id_coach_to_is_change_coach(data)
    # check_prosense_nan_values(data, data_org)

    # print('fill zero nan data')
    data = data.fillna(0)
    data['is_cup'] = data['is_cup'].astype('int', errors='ignore')

    check_prosense_nan_values(data, data_org)

    return data


def check_prosense_nan_values(data, data_org):
    print('percent of object with nan value and orginals: ',
          f'{(data.shape[0] - data.dropna().shape[0]) / data.shape[0] * 100.:0.2f}, {data.shape[0] / data_org.shape[0] * 100.:02f}')


def date_col_to_datetime_type(data):
    date_columns = [column for column in data.columns.values if 'date' in column]

    data[date_columns] = data[date_columns].astype('datetime64')
    data = data[(data[date_columns].isna().sum(axis=1) < 1)]
    return data


def remove_coach_cols(train):
    columns_without_coach_col = [column for column in train.columns if 'coach' not in column]
    train = train[columns_without_coach_col]
    return train


def change_id_coach_to_is_change_coach(data):
    coach_col = [x for x in data.columns if 'coach' in x]
    home_coach_col = [x for x in coach_col if 'home' in x]
    away_coach_col = [x for x in coach_col if 'away' in x]

    for i in range(len(home_coach_col) - 1):
        data[home_coach_col[i]] = (data[home_coach_col[i]] != data[home_coach_col[i + 1]]) * 1
        data[away_coach_col[i]] = (data[away_coach_col[i]] != data[away_coach_col[i + 1]]) * 1

    data = data.drop(columns=[home_coach_col[-1], away_coach_col[-1]])
    return data


def fillna_with_zero_coach_cols(train):
    coach_col = [column for column in train.columns if 'coach' in column]
    train[coach_col] = train[coach_col].fillna(0)
    return train


def remove_history_col(train, number_of_history_matches):
    number_history_col = list(map(str, range(1, number_of_history_matches + 1)))

    current_match_columns = [column for column in train.columns.values if 'history' not in column]
    columns_history = [x for x in train.columns for number in number_history_col if number in x and '0' not in x]
    return train[current_match_columns + columns_history]


def convert_historical_date_to_date_difference(train):
    home_date_columns = [column for column in train.columns.values if
                         ('date' in column and 'home' in column) or column == 'match_date']
    away_date_columns = [column for column in train.columns.values if
                         ('date' in column and 'away' in column) or column == 'match_date']
    # train[date_columns]
    train[home_date_columns[1:]] = train[home_date_columns].diff(periods=1, axis=1).apply(
        lambda x: x / np.timedelta64(-1, 'h')).iloc[:, 1:].astype('int64')
    train[away_date_columns[1:]] = train[away_date_columns].diff(periods=1, axis=1).apply(
        lambda x: x / np.timedelta64(-1, 'h')).iloc[:, 1:].astype('int64')

    return train


def remove_described_col_and_set_index_id(train):
    train = train.set_index('id')

    col_to_not_remove = [x for x in train.columns if
                         ('league_id_ratting' in x) or (is_numeric_dtype(train[x]) and 'id' not in x) or (
                                 x == 'target')]

    train = train[col_to_not_remove]
    return train


def map_target(train):
    di = {'home': 1, 'draw': 0, 'away': -1}
    train = train.replace({"target": di})
    return train
