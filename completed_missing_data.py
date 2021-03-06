import pandas as pd
import numpy as np


def completed_missing_data(train):
    train_org = train.copy()

    print('correction col type..')
    train = correction_col_type(train)
    check_prosense_nan_values(train, train_org)

    print('remove coach columns..')
    train = remove_coach_cols(train)
    check_prosense_nan_values(train, train_org)

    print('remove object with more that  col 80 empty...')
    train = train[train.isnull().sum(axis=1) < 80]
    check_prosense_nan_values(train, train_org)

    print('remove 9 and 10 day historii')
    train = remove_9_10_col_dates(train)
    check_prosense_nan_values(train, train_org)

    print('remove all nan data')
    train = train.dropna()
    check_prosense_nan_values(train, train_org)

    return train


def check_prosense_nan_values(data, data_org):
    print('percent of object with nan value: ',
          f'{(data_org.shape[0] - data.dropna().shape[0]) / data_org.shape[0] * 100.:0.2f}')


def correction_col_type(data):
    train = data
    date_columns = [column for column in train.columns.values if 'date' in column]
    train[date_columns] = train[date_columns].astype('datetime64')
    train['is_cup'] = train['is_cup'].astype('int', errors='ignore').fillna(0)
    return train


def remove_coach_cols(train):
    columns_without_coach_col = [column for column in train.columns if 'coach' not in column]
    train = train[columns_without_coach_col]
    return train


def remove_9_10_col_dates(train):
    columns_without_9_10_day = [x for x in train.columns.values if '9' not in x and '10' not in x]
    train = train[columns_without_9_10_day]
    return train


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


def remove_described_col(train):
    col_to_not_remove = [x for x in train.columns if "name" not in x and "id" not in x and x != 'match_date']
    train = train[col_to_not_remove]
    return train


def map_target(train):
    di = {'home': 1, 'draw': 2, 'away': 3}
    train = train.replace({"target": di})
    return train
