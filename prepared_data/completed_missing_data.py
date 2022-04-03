import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


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


def completed_missing_data_for_test_drop(test):
    test_org = test.copy()

    print('correction col type..')
    test = correction_col_type(test)
    check_prosense_nan_values(test, test_org)

    print('remove coach columns..')
    test = remove_coach_cols(test)
    check_prosense_nan_values(test, test_org)

    print('remove object with more that  col 80 empty...')
    test = test[test.isnull().sum(axis=1) < 80]
    check_prosense_nan_values(test, test_org)

    print('remove all day historii without 1')
    test = remove_all_col_dates_only_stayed_1(test)
    check_prosense_nan_values(test, test_org)

    print('remove all nan data')
    test = test.dropna()
    check_prosense_nan_values(test, test_org)

    return test

def check_prosense_nan_values(data, data_org):
    print('percent of object with nan value: ',
          f'{(data_org.shape[0] - data.dropna().shape[0]) / data_org.shape[0] * 100.:0.2f}')


def correction_col_type(data):
    train = data
    date_columns = [column for column in train.columns.values if 'date' in column]
    train[date_columns] = train[date_columns].astype('datetime64')
    train['is_cup'] = train['is_cup'].fillna(0).astype('int', errors='ignore')
    return train


def remove_coach_cols(train):
    columns_without_coach_col = [column for column in train.columns if 'coach' not in column]
    train = train[columns_without_coach_col]
    return train


def remove_9_10_col_dates(train):
    columns_without_9_10_day = [x for x in train.columns.values if '9' not in x and '10' not in x]
    train = train[columns_without_9_10_day]
    return train


def remove_all_col_dates_only_stayed_1(train):
    used = set()
    columns_only_first_hist = [column for column in train.columns.values if
                               re.sub('\d', '', column) not in used and (used.add(re.sub('\d', '', column)) or True)]

    return train[columns_only_first_hist]


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
                         ('league_id_ratting' in x) or (is_numeric_dtype(train[x]) and 'id' not in x)]

    train = train[col_to_not_remove]
    return train


def map_target(train):
    di = {'home': 1, 'draw': 0, 'away': -1}
    train = train.replace({"target": di})
    return train
