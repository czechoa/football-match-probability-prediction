import pandas as pd
import numpy as np


def team_regeneration(data_f, team_h_or_a, index, history_matches_amount):
    i_str = str(index)
    if index == 1:
        data_f[team_h_or_a + '_team_mean_regeneration_time'] = data_f[
                                                                   team_h_or_a + '_team_history_match_date_' + i_str] / history_matches_amount
    else:
        data_f[team_h_or_a + '_team_mean_regeneration_time'] += data_f[
                                                                    team_h_or_a + '_team_history_match_date_' + i_str] / history_matches_amount
    return data_f


def history_target(data_f, team_h_or_a, index):
    i_str = str(index)
    team_history_goal = team_h_or_a + '_team_history_goal_' + i_str

    team_history_opponent_goal = team_h_or_a + '_team_history_opponent_goal_' + i_str

    data_f[team_h_or_a + '_team_history_target_' + i_str] = np.sign(
        data_f[team_history_goal] - data_f[team_history_opponent_goal])
    return data_f


def league_team_mean_ratting(data_f, team_h_or_a, index):
    i_str = str(index)
    last_match_columns = [x for x in data_f.columns if i_str in x]
    rating_col_name = team_h_or_a + '_team_history_league_id_' + i_str

    league_mean_ratting = data_f[last_match_columns].groupby(by=rating_col_name).mean()
    league_mean_ratting['mean_ratting_' + team_h_or_a + '_' + i_str] = (league_mean_ratting[
                                                                            team_h_or_a + '_team_history_rating_' + i_str]
                                                                        + league_mean_ratting[
                                                                            team_h_or_a + '_team_history_opponent_rating_' + i_str]) / 2

    return league_mean_ratting['mean_ratting_' + team_h_or_a + '_' + i_str]


def teams_mean_ratting(data_f, team_h_or_a, index):
    i_str = str(index)
    team_col_name = team_h_or_a + '_team_name'

    team_mean_ratting = data_f.groupby(by=team_col_name).mean()

    return team_mean_ratting[team_h_or_a + '_team_history_rating_' + i_str]


def adding_new_features(train, number_of_history_matches=8):
    league_mean_ratting_all = pd.DataFrame()
    teams_mean_ratting_all = pd.DataFrame()

    for i in range(1, number_of_history_matches + 1):
        for home_or_away in ['home', 'away']:
            league_mean_ratting_all = league_mean_ratting_all.append(league_team_mean_ratting(train, home_or_away, i))
            teams_mean_ratting_all = teams_mean_ratting_all.append(teams_mean_ratting(train, home_or_away, i))
            train = history_target(train, home_or_away, i)
            train = team_regeneration(train, home_or_away, i, number_of_history_matches)
            if i == number_of_history_matches - 1:
                team_name_col = home_or_away + '_team_name'
                teams_ratting = teams_mean_ratting_all.mean().reset_index().rename(
                    columns={'index': team_name_col, 0: home_or_away + '_team_mean_ratting'})
                train = train.merge(teams_ratting, on=team_name_col)

    league_mean_ratting_all = league_mean_ratting_all.mean().reset_index().rename(
        columns={'index': 'league_id', 0: 'league_id_ratting'})

    train = train.merge(league_mean_ratting_all, on='league_id').sort_values(by='league_id')

    return train
