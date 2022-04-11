import pandas as pd

from prepared_data.get_prepared_data import get_prepared_data

train_org = pd.read_csv('data/train.csv')
# train,_,_ = get_prepared_data(train_org,number_of_history_matches=6)

home_columns = [x for x in train_org.columns if 'home' == x[:4]]
away_columns = [x for x in train_org.columns if 'away' == x[:4]]
not_home_away = [x for x in train_org.columns if 'away' != x[:4] and 'home' != x[:4]]

home_team_matches = train_org[not_home_away + home_columns]
away_team_matches = train_org[not_home_away + away_columns]
columns_home_dict = {x:x[5:] for x in home_columns}
columns_away_dict = {x:x[5:] for x in away_columns}


home_team_matches = home_team_matches.rename(columns=columns_home_dict)
away_team_matches = away_team_matches.rename(columns=columns_away_dict)
# %%
reshaped_train = home_team_matches.append(away_team_matches,ignore_index=True)
