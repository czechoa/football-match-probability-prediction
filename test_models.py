import numpy as np
from sklearn.ensemble import RandomForestClassifier

from prepared_data.get_prepared_data import get_prepared_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

train_org = pd.read_csv('data/train.csv')
train, league_mean_ratting_all, teams_ratting_all = get_prepared_data(train_org)

pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=1000, n_jobs=-1))
scores = cross_val_score(pipe,  train.iloc[:, 1:], train.iloc[:, 0] , cv=5)
print( np.median(scores))

