from sklearn.ensemble import RandomForestClassifier

from prepared_data.get_prepared_data import get_prepared_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from prepared_data.prepared_test_results import result_predict_prob_to_dataFrame, not_duplicate_elements_in_dataframes
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

train_org = pd.read_csv('data/train.csv')
train, league_mean_ratting_all, teams_ratting_all = get_prepared_data(train_org)
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.33,
                                                    random_state=42)

pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, max_depth=5))

pipe.fit(X_train, y_train)
results = pipe.predict(X_train)
print(f'accuracy: {pipe.score(X_test, y_test):.04f}')
# %%
