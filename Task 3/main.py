import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


amino_acids = np.array(['R', 'H' ,'K' ,'D' ,'E' ,'S' ,'T' ,'N' ,'Q' ,'C' ,'U' ,'G' ,'P' ,'A' ,'I' ,'L' ,'M' ,'F' ,'W' ,'Y' ,'V'])

train_data = pd.read_csv('./datasets/train.csv')
test_data = pd.read_csv('./datasets/test.csv')


def feature_processing(data):
    X = []
    enc = OneHotEncoder()
    enc.fit(amino_acids.reshape(-1, 1))
    for row in data.values:
        seq = np.array(list(row[0]))
        seq_onehot = enc.transform(seq.reshape(-1, 1)).toarray().ravel()
        X.append(seq_onehot)
    return X


X_train = feature_processing(train_data)
y_train = train_data['Active'].values
X_test = feature_processing(test_data)


clf = HistGradientBoostingClassifier(l2_regularization=1.0, learning_rate=0.15, max_iter=200, max_leaf_nodes=150, min_samples_leaf=50, scoring='f1')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

score = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1').mean()
print(score)

y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv('sub.csv', index=False, header=False)
