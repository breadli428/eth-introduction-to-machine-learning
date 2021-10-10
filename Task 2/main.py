import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

train_data = pd.read_csv('./datasets/train_features.csv')
labels = pd.read_csv('./datasets/train_labels.csv')
test_data = pd.read_csv('./datasets/test_features.csv')

def features_engineering(data, n):
    x = []
    features = [np.nanmedian, np.nanmean, np.nanvar, np.nanmin, np.nanmax]
    for index in range(int(data.shape[0] / n)):
        patient_data = data[n * index: n * (index + 1), 2:]
        feature_values = np.empty((len(features), data[:, 2:].shape[1]))
        for i, feature in enumerate(features):
            feature_values[i] = feature(patient_data, axis=0)
        x.append(feature_values.ravel())
    return np.array(x)

x_train = features_engineering(train_data.to_numpy(), 12)
x_test = features_engineering(test_data.to_numpy(), 12)

# Task 1

task1_labels = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
y_train = labels[task1_labels].to_numpy()

pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), HistGradientBoostingClassifier(l2_regularization=1.0))

# for i, label in enumerate(task1_labels):
#     scores = cross_val_score(pipeline, x_train, y_train[:, i], cv=5, scoring='roc_auc', verbose=True)
#     print("Cross-validation score is {score:.3f},"
#           " standard deviation is {err:.3f}"
#           .format(score = scores.mean(), err = scores.std()))


df = pd.DataFrame({'pid': test_data.iloc[0::12, 0].values})

for i, label in enumerate(task1_labels):
    pipeline.fit(x_train, y_train[:, i].ravel())
    # print("Training score:", metrics.roc_auc_score(y_train[:, i], pipeline.predict_proba(x_train)[:, 1]))
    predictions = pipeline.predict_proba(x_test)[:, 1]
    df[label] = predictions


# Task 2

task2_labels = ['LABEL_Sepsis']
y_train = labels[task2_labels].to_numpy().ravel()

pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), HistGradientBoostingClassifier(l2_regularization=1.0))

# scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring='roc_auc', verbose=True)
# print("Cross-validation score is {score:.3f},"
#       " standard deviation is {err:.3f}"
#       .format(score = scores.mean(), err = scores.std()))


pipeline.fit(x_train, y_train)
predictions = pipeline.predict_proba(x_test)[:, 1]
# print("Training score:", metrics.roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1]))
df[task2_labels[0]] = predictions


# Task 3

task3_labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
y_train = labels[task3_labels].to_numpy()

pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), HistGradientBoostingRegressor(max_depth=3))

# for i, label in enumerate(task3_labels):
#     scores = cross_val_score(pipeline, x_train, y_train[:, i],
#                             cv=5,
#                             scoring='r2',
#                             verbose=True)
#     print("Cross-validation score is {score:.3f},"
#           " standard deviation is {err:.3f}"
#           .format(score = scores.mean(), err = scores.std()))


for i, label in enumerate(task3_labels):
    pipeline.fit(x_train, y_train[:, i])
    predictions = pipeline.predict(x_test)
    # print("Training score:", metrics.r2_score(y_train[:, i], pipeline.predict(x_train)))
    df[label] = predictions

compression_options = dict(method='zip', archive_name='prediction.csv')
df.to_csv('prediction.zip', index=False, float_format='%.3f', compression=compression_options)

