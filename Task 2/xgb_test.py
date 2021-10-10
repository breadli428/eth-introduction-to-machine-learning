import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
import xgboost as xgb
from sklearn import linear_model


def features_engineering(df_ori):
    df_ori = df_ori.drop(columns=['Time'])
    feature = df_ori.groupby(['pid'], sort=False).agg([np.nanmax, np.nanmin, np.nanmean, np.nanmedian, np.nanstd]) # feature augmentation
    feature = feature.fillna(feature.mean()) # fill nan with mean
    transformer = RobustScaler()
    X = transformer.fit_transform(feature)
    return X

# read data
df_train = pd.read_csv('./datasets/train_features.csv')
X_train = features_engineering(df_train)
df_label = pd.read_csv('./datasets/train_labels.csv')
labels = list(df_label.columns)
labels.remove('pid')

df_test = pd.read_csv('./datasets/test_features.csv')
X_test = features_engineering(df_test)
pid_test = df_test['pid'].drop_duplicates().to_numpy()
y_pred = {'pid': pid_test}
print('Preprocessed!\n')

# print(X_train)
# print(X_test)

# parameters found by grid search
# uncomment commented lines to run grid search
# learning rate, max_depth, subsample
param = np.zeros((11, 3))
param[0] = [0.1, 5.0, 0.9] # LABEL_BaseExcess
param[1] = [0.1, 3.0, 0.75] # LABEL_Fibrinogen
param[2] = [0.15, 3.0, 0.9] # LABEL_AST
param[3] = [0.15, 3.0, 1.0] # LABEL_Alkalinephos
param[4] = [0.1, 3.0, 0.7] # LABEL_Bilirubin_total
param[5] = [0.1, 3.0, 0.75] # LABEL_Lactate
param[6] = [0.1, 3.0, 0.5] # LABEL_TroponinI
param[7] = [0.05, 5.0, 0.75] # LABEL_SaO2
param[8] = [0.05, 3.0, 0.75] # LABEL_Bilirubin_direct
param[9] = [0.05, 5.0, 0.75] # LABEL_EtCO2
param[10] = [0.05, 3.0, 1.0] # LABEL_Sepsis

# # Task 1 & 2
for i in range(11):
    label = labels[i]
    y_train = df_label[label]

    # # imbalanced rate
    # neg, pos = np.bincount(y_train)
    # total = neg + pos
    # print(label, ': Positive: {} ({:.2f}% of total)'.format(pos, 100 * pos / total))

    # # grid search
    # parameters = {
    #     'max_depth':[3, 4, 5, 6, 7, 8, 9], 
    #     'learning_rate':[0.05, 0.10, 0.15, 0.20],
    #     'subsample':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # }
    # count = Counter(y_train)
    # imbalance = count[0] / count[1]
    # xgbclf = xgb.XGBClassifier(scale_pos_weight=imbalance, use_label_encoder=False, eval_metric='logloss')
    # clf = GridSearchCV(estimator=xgbclf, param_grid=parameters, cv=5, scoring='roc_auc', verbose=4, n_jobs=-1)
    # clf.fit(X_train, y_train)
    # print(clf.cv_results_)
    # results = pd.DataFrame(clf.cv_results_)
    # name = label + '.csv'
    # results.to_csv(name)

    count = Counter(y_train)
    imbalance = count[0] / count[1]
    xgbclf = xgb.XGBClassifier(
        learning_rate=param[i][0],
        max_depth=int(param[i][1]), 
        subsample=param[i][2],
        scale_pos_weight=imbalance, 
        use_label_encoder=False, 
        eval_metric='logloss'
        )

    # cv_score = cross_val_score(xgbclf, X_train, y_train, cv=5, scoring='roc_auc')
    # print('CV_score: ', np.mean(cv_score))

    model = xgbclf.fit(X_train, y_train)
    prob = model.predict_proba(X_test)
    prob = np.array(prob[:, 1])
    y_pred[label] = prob

    print(label, ': finished!\n')


# Task 3
for i in range(11, len(labels)):
    label = labels[i]
    y_train = df_label[label]

    # grid search
    # parameters = {
    #     'learning_rate':[0.05, 0.10, 0.15, 0.20],
    #     'max_depth':[3, 4, 5, 6, 7, 8, 9], 
    #     'subsample':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # }

    xgbreg = xgb.XGBRegressor(use_label_encoder=False)
    # model = GridSearchCV(estimator=xgbreg, param_grid=parameters, cv=5, scoring='r2', verbose=4, n_jobs=-1)
    # model.fit(X_train, y_train)
    # print(model.cv_results_)
    # results = pd.DataFrame(model.cv_results_)
    # name = label + '.csv'
    # results.to_csv(name)

    mdl = linear_model.Lasso(alpha=0.1)

#     cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
#     print('CV_score: ', np.mean(cv_score))

    model = mdl
    model = model.fit(X_train, y_train)
    y_pred_label = model.predict(X_test)
    y_pred[label] = y_pred_label

    print(label, ': finished! \n')

df_pred = pd.DataFrame(y_pred)

compression_options = dict(method='zip', archive_name='prediction.csv')
df_pred.to_csv('prediction.zip', index=False, float_format='%.3f', compression=compression_options)

print('All finished!')