from sklearn import linear_model
import statistics
import csv
import numpy as np

with open('./datasets/train.csv', 'r') as file:
    reader = csv.reader(file)
    header_row = next(reader)
    X = []
    y = []
    for row in reader:
        X.append(np.array(list(map(float, row[2:]))))
        y.append(float(row[1]))

X_raw = np.array(X)
y = np.array(y)

X_transformed = np.concatenate((X_raw, np.power(X_raw, 2), np.exp(X_raw), np.cos(X_raw), np.ones([len(X_raw), 1])), axis=1)

model = linear_model.Ridge(alpha=0.01, fit_intercept=False)
reg = model.fit(X_transformed, y)

np.savetxt('./sub.csv', reg.coef_, newline='\n')
