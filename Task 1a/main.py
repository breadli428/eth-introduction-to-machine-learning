from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import statistics
import csv


with open('./datasets/train.csv', 'r') as file:
    reader = csv.reader(file)
    header_row = next(reader)
    X = []
    y = []
    for row in reader:
        X.append(list(map(float, row[1:])))
        y.append(float(row[0]))

regularization_parameter = [0.1, 1, 10, 100, 200]
avg_scores = []

for alpha in regularization_parameter:

    model = linear_model.Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=10)
    avg_scores.append(statistics.mean(-scores))

with open('./sub.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in range(len(avg_scores)):
        writer.writerow([avg_scores[row],])
