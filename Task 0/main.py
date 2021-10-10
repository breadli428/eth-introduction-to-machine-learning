from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import statistics
import csv


with open('./datasets/train.csv', 'r') as file:
    reader = csv.reader(file)
    header_row = next(reader)
    X_train = []
    y_train = []
    for row in reader:
        X_train.append(list(map(float, row[2:])))
        y_train.append(float(row[1]))
 

with open('./datasets/test.csv', 'r') as file:
    reader = csv.reader(file)
    header_row = next(reader)
    X_test = []
    y_test = []
    Id = []
    for row in reader:
        Id.append(row[0])
        X_test.append(list(map(float, row[1:])))
        y_test.append(statistics.mean(list(map(float, row[1:]))))


model = linear_model.Ridge(alpha=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(rmse)

with open('./sub.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Id', 'y'])
    for row in range(len(y_pred)):
        writer.writerow([Id[row], y_pred[row]])
