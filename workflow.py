import pipeline
import logistic_regression
import numpy as np
import pandas as pd

d = pd.read_csv('data/creditcard.csv')
d_train, d_test = pipeline.train_test_split(d)
d_train, d_valid = pipeline.train_test_split(d_train)

def clean_data(d):
    d = d.copy()
    d.columns = [col.lower() for col in d.columns]
    d = d.rename(columns={
        'default payment next month': 'default',
        'pay_0': 'pay_1'})
    d['education'] = d['education'].replace({x: 0 for x in [0, 4, 5, 6]})
    d['marriage'] = d['marriage'].replace({0: 3})

    # update payment status
    cols = [f'pay_{i}' for i in range(1, 7)]
    d[cols] = d[cols].mask(lambda x: x < 0).fillna(0)

    # remove column ID
    del d['id']
    return d


def feature_engineering(d):
    last_column = d.columns[-1]
    d['any_late_pay'] = (d[[f'pay_{i}' for i in range(1, 7)]]>0).sum(axis=1)
    return d[[col for col in d.columns if col != last_column]+[last_column]]


def feature_keeper(d, features, y_column):
    return d[features+[y_column]]


def process_data(d):
    d = clean_data(d)
    d = feature_engineering(d)
    d = feature_keeper(d, ['pay_1', 'limit_bal', 'any_late_pay'], 'default')
    X,y = pipeline.get_X_y(d)
    return d, X, y

d_train, X_train, y_train = process_data(d_train)
d_valid, X_valid, y_valid = process_data(d_valid)
d_test, X_test, y_test = process_data(d_test)


## train model
scaler = pipeline.MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

lr = logistic_regression.LogisticRegression(learning_rate=0.1, num_iterations=10**3, verbose=100)
lr.fit(X_train, y_train)

# plot relative feature importance
pd.Series(lr.coef_[1:]/np.abs(lr.coef_[1:]).sum(), index=d_train.columns[:-1]).plot.bar(title='Relative feature importance')
pd.Series(lr.coef_[1:]/np.abs(lr.coef_[1:]).sum(), index=d_train.columns[:-1]).abs().sort_values(ascending=False).plot.bar(title='Relative feature importance')


## validate model
X_valid_scaled = scaler.transform(X_valid)
pm = pipeline.PerformanceMetric(lr.predict(X_valid_scaled), y_valid)
print('f1:', pm.f1_score)

pr_curve = pipeline.PRCurve(lr.predict_proba(X_valid_scaled), y_valid)
pr_curve.plot()

roc_curve = pipeline.ROCCurve(lr.predict_proba(X_valid_scaled), y_valid)
roc_curve.plot()

