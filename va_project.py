
import collections


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('dataset/weatherAUS.csv')



plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

df.isnull().sum().sort_values(ascending=False)

df.isnull().sum().sort_values(ascending=False) * 100 / len(df)

df.describe().transpose()

plt.figure(figsize=(16,8))
ax = sns.boxplot(x='Evaporation', data=df)

plt.figure(figsize=(16,8))
sns.boxplot(x='Rainfall', data=df)

plt.figure(figsize=(16,8))
sns.boxplot(x='WindGustSpeed', data=df)

plt.figure(figsize=(16,8))
sns.boxplot(x='WindSpeed9am', data=df)





plt.figure(figsize=(16,8))
sns.boxplot(x='WindSpeed3pm', data=df)

pd.to_datetime(df['Date']).sort_values()

df.duplicated().value_counts()

df.drop('Date', axis=1, inplace=True)

from typing import List

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    #print(df.select_dtypes(include='number'))
    return list(df.select_dtypes(include='number').columns.values)

def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    #print(df.select_dtypes(include='object'))
    return list(df.select_dtypes(include='object').columns.values)

def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    numeric_cols = get_numeric_columns(df)
    if(column in numeric_cols):
        temp_df = df.copy()
        #Using mean - standard deviation rule to find the lowest and highest limit
        highest_limit = temp_df[column].mean() + 3*temp_df[column].std()
        lowest_limit = temp_df[column].mean() - 3*temp_df[column].std()
        #Replace values less than lower limit to lower limit and values higher than upper limit to upper limit value
        temp_df[column] = np.where(temp_df[column] > highest_limit,highest_limit,np.where(temp_df[column] < lowest_limit,lowest_limit,temp_df[column]))
        return temp_df
    return df

numeric_cols = get_numeric_columns(df)
print(numeric_cols)

for col in numeric_cols:
  df = fix_outliers(df, col)

plt.figure(figsize=(16,8))
ax = sns.boxplot(x='Evaporation', data=df)

plt.figure(figsize=(16,8))
sns.boxplot(x='WindSpeed9am', data=df)

# def replace_nan_with_mean(col):
#   df[col] = df[col].fillna(df[col].mean())


# for col in numeric_cols:
#   replace_nan_with_mean(col)

# df.isnull().sum().sort_values(ascending=False) * 100 / len(df)

df.head()

df = df.dropna()

df.isnull().sum().sort_values(ascending=False) * 100 / len(df)

df.shape

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'].values)

df.head()

categorical_cols = get_text_categorical_columns(df)
print(categorical_cols)

df.Location.unique()

df = pd.get_dummies(df, columns=categorical_cols)

df.head()

y = df['RainTomorrow']
X = df.drop('RainTomorrow', axis=1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)

X = scaler.transform(X)

X



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf_model = RandomForestClassifier(n_estimators=2, max_depth=3)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Calculating Scores
rf_confusion_matrix = confusion_matrix(y_test, y_pred)
rf_classification_report = classification_report(y_test, y_pred)

print(rf_classification_report)

print(rf_confusion_matrix)

# # parameter_tuning = {'C': [0.1, 1, 10, 100, 1000],
# #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
# #               'kernel': ['rbf']}
# # tuned_model = GridSearchCV(svm.SVC(),parameter_tuning, refit=True, verbose=3)
# # tuned_model.fit(X_train,y_train)
# # y_pred_tuned = tuned_model.predict(X_test)

# svm_model = svm.SVC()
# svm_model.fit(X_train, y_train)
# y_pred = svm_model.predict(X_test)

# # Calculating Scores
# svm_confusion_matrix = confusion_matrix(y_test, y_pred)
# svm_classification_report = classification_report(y_test, y_pred)

# print(svm_classification_report)

# tuned_model.best_params_

# tuned_model.best_estimator_

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred_xgb = xgb.predict(X_test)

# Making the Confusion Matrix
print(confusion_matrix(y_test, y_pred_xgb))

print(classification_report(y_test,y_pred_xgb))

print(accuracy_score(y_test,y_pred_xgb))



rf = RandomForestClassifier()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# n_estimators = range(100,1000,50)

# hyperF = dict(n_estimators = n_estimators)

# random_forest = RandomizedSearchCV(rf, hyperF, cv = 3, verbose = 10, 
#                       n_jobs = -1)
# random_forest.fit(X_train, y_train)

X_train.shape

#Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model

model = Sequential()
model.add(Dense(units=92,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=46,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=23,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

model.fit(x=X_train, 
          y=y_train, 
          epochs=100,
          batch_size=256,
          validation_data=(X_test, y_test),
          callbacks=[early_stop]
          )

# model.save('Project.h5')

loss = pd.DataFrame(model.history.history)
loss[['loss','val_loss']].plot()

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(np.int)

print(classification_report(y_test,y_pred))

# Making the Confusion Matrix
print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test,y_pred))



# from sklearn.metrics import classification_report,confusion_matrix
# y_pred = model.predict(X_test)
# y_pred = np.round(y_pred).astype(int)
# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
# sns.countplot(data=dataset,x='loan_repaid')



# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe



space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

def objective(space):
    clf=XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05],#, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3],#, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1],#, 3, 5, 7 ],
 "gamma"            : [ 0.0],#, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3],#, 0.4, 0.5 , 0.7 ]
    
}

classifier=XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_train,y_train)
timer(start_time) # timing ends here for "start_time" variable

random_search.best_estimator_

random_search.best_params_

classifier=XGBClassifier(colsample_bytree=0.7, gamma=0.4, learning_rate=0.05, max_depth=15, min_child_weight=1)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_xgb = classifier.predict(X_test)

# Making the Confusion Matrix
print(confusion_matrix(y_test, y_pred_xgb))

print(classification_report(y_test,y_pred_xgb))

print(accuracy_score(y_test,y_pred_xgb))





from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X_train,y_train,cv=10)

score

score.mean()

