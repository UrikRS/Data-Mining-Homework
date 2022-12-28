from data_handler import load_data
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

x, y = load_data('Data Mining/Homework')

p_grid = {
    'criterion': ['entropy', 'gini'], 
    'splitter': ['best', 'random'], 
    'max_depth': [5, 7, 9], 
    'min_samples_leaf': [1, 3, 5], 
    'min_samples_split': [2, 3, 4], 
    'max_features': [6, 7, 8], 
    'random_state': [123]
}

dtc = GridSearchCV(DecisionTreeClassifier(), p_grid, cv=10)
dtc.fit(x, y)

print('Best parameters :', dtc.best_params_, '\n')
print('Best score :', dtc.best_score_, '\n')
print('Classification report :\n', classification_report(y, dtc.predict(x)))