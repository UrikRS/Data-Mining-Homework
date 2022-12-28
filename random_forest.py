from data_handler import load_data
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

x, y = load_data('Data Mining/Homework')

p_grid = {
    'n_estimators': [50, 100, 120],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [7, 8, 9],
    'max_features': [7, 8, 9],
    'min_samples_split': [3, 5, 7]
}

rfc = GridSearchCV(RandomForestClassifier(), p_grid, cv=10)
rfc.fit(x, y.to_numpy().ravel())

print('Best parameters :', rfc.best_params_, '\n')
print('Best score :', rfc.best_score_, '\n')
print('Classification report :\n', classification_report(y, rfc.predict(x)))