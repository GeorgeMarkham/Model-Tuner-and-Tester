from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#     AdaBoost
params = {
    'base_estimator' : [GradientBoostingClassifier(), RandomForestClassifier(), SVC(), XGBClassifier()],
    'n_estimators'   : [50, 100, 200, 300, 400, 500],
    'learning_rate'  : [10**x for x in list(range(-5, -1, 1))],
}

estimator = AdaBoostClassifier(algorithm='SAMME', random_state=42)

estimator_name = str(estimator).split('(')[0]

print("Tuning {}".format(estimator_name))

model = GridSearchCV(estimator, params, cv=5, n_jobs=-1, refit=True)
model.fit(X_train.values, y_train.values)

pred = model.predict(np.array(X_test))

classificationReport = classification_report(np.array(y_test), np.array(pred))

print('-'*50)
print('\n')
print('REPORT - {} \n'.format(estimator_name))
print(str(classificationReport))
print('\n'*2)
report_name = "./{}_report.txt".format(estimator_name)
print(report_name)
print("Saving Report...")

print('-'*50)
print('\n'*2)
with open(report_name, 'w') as of:
    of.write('REPORT - {} \n'.format(estimator_name))
    of.write('-'*50)
    of.write(str(model.best_estimator_))
    of.write("\n\n")
    of.write(str(model.best_params_))
    of.write("\n\n")
    of.write("Classification Report:\n" + str(classificationReport))
    of.write("\n\n")
    of.write("Confusion Matrix:\t" + str(confusion_matrix(np.array(y_test), np.array(pred))))
    of.write("\n\n")
    of.write("Balanced Accuracy:\t" + str(balanced_accuracy_score(np.array(y_test), np.array(pred))))
    of.write("\n\n")
    of.write('-'*50)


#     GradientBoostingClassifier
params = {
    'criterion'      : ['friedman_mse', 'mse', 'mae'], #3
    'n_estimators'   : [50, 100, 200, 300, 400, 500], #6
    'learning_rate'  : [10**x for x in list(range(-5, -1, 1))], #4
    'max_features'   : ['auto', 'sqrt', 'log2', None], #4
}

estimator = GradientBoostingClassifier()

estimator_name = str(estimator).split('(')[0]

print("Tuning {}".format(estimator_name))

model = GridSearchCV(estimator, params, cv=5, n_jobs=-1, refit=True)
model.fit(X_train.values, y_train.values)

pred = model.predict(np.array(X_test))

classificationReport = classification_report(np.array(y_test), np.array(pred))

print('-'*50)
print('\n')
print('REPORT - {} \n'.format(estimator_name))
print(str(classificationReport))
print('\n'*2)
report_name = "./{}_report.txt".format(estimator_name)
print(report_name)
print("Saving Report...")

print('-'*50)
print('\n'*2)
with open(report_name, 'w') as of:
    of.write('REPORT - {} \n'.format(estimator_name))
    of.write('-'*50)
    of.write(str(model.best_estimator_))
    of.write("\n\n")
    of.write(str(model.best_params_))
    of.write("\n\n")
    of.write("Classification Report:\n" + str(classificationReport))
    of.write("\n\n")
    of.write("Confusion Matrix:\t" + str(confusion_matrix(np.array(y_test), np.array(pred))))
    of.write("\n\n")
    of.write("Balanced Accuracy:\t" + str(balanced_accuracy_score(np.array(y_test), np.array(pred))))
    of.write("\n\n")
    of.write('-'*50)