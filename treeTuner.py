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

# ## 1. Get the data

# In[2]:

import warnings
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np

# './IEMOCAP_features_2.csv'
features_IME = pd.read_csv('https://raw.githubusercontent.com/GeorgeMarkham/Model-Tuner-and-Tester/master/IEMOCAP_features_2_short_framing.csv')

# './RAVDASS_features_2.csv'
feature_RAVDASS = pd.read_csv('https://raw.githubusercontent.com/GeorgeMarkham/Model-Tuner-and-Tester/master/RAVDASS_features_2_short_framing.csv')


features_IME = features_IME.drop(columns=["File_Name", "Session", "val", "act", "dom", "wav_file_name"])
feature_RAVDASS = feature_RAVDASS.drop(columns=["File_Name", "Modality", "Vocal_Channel ", "Emotional_Intensity", "Statement", "Repetition", "Actor"])

data = pd.concat([features_IME, feature_RAVDASS])

df = data

lab = data.drop(columns = ['Signal_Mean', 'Signal_StdDeviation', 'Rms_Vec_Mean',
      'Rms_Vec_StdDeviation', 'Autocorrelation_Max',
      'Autocorrelation_StdDeviation', 'Silence', 'Harmonic_Mean']) #"Unnamed: 0"

df = df.drop(columns=['Emotion'])


# ## 2. Split the data into training and testing sets
# If the data was using raw emotion values, e.g 'neu' instead of 0 then one would need to use a label encoder to encode each unique label as an integer between 0 and n_classes-1. Label encoding can be done using the Sci-Kit Learn LabelEncoder class https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html.

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# y = le.fit_transform(lab)
y = lab
x = df 

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=y)

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