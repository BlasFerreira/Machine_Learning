
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('diabetes.csv')


X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = df[['Outcome']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=4)
scaler = StandardScaler().fit(X_train)
X_train_scal = scaler.transform( X_train )
X_test_scal = scaler.transform( X_test ) 

param_grid = {
    'n_estimators': np.arange(1,100,5),
    'criterion' :['gini', 'entropy', 'log_loss'],
    'max_depth': np.arange(1,10,5),
    'max_features': ['auto', 'sqrt', 'log2'],
     }

rfc = RandomForestClassifier()

CV_rfc = GridSearchCV(estimator=rfc, 
                      param_grid=param_grid, 
                      cv= 5,
                      refit=True)

CV_rfc.fit(X_train_scal, y_train)

print(CV_rfc.best_params_)


