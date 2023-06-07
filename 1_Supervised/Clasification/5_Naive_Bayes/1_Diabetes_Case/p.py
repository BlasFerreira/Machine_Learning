import pandas as pd
from sklearn.model_selection import train_test_split
# Standar Scaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



df = pd.read_csv('diabetes.csv')


X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = df[['Outcome']]


X_train, X_test2, y_train, y_test2 = train_test_split( X , y , test_size = 0.3 , random_state=4 )

scaler = StandardScaler().fit(X_train)

X_train_scal = scaler.transform( X_train )
X_test2_scal = scaler.transform( X_test2 ) 

from sklearn.tree import plot_tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf = clf.fit(X_train_scal, y_train)

plt.figure(  )
plot_tree(clf, filled=True)
plt.title("Decision tree  features")
plt.show()
