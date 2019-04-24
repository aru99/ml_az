"""
random forest classifier
Author : Mohammad Arman
Date : 24/4/19

"""

# ===libraries===
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# for splitting the dataset
from sklearn.model_selection import train_test_split
# for feature scaling
from sklearn.preprocessing import StandardScaler
# for classifier
from sklearn.ensemble import RandomForestClassifier
# confusionn matrix
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
# ===============

# importing dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# dataset split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

# feature scaling
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fitting random forest classification to the traning set
# classifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
# Predictiing the test set result
Y_pred = classifier.predict(X_test)


# empty list for false result storage
lis2 = []

for i in range(0, len(Y_pred)):
    if Y_test[i] != Y_pred[i]:
        lis2.append(i)
print(lis2, "\n", len(lis2))

# confusionn matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


# Visualising the Training set results


X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('white', 'yellow'))(i), label=j)
plt.title('Decision tree classification')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results


X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('white', 'yellow'))(i), label=j)
plt.title('svm (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()






