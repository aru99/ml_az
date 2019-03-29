# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# immporting another library for traning set and test set
from sklearn.model_selection import train_test_split
# importing library for feature scaling
from sklearn.preprocessing import StandardScaler
# importing library for logisticRegression
from sklearn.linear_model import LogisticRegression
# importing library for confusion matrix
from sklearn.metrics import confusion_matrix
# importing library for visualisation
from matplotlib.colors import ListedColormap

# importing the data set 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# splitting the dataset into traning set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# fitting the logistic regression to the data set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# predicting the results
Y_pred = classifier.predict(X_test)

# making the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)

# visualising the results for traning set
X_set, Y_Set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_set[Y_Set == j, 0], X_set[Y_Set == j, 1], c=ListedColormap(('white', 'yellow'))(i), label=j)

plt.title('logistic regression traning set')
plt.xlabel('Age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()


# visualising the results for test set
X_set, Y_Set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_set[Y_Set == j, 0], X_set[Y_Set == j, 1], c=ListedColormap(('white', 'yellow'))(i), label=j)

plt.title('logistic regression test set')
plt.xlabel('Age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()
