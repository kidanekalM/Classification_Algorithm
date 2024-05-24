# Step 1: Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load your dataset
data = pd.read_csv("dataset.csv")

# Step 3: Split the dataset into features and target
X = data.drop('condition', axis=1)  # Features
y = data['condition']  # Target

X = pd.get_dummies(X)
# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize the Decision Tree Classifier
clf = tree.DecisionTreeClassifier()

# Step 6: Train the model using the training sets
clf.fit(X_train, y_train)

# Step 7: Make predictions on the testing set
y_pred = clf.predict(X_test)

# Step 8: Check the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: To see the decision tree
tree.plot_tree(clf)
plt.show()
plt.savefig('tree.png')
