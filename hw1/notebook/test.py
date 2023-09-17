import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# Load the dataset from the provided files
data = pd.read_csv('hw1/data/column_2C.dat', sep=' ', header=None, names=['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'grade_spondylolisthesis', 'class'])

# Map class labels to binary values (NO=0, AB=1)
data['class'] = data['class'].map({'NO': 0, 'AB': 1})

# Create separate datasets for Class 0 and Class 1 for easliy drawing plots.
class_0_data = data[data['class'] == 0]
class_1_data = data[data['class'] == 1]

# print(data)
#i. Scatterplots of independent variables
# make class0 to be blue and class 1 to be red
colors = ['blue' if c == 0 else 'red' for c in data['class']]
pd.plotting.scatter_matrix(data.iloc[:, :-1], c=colors, alpha=0.6, figsize=(12, 12), marker='o', hist_kwds={'bins': 20})
plt.suptitle('Scatterplots of Independent Variables (0: NO, 1: AB)', fontsize=16)
plt.show()

plt.figure(figsize=(12, 6))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(2, 3, i + 1)
    boxprops = dict(color='blue', facecolor='lightblue')
    data[data['class'] == 0].boxplot(positions = [0], column=col, patch_artist=True, boxprops=boxprops)
    boxprops = dict(color='red', facecolor='lightcoral')
    data[data['class'] == 1].boxplot(positions = [1],column=col, patch_artist=True, boxprops=boxprops)
    plt.title(col)
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
plt.suptitle('Boxplots of Independent Variables (0: NO, 1: AB)', fontsize=16)
plt.tight_layout()
plt.show()

# iii. Select the first 70 rows of Class 0 and the first 140 rows of Class 1 as the training set
# and the rest of the data as the test set
class_0_train = class_0_data.head(70)
class_1_train = class_1_data.head(140)
training_data = pd.concat([class_0_train, class_1_train], axis=0)
test_data = data.drop(training_data.index)

# Separate features (X) and labels (y)
y_train = training_data['class']
X_train = training_data.drop('class', axis=1)
y_test = test_data['class']
X_test = test_data.drop('class', axis=1)


# Define values of k to test
k_values = list(range(208, 0, -3))

# Initialize lists to store train and test errors
train_errors = []
test_errors = []

for k in k_values:
    # Create a KNN classifier with the current value of k
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Predict on the training data
#     y_train_pred = knn.predict(X_train)
    
    # Predict on the testing data
    y_test_pred = knn.predict(X_test)
    
    # Calculate training and testing errors
    train_error = 1 - knn.score(X_train, y_train)
    test_error = 1 - knn.score(X_test, y_test)
    
    # Append errors to the lists
    train_errors.append(train_error)
    test_errors.append(test_error)

# Find the index of the minimum test error
best_k_index = np.argmin(test_errors)
best_k = k_values[best_k_index]

# Plot train and test errors in terms of k
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_errors, label='Train Error', marker='o')
plt.plot(k_values, test_errors, label='Test Error', marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate')
plt.title('Train and Test Errors vs. Number of Neighbors (k)')
plt.legend()
plt.grid(True)
plt.show()
