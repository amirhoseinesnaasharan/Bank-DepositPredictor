from mlxtend.plotting import plot_decision_regions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification

# Original Dataset:
data = pd.read_csv('bank.csv')

# Convert label feature to 0 and 1
data['deposit'] = data['deposit'].map({'no': 0, 'yes': 1})

# Separate Nominal and Numerical
numerical_features = data.select_dtypes(include=['int64', 'float64'])
categorical_features = data.select_dtypes(include=['object'])

# Normalize numerical features:
scaler = StandardScaler()
normalized_numerical = scaler.fit_transform(numerical_features)
normalized_numerical_df = pd.DataFrame(
    normalized_numerical, columns=numerical_features.columns)

# One-hot encode categorical features:
one_hot_encoder = OneHotEncoder()
encoded_categorical = one_hot_encoder.fit_transform(categorical_features)
encoded_categorical_df = pd.DataFrame(
    encoded_categorical.toarray(), columns=one_hot_encoder.get_feature_names_out())

# Combine encoded features:
final_data = pd.concat(
    [normalized_numerical_df, encoded_categorical_df], axis=1)

# Calculate the Pearson correlation coefficients after converting labels
correlation_matrix = final_data.corr()

# Identify the least important feature
least_important_feature = correlation_matrix.abs().div(
    correlation_matrix.abs().max(axis=0), axis=0).apply(pd.Series.idxmin).index[0]

# Remove the least important feature from the final_data Dataset
final_data_without_least_important_feature = final_data.drop(
    least_important_feature, axis=1)
print("Least important feature:", least_important_feature)

# Identify the 2 most important features
most_important_features = correlation_matrix['deposit'].abs().nlargest(
    3).index[1:]

# Print the 2 most important features
print("Most important features:", most_important_features)

# Create a scatter plot with different markers for each day and circles for 'duration'
plt.figure(figsize=(10, 8))
sns.scatterplot(data=final_data_without_least_important_feature, x=most_important_features[0], y=most_important_features[1], hue='deposit',
                style=data['day'], size=data['duration'], sizes=(10, 200), marker='o', palette='viridis')

plt.title('Scatter plot of the 2 most important features')
plt.show()

# Split the data into features (X) and target variable (y)
X = final_data_without_least_important_feature  # Features
y = data['deposit']  # Target variable

# Split the data into train and test sets (80% train, 20% test)
train_data, test_data, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train an SVM with a linear kernel on the training data
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(train_data, train_labels)

# Make predictions on the test data using the linear SVM
linear_test_predictions = linear_svm.predict(test_data)

# Calculate the accuracy score, F1 score, precision, recall, and confusion matrix for the linear SVM on the test data
linear_accuracy = accuracy_score(test_labels, linear_test_predictions)
linear_f1 = f1_score(test_labels, linear_test_predictions)
linear_precision = precision_score(test_labels, linear_test_predictions)
linear_recall = recall_score(test_labels, linear_test_predictions)
linear_confusion_matrix = confusion_matrix(
    test_labels, linear_test_predictions)

# Train an SVM with an RBF kernel on the training data
rbf_svm = SVC(kernel='rbf', C=1.0)
rbf_svm.fit(train_data, train_labels)

# Make predictions on the test data using the RBF SVM
rbf_test_predictions = rbf_svm.predict(test_data)

# Calculate the accuracy score, F1 score, precision, recall, and confusion matrix for the RBF SVM on the test data
rbf_accuracy = accuracy_score(test_labels, rbf_test_predictions)
rbf_f1 = f1_score(test_labels, rbf_test_predictions)
rbf_precision = precision_score(test_labels, rbf_test_predictions)
rbf_recall = recall_score(test_labels, rbf_test_predictions)
rbf_confusion_matrix = confusion_matrix(test_labels, rbf_test_predictions)

# Print the accuracy score, F1 score, precision, recall, and confusion matrix for both models on the test data
print("Linear SVM:")
print("Accuracy score:", linear_accuracy)
print("F1 score:", linear_f1)
print("Precision:", linear_precision)
print("Recall:", linear_recall)
print("Confusion matrix:\n", linear_confusion_matrix)
print("\nRBF SVM:")
print("Accuracy score:", rbf_accuracy)
print("F1 score:", rbf_f1)
print("Precision:", rbf_precision)
print("Recall:", rbf_recall)
print("Confusion matrix:\n", rbf_confusion_matrix)

# Print the accuracy score, F1 score, precision, recall, and confusion matrix for both models on the test data
print("Linear SVM:")
print("Accuracy score:", linear_accuracy)
print("F1 score:", linear_f1)
print("Precision:", linear_precision)
print("Recall:", linear_recall)
print("Confusion matrix:\n", linear_confusion_matrix)
print("\nRBF SVM:")
print("Accuracy score:", rbf_accuracy)
print("F1 score:", rbf_f1)
print("Precision:", rbf_precision)
print("Recall:", rbf_recall)
print("Confusion matrix:\n", rbf_confusion_matrix)


def plot_decision_boundary(model, X, y, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm,
                edgecolors='k', marker='o')

    plt.title(title)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.show()


# Assuming linear_svm and rbf_svm were trained with all features
linear_feature_indices = [0, 1]
rbf_feature_indices = [0, 1]

# Plot decision boundary for Linear SVM
plot_decision_boundary(
    linear_svm, test_data.values[:, linear_feature_indices], test_labels.values, 'Linear SVM')

# Plot decision boundary for RBF SVM
plot_decision_boundary(
    rbf_svm, test_data.values[:, rbf_feature_indices], test_labels.values, 'RBF SVM')
