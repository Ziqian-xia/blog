---
title: ESE 2023 Open Code
author: Ziqian Xia
date: '2023-08-19'
slug: ese-2023-open-code
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2023-08-19T12:26:13+08:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

This is the open code repo for ESE 2023.

## Table of Contents

1. [K-Means Clustering Tutorial](#K-Means)
2. [CART Regression with Public Dataset](#CART_regression)
3. [Unveiling the Power of Neural Networks](#Neural_Network)
4. [Conclusion](#Further_Reading)


# K-Means

In this tutorial, we will explore the popular k-means clustering algorithm using Python. K-means is an unsupervised machine learning technique that divides a dataset into a given number of clusters. We'll use the Iris dataset and walk through the process of determining the optimal number of clusters using a scree plot.

## Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
```

## Load and Prepare Data

Let's start by loading the Iris dataset and standardizing the features:

```python
data = load_iris()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Scree Plot for Optimal Number of Clusters

We need to decide on the optimal number of clusters (k) for our k-means algorithm. One way to do this is by using a scree plot:

```python
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Scree Plot')
plt.show()
```

From the scree plot, identify the "elbow" point where the decrease in the sum of squared distances starts to slow down. This point suggests the optimal number of clusters.

## Perform K-Means Clustering

Assuming we find the optimal k to be 3, let's perform k-means clustering:

```python
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

data_with_clusters = pd.DataFrame(data.data, columns=data.feature_names)
data_with_clusters['Cluster'] = clusters
```

## Visualize the Clusters

You can visualize the clusters by creating scatter plots of different feature pairs:

```python
plt.scatter(data_with_clusters['sepal length (cm)'], data_with_clusters['sepal width (cm)'], c=data_with_clusters['Cluster'], cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.show()
```

Repeat the visualization for other pairs of features.

That's it! You've successfully performed k-means clustering using Python and determined the optimal number of clusters using a scree plot. Feel free to explore this technique with other datasets and experiment with different values of k.


#CART_regression

## CART Regression with Diabetes Dataset

In the previous section, we explored the world of k-means clustering. Now, let's dive into another powerful machine learning technique: Classification and Regression Trees (CART). CART is a decision tree-based algorithm that can be used for both classification and regression tasks.

### Introduction to CART Regression

CART regression is particularly useful when you're dealing with continuous target variables. The algorithm works by recursively splitting the data into subsets based on feature values, with the goal of minimizing the variance of the target variable within each subset.

In this section, we'll apply CART regression to the Diabetes dataset, which contains ten baseline variables (age, sex, BMI, average blood pressure, and six blood serum measurements) and a quantitative measure of disease progression one year after baseline. Our aim is to predict this quantitative measure using the CART regression model.

### Loading the Data

Let's start by loading the Diabetes dataset and preparing it for modeling:

```python
# Load the Diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Hyperparameter Tuning with GridSearchCV

CART regression comes with hyperparameters that control the depth of the tree and how nodes are split. To find the best combination of hyperparameters, we'll use GridSearchCV:

```python
# Define hyperparameters to tune
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Initialize the Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=0)

# Initialize GridSearchCV
grid_search = GridSearchCV(regressor, param_grid, cv=5)

# Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_regressor = grid_search.best_estimator_
```

### Model Evaluation and Visualization

With the best model in hand, let's evaluate its performance on the test set and visualize the decision tree it learned:

```python
# Predict on the test set using the best model
y_pred = best_regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Best Mean Squared Error: {mse:.2f}")

# Visualize the Decision Tree of the best model
plt.figure(figsize=(15, 10))
plot_tree(best_regressor, feature_names=data.feature_names, filled=True, rounded=True)
plt.title("Best Decision Tree Regressor")
plt.show()
```

In this section, we covered the basics of CART regression, from loading the data to hyperparameter tuning and model evaluation. The visualization of the decision tree provides insights into how the model makes predictions based on different features.

Next, we'll continue our journey into machine learning with more advanced techniques.

#Neural_Network
## Neural Networks: Unveiling the Power of Deep Learning

In the realm of machine learning, Neural Networks (NNs) have risen to prominence due to their exceptional capabilities in capturing complex patterns in data. A Neural Network is inspired by the human brain's structure and consists of interconnected neurons organized in layers.

### Understanding Neural Networks

Neural Networks are particularly adept at handling tasks like classification, regression, and more recently, even tasks like image recognition, natural language processing, and game playing. They consist of layers:

- **Input Layer**: Receives the initial data.
- **Hidden Layers**: Process the data using weighted connections.
- **Output Layer**: Produces the final prediction or output.

Each neuron in a layer receives inputs, processes them using weights, and passes the result through an activation function. This activation function introduces non-linearity, enabling the network to learn complex relationships.

### Applying Neural Networks to the Iris Dataset

Now, let's apply a Neural Network to the Iris dataset, continuing from where we left off in the previous sections:

### Loading and Preprocessing Data

```python
# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Building and Training the Neural Network

Now, let's create a simple Neural Network using the `MLPClassifier` from scikit-learn:

```python
# Initialize the Neural Network Classifier
classifier = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000, random_state=0)

# Train the model
classifier.fit(X_train_scaled, y_train)
```

### Evaluating the Neural Network

After training, we can evaluate the Neural Network's performance using accuracy and visualize its predictions using a confusion matrix:

```python
# Predict on the test set
y_pred = classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

### The Power of Neural Networks

In this section, we've delved into the world of Neural Networks, exploring their architecture and capabilities. By applying a simple Neural Network to the Iris dataset, we've witnessed how it can learn intricate patterns from data and make accurate predictions. The visualization of the confusion matrix offers insights into the model's strengths and weaknesses.


#Further_Reading

If you're eager to dive deeper into the topics covered in this tutorial, here are some resources to explore:

### K-Means Clustering:

- [Scikit-Learn Documentation on K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Understanding K-Means Clustering](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)

### CART Regression:

- [Scikit-Learn Documentation on Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Introduction to Decision Trees](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)

### Neural Networks and Deep Learning:

- [Introduction to Neural Networks](https://www.tensorflow.org/guide/keras/sequential_model)
- [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)

### Machine Learning Blogs and Communities:

- [Towards Data Science](https://towardsdatascience.com/)
- [Kaggle](https://www.kaggle.com/)

## Conclusion

In this tutorial, we embarked on a journey through fundamental machine learning techniques, from K-Means Clustering to CART Regression and Neural Networks. We've explored the power of these methods using real-world datasets and visualizations. Armed with this knowledge, you're ready to tackle diverse machine learning challenges and continue your exploration into the realm of data science.


Happy learning!
