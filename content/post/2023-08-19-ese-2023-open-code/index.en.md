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
0. [Pre-requisite](#pre-requisite)
1. [K-Means Clustering Tutorial](#k-means)
2. [CART Regression with Public Dataset](#cart-regression)
3. [Unveiling the Power of Neural Networks](#neural-network)
4. [Conclusion](#further-reading)
5. [Appendix and other codes](#appendix)
    - [Linear Regression](#linear-regression)
    - [Hierarchy clustering](#hierarchy-clustering-seed-data)
    - [Decision tree (classification mode)](#decision-tree-classification)
    - [Decision tree (regression mode)](#decision-tree-regression)
    - [Random forest (regression mode)](#random-forest-regression)
    - [Random forest (classification mode)](#random-forest-classification)


## Pre-requisite

1. **Basic Programming Knowledge:** Familiarity with programming concepts is essential. Understanding variables, data types, loops, and conditional statements will help you grasp the fundamentals of Python more easily.

2. **Mathematics Fundamentals:** Machine learning involves various mathematical concepts, including linear algebra, calculus, and statistics. While you don't need to be a math expert, having a solid foundation in these areas will greatly enhance your understanding of ML algorithms.

3. **Python Programming Language:** Python is the go-to language for machine learning due to its simplicity and a wide range of libraries. Before diving into ML, make sure you have a good grasp of Python syntax, data structures, and basic libraries like NumPy and Pandas.

4. **Data Handling Skills:** Dealing with data is at the heart of machine learning. Learn how to clean, preprocess, and manipulate data using libraries like Pandas. This skill is crucial for preparing your data for model training.

5. **Linear Algebra:** Concepts like vectors, matrices, and matrix operations are fundamental to machine learning algorithms. Brush up on these topics to understand how data is represented and manipulated in ML.

6. **Statistics and Probability:** ML algorithms often rely on statistical principles and probability theory. Understanding concepts like mean, median, variance, and probability distributions will help you interpret and evaluate your models effectively.

7. **Machine Learning Basics:** Familiarize yourself with the basic concepts of machine learning, including supervised and unsupervised learning, classification, regression, clustering, and overfitting. This knowledge forms the groundwork for more advanced topics.

**Useful Resources:**

1. [Coursera - Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning): A highly recommended course that covers the fundamentals of ML. The course provides video lectures, assignments, and hands-on practice using Python.

2. [Bilibili (哔哩哔哩)](https://www.bilibili.com/video/BV1zf4y1Z7zF/?spm_id_from=333.337.search-card.all.click)

3. [Python.org](https://www.python.org/): The official Python website offers extensive documentation and tutorials for beginners. These resources will help you get comfortable with Python's syntax and features.

4. [DataCamp](https://www.datacamp.com/): DataCamp offers interactive courses on Python, data manipulation, and machine learning. Their courses provide practical exercises to reinforce your skills.

5. **Books:** ***"Python Machine Learning"*** by Sebastian Raschka and Vahid Mirjalili, and ***"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"*** by Aurélien Géron are highly recommended books to deepen your ML understanding.



## K Means

In this tutorial, we will explore the popular k-means clustering algorithm using Python. K-means is an unsupervised machine learning technique that divides a dataset into a given number of clusters. We'll use the Iris dataset and walk through the process of determining the optimal number of clusters using a scree plot.

### Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

### Load and Prepare Data

Let's start by loading the Iris dataset and standardizing the features:

```python
data = load_iris()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Scree Plot for Optimal Number of Clusters

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

### Perform K-Means Clustering

Assuming we find the optimal k to be 3, let's perform k-means clustering:

```python
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

data_with_clusters = pd.DataFrame(data.data, columns=data.feature_names)
data_with_clusters['Cluster'] = clusters
```

### Visualize the Clusters

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


## CART regression

### CART Regression with Diabetes Dataset

In the previous section, we explored the world of k-means clustering. Now, let's dive into another powerful machine learning technique: Classification and Regression Trees (CART). CART is a decision tree-based algorithm that can be used for both classification and regression tasks.

### Introduction to CART Regression

CART regression is particularly useful when you're dealing with continuous target variables. The algorithm works by recursively splitting the data into subsets based on feature values, with the goal of minimizing the variance of the target variable within each subset.

In this section, we'll apply CART regression to the Diabetes dataset, which contains ten baseline variables (age, sex, BMI, average blood pressure, and six blood serum measurements) and a quantitative measure of disease progression one year after baseline. Our aim is to predict this quantitative measure using the CART regression model.

### Loading the Data

Let's start by loading the Diabetes dataset and preparing it for modeling:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree

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

## Neural Network
### Neural Networks: Unveiling the Power of Deep Learning

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
classifier = MLPClassifier(max_iter=10000, random_state=0)

# Hyperparameter grid for grid search
param_grid = {
    'hidden_layer_sizes': [(8,), (16,), (32,)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_classifier = grid_search.best_estimator_

# Predict on the test set
y_pred = best_classifier.predict(X_test_scaled)
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

### NN Regression

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the California housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Neural Network Regressor
regressor = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=0)

# Train the model
regressor.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test_scaled)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize predicted vs. actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs. Actual Prices')
plt.show()
```

### The Power of Neural Networks

In this section, we've delved into the world of Neural Networks, exploring their architecture and capabilities. By applying a simple Neural Network to the Iris dataset, we've witnessed how it can learn intricate patterns from data and make accurate predictions. The visualization of the confusion matrix offers insights into the model's strengths and weaknesses.


## Further Reading

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

## Appendix

### Linear regression

```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 读取CSV文件
data = pd.read_csv(r'C:\Users\29153\Desktop\carbon.csv')
print(data)

# 提取高收入国家(HI)的2016年以前的数据
hi_data = data[(data['Income'] == 'HI') & (data['Time'] <= 2016)]
# 剔除含有NaN项的数据
hi_data = hi_data.dropna(subset=['Carbon', 'Pop', 'GDP_per_pop', 'Carbon_per_GDP'])

print(hi_data)

# 提取预测变量和响应变量
hi_data['GDP_per_pop_squared'] = hi_data['GDP_per_pop'] ** 2
X = hi_data[['Pop', 'Carbon_per_GDP', 'GDP_per_pop_squared']]
y = hi_data['Carbon']

# 添加截距列
X = sm.add_constant(X)

# 进行多元线性回归拟合
model = sm.OLS(y, X).fit()

# 打印拟合结果摘要
print(model.summary())


# 模型诊断
# 获取模型预测值和残差
# 模型预测值为model.predict(X)
# 模型残差值为model.resid
fitted_values = model.predict()
residuals = model.resid

# 同方差性/线性性（Homoscedasticity / linearity）检验
# 残差e的方差会随着x变动而变动,因此方差是异质性的. 这被称为异方差问题，异方差存在的时候,大多数情况下,OLS估计出的方差会比实际的方差要小，因此会过高地估计系数b的显著性

plt.scatter(fitted_values, residuals)
# 拟合拟合值和残差的趋势线
lowess = sm.nonparametric.lowess(residuals, fitted_values)
plt.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2)  # 添加红色趋势线

plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity / Linearity Check')
plt.axhline(y=0, color='green', linestyle='--')
plt.show()


# 正态分布随机误差（Normally distributed random errors）检验
sm.qqplot(residuals, line='s')
plt.title('Normality of Residuals')
plt.show()


# 使用boxcox函数进行转化
y_transformed, lmbda_best = boxcox(y, lmbda=None, alpha=None)
print('lambda best ==', lmbda_best)

model_2 = sm.OLS(y_transformed, X).fit()
print(model_2.summary())


fitted_values_2 = model_2.predict()
residuals_2 = model_2.resid
plt.scatter(fitted_values_2, residuals_2)
# 拟合拟合值和残差的趋势线
lowess = sm.nonparametric.lowess(residuals_2, fitted_values_2)
plt.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2)  # 添加红色趋势线
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity / Linearity Check')
plt.axhline(y=0, color='green', linestyle='--')
plt.show()

# 正态分布随机误差（Normally distributed random errors）检验
sm.qqplot(residuals_2, line='s')
plt.title('Normality of Residuals')
plt.show()


# 共线性（collinearity）检验
# 使用VIF进行检验的方法主要为，对某一因子和其余因子进行回归，得到R^2，计算VIF，剔除因子中VIF高的因子，保留VIF较低的因子，以此类推，直到得到一个相关性较低的因子组合来增强模型的解释能力。
# 在实际测试过程中，并非要指定一个VIF阈值，比如某因子的VIF值超过阈值才剔除，而是通过观察所有因子值的VIF值，如果发现该值较大（显著离群），剔除该因子即可。
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("VIF:")
print(vif_data)

# 自相关性（autocorrelation）检验
sm.graphics.tsa.plot_acf(residuals, lags=20)
plt.title('Autocorrelation of Residuals')
plt.show()

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
data = pd.read_csv(r'C:\Users\29153\Desktop\carbon.csv')
print(data)

# 提取高收入国家(HI)的2016年以前的数据
hi_data = data[(data['Income'] == 'HI') & (data['Time'] <= 2016)]
# 剔除含有NaN项的数据
hi_data = hi_data.dropna(subset=['Carbon', 'Pop'])

print(hi_data)


# 提取预测变量和响应变量
X = hi_data['Pop']
y = hi_data['Carbon']

# 添加截距列
X = sm.add_constant(X)

# 进行多元线性回归拟合
model = sm.OLS(y, X).fit()

# 打印拟合结果摘要
print(model.summary())

# 计算95%置信区间
conf_int = model.conf_int(alpha=0.05)
print("95% 置信区间:")
print(conf_int)

# 绘制散点图和回归线
plt.figure(figsize=(8, 6))
sns.scatterplot(data=hi_data, x='Pop', y='Carbon', label='Pop')
plt.plot(hi_data['Pop'], model.predict(X), color='red', label='Regression line (Pop)')
plt.xlabel('Pop')
plt.ylabel('Carbon')
plt.title('Scatter Plot and Regression Line')
plt.legend()
plt.show()

# 提取高收入国家(HI)的2014年的数据
hi_data_2014 = data[(data['Income'] == 'HI') & (data['Time'] ==2014)]
# 剔除含有NaN项的数据
hi_data_2014 = hi_data_2014.dropna(subset=['Carbon', 'Pop'])

print(hi_data_2014)


# 提取预测变量和响应变量
X_new = hi_data_2014[['Pop']]
y_new = hi_data_2014['Carbon']

# 添加截距列
X_new = sm.add_constant(X_new)

# 使用已拟合的模型进行预测
y_pred_new = model.predict(X_new)

# 打印预测结果
print("Predicted Carbon values:", y_pred_new)

```
### Hierarchy Clustering (seed data)
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# 读取CSV文件
data = pd.read_csv(r'C:\Users\29153\Downloads\seed.csv')

# 提取特征列
X = data.iloc[:, :-1].values


# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# 计算距离矩阵
# 使用平均链接法（average linkage）进行层次聚类
# 使用欧氏距离作为距离度量
distance_matrix = pdist(scaled_data, metric='euclidean')
# 进行层次聚类
linkage_matrix = linkage(distance_matrix, method='average')

# 绘制树状图
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.title('Dendrogram of Agglomerative Hierarchical Clustering')
plt.show()
# 在层次聚类中，开始时，每个数据点都被视为一个单独的簇。随着聚类的进行，相似的簇逐渐合并成更大的簇。树状图从底部开始，每个叶节点代表一个初始数据点或簇，然后逐渐向上合并，直到达到一个根节点，表示整个数据集。在合并过程中，树状图的高度表示样本间的距离或相似性。
# 簇的结构: 树状图可以显示数据点如何逐渐合并成簇。簇越靠近树状图的根部，表示越大的簇，而叶节点表示单个数据点或较小的簇。
# 样本相似性: 树状图的高度代表了样本间的距离或相似性。簇内的样本越相似，它们在树状图上就会更早合并，并形成较短的分支。分支越长，表示样本之间的距离较大。
# 聚类的数量: 通过观察树状图的分支和高度，可以帮助我们判断最佳的聚类数。当分支开始快速延长或形成明显的“肘部”时，可能是数据的一个自然分割点。

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# 读取CSV文件
data = pd.read_csv(r'C:\Users\29153\Downloads\seed.csv')

# 提取特征列
X = data.iloc[:, :-1].values

# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# 计算层次聚类
# 使用最近邻（单链接）法来计算簇之间的距离
linkage_matrix = linkage(scaled_data, method='single', metric='euclidean')

# 绘制树状图
dendrogram(linkage_matrix)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# 读取CSV文件
data = pd.read_csv(r'C:\Users\29153\Downloads\seed.csv')

# 提取特征列
X = data.iloc[:, :-1].values

# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# 计算层次聚类
# 使用"complete linkage"方法（完全连接法）来计算簇之间的距离
Z = linkage(scaled_data, method='complete', metric='euclidean')

# 绘制层次聚类的树状图
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Index')
plt.ylabel('Distance')
plt.show()
```

### Decision tree (Classification)
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 加载Iris数据集
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target_names[iris.target]
X = iris.data
y = iris.target

# 查看摘要统计信息
summary = data.describe()

print(summary)

# 创建决策树分类器
clf = DecisionTreeClassifier()

# 训练模型
clf.fit(X, y)

# 绘制决策树的树状图
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names.tolist())
plt.show()

# 对数据集进行预测
X_pred = iris.data
y_true = iris.target

# 进行预测
y_pred = clf.predict(X_pred)

# 导入pandas库，将预测结果转换为DataFrame
y_pred_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_true})

# 计算混淆矩阵
confusion = confusion_matrix(y_true, y_pred)

# 打印混淆矩阵
print("Confusion Matrix:\n", confusion)

# 生成分类报告
class_report = classification_report(y_true, y_pred, target_names=iris.target_names)

# 打印分类报告
print("Classification Report:\n", class_report)
```

### Decision tree (regression)
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the airquality dataset
airquality = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv', index_col=0)

# Check summary statistics
summary = airquality.describe()
print(summary)

# Remove rows with NaN values
airquality = airquality.dropna()

# Separate features and target
X = airquality.drop(columns='Ozone')
y = airquality['Ozone']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# Create a DecisionTreeRegressor with max_depth=4
regressor = DecisionTreeRegressor(max_depth=3)

# Train the model
regressor.fit(X_train, y_train)

# Display the decision tree plot
plt.figure(figsize=(12, 8))
plt.title("Decision Tree Regressor")
plot_tree(regressor, filled=True, feature_names=X.columns.tolist())
plt.show()

# Predict on the test set
y_pred = regressor.predict(X_test)

# Compare actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='gray')  # Diagonal line
plt.xlabel('Actual Ozone')
plt.ylabel('Predicted Ozone')
plt.title('Actual vs Predicted Ozone')
plt.show()

# Calculate and display R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

```


### Random forest (regression)

```python
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the airquality dataset
airquality = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv', index_col=0)

# Check summary statistics
summary = airquality.describe()
print(summary)

# Remove rows with NaN values
airquality = airquality.dropna()

# 数据拆分
X = airquality.drop(columns=['Ozone'])
y = airquality['Ozone']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 定义超参数网格
param_grid = {
    'n_estimators': [100, 500, 800],
    'max_features': [1, 2, 3, 4],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 创建随机森林回归器
rf_regressor = RandomForestRegressor(random_state=1)

# 使用网格搜索寻找最佳超参数
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train, y_train)

# 打印最佳参数
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 使用最佳参数的随机森林回归器进行预测
best_rf_regressor=RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                 max_features=best_params['max_features'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 random_state=1)

# 训练模型
best_rf_regressor.fit(X_train, y_train)

# 进行预测
predictions=best_rf_regressor.predict(X_test)

# 计算R2分数
r2 = r2_score(y_test, predictions)
print("Best R-squared:", r2)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the airquality dataset
airquality = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv', index_col=0)

# Check summary statistics
summary = airquality.describe()
print(summary)

# Remove rows with NaN values
airquality = airquality.dropna()

# 数据拆分
idx = np.random.choice(range(len(airquality)), size=int(len(airquality) * 0.7), replace=False)
train = airquality.iloc[idx]
test = airquality.iloc[np.delete(range(len(airquality)), idx)]

# 创建随机森林回归器
rf_regressor = RandomForestRegressor(n_estimators=500, max_features=2, random_state=1)

# 训练随机森林模型
rf_regressor.fit(train.drop(columns=['Ozone']), train['Ozone'])

# 预测测试集
predictions = rf_regressor.predict(test.drop(columns=['Ozone']))

# 计算R2分数
r2 = r2_score(test['Ozone'], predictions)
print("R-squared:", r2)

# 绘制预测值和真实值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(test['Ozone'], predictions, alpha=0.7)
plt.plot([test['Ozone'].min(), test['Ozone'].max()], [test['Ozone'].min(), test['Ozone'].max()], linestyle='--', color='gray')  # Diagonal line
plt.xlabel('Actual Ozone')
plt.ylabel('Predicted Ozone')
plt.title('Actual vs Predicted Ozone')
plt.show()
```

### Random forest (classification)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_clf = RandomForestClassifier(**best_params, random_state=42)

best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Best Model Accuracy:", accuracy)

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# 设置随机数种子
np.random.seed(1)

# 加载鸢尾花数据集
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target_names[iris.target]
X = iris.data
y = iris.target

# 查看摘要统计信息
summary = data.describe()

print(summary)

# 数据拆分
idx = np.random.choice(range(len(iris.target)), size=int(len(iris.target) * 0.7), replace=False)
train = X[idx]
train_labels = y[idx]
test = np.delete(X, idx, axis=0)
test_labels = np.delete(y, idx, axis=0)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=1000, max_features=3, random_state=1)

# 训练随机森林模型
rf_classifier.fit(train, train_labels)

# 预测测试集
predictions = rf_classifier.predict(test)

# 计算混淆矩阵
confusion = confusion_matrix(test_labels, predictions)

# 打印混淆矩阵
print("Confusion Matrix:\n", confusion)

# 计算分类报告
class_report = classification_report(test_labels, predictions, target_names=iris.target_names)

# 打印分类报告
print("Classification Report:\n", class_report)

# 计算整体准确度
overall_accuracy = accuracy_score(test_labels, predictions)

# 打印整体准确度
print("Overall Accuracy:", overall_accuracy)

```