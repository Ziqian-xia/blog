---
title: K-MEANS习题答案
author: Ziqian Xia
date: '2023-08-21'
slug: kmeans
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2023-08-21T12:26:13+08:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


## 源数据生成

```python
import numpy as np
import pandas as pd

# 生成模拟数据
np.random.seed(42)
num_users = 300
num_features = 4

# 为每个簇指定均值和标准差
cluster_params = [
    {'mean': [3, 50, 2, 10], 'std': [1, 10, 0.5, 5]},   # Cluster 0
    {'mean': [7, 70, 4, 30], 'std': [1, 8, 0.8, 6]},    # Cluster 1
    {'mean': [5, 60, 3, 20], 'std': [0.8, 7, 0.7, 5]}   # Cluster 2
]

cluster_labels = np.random.choice([0, 1, 2], size=num_users)
data = np.zeros((num_users, num_features))

for i, label in enumerate(cluster_labels):
    cluster_mean = cluster_params[label]['mean']
    cluster_std = cluster_params[label]['std']
    data[i] = np.random.normal(cluster_mean, cluster_std)

# 创建DataFrame
df = pd.DataFrame(data, columns=['webs', 'time', 'cart', 'click'])

df.to_csv('user_behavior_data.csv', index=False)
```

## 答案

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the user behavior data
df = pd.read_csv('user_behavior_data.csv')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Scree Plot for Optimal Number of Clusters
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

# From the scree plot, identify the “elbow” point where the decrease in the sum of squared distances starts to slow down.
# This point suggests the optimal number of clusters.

# Perform K-Means Clustering
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

df_with_clusters = df.copy()
df_with_clusters['Cluster'] = clusters

# Visualize the Clusters
plt.scatter(df_with_clusters['time'], df_with_clusters['cart'],
            c=df_with_clusters['Cluster'], cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Cart Items')
plt.title('User Behavior Clusters')
plt.show()

```
解释：

加载模拟的用户行为数据集： 我们假设已经有一个名为 user_behavior_data.csv 的数据文件，其中包含了模拟的用户行为数据，包括网页数、停留时间、购物车商品数和点击广告数。

数据标准化： 我们使用 StandardScaler 对数据进行标准化，确保不同特征在相同的尺度上，从而避免某些特征对聚类结果的影响过大。

使用不同的聚类数目进行 K-Means： 我们尝试不同的聚类数目，并计算每个聚类数目对应的簇内方差（inertia）。这些值会被用于绘制 Scree Plot。

绘制 Scree Plot： Scree Plot 是一种通过绘制不同聚类数目与簇内方差之间的关系图来帮助选择合适聚类数目的方法。通常，在聚类数目增加时，簇内方差会逐渐减小。我们通过观察 Scree Plot 来选择一个“拐点”（elbow point），它对应于一个合适的聚类数目。

选择最佳聚类数目： 通过观察 Scree Plot，我们选择拐点对应的聚类数目作为最佳的聚类数目。

使用最佳聚类数目进行 K-Means： 我们使用上一步中选择的最佳聚类数目，应用 K-Means 聚类算法。

输出每个群体的统计信息： 我们输出每个群体的平均值，以便更好地了解不同群体的用户行为特点。
