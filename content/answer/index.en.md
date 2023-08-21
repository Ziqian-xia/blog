---
title: K-MEANS ANSWER
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
num_users = 200
num_features = 4
data = np.random.rand(num_users, num_features) * np.array([10, 100, 5, 20]) + np.array([1, 10, 0.1, 5])

# 创建DataFrame
df = pd.DataFrame(data, columns=['网页数', '停留时间', '购物车商品数', '点击广告数'])

# 将数据保存为CSV文件
df.to_csv('user_behavior_data.csv', index=False)
```

## 答案

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 加载模拟的用户行为数据集
df = pd.read_csv('user_behavior_data.csv')

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# 使用不同的聚类数目进行 K-Means
num_clusters_range = range(2, 10)
inertia_values = []

for num_clusters in num_clusters_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data_scaled)
    inertia_values.append(kmeans.inertia_)

# 绘制 Scree Plot
plt.plot(num_clusters_range, inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Scree Plot for K-Means')
plt.show()

# 选择最佳聚类数目
best_num_clusters = np.argmin(np.diff(inertia_values)) + 2
print(f"Best number of clusters: {best_num_clusters}")

# 使用最佳聚类数目进行 K-Means
best_kmeans = KMeans(n_clusters=best_num_clusters, random_state=0)
df['cluster'] = best_kmeans.fit_predict(data_scaled)

# 输出每个群体的统计信息
cluster_stats = df.groupby('cluster').mean()
print(cluster_stats)

```
解释：

加载模拟的用户行为数据集： 我们假设已经有一个名为 user_behavior_data.csv 的数据文件，其中包含了模拟的用户行为数据，包括网页数、停留时间、购物车商品数和点击广告数。

数据标准化： 我们使用 StandardScaler 对数据进行标准化，确保不同特征在相同的尺度上，从而避免某些特征对聚类结果的影响过大。

使用不同的聚类数目进行 K-Means： 我们尝试不同的聚类数目，并计算每个聚类数目对应的簇内方差（inertia）。这些值会被用于绘制 Scree Plot。

绘制 Scree Plot： Scree Plot 是一种通过绘制不同聚类数目与簇内方差之间的关系图来帮助选择合适聚类数目的方法。通常，在聚类数目增加时，簇内方差会逐渐减小。我们通过观察 Scree Plot 来选择一个“拐点”（elbow point），它对应于一个合适的聚类数目。

选择最佳聚类数目： 通过观察 Scree Plot，我们选择拐点对应的聚类数目作为最佳的聚类数目。

使用最佳聚类数目进行 K-Means： 我们使用上一步中选择的最佳聚类数目，应用 K-Means 聚类算法。

输出每个群体的统计信息： 我们输出每个群体的平均值，以便更好地了解不同群体的用户行为特点。
