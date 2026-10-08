# customer-segmentation-kmeans
Customer Segmentation using K-Means clustering
## K-Means Algorithm Summary

K-Means is an unsupervised machine learning algorithm that groups data into K clusters based on similarity.

**How it works:**
1. Choose the number of clusters (K).
2. Initialize K centroids (k-means++ is used to pick well-spread starting points).
3. Assign each data point to its nearest centroid.
4. Recalculate each centroid as the mean of its assigned points.
5. Repeat steps 3 and 4 until convergence (centroids stop moving significantly or the maximum iterations are reached).

**Use in this project:** Customers are segmented using features like Annual Income and Spending Score. The Elbow method helps choose the best K.
