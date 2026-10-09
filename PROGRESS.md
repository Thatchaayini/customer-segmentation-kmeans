# Week 1 Progress Report

## Summary

Built a complete customer segmentation pipeline using K-Means with scikit-learn. It generates a synthetic customer dataset, preprocesses and validates it, clusters the customers into segments, and saves the results. 11 automated tests pass.

## Work Completed

| Day | Focus | Outcome |
|---|---|---|
| 1 | Setup and research | Python environment, libraries, GitHub repo |
| 2 | Core concept study | K-Means study, notes, README summary |
| 3 | First implementation | Basic K-Means on sample data |
| 4 | Data setup | 200-customer dataset, preprocessing, validation |
| 5 | Core feature build | Segmentation with edge case handling, 8 unit tests |
| 6 | Integration | End-to-end pipeline (main.py), 3 pipeline tests |

## Code Review of K-Means Implementation

**What works well**
- Pipeline is split into clear modules: data generation, preprocessing, validation, segmentation.
- Edge cases are handled: empty data, invalid K, K larger than the data, empty clusters.
- Results are reproducible (fixed random_state, seeded data generation).
- Tests cover both the clustering logic and the full workflow.

**Improvements identified**
- K is fixed at 3. The Elbow method and silhouette score should be used to choose K from the data.
- The model is only judged by cluster sizes. Evaluation metrics are needed (silhouette, inertia, Davies-Bouldin).
- There are no visualizations yet, so the segments are hard to interpret.
- Data is synthetic. The pipeline should be tested on a real dataset (e.g. Mall Customers).
- The generated file data/customers_segmented.csv could be left out of version control.

## Week 2 Plan: Evaluation Metrics and Visualization

1. Implement the Elbow method (inertia for K = 1 to 10) and plot it.
2. Compute silhouette score and Davies-Bouldin index for different K values.
3. Choose the best K automatically from the metrics.
4. Visualize clusters in 2D using PCA.
5. Plot cluster profiles (average income, frequency, order value per segment).
6. Add tests for the metrics, and update the README with charts and results.