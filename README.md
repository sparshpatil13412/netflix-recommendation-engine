# Netflix Recommendation Engine

A machine learning project exploring **unsupervised learning with K-Means clustering** on the Netflix Movies and TV Shows dataset.

The project focuses on discovering groups of similar Netflix content using metadata such as content type, genres, ratings, release year, and duration.

> **Current status:** Offline ML project focused on clustering and analysis. No web interface or deployment is currently included.

---

## Overview

The goal of this project is to explore how Netflix titles can be grouped into meaningful clusters using **K-Means**.

The project focuses on:

* Data preprocessing
* Feature engineering
* Categorical encoding
* Feature scaling
* K-Means clustering
* Cluster evaluation
* PCA-based visualization
* Cluster interpretation
* Recommendation

The current implementation is focused on **content clustering**, rather than personalized recommendations based on individual user history.

---

## Dataset

The project uses the **Netflix Movies and TV Shows** dataset from Kaggle by `debayank2024`.

The dataset contains information including:

* Title
* Content type
* Director
* Country
* Release year
* Rating
* Duration
* Genres/categories
* Description

---

## Machine Learning Approach

The general workflow is:

```text
Netflix Dataset
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Categorical Encoding
      ↓
Feature Scaling
      ↓
K-Means Clustering
      ↓
Cluster Evaluation
      ↓
Visualization & Interpretation
      ↓
Recommendation
```

### K-Means

K-Means is used to group titles based on similarity in the processed feature space.

Different values of `K` are tested to investigate how the structure of the clusters changes.

The project does not assume that the mathematically highest-scoring value of `K` is automatically the most useful. Cluster interpretability is also considered.

---

## Feature Engineering

The preprocessing pipeline includes:

* Handling missing values
* Removing non-informative fields
* Separating movie duration from TV-show season count
* Processing categorical variables
* Encoding genres and other categorical information
* Scaling appropriate features

The resulting feature matrix is used as the input for clustering.

---

## Cluster Evaluation

Several clustering evaluation methods are used to compare different values of `K`:

* **Silhouette Score** — measures cluster cohesion and separation.
* **Davies-Bouldin Index** — evaluates similarity between clusters.
* **Calinski-Harabasz Score** — compares between-cluster and within-cluster dispersion.
* **Elbow Method** — examines K-Means inertia as the number of clusters increases.

Using multiple methods provides a more reliable view of the clustering structure than relying on a single metric.

---

## Visualization

Because the processed feature space can contain many dimensions, **PCA (Principal Component Analysis)** is used to create lower-dimensional representations for visualization.

PCA is used primarily for **visualization and dimensionality reduction**.

It is not required for K-Means itself.

The clustering model can operate on the original processed feature space, while PCA can be used separately to visualize the resulting clusters in 2D or 3D.

---

## Cluster Interpretation

After clustering, the groups are analyzed to understand their characteristics.

Examples of questions considered include:

* Which genres are common in a cluster?
* Are Movies or TV Shows more prevalent?
* What types of ratings are represented?
* Are there recognizable differences between clusters?

This helps determine whether the mathematical clusters correspond to meaningful content groups.

---

## Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Plotly**
* **Jupyter / VS Code**
* **Git / GitHub**

---

## Project Structure

```text
netflix-recommendation-engine/
│
├── data/
│   └── netflix_titles.csv
│
├── main.py
│
├── README.md
└── requirements.txt
```

The exact structure may change as the project develops.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sparshpatil13412/netflix-recommendation-engine
cd netflix-recommendation-engine
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Current Scope

### Implemented

* Netflix dataset preprocessing
* Feature engineering
* Categorical encoding
* K-Means clustering
* Multiple cluster evaluation methods
* PCA visualization
* Cluster analysis

### Not currently implemented

* Personalized user recommendations
* User accounts or watch history
* Collaborative filtering
* Web interface
* Flask / Streamlit
* Deployment

The project is currently focused on understanding and developing the **unsupervised learning and clustering pipeline**.

---

## Future Improvements

Possible future directions include:

* Incorporating user preferences
* Comparing K-Means with other clustering algorithms
* Exploring text-based features
* Improving cluster analysis
* Developing a user-facing recommendation system

These are future directions and are not part of the current implementation.

---

## License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for the full license text.

> **Note:** The MIT License applies to the project's source code. The Netflix dataset is not owned by this project and remains subject to the terms and licensing conditions of its original source.

---

## Author

**Sparsh**