"""
The main goal of this project is to create a Netflix recommendation engine that can recommend movies and TV shows to users based on their preferences.
The project uses K-Means clustering to group similar movies and TV shows together based on their genres, ratings, and other features.
The optimal number of clusters is determined using various methods such as silhouette score, Davies-Bouldin score, Calinski-Harabasz score, and the elbow method.
The final clusters are visualized using PCA to reduce the dimensionality of the data to 3D for better understanding of the clusters.
The final clusters will be then differentiated into more clusters creating sub-clusters to recommend movies and TV shows to users based on their preferences in detail.
The following libraries are used in this project:
    1. pandas: for data manipulation and analysis
    2. numpy: for numerical computations
    3. matplotlib: for data visualization
    4. sklearn: for machine learning algorithms and metrics
    5. plotly: for interactive data visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

##LOADING THE DATASET
csv_path = 'data//netflix_titles.csv'
df =  pd.read_csv(csv_path)

print(df.head(2)) #checking if the dataset has been loaded properly and correctly

##REMOVING UNNECESSARY COLUMNS
df.drop(['show_id','date_added', 'cast', 'description', 'country', 'director'],axis=1,inplace=True) # Removing columns not required for genre-based K-Means clustering
df['movie_duration'] = 0
df['series_duration'] = 0
for value in df['type'].unique():
    if value == "Movie":
        df.loc[df['type'] == "Movie", 'movie_duration'] = df.loc[df['type'] == "Movie", 'duration'].str.replace(' min','').astype(float)
        df.loc[df['type'] == "Movie", 'series_duration'] = 0
    elif value == "TV Show":
        df.loc[df['type'] == "TV Show", 'series_duration'] = df.loc[df['type'] == "TV Show", 'duration'].str.replace(' Seasons','').str.replace(' Season', '').astype(float)
        df.loc[df['type'] == "TV Show", 'movie_duration'] = 0
    else:
        print("Unknown type found in the dataset")

df = df.drop(['duration'],axis=1) #dropping the duration column as it has been divided into two columns


##DATA CLEANING
print("Info about the dataset:")
df.info()#info about the dataset

print("Null values in each column:\n",df.isnull().sum())#checking for null values

df['movie_duration'] = df['movie_duration'].fillna(df['movie_duration'].median()) #filling null values in movie_duration with the median value
df['rating'] = df['rating'].fillna(df['rating'].mode()[0]) #filling null values in rating with the mode value
print("Null values in each column after filling:\n",df.isnull().sum())#checking for null values after filling

## SCALING THE DATA
"""
The following three methods will be used to scale the data:
    1. StandardScaler for scaling numerical features
    2. OneHotEncoder for encoding categorical features
    3. MultiLabelBinarizer for encoding multi-label categorical features
"""
std_scaler = StandardScaler()
one_hot_encoder = OneHotEncoder(sparse_output=False)
multi_label_binarizer = MultiLabelBinarizer()

# Encoding genres/categories
genre_encoded = multi_label_binarizer.fit_transform(
    df['listed_in'].str.split(', ')
)

# Encoding type and rating
one_hot_encoded = one_hot_encoder.fit_transform(
    df[['type', 'rating']]
)

# Scaling numerical features
scaled_features = std_scaler.fit_transform(
    df[['movie_duration', 'series_duration', 'release_year']]
)

##COMBINING ALL FEATURES INTO A SINGLE DATAFRAME
features = np.hstack((scaled_features, one_hot_encoded, genre_encoded))

scaled_df = pd.DataFrame(features)
print("Final DataFrame:", scaled_df.head(2)) #checking the final dataframe after scaling and encoding

##K-MEANS CLUSTERING
"""
The following four methods will be used to find the optimal number of clusters for k-means clustering:
    1.Silhouette Score
    2.Davies-Bouldin Score
    3.Calinski-Harabasz Score
    4.Elbow Method
"""
range_n_clusters = range(2, 21) #setting the range of clusters to be tested
elbow_scores = []
results = []
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_df)
    elbow_scores.append(kmeans.inertia_)
    silhouette = silhouette_score(scaled_df, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(scaled_df, kmeans.labels_)
    calinski_harabasz = calinski_harabasz_score(scaled_df, kmeans.labels_)
    print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette}")
    print(f"For n_clusters = {n_clusters}, the Davies-Bouldin score is {davies_bouldin}")
    print(f"For n_clusters = {n_clusters}, the Calinski-Harabasz score is {calinski_harabasz}\n\n")
    results.append([n_clusters, silhouette, davies_bouldin, calinski_harabasz])

#checking the results dataframe to see if the silhouette score is winning in only terms of the mathematical differnece
results_df = pd.DataFrame(results, columns=['n_clusters', 'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score'])
print("Results DataFrame:\n", results_df.sort_values('silhouette_score', ascending=False).head(10))#finding the top 10 silhouette scores to find the optimal number of clusters

#plotting the elbow method graph to find the optimal number of clusters
plt.figure(figsize=(14, 7))
plt.plot(range_n_clusters, elbow_scores, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

"""
After analyzing the silhouette score, Davies-Bouldin score, Calinski-Harabasz score, and the elbow the method it was concluded that the optimal number of clusters for the project was one of the following: 2, 4, or 10.

To select the best no of clusters from the following three options, the methods used are as follows:
    1. Genre distribution analysis for each cluster
    2. Cluster visualization using PCA to reduce the dimensionality of the data to 3D for better visualization and understanding of the clusters.
"""

#plotting n_clusters = 2, 4, 10 according to the silhouette score, davies bouldin score and elbow method
def plot_cluster_genres(df, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_df)

    df_plot = df.copy()
    df_plot['cluster'] = labels

    fig, axes = plt.subplots(
        k, 1,
        figsize=(14, 3.5 * k)
    )

    if k == 1:
        axes = [axes]

    for cluster, ax in enumerate(axes):

        genres = (
            df_plot[df_plot['cluster'] == cluster]['listed_in']
            .str.split(', ')
            .explode()
            .value_counts()
            .head(8)
            .sort_values()
        )

        genres.plot(
            kind='barh',
            ax=ax
        )

        ax.set_title(
            f'Cluster {cluster}',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Number of Titles', fontsize=11)
        ax.set_ylabel('Genre', fontsize=11)

        ax.tick_params(
            axis='y',
            labelsize=10
        )

    plt.tight_layout(h_pad=2)
    plt.show()

plot_cluster_genres(df, 2)
plot_cluster_genres(df, 4)
plot_cluster_genres(df, 10)

pca = PCA(n_components=3, random_state=42)#using PCA to reduce dimesionality to 3 for visualization purposes
X_pca = pca.fit_transform(scaled_df)

def plot_clusters(k):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(scaled_df)

    # Transform cluster centers into PCA space
    centers_pca = pca.transform(kmeans.cluster_centers_)

    # Create DataFrame for Plotly
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Cluster': labels.astype(str)
    })

    # Plot all data points
    fig = px.scatter_3d(
        plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        title=f'K-Means Clusters (K={k})',
        opacity=0.6
    )

    # Add cluster centroids
    fig.add_trace(
        go.Scatter3d(
            x=centers_pca[:, 0],
            y=centers_pca[:, 1],
            z=centers_pca[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color='black',
                symbol='diamond'
            ),
            name='Centroids'
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3'
        )
    )

    fig.show()

plot_clusters(2)
plot_clusters(4)
plot_clusters(10)

"""
After the methods were applied it was concluded:
    1. When n_clusters was taken as k-means was diividing the dataset into two major groups-movies and web series as was evident from the genre distribution analysis and the PCA visualization. This was not a good choice as it was not providing any new insights into the dataset.
    2. When n_clusters was taken as 4, the dataset was divided into four clusters with each cluster having a unique genre distribution and the PCA visualization also showed that the clusters were well separated and detailed while still being distinct. This was a good choice as it provided new insights into the dataset.
    3. When n_clusters was taken as 10, the dataset was divided into ten clusters providuing fine details about the dataset revealing new patterns. But PCA visualization showed that the cluster were not distinct and there was a lot of overlap between the clusters.

After analyzing the results of the three methods it was concluded that the best choice for n_clusters was 4 due to its detailing and fine separation of the clusters wihtout any overlap. 
"""

#using k means with n_clusters = 4 to create the final model
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(scaled_df)
df['cluster'] = final_labels
scaled_df['cluster'] = final_labels
print("Final KMeans model created with n_clusters = 4")
print("Final dataframe after clustering:", df.head(10))
print("Final dataframe with cluster names:\n", df['cluster'].head(10))

##USING K-MEANS TO DIFFERENTIATE THE FOUR CLUSTERS INTO SUB-CLUSTERS FOR FINER DETAILING
"""
Making dataframes of the four clusters got from K-Means.
The following four clusters will be separated into sub-clusters for finer detailing and better recommendations.
"""
df_c0 = scaled_df[scaled_df['cluster']==0]#creating dataframe with titles in cluster 0
print("Titles in cluster 1:",df_c0.head(2))
df_c0 = df_c0.drop(columns=['cluster'])

df_c1 = scaled_df[scaled_df['cluster']==1]#creating dataframe with titles in cluster 1
print("Titles in cluster 2:",df_c1.head(2))
df_c1 = df_c1.drop(columns=['cluster'])

df_c2 = scaled_df[scaled_df['cluster']==2]#creating dataframe with titles in cluster 2
print("Titles in cluster 3:",df_c2.head(2))
df_c2 = df_c2.drop(columns=['cluster'])

df_c3 = scaled_df[scaled_df['cluster']==3]#creating dataframe with titles in cluster 3
print("Titles in cluster 4:",df_c3.head(2))
df_c3 = df_c3.drop(columns=['cluster'])

"""
Using the cluster dataframes, sub-clusters can be made.
The optimum number of clusters in existing clusters to make sub-clusters, the following parameters will be taken into consideration:
    1.Elbow Method
    2.Silhouette Score
    3.Davies-Bouldin Score
    4.Calinski-Harabsz Score
"""

def optimum_clusters(df, df_name):
    range_n_clusters = range(2, 21) #setting the range of clusters to be tested
    elbow_scores = []
    results = []
    for n_clusters in range_n_clusters:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        kmeans.fit(df)
        elbow_scores.append([n_clusters, kmeans.inertia_])
        silhouette = silhouette_score(df, kmeans.labels_)
        davies_bouldin = davies_bouldin_score(df, kmeans.labels_)
        calinski_harabasz = calinski_harabasz_score(df, kmeans.labels_)
        print(f"For n_clusters = {n_clusters} in {df_name}, the silhouette score is {silhouette}")
        print(f"For n_clusters = {n_clusters} in {df_name}, the Davies-Bouldin score is {davies_bouldin}")
        print(f"For n_clusters = {n_clusters} in {df_name}, the Calinski-Harabasz score is {calinski_harabasz}\n\n")
        results.append([n_clusters, silhouette, davies_bouldin, calinski_harabasz])

    #checking the results dataframe to see if the silhouette score is winning in only terms of the mathematical differnece
    results_df = pd.DataFrame(results, columns=['n_clusters', 'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score'])
    print(f"Results DataFrame of {df_name}:\n", results_df.sort_values('silhouette_score', ascending=False).head(5))#finding the top 5 silhouette scores to find the optimal number of clusters
    elbow_scores_df = pd.DataFrame(elbow_scores, columns=['n_clusters', 'inertia'])
    print(f"Interias of the the sub-clusters of the cluster {df_name}:\n", elbow_scores_df.head(20))

    #plotting the elbow method graph to find the optimal number of clusters
    plt.figure(figsize=(14, 7))
    plt.plot(elbow_scores_df['n_clusters'], elbow_scores_df['inertia'], marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    return results_df, elbow_scores_df

results_c0, elbow_scores_c0 = optimum_clusters(
    df_c0,
    "df_c0"
    )
results_c1, elbow_scores_c1 = optimum_clusters(
    df_c1,
    "df_c1"
    )
results_c2, elbow_scores_c2 = optimum_clusters(
    df_c2,
    "df_c2"
    )
results_c3, elbow_scores_c3 = optimum_clusters(
    df_c3,
    "df_c3"
    )

"""
After viewing the silhouette scores, interias, davies-bouldin score and calinski-harabasz score, the optimum number of clusters are chosen:
    1.df_c0:2
    2.df_c1:2
    3.df_c2:19
    4.df_c3:19
"""

def clustering(df, n_clusters):
    k_means = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    final_labels = k_means.fit_predict(df)
    df['cluster'] = final_labels

    return df, k_means

df_c0, kmeans_c0 = clustering(df_c0, 2)
df_c1, kmeans_c1 = clustering(df_c1, 2)
df_c2, kmeans_c2= clustering(df_c2, 19)
df_c3, kmeans_c3 = clustering(df_c3, 19)

#adding the sub-clusters to the original DataFrame
df.loc[df_c0.index, 'sub_cluster'] = df_c0['cluster']
df.loc[df_c1.index, 'sub_cluster'] = df_c1['cluster']
df.loc[df_c2.index, 'sub_cluster'] = df_c2['cluster']
df.loc[df_c3.index, 'sub_cluster'] = df_c3['cluster']

print(df.head(2))#checking if the sub-clusters are added
