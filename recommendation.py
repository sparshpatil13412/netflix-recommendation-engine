from main import std_scaler, one_hot_encoder, multi_label_binarizer, kmeans_final, kmeans_c0, kmeans_c1, kmeans_c2, kmeans_c3, df
from datetime import datetime
import pandas as pd
import numpy as np

##RECOMMENDATION TO USER
def recommend(title, title_type, genres, rating, release_year, duration):
    title = title.strip()
    title_type = title_type.strip().title()
    if title_type not in ["Movie", "Tv Show"]:
        print("Invalid type")

    if title_type == "Tv Show":
        title_type = "TV Show"

    genres = [
        genre.strip()
        for genre in genres.split(",")
        if genre.strip()
    ]

    rating = rating.strip().upper()
    valid_ratings = df["rating"].dropna().unique()

    if rating not in valid_ratings:
        raise ValueError("Rating not found in dataset")

    release_year = int(release_year)
    current_year = datetime.now().year

    if release_year < 1900 or release_year > current_year:
        raise ValueError("Invalid release year")

    movie_duration = 0
    series_duration = 0
    if title_type == 'Movie':
        movie_duration = duration
        series_duration = 0
    else:
        movie_duration = 0
        series_duration = duration

    #Creating Dataframe as one_hot_encoded was fitted on DataFrame
    input_categorical = pd.DataFrame({
        'type': [title_type],
        'rating': [rating]
    })
    #Creating Dataframe as std_scaler was fitted on DataFrame
    scaled_features = pd.DataFrame({
        'movie_duration': [movie_duration],
        'series_duration': [series_duration],
        'release_year': [release_year]
    })

    one_hot_encoded = one_hot_encoder.transform(input_categorical)
    genre_encoded = multi_label_binarizer.transform([genres])#creating list as multi_label_binarizer was fitted on list
    scaled_features = std_scaler.transform(scaled_features)

    prediction_features = np.hstack((scaled_features, one_hot_encoded, genre_encoded))
    main_cluster = kmeans_final.predict(prediction_features)[0]#predicting main cluster

    if main_cluster == 0:
        sub_cluster = kmeans_c0.predict(prediction_features)[0]#predicting sub-cluster
    elif main_cluster == 1:
        sub_cluster = kmeans_c1.predict(prediction_features)[0]#predicting sub-cluster
    elif main_cluster == 2:
        sub_cluster = kmeans_c2.predict(prediction_features)[0]#predicting sub-cluster
    elif main_cluster == 3:
        sub_cluster = kmeans_c3.predict(prediction_features)[0]#predicting sub-cluster

    recommend_titles = df[
        (df['cluster'] == main_cluster) &
        (df['sub_cluster'] == sub_cluster)
        ]
    recommend_titles = recommend_titles.sample(10)

    print(f"Because you watched {title}:\n{recommend_titles}")

if __name__ == '__main__':
    title = input("Enter the title:")
    title_type = input("Enter whether the title is Movie or TV show:")
    genres = input("Enter all  the genres of the title:")
    rating = input("Enter the age ratings of the title:")
    release_year = input("Enter the release year of the title:")
    if title_type.strip().lower() == 'movie':
        duration = input("Enter the duration of the title in minutes(don't put the unit mins in front of the answer):")
    else:
        duration = input("Enter the no. of seasons of the title(only the no.):")

    recommend(title, title_type, genres, rating, release_year, duration)
