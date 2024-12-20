import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(engine):
    try:
        with engine.connect() as conn:
            # Make sure the query contains the necessary columns
            movies_df = pd.read_sql(text("SELECT movie_id, release_year, title FROM movies;"), conn)
            episodes_df = pd.read_sql(
                text("SELECT episode_id, movie_id, release_year, episode_title, episode_number FROM episodes;"), conn)
        print("\nMovies DataFrame Sample:")
        print(movies_df[['movie_id', 'release_year', 'title']].head())
        print("\nEpisodes DataFrame Sample:")
        print(episodes_df[['episode_id', 'movie_id', 'release_year', 'episode_title', 'episode_number']].head())
        print("Data loaded successfully from the database.")
        return movies_df, episodes_df
    except SQLAlchemyError as e:
        print("Error loading data from the database:")
        print(e)
        raise


def clean_data(movies_df, episodes_df):
    # Drop duplicates and handle missing values
    print("Cleaning movies data...")
    initial_movies_count = len(movies_df)
    movies_df = movies_df.drop_duplicates(subset=['movie_id'])
    movies_df = movies_df.dropna(subset=['title'])  # Assuming title is essential
    movies_df['release_year'] = movies_df['release_year'].fillna(movies_df['release_year'].median())
    final_movies_count = len(movies_df)
    print(f"Movies data cleaned: {initial_movies_count - final_movies_count} duplicates or missing titles removed.")

    print("Cleaning episodes data...")
    initial_episodes_count = len(episodes_df)
    episodes_df = episodes_df.drop_duplicates(subset=['episode_id'])
    episodes_df = episodes_df.dropna(
        subset=['episode_title', 'episode_number'])  # Assuming episode_title and episode_number are essential
    episodes_df['release_year'] = episodes_df['release_year'].fillna(episodes_df['release_year'].median())
    final_episodes_count = len(episodes_df)
    print(
        f"Episodes data cleaned: {initial_episodes_count - final_episodes_count} duplicates or missing titles removed.")

    return movies_df, episodes_df


def feature_extraction(movies_df, episodes_df):
    print("Performing feature extraction...")
    # Merge movies and episodes on movie_id, including 'title' and 'episode_title'
    merged_df = pd.merge(
        episodes_df,
        movies_df[['movie_id', 'release_year', 'title']],  # Ensure 'title' is included
        on='movie_id',
        how='left',
        suffixes=('_episode', '_movie')
    )

    # Print merged data sample
    print("\nMerged DataFrame Sample:")
    print(merged_df[['movie_id', 'release_year_episode', 'release_year_movie', 'title', 'episode_title']].head())

    if 'episode_number' in merged_df.columns:
        # Extract season and episode number from episode_number (e.g., 'S01E02')
        merged_df['season'] = merged_df['episode_number'].str.extract(r'S(\d+)E\d+', expand=False)
        merged_df['episode'] = merged_df['episode_number'].str.extract(r'S\d+E(\d+)', expand=False)

        # Convert extracted season and episode to numeric, handle missing values
        merged_df['season'] = pd.to_numeric(merged_df['season'], errors='coerce').fillna(0).astype(int)
        merged_df['episode'] = pd.to_numeric(merged_df['episode'], errors='coerce').fillna(0).astype(int)
    else:
        print("Warning: 'episode_number' column not found. Skipping season and episode extraction.")
        merged_df['season'] = 0
        merged_df['episode'] = 0

    # Calculate movie age at the time of episode release
    merged_df['movie_age'] = merged_df['release_year_episode'] - merged_df['release_year_movie']

    # Handle possible negative ages
    merged_df['movie_age'] = merged_df['movie_age'].apply(lambda x: x if x >= 0 else 0)

    # Check release_years
    invalid_movie_age = merged_df[merged_df['movie_age'] < 0]
    if len(invalid_movie_age) > 0:
        print(
            f"\nFound {len(invalid_movie_age)} records where 'release_year_episode' < 'release_year_movie'. Setting 'movie_age' to 0.")
        merged_df.loc[merged_df['movie_age'] < 0, 'movie_age'] = 0
    else:
        print("\nAll 'release_year_episode' are greater than or equal to 'release_year_movie'.")

    # Print 'movie_age' statistics
    print("\n'movie_age' statistics after calculation:")
    print(merged_df['movie_age'].describe())

    # Print some non-zero 'movie_age' samples
    non_zero_ages = merged_df[merged_df['movie_age'] > 0]
    print("\nTop 10 samples where 'movie_age' is not zero:")
    print(non_zero_ages[['release_year_movie', 'release_year_episode', 'movie_age']].head())

    # Visualize 'movie_age' distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(non_zero_ages['movie_age'], bins=50, kde=True)
    plt.title('Distribution of movie_age (Non-zero)')
    plt.xlabel('movie_age')
    plt.ylabel('Frequency')
    plt.show()

    print("Feature extraction completed.")
    return merged_df


def frequency_encoding(df, column):
    freq = df[column].value_counts() / len(df)
    return df[column].map(freq)


def encode_and_scale(merged_df):
    print("Encoding categorical variables and scaling numerical features...")

    # Identify columns
    categorical_features = ['title', 'episode_title']
    numerical_features = ['release_year_movie', 'release_year_episode', 'season', 'episode']  # Exclude 'movie_age'

    # Apply Frequency Encoding to categorical features
    for col in categorical_features:
        if col in merged_df.columns:
            merged_df[col + '_freq_enc'] = frequency_encoding(merged_df, col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    # Drop original categorical columns after encoding
    merged_df = merged_df.drop(columns=[col for col in categorical_features if col in merged_df.columns])

    # Select numerical features for scaling
    scaler = StandardScaler()
    merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])

    print("Encoding and scaling completed.")
    return merged_df


def save_preprocessed_data(preprocessed_df):
    csv_filename = 'preprocessed_data.csv'
    preprocessed_df.to_csv(csv_filename, index=False)
    print(f"Preprocessed data saved to {csv_filename}.")


def main():
    DB_USER = "postgres"
    DB_PASSWORD = "postgres"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "project"

    # Create Database Engine
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # Step 1: Load data from database
    print("Step 1: Loading data from the database...")
    movies_df, episodes_df = load_data(engine)

    # Step 2: Clean the data
    print("\nStep 2: Cleaning the data...")
    movies_df, episodes_df = clean_data(movies_df, episodes_df)

    # Step 3: Feature extraction
    print("\nStep 3: Extracting features...")
    merged_df = feature_extraction(movies_df, episodes_df)

    # Step 4: Encode and scale
    print("\nStep 4: Encoding and scaling features...")
    preprocessed_df = encode_and_scale(merged_df)

    # Step 5: Save preprocessed data
    print("\nStep 5: Saving preprocessed data...")
    save_preprocessed_data(preprocessed_df)

    print("\nData preprocessing completed successfully.")


if __name__ == "__main__":
    main()
