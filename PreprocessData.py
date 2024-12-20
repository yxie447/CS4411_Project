import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# Database Connection Configuration
DB_NAME = "project"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create Database Engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')


def load_data(engine):
    """
    Load data from the database into Pandas DataFrames.

    :param engine: SQLAlchemy engine object
    :return: Tuple of (movies_df, episodes_df)
    """
    try:
        with engine.connect() as conn:
            movies_df = pd.read_sql(text("SELECT * FROM movies;"), conn)
            episodes_df = pd.read_sql(text("SELECT * FROM episodes;"), conn)
        print("Data loaded successfully from the database.")
        return movies_df, episodes_df
    except SQLAlchemyError as e:
        print("Error loading data from the database:")
        print(e)
        raise


def clean_data(movies_df, episodes_df):
    """
    Clean the loaded data by handling missing values and duplicates.

    :param movies_df: DataFrame containing movies data
    :param episodes_df: DataFrame containing episodes data
    :return: Tuple of cleaned (movies_df, episodes_df)
    """
    # Handle missing values in movies_df
    print("Cleaning movies data...")
    initial_movies_count = len(movies_df)
    movies_df = movies_df.drop_duplicates(subset=['movie_id'])
    movies_df = movies_df.dropna(subset=['title'])  # Assuming title is essential
    movies_df['release_year'] = movies_df['release_year'].fillna(movies_df['release_year'].median())
    movies_df['additional_info'] = movies_df['additional_info'].fillna('')
    final_movies_count = len(movies_df)
    print(f"Movies data cleaned: {initial_movies_count - final_movies_count} duplicates or missing titles removed.")

    # Handle missing values in episodes_df
    print("Cleaning episodes data...")
    initial_episodes_count = len(episodes_df)
    episodes_df = episodes_df.drop_duplicates(subset=['episode_id'])
    episodes_df = episodes_df.dropna(subset=['episode_title'])  # Assuming episode_title is essential
    episodes_df['release_year'] = episodes_df['release_year'].fillna(episodes_df['release_year'].median())
    episodes_df['episode_number'] = episodes_df['episode_number'].fillna('Unknown')
    final_episodes_count = len(episodes_df)
    print(
        f"Episodes data cleaned: {initial_episodes_count - final_episodes_count} duplicates or missing titles removed.")

    return movies_df, episodes_df


def feature_extraction(movies_df, episodes_df):
    """
    Perform feature extraction on the data.

    :param movies_df: Cleaned movies DataFrame
    :param episodes_df: Cleaned episodes DataFrame
    :return: Merged DataFrame with extracted features
    """
    print("Performing feature extraction...")
    # Merge movies and episodes on movie_id
    merged_df = pd.merge(episodes_df, movies_df, on='movie_id', how='left', suffixes=('_episode', '_movie'))

    # Extract season and episode number from episode_number (e.g., 'S01E02')
    merged_df['season'] = merged_df['episode_number'].str.extract(r'S(\d+)E\d+', expand=False)
    merged_df['episode'] = merged_df['episode_number'].str.extract(r'S\d+E(\d+)', expand=False)

    # Convert extracted season and episode to numeric, handle missing values
    merged_df['season'] = pd.to_numeric(merged_df['season'], errors='coerce').fillna(0).astype(int)
    merged_df['episode'] = pd.to_numeric(merged_df['episode'], errors='coerce').fillna(0).astype(int)

    # Calculate movie age at the time of episode release
    merged_df['movie_age'] = merged_df['release_year_episode'] - merged_df['release_year_movie']

    # Handle possible negative ages
    merged_df['movie_age'] = merged_df['movie_age'].apply(lambda x: x if x >= 0 else 0)

    print("Feature extraction completed.")
    return merged_df


def frequency_encoding(df, column):
    """
    Perform frequency encoding on a categorical column.

    :param df: DataFrame
    :param column: Column name to encode
    :return: Series with frequency-encoded values
    """
    freq = df[column].value_counts() / len(df)
    return df[column].map(freq)


def encode_and_scale(merged_df):
    """
    Encode categorical variables and scale numerical features.

    :param merged_df: DataFrame with extracted features
    :return: Preprocessed DataFrame
    """
    print("Encoding categorical variables and scaling numerical features...")

    # Identify columns
    # Instead of OneHotEncoding high-cardinality features, use Frequency Encoding
    categorical_features = ['title', 'episode_title']
    numerical_features = ['release_year_movie', 'release_year_episode', 'season', 'episode', 'movie_age']

    # Apply Frequency Encoding to categorical features
    for col in categorical_features:
        merged_df[col + '_freq_enc'] = frequency_encoding(merged_df, col)

    # Drop original categorical columns after encoding
    merged_df = merged_df.drop(columns=categorical_features)

    # Select numerical features for scaling
    scaler = StandardScaler()
    merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])

    # Optionally, handle other categorical variables or features as needed

    print("Encoding and scaling completed.")
    return merged_df


def save_preprocessed_data(preprocessed_df):
    """
    Save the preprocessed data to a CSV file.

    :param preprocessed_df: Preprocessed DataFrame
    """
    csv_filename = 'preprocessed_data.csv'
    preprocessed_df.to_csv(csv_filename, index=False)
    print(f"Preprocessed data saved to {csv_filename}.")


def main():
    # Step 1: Load data from the database
    print("Step 1: Loading data from the database...")
    movies_df, episodes_df = load_data(engine)

    # Step 2: Clean the data
    print("\nStep 2: Cleaning the data...")
    movies_df, episodes_df = clean_data(movies_df, episodes_df)

    # Step 3: Feature extraction
    print("\nStep 3: Extracting features...")
    merged_df = feature_extraction(movies_df, episodes_df)

    # Step 4: Encode categorical variables and scale numerical features
    print("\nStep 4: Encoding and scaling features...")
    preprocessed_df = encode_and_scale(merged_df)

    # Step 5: Save the preprocessed data
    print("\nStep 5: Saving preprocessed data...")
    save_preprocessed_data(preprocessed_df)

    print("\nFeature extraction and data preprocessing completed successfully.")


if __name__ == "__main__":
    main()
