import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import psycopg2

# Database Connection Configuration
DB_NAME = "project"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create Database Engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')


def drop_tables(engine):
    """
    Drop existing episodes and movies tables if they exist.
    """
    try:
        with engine.begin() as conn:  # Use transaction to ensure DROP TABLE is committed
            conn.execute(text("DROP TABLE IF EXISTS episodes CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS movies CASCADE;"))
            print("Dropped existing episodes and movies tables (if they existed).")
    except SQLAlchemyError as e:
        print("Error dropping tables:")
        print(e)
        raise


def create_tables(engine):
    """
    Create new movies and episodes tables with appropriate schema.
    """
    try:
        with engine.begin() as conn:
            # Create movies table
            conn.execute(text("""
                CREATE TABLE movies (
                    movie_id INTEGER PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    release_year INTEGER,
                    additional_info TEXT
                );
            """))
            print("Created movies table.")

            # Create episodes table
            conn.execute(text("""
                CREATE TABLE episodes (
                    episode_id INTEGER PRIMARY KEY,
                    movie_id INTEGER REFERENCES movies(movie_id),
                    episode_title TEXT,
                    episode_number VARCHAR(10),
                    release_year INTEGER
                );
            """))
            print("Created episodes table.")
    except SQLAlchemyError as e:
        print("Error creating tables:")
        print(e)
        raise


def check_duplicates(df, column_name):
    """
    Check for duplicates in the specified column of the DataFrame.
    """
    duplicates = df[df.duplicated(subset=[column_name], keep=False)]
    if not duplicates.empty:
        print(f"Duplicate {column_name} found:")
        print(duplicates)
        return True
    else:
        print(f"All {column_name} values are unique.")
        return False


def import_data(engine, movies_csv, episodes_csv):
    """
    Import data from movies.csv and episodes.csv into the database.
    """
    try:
        # Read CSV files
        movies_df = pd.read_csv(movies_csv)
        episodes_df = pd.read_csv(episodes_csv)

        # Check if movie_id in movies.csv is unique
        print("\nChecking for duplicate movie_id in movies.csv:")
        if check_duplicates(movies_df, 'movie_id'):
            print("Please fix duplicate movie_id values in movies.csv and retry.")
            return

        # Check if episode_id in episodes.csv is unique
        print("\nChecking for duplicate episode_id in episodes.csv:")
        if check_duplicates(episodes_df, 'episode_id'):
            print("Please fix duplicate episode_id values in episodes.csv and retry.")
            return

        # Import movies.csv data
        print("\nImporting movies.csv data into movies table...")
        try:
            movies_df.to_sql('movies', engine, if_exists='append', index=False)
            print("Successfully imported movies.csv data.")
        except SQLAlchemyError as e:
            print("Error importing movies.csv:")
            print(e)
            raise

        # Import episodes.csv data
        print("\nImporting episodes.csv data into episodes table...")
        try:
            episodes_df.to_sql('episodes', engine, if_exists='append', index=False)
            print("Successfully imported episodes.csv data.")
        except SQLAlchemyError as e:
            print("Error importing episodes.csv:")
            print(e)
            raise

    except Exception as e:
        print("An error occurred during data import:")
        print(e)
        raise


def verify_import(engine):
    """
    Verify that the data has been correctly imported into the database.
    """
    try:
        with engine.connect() as conn:
            # Get count of records in movies table
            result = conn.execute(text("SELECT COUNT(*) FROM movies;"))
            count_movies = result.fetchone()[0]
            print(f"\nNumber of records in movies table: {count_movies}")

            # Get count of records in episodes table
            result = conn.execute(text("SELECT COUNT(*) FROM episodes;"))
            count_episodes = result.fetchone()[0]
            print(f"Number of records in episodes table: {count_episodes}")

            # Display first 5 records from movies table
            print("\nFirst 5 records in movies table:")
            result = conn.execute(text("SELECT * FROM movies LIMIT 5;"))
            for row in result.fetchall():
                print(row)

            # Display first 5 records from episodes table
            print("\nFirst 5 records in episodes table:")
            result = conn.execute(text("SELECT * FROM episodes LIMIT 5;"))
            for row in result.fetchall():
                print(row)

            # Check if all movie_id in episodes exist in movies table
            print("\nVerifying that all movie_id in episodes exist in movies table:")
            result = conn.execute(text("""
                SELECT e.movie_id
                FROM episodes e
                LEFT JOIN movies m ON e.movie_id = m.movie_id
                WHERE m.movie_id IS NULL;
            """))
            missing_movie_ids = result.fetchall()
            if missing_movie_ids:
                print("The following movie_id values in episodes do not exist in movies table:")
                for row in missing_movie_ids:
                    print(row[0])
            else:
                print("All movie_id values in episodes exist in movies table.")
    except SQLAlchemyError as e:
        print("Error verifying import:")
        print(e)
        raise


def main():
    # Define CSV file paths
    movies_csv = 'movies.csv'
    episodes_csv = 'episodes.csv'

    # Step 1: Drop existing tables if they exist
    print("Step 1: Dropping existing tables (if any)...")
    drop_tables(engine)

    # Step 2: Create new tables
    print("\nStep 2: Creating new tables...")
    create_tables(engine)

    # Step 3: Import data
    print("\nStep 3: Importing data...")
    import_data(engine, movies_csv, episodes_csv)

    # Step 4: Verify import results
    print("\nStep 4: Verifying import results...")
    verify_import(engine)

    print("\nData import process completed successfully.")


if __name__ == "__main__":
    main()
