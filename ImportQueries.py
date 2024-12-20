import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Database Connection Configuration
DB_NAME = "project"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create Database Engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')


def execute_query(engine, query, params=None):
    """
    Execute a SQL query and return the result as a Pandas DataFrame.

    :param engine: SQLAlchemy engine object
    :param query: SQL query string
    :param params: Dictionary of parameters for the query
    :return: Pandas DataFrame with query results
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except SQLAlchemyError as e:
        print("Error executing query:")
        print(e)
        return pd.DataFrame()  # Return empty DataFrame on error


def collect_selected_values(engine):
    """
    Define and execute SQL queries to collect selected values from the database.

    :param engine: SQLAlchemy engine object
    :return: Dictionary of DataFrames with query results
    """
    queries = {
        "total_movies": """
            SELECT COUNT(*) AS total_movies
            FROM movies;
        """,
        "total_episodes": """
            SELECT COUNT(*) AS total_episodes
            FROM episodes;
        """,
        "episodes_per_movie": """
            SELECT m.title, COUNT(e.episode_id) AS episode_count
            FROM movies m
            LEFT JOIN episodes e ON m.movie_id = e.movie_id
            GROUP BY m.title
            ORDER BY episode_count DESC;
        """,
        "movies_by_release_year": """
            SELECT release_year, COUNT(*) AS movie_count
            FROM movies
            GROUP BY release_year
            ORDER BY release_year;
        """,
        "episodes_by_release_year": """
            SELECT release_year, COUNT(*) AS episode_count
            FROM episodes
            GROUP BY release_year
            ORDER BY release_year;
        """,
        "top_movies_with_most_episodes": """
            SELECT m.title, COUNT(e.episode_id) AS episode_count
            FROM movies m
            JOIN episodes e ON m.movie_id = e.movie_id
            GROUP BY m.title
            ORDER BY episode_count DESC
            LIMIT 10;
        """,
        "average_episodes_per_movie": """
            SELECT AVG(episode_count) AS average_episodes
            FROM (
                SELECT COUNT(e.episode_id) AS episode_count
                FROM movies m
                LEFT JOIN episodes e ON m.movie_id = e.movie_id
                GROUP BY m.movie_id
            ) subquery;
        """
    }

    results = {}
    for key, query in queries.items():
        print(f"Executing query: {key}")
        df = execute_query(engine, query)
        if not df.empty:
            results[key] = df
            print(f"Query {key} executed successfully.\n")
        else:
            print(f"Query {key} failed or returned no results.\n")
    return results


def main():
    # Step 1: Collect selected values by executing queries
    print("Step 1: Executing queries to collect selected values...\n")
    collected_data = collect_selected_values(engine)

    # Step 2: Display the collected data
    for key, df in collected_data.items():
        print(f"--- {key} ---")
        print(df.to_string(index=False))
        print("\n")

    # Optional: Save the collected data to CSV files for further processing
    save_to_csv = input("Do you want to save the collected data to CSV files? (y/n): ").strip().lower()
    if save_to_csv == 'y':
        for key, df in collected_data.items():
            csv_filename = f"{key}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Saved {key} to {csv_filename}")
        print("All data saved to CSV files successfully.")
    else:
        print("Data not saved to CSV files.")


if __name__ == "__main__":
    main()
