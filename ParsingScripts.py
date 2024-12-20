import re
import pandas as pd
import chardet

# Define the file path
file_path = '../movies.list'

# Detecting file encoding
with open(file_path, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    print(f"Detected encoding: {encoding} with confidence {confidence}")

# Initializing List Storage Data
movies = []
episodes = []

# regular expression pattern
movie_pattern = re.compile(r'^"(.+?)"\s+\((\d{4})\)(?:\s+\{(.+?)\})?\s+(\d{4}(?:-\?{4})?)?$')

# Read and parse the file
with open(file_path, 'r', encoding=encoding, errors='replace') as f:
    for line in f:
        line = line.strip()
        # Skip irrelevant lines
        if not line or line.startswith('CRC') or line.startswith('Copyright') or \
                line.startswith('http') or line.startswith('MOVIES LIST') or \
                line.startswith('=') or line.startswith('-'):
            continue

        match = movie_pattern.match(line)
        if match:
            title = match.group(1)
            release_year = match.group(2)
            episode_info = match.group(3)
            additional_year_info = match.group(4)

            movie_id = len(movies) + 1
            movies.append({
                'movie_id': movie_id,
                'title': title,
                'release_year': release_year,
                'additional_info': additional_year_info if additional_year_info else ''
            })

            if episode_info:
                # Further parsing of episode information
                episode_match = re.match(r'(.+?)\s+\(#(\d+\.\d+)\)', episode_info)
                if episode_match:
                    episode_title = episode_match.group(1)
                    episode_number = episode_match.group(2)
                    episodes.append({
                        'episode_id': len(episodes) + 1,
                        'movie_id': movie_id,
                        'episode_title': episode_title,
                        'episode_number': episode_number,
                        'release_year': release_year
                    })

# Convert to DataFrame
movies_df = pd.DataFrame(movies)
episodes_df = pd.DataFrame(episodes)

# Save as CSV
movies_df.to_csv('movies.csv', index=False)
episodes_df.to_csv('episodes.csv', index=False)

print(movies_df.head())
print(episodes_df.head())
