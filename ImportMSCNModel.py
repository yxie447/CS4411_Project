import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data(filepath):
    """
    Load preprocessed data from a CSV file.

    :param filepath: Path to the preprocessed CSV file
    :return: Pandas DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print("Preprocessed data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        raise


def define_target_variable(df):
    """
    Define the target variable for the model.

    :param df: DataFrame containing the data
    :return: Tuple of (features DataFrame, target Series)
    """
    target = 'movie_age'
    if target not in df.columns:
        print(f"Target variable '{target}' not found in the data.")
        raise KeyError(f"Target variable '{target}' not found.")

    X = df.drop(columns=[target])
    y = df[target]

    # Print feature columns to verify
    print(f"Features columns: {X.columns.tolist()}")

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    :param X: Features DataFrame
    :param y: Target Series
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random seed for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    print(f"Data split into training and testing sets with test size = {test_size}")
    return X_train, X_test, y_train, y_test


def print_target_statistics(y, dataset_name='Dataset'):
    """
    Print statistics of the target variable.

    :param y: Target Series
    :param dataset_name: Name of the dataset (e.g., 'Train', 'Test')
    """
    print(f"\n{dataset_name} Target Variable Statistics:")
    print(y.describe())

    # Check for unique values
    unique_values = y.unique()
    print(f"Unique values in {dataset_name} target: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")


def check_target_distribution(df, target='movie_age'):
    """
    Check the distribution of the target variable.

    :param df: DataFrame
    :param target: Target variable name
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target], bins=50, kde=True)
    plt.title(f'Distribution of {target}')
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.show()

    print(f"Target variable '{target}' statistics:")
    print(df[target].describe())


def check_movie_age_details(df):
    """
    Check the specific value and distribution of movie_age

    :param df: DataFrame
    """
    print("\nCheck the details of 'movie_age'.")
    print(df['movie_age'].describe())

    # View non-zero 'movie_age'
    non_zero_ages = df[df['movie_age'] > 0]['movie_age']
    print(f"\nStatistics for non-zero 'movie_age'.")
    print(non_zero_ages.describe())

    # See the percentage of samples where 'movie_age' is zero
    zero_age_count = len(df[df['movie_age'] == 0])
    total_count = len(df)
    print(f"\nProportion of samples where 'movie_age' is zero: {zero_age_count / total_count * 100:.2f}%")

    # View the top 10 samples where 'movie_age' is not zero
    sample_non_zero = df[df['movie_age'] > 0].head(10)
    print("\nTop 10 samples where 'movie_age' is not zero: ")
    print(sample_non_zero[['release_year_movie', 'release_year_episode', 'movie_age']])


def build_mscn_model(input_dim):
    """
    Build a Multi-Scale Convolutional Network (MSCN) model.

    :param input_dim: Number of input features
    :return: Compiled Keras model
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Use Input layer to avoid warnings
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # 降低学习率以确保稳定

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    print("MSCN model built and compiled.")
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
    """
    Train the MSCN model with early stopping and TensorBoard.

    :param model: Compiled Keras model
    :param X_train: Training features
    :param y_train: Training target
    :param X_val: Validation features
    :param y_val: Validation target
    :param epochs: Maximum number of epochs
    :param batch_size: Batch size for training
    :return: Training history
    """
    # Create TensorBoard callback
    log_dir = "logs/fit/" + pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Early Stopping callback with adjusted patience
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, tensorboard_callback],
        verbose=1
    )
    print("Model training completed.")
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.

    :param model: Trained Keras model
    :param X_test: Testing features
    :param y_test: Testing target
    """
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # Predict and calculate RMSE
    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")


def print_predictions(model, X_test, y_test, num_samples=10):
    """
    Print a few predictions alongside actual values.

    :param model: Trained Keras model
    :param X_test: Testing features
    :param y_test: Testing target
    :param num_samples: Number of samples to display
    """
    y_pred = model.predict(X_test).flatten()
    print("\nSample Predictions vs Actuals:")
    for i in range(num_samples):
        print(f"Predicted: {y_pred[i]:.4f}, Actual: {y_test.iloc[i]:.4f}")


def plot_training_history(history):
    """
    Plot the training and validation loss and MAE over epochs.

    :param history: Keras History object
    """
    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_model_to_file(model):
    """
    Save the trained model to a file.

    :param model: Trained Keras model
    """
    model.save('mscn_model.h5')
    print("Model saved to 'mscn_model.h5'.")


def main():
    # Step 1: Load preprocessed data
    print("Step 1: Loading preprocessed data...")
    df = load_preprocessed_data('preprocessed_data.csv')

    # Step 2: Define target variable
    print("\nStep 2: Defining target variable...")
    X, y = define_target_variable(df)

    # Step 2.1: Check target distribution
    print("\nStep 2.1: Checking target distribution...")
    check_target_distribution(df, target='movie_age')

    # Step 2.2: Check detailed movie_age
    check_movie_age_details(df)

    # Step 3: Split data into training and testing sets
    print("\nStep 3: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Print target statistics
    print_target_statistics(y_train, dataset_name='Train')
    print_target_statistics(y_test, dataset_name='Test')

    # Further split training set into training and validation sets
    print("\nSplitting training data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Step 4: Build the MSCN model
    print("\nStep 4: Building the MSCN model...")
    input_dim = X_train.shape[1]
    model = build_mscn_model(input_dim)

    # Step 5: Train the model
    print("\nStep 5: Training the model...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Step 6: Evaluate the model
    print("\nStep 6: Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Print sample predictions
    print_predictions(model, X_test, y_test, num_samples=10)

    # Step 7: Plot training history
    print("\nStep 7: Plotting training history...")
    plot_training_history(history)

    # Step 8: Save the model (optional)
    save_choice = input("\nDo you want to save the trained model? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_model_to_file(model)
    else:
        print("Model not saved.")

    print("\nMSCN model implementation completed successfully.")


if __name__ == "__main__":
    main()
