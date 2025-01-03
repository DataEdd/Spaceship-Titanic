import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from data_cleaner import DataCleaner

def train_and_save_model(model, model_name, X_train, y_train, X_val, y_val, X_test, passenger_ids):
    # Fit the model pipeline
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    cm = confusion_matrix(y_val, val_preds)
    acc = accuracy_score(y_val, val_preds)

    print(f"\n=== {model_name.upper()} ===")
    print("Confusion Matrix (Validation):")
    print(cm)
    print(f"Validation Accuracy: {acc:.4f}")

    # Predict on test set
    test_preds = model.predict(X_test).astype(bool)
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Transported": test_preds
    })

    # Save submission file
    out_dir = f"submissions/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"{model_name} submission saved to: {out_path}")

def align_columns(reference_df, *other_dfs):
    """Align columns of other_dfs to match reference_df."""
    aligned_dfs = []
    for df in other_dfs:
        for col in reference_df.columns:
            if col not in df.columns:
                df[col] = 0
        for col in df.columns:
            if col not in reference_df.columns:
                df.drop(columns=[col], inplace=True)
        df = df[reference_df.columns]
        aligned_dfs.append(df)
    return aligned_dfs

def main():
    # Load datasets
    train_df = pd.read_csv("train1-SpaceshipTitanic.csv")
    test_df = pd.read_csv("test1-SpaceshipTitanic.csv")
    X_full = train_df.drop(columns=["Transported"])
    y_full = train_df["Transported"].astype(int)
    passenger_ids = test_df["PassengerId"]

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # DataCleaner
    cleaner = DataCleaner()

    # Fit the cleaner only on training data
    cleaner.fit(X_train, y_train)
    X_train_cleaned = cleaner.transform(X_train)
    X_val_cleaned = cleaner.transform(X_val)
    X_test_cleaned = cleaner.transform(test_df)

    # Print transformation steps
    print("\n=== Data Cleaning Transformation Steps ===")
    for step in cleaner.transform_steps_:
        print(step)

    # Align columns using the cleaned training dataset as the reference
    X_val_cleaned, X_test_cleaned = align_columns(X_train_cleaned, X_val_cleaned, X_test_cleaned)

    # Identify numeric and binary columns
    binary_columns = [col for col in X_train_cleaned.columns if set(X_train_cleaned[col].unique()) <= {0, 1}]
    numeric_columns_to_scale = [
        col for col in X_train_cleaned.select_dtypes(include=["float64", "int64"]).columns
        if col not in binary_columns
    ]

    # Scaling: Fit on training set and apply to validation/test sets
    scaler = StandardScaler()
    X_train_scaled_numeric = scaler.fit_transform(X_train_cleaned[numeric_columns_to_scale])
    X_val_scaled_numeric = scaler.transform(X_val_cleaned[numeric_columns_to_scale])
    X_test_scaled_numeric = scaler.transform(X_test_cleaned[numeric_columns_to_scale])

    # Reassemble the datasets
    def combine_scaled_and_binary(scaled_numeric, original_df, binary_cols):
        combined = pd.DataFrame(scaled_numeric, columns=numeric_columns_to_scale)
        combined[binary_cols] = original_df[binary_cols].reset_index(drop=True)
        return combined

    X_train_scaled = combine_scaled_and_binary(X_train_scaled_numeric, X_train_cleaned, binary_columns)
    X_val_scaled = combine_scaled_and_binary(X_val_scaled_numeric, X_val_cleaned, binary_columns)
    X_test_scaled = combine_scaled_and_binary(X_test_scaled_numeric, X_test_cleaned, binary_columns)

    # Define models and pipelines
    models = [
        (Pipeline([("model", LogisticRegression(max_iter=1000))]), "logistic_regression"),
        (Pipeline([("model", SVC(probability=True))]), "svm"),
        (XGBClassifier(eval_metric="logloss"), "xgboost"),
        (Pipeline([("model", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500))]), "neural_network"),
        (CatBoostClassifier(verbose=0), "catboost")  # CatBoost supports string features
    ]

    # Train and save for each model
    for model, model_name in models:
        train_and_save_model(
            model=model,
            model_name=model_name,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            X_test=X_test_scaled,
            passenger_ids=passenger_ids
        )

if __name__ == "__main__":
    # Suppress specific warnings
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main()
