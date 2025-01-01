import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from data_cleaner import DataCleaner

def main():
    # 1) Load TRAINING data (with 'Transported' column)
    train_df = pd.read_csv("train1-SpaceshipTitanic.csv")
    X_train = train_df.drop(columns=["Transported"])
    y_train = train_df["Transported"].astype(int)  # or bool

    # 2) Build the pipeline
    pipeline = Pipeline([
        ("cleaner", DataCleaner()),
        ("model",   LogisticRegression())
    ])

    # 3) Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # ================================
    # Evaluate on the SAME TRAIN DATA
    # (Or you could do a validation split)
    # ================================
    train_preds = pipeline.predict(X_train)

    # Compute confusion matrix
    cm = confusion_matrix(y_train, train_preds)
    print("Confusion Matrix on TRAIN data:")
    print(cm)

    # Compute accuracy
    acc = accuracy_score(y_train, train_preds)
    print("Accuracy on TRAIN data:", acc)

    # =====================
    # 4) Load TEST data
    # =====================
    test_df = pd.read_csv("test1-SpaceshipTitanic.csv")

    # Save PassengerId for submission
    passenger_ids = test_df["PassengerId"].copy()

    # 5) Predict on TEST data (no ground truth, so we can't do confusion matrix here)
    test_preds = pipeline.predict(test_df)

    # 6) Create submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Transported": test_preds.astype(bool)  # or keep as 0/1 if Kaggle allows
    })

    # 7) Save to CSV
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv!")

if __name__ == "__main__":
    main()
