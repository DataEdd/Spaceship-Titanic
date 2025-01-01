# train_and_predict.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Import the DataCleaner from the new file
from data_cleaner import DataCleaner

if __name__ == "__main__":
    # 1) Load TRAINING data
    train_df = pd.read_csv("train1-SpaceshipTitanic.csv")  # has 'Transported'
    X_train = train_df.drop(columns=["Transported"])
    y_train = train_df["Transported"].astype(int)

    # 2) Build the pipeline
    pipeline = Pipeline([
        ("cleaner", DataCleaner()),
        ("model",   LogisticRegression())
    ])

    # 3) Fit the pipeline on training data
    pipeline.fit(X_train, y_train)

    # 4) Load the TEST data
    test_df = pd.read_csv("test1-SpaceshipTitanic.csv")
    
    # Preserve PassengerId to rebuild final submission
    passenger_ids = test_df["PassengerId"].copy()

    # 5) Predict
    predictions = pipeline.predict(test_df)  # Usually an array of 0/1 or bool

    # 6) Create submission DataFrame
    predictions_bool = predictions.astype(bool)
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Transported": predictions_bool
    })

    # 7) Save to CSV
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv!")
