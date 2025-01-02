SPACESHIP TITANIC PROJECT
===============================================================================

This repository was created for the Spaceship Titanic challenge on Kaggle. 
The objective is to predict which passengers were transported to an alternate 
dimension following the spaceship’s collision with a cosmic anomaly.

-------------------------------------------------------------------------------
COMPETITION & DATASET
-------------------------------------------------------------------------------
Competition Link: https://www.kaggle.com/competitions/spaceship-titanic  
Dataset Source:  https://www.kaggle.com/competitions/spaceship-titanic/data

The training dataset contains approximately 8,700 rows and includes the 
"Transported" label (True/False). The test dataset contains about 4,300 rows 
without the label, and the task is to predict it. The evaluation metric for the 
competition is classification accuracy.

-------------------------------------------------------------------------------
PROJECT FILES
-------------------------------------------------------------------------------
The repository is organized as follows:

 - **data_cleaner.py**
     A custom scikit-learn transformer class (`DataCleaner`) that handles:
       - Splitting columns like `PassengerId` into components.
       - Filling missing values for numerical and categorical fields.
       - One-hot encoding categorical variables (e.g., Deck, Side, HomePlanet, Destination).
       - Preparing datasets for use in pipelines.

 - **train_and_predict_multiple_models_scaled.py**
     The main script for the project:
       - Implements advanced models such as Logistic Regression, SVM, XGBoost, and Neural Networks.
       - Includes preprocessing pipelines with scaling, encoding, and train/validation splitting.
       - Writes separate submission files for each model to their respective folders under `submissions/`.

 - **train1-SpaceshipTitanic.csv**
     The training dataset, including the `Transported` column.

 - **test1-SpaceshipTitanic.csv**
     The test dataset, without the `Transported` column.

 - **submissions/**
     A folder containing the output predictions for each model. For example:
       - `submissions/logistic_regression/submission.csv`
       - `submissions/svm/submission.csv`
       - `submissions/xgboost/submission.csv`
       - `submissions/neural_network/submission.csv`

 - **README.txt**
     This file, providing an overview of the project.

-------------------------------------------------------------------------------
WORKFLOW
-------------------------------------------------------------------------------
1. **Setup Environment**:
   - Ensure Python is installed along with required libraries (`pandas`, `numpy`, `scikit-learn`, `xgboost`).
   - Place the dataset files (`train1-SpaceshipTitanic.csv` and `test1-SpaceshipTitanic.csv`) into the project directory.

2. **Run the Main Script**:
   - Open the project directory in a terminal and run:
     ```bash
     python train_and_predict_multiple_models_scaled.py
     ```
   - This script trains multiple models on the training data, evaluates them on 
     a validation split, and generates predictions for the test data. 
     Submission files for each model are saved in the `submissions/` folder.

3. **Check Results**:
   - Each model's validation accuracy and confusion matrix are printed to the 
     terminal. Use this information to compare model performance.

4. **Submit to Kaggle**:
   - Submit the file via the Kaggle CLI:
     ```bash
     kaggle competitions submit -c spaceship-titanic -f submissions/<model>/submission.csv -m "Submission from <model>"
     ```

-------------------------------------------------------------------------------
RESULTS & NEXT STEPS
-------------------------------------------------------------------------------
**Current Results**:
   - Logistic Regression: ~78% validation accuracy.
   - SVM: ~79% validation accuracy.
   - XGBoost: Pending further tuning.
   - Neural Network: Pending further tuning.

**Future Work**:
   - Hyperparameter tuning for advanced models (e.g., XGBoost, Neural Network).
   - Explore additional feature engineering, such as extracting insights from the 
     `last_name` column.
   - Evaluate performance using cross-validation for more robust results.
   - Automate model comparison and ensemble techniques to boost accuracy.

===============================================================================
