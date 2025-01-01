===============================================================================
SPACESHIP TITANIC PROJECT
===============================================================================

I created this repository for the Spaceship Titanic challenge on Kaggle. 
The core objective is to predict which passengers were transported to 
an alternate dimension following the spaceship’s collision with a cosmic anomaly.

-------------------------------------------------------------------------------
COMPETITION & DATASET
-------------------------------------------------------------------------------
Competition Link: https://www.kaggle.com/competitions/spaceship-titanic
Dataset Source:  https://www.kaggle.com/competitions/spaceship-titanic/data

The training data (about 8,700 rows) contains a "Transported" label 
(True/False). The test data (about 4,300 rows) does not include the label, 
so I must predict it. The final metric is classification accuracy.

-------------------------------------------------------------------------------
PROJECT FILES
-------------------------------------------------------------------------------
The files and scripts in the repo are organized as follows:

 - data_cleaner.py
     A custom scikit‐learn transformer class (DataCleaner). It splits 
     columns like PassengerId, handles missing values (e.g., Age, 
     RoomService, VIP, CryoSleep), and one‐hot‐encodes categorical 
     fields like Deck, Side, HomePlanet, and Destination.

 - train_and_predict.py
     Builds a Pipeline (DataCleaner + LogisticRegression), fits it on 
     train1-SpaceshipTitanic.csv, then predicts on test1-SpaceshipTitanic.csv, 
     finally writing a submission.csv file.

 - main.py
     Similar to train_and_predict.py, but also prints a confusion matrix 
     and accuracy score on the training data (since I know those labels). 
     It then creates the same submission.csv file.

 - train1-SpaceshipTitanic.csv
     The training dataset, including the Transported column.

 - test1-SpaceshipTitanic.csv
     The test dataset, without Transported. 

 - notebook.ipynb
     A Jupyter notebook I use for exploratory data analysis and 
     visualizations.

 - README.txt
     This file (plain‐text overview of the project).

-------------------------------------------------------------------------------
WORKFLOW
-------------------------------------------------------------------------------
1. Ensure that pandas, numpy, and scikit‐learn installed.

2. Place the CSVs (train1-SpaceshipTitanic.csv and test1-SpaceshipTitanic.csv) 
   from Kaggle into this the local repo.

3. Open project directory in terminal and run: python train_and_predict.py
   That fits the data (cleaning + model) and writes submission.csv. 

4. To see the model’s performance on the training set (confusion 
   matrix and accuracy), run : python main.py
   It prints the metrics and still produces submission.csv for Kaggle.

-------------------------------------------------------------------------------
RESULTS & NEXT STEPS
-------------------------------------------------------------------------------
My baseline LogisticRegression approach currently achieves around 78–79% 
accuracy on the training data. The confusion matrix reveals a moderate number 
of false positives and false negatives, so there is room for improvement.

In the future, I plan to:
 - Experiment with more advanced models like Knn or XGBoost.
 - Perform hyperparameter tuning and possibly feature engineering, 
   for instance extracting info from the "last_name" column.
 - Incorporate a train/validation split or cross‐validation for a more 
   reliable measure of generalization.
===============================================================================
