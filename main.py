import numpy as np  # type: ignore
import pandas as pd # type: ignore

train = pd.read_csv('Kaggle Comps/Spaceship Titanic/train-SpaceshipTitanic.csv')
test = pd.read_csv('Kaggle Comps/Spaceship Titanic/test-SpaceshipTitanic.csv')

#print(train.head())
print(train.describe())


