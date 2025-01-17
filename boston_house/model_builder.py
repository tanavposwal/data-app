import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

model_pkl_file = "housing_model.pkl"
# Loads the Boston House Price Dataset
boston = pd.read_csv("BostonHousing.csv")
X = boston.drop(columns=["MEDV"])
Y = boston[["MEDV"]]

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)

# Save Model
with open(model_pkl_file, "wb") as file:
    pickle.dump(model, file)
