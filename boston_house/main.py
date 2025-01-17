import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

st.header("Boston House Price Prediction", divider="rainbow")

# Loads the Boston House Price Dataset
boston = pd.read_csv("BostonHousing.csv")
X = boston.drop(columns=["MEDV"])
Y = boston[["MEDV"]]

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header("Specify Input Parameters")


def user_input_features():
    # CRIM - per capita crime rate by town
    CRIM = st.sidebar.slider(
        "CRIM - per capita crime rate", X.CRIM.min(), X.CRIM.max(), X.CRIM.mean()
    )
    # ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    ZN = st.sidebar.slider(
        "ZN - residential land zoned for lots over 25,000 square feet",
        X.ZN.min(),
        X.ZN.max(),
        X.ZN.mean(),
    )
    # INDUS - proportion of non-retail business acres per town
    INDUS = st.sidebar.slider(
        "INDUS - percentage of land used for non retail",
        X.INDUS.min(),
        X.INDUS.max(),
        X.INDUS.mean(),
    )
    # CHAS - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    CHAS = st.sidebar.slider(
        "CHAS - located near the Charles River",
        float(X.CHAS.min()),
        float(X.CHAS.max()),
        X.CHAS.mean(),
    )
    # NOX - nitric oxides concentration (parts per 10 million)
    NOX = st.sidebar.slider(
        "NOX - nitric oxides concentration", X.NOX.min(), X.NOX.max(), X.NOX.mean()
    )
    # RM - average number of rooms per dwelling
    RM = st.sidebar.slider(
        "RM - average size of the houses (rooms)", X.RM.min(), X.RM.max(), X.RM.mean()
    )
    # AGE - proportion of owner-occupied units built prior to 1940
    AGE = st.sidebar.slider(
        "AGE - proportion of owner-occupied units built before 1940",
        X.AGE.min(),
        X.AGE.max(),
        X.AGE.mean(),
    )
    # DIS - weighted distances to five Boston employment centres
    DIS = st.sidebar.slider(
        "DIS - distance to employment centers", X.DIS.min(), X.DIS.max(), X.DIS.mean()
    )
    # RAD - index of accessibility to radial highways
    RAD = st.sidebar.slider(
        "RAD - accessibility to radial highways",
        float(X.RAD.min()),
        float(X.RAD.max()),
        X.RAD.mean(),
    )
    # TAX - full-value property-tax rate per $10,000
    TAX = st.sidebar.slider(
        "TAX - tax per $10k", float(X.TAX.min()), float(X.TAX.max()), X.TAX.mean()
    )
    PTRATIO = st.sidebar.slider(
        "PTRATIO - Pupil-teacher ratio",
        X.PTRATIO.min(),
        X.PTRATIO.max(),
        X.PTRATIO.mean(),
    )
    B = st.sidebar.slider(
        "B - proportions of african american residents",
        X.B.min(),
        X.B.max(),
        X.B.mean(),
    )
    LSTAT = st.sidebar.slider(
        "LSTAT - of lower status population",
        X.LSTAT.min(),
        X.LSTAT.max(),
        X.LSTAT.mean(),
    )
    data = {
        "CRIM": CRIM,
        "ZN": ZN,
        "INDUS": INDUS,
        "CHAS": CHAS,
        "NOX": NOX,
        "RM": RM,
        "AGE": AGE,
        "DIS": DIS,
        "RAD": RAD,
        "TAX": TAX,
        "PTRATIO": PTRATIO,
        "B": B,
        "LSTAT": LSTAT,
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.header("Specified Input parameters")
st.write(df)
st.write("---")

# import model with pickle
model = pickle.load(open("model.pkl", "rb"))

# Apply Model to Make Prediction
prediction = model.predict(df)
st.header("Prediction of MEDV (median house price)")
st.write(prediction)
st.write("---")

# Explining the model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header("Feature Importance")
fig = plt.figure()
plt.title("Feature importance based on SHAP values")
shap.summary_plot(shap_values, X)
st.pyplot(fig, bbox_inches="tight")
st.write("---")

fig = plt.figure()
plt.title("Feature importance based on SHAP values (Bar)")
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(fig, bbox_inches="tight")
