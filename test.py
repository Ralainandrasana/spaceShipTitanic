import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# ============================================================
# 1. CHARGEMENT
# ============================================================
data = pd.read_csv("train.csv")

data["numGrp"] = data["PassengerId"].str.split("_").str[0]
data["grpSize"] = data.groupby("numGrp")["PassengerId"].transform("count")
data["isalone"] = (data["grpSize"] == 1).astype(int)



print(data[["numGrp", "Name", "Transported"]][data["numGrp"] == "0221"])