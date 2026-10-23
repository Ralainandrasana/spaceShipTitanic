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
dfOrig = pd.read_csv("train.csv")
dataTestOrig = pd.read_csv("test.csv")


# ============================================================
# 2. FEATURE ENGINEERING (aucune statistique apprise ici)
# ============================================================
def add_features(data):
    data["numGrp"] = data["PassengerId"].str.split("_").str[0]
    data["grpSize"] = data.groupby("numGrp")["PassengerId"].transform("count")

    data["isalone"] = (data["grpSize"] == 1).astype(int)

    data["grpType"] = "large group"
    data.loc[data["grpSize"] == 1, "grpType"] = "alone"
    data.loc[data["grpSize"] == 2, "grpType"] = "duo"
    data.loc[data["grpSize"] == 3, "grpType"] = "trio"
    data = data.drop(["PassengerId", "Name", "numGrp"], axis=1)

    data["Cabin_deck"] = data["Cabin"].str.split("/").str[0]
    data["Cabin_side"] = data["Cabin"].str.split("/").str[2]
    data["HasCabin"] = data["Cabin"].notnull().astype(int)
    data = data.drop(["Cabin"], axis=1)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    data["TotalSpent"] = data[spend_cols].sum(axis=1, skipna=True)
    data["NoSpending"] = (data["TotalSpent"] == 0).astype(int)

    for c in ["CryoSleep", "VIP"]:
        data[c] = data[c].astype("Int64")
        data[c] = data[c].astype(float)

    return data


df = add_features(dfOrig.copy())
dataTest = add_features(dataTestOrig.copy())

# ============================================================
# 3. DÉFINITION DES COLONNES
# ============================================================
colsNum = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpent",
]
colsCat = ["HomePlanet", "Destination", "Cabin_deck", "Cabin_side", "grpSize"]
colsBin = ["CryoSleep", "VIP", "HasCabin", "NoSpending", "isalone"]

X = df.drop("Transported", axis=1)
y = df["Transported"].astype(int)

# ============================================================
# 4. PIPELINE DE PRÉTRAITEMENT
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            colsNum,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
                ]
            ),
            colsCat,
        ),
        (
            "bin",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
            ),
            colsBin,
        ),
    ],
    remainder="drop",
)

# ============================================================
# 5. PIPELINES : LR, RF, XGB
# ============================================================
pipe_lr = Pipeline(
    [
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
)

pipe_rf = Pipeline(
    [
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        )),
    ]
)

pipe_xgb = Pipeline(
    [
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )),
    ]
)

# ============================================================
# 6. VALIDATION CROISÉE — 1 SEULE MÉTRIQUE (accuracy)
# ============================================================
pipelines = {
    "Logistic Regression": pipe_lr,
    "Random Forest":       pipe_rf,
    "XGBoost":             pipe_xgb,
}

print("=" * 60)
print("COMPARAISON DES MODÈLES (CV=5, métrique = accuracy)")
print("=" * 60)

scores = {}
for name, pipe in pipelines.items():
    s = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    scores[name] = s.mean()
    print(f"{name:20s} : {s.mean():.4f} ± {s.std():.4f}")

# Meilleur modèle
best_name = max(scores, key=scores.get)
print(f"\n🏆 Meilleur modèle : {best_name} ({scores[best_name]:.4f})")

# ============================================================
# 7. PRÉDICTION FINALE AVEC LE MEILLEUR MODÈLE
# ============================================================
final_pipe = pipelines[best_name]
final_pipe.fit(X, y)
predTest = final_pipe.predict(dataTest)

resCsv = pd.DataFrame(
    {
        "PassengerId": dataTestOrig["PassengerId"],
        "Transported": predTest.astype(bool),
    }
)
resCsv.to_csv("spaceShipTitanic.csv", index=False)

print("\n✅ Fichier spaceShipTitanic.csv généré")
print(resCsv.head())
print(f"\nNombre de passagers prédits 'Transported' : {resCsv['Transported'].sum()} / {len(resCsv)}")
print(f"Proportion : {resCsv['Transported'].mean():.2%}  (train ≈ 50.4%)")