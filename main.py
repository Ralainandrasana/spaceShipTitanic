import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

dfOrig = pd.read_csv('train.csv')
df = dfOrig.copy()
dataTestOrig = pd.read_csv('test.csv')
dataTest = dataTestOrig.copy()

def pretraitement(data):
    # suppression variable non importants
    data = data.drop(["PassengerId", "Name"], axis=1)

    # imputation
    colsCat = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    colsNum = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in colsCat:
        data[col] = data[col].fillna(data[col].mode()[0])
        data[col] = data[col].astype(str)

    for col in colsNum:
        data[col] = data[col].fillna(data[col].median())

    data["Age"] = data["Age"].fillna(data["Age"].median())

    # feature engineering
    # data[["Cabin_deck", "Cabin_num", "Cabin_side"]] = data["Cabin"].str.split("/", expand=True)
    data["HasCabin"] = data["Cabin"].notnull().astype(int)
    data = data.drop(["Cabin"], axis=1)

    # encodage des variables catégorielles
    data = pd.get_dummies(data, columns=colsCat, drop_first=True)

    return data

df = pretraitement(df)
dataTest = pretraitement(dataTest)

# modele
model = LogisticRegression(max_iter=3000)

# cross validation
x = df.drop("Transported", axis=1)
y = df["Transported"]

for metric in ['f1', 'accuracy', 'precision', 'recall']:
    s = cross_val_score(model, x, y, cv=5, scoring=metric)
    print(f"{metric:10s} : {s.mean():.4f} ± {s.std():.4f}")

# entrainement du modèle
model.fit(x, y)

# prediction
predTest = pd.Series(model.predict(dataTest))
predTest.name = 'Transported'
PassengerId = dataTestOrig['PassengerId']
resCsv = pd.concat([PassengerId, predTest], axis=1)
resCsv.to_csv('spaceShipTitanic.csv', index=False)