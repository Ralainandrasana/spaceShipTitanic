import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')

""" EDA """
# 1. Comprendre la structure du dataset

def eda1(data):
    print(f"dimensions: {df.shape}") #dimensions
    print(f"noms des colonnes: {df.columns}") #dimensions
    print(f"types: {df.info()}") #types des données
    print(df.head(5)) #5 premier
    print(df.tail(5)) #5 dernier
# eda1(df)

# 1. Statistique descriptif

# cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# print(df.describe())
#variable categorielles
# for col, ax in zip(cols, axes.flatten()):
#     df[col].value_counts().plot(
#         kind="bar",
#         ax=ax,
#         edgecolor="black"
#     )

#     ax.set_title(f"Répartition de {col}")
#     ax.set_xlabel(col)
#     ax.set_ylabel("Fréquence")
#     ax.tick_params(axis="x", rotation=45)

# plt.tight_layout()
# plt.show()

# cols = ["Age", "RoomService", "FoodCourt",
#         "ShoppingMall", "Spa", "VRDeck"]

# df[cols].hist(figsize=(12, 8), bins=30)
# plt.tight_layout()
# plt.show()

# 3. Detection des problèmes de qualité des données
#valeurs manquantes
print(df.duplicated().sum())
cols = ["Age", "RoomService", "FoodCourt",
        "ShoppingMall", "Spa", "VRDeck"]

# Sélectionner les variables numériques
num_df = df.select_dtypes(include="number")

# Calculer la matrice de corrélation
corr = num_df.corr()

# Afficher la heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()