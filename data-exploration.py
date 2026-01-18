import kagglehub
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")

print("Path to dataset files:", path)
########################################
books = pd.read_csv(f"{path}/books.csv")
########################################
books.describe()
books.isnull().sum()  # Verificando dados nulos
books.nunique()  # Contando valores distintos
########################################
ax = plt.axes()
sn.heatmap(books.isna().transpose(), cbar=False, ax=ax)

plt.xlabel("Columns")
plt.ylabel("Missing Values")

plt.savefig("images/heat-mape-for-missing-values.png")
#########################################
books["missing_descriptions"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2026 - books["published_year"]
#########################################
columns_of_interest = [
    "num_pages",
    "age_of_book",
    "missing_descriptions",
    "average_rating",
]

correlation_matrix = books[columns_of_interest].corr(method="spearman")

sn.set_theme(style="white")
plt.figure(figsize=(8, 10))
heatmap = sn.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"label": "Spearman Correlation"},
)

heatmap.set_title(label="Correlation Heat Map")
plt.savefig("images/heat-map-corr-missing-description.png")
##########################################
books[
    (books["description"].isna())
    | (books["num_pages"].isna())
    | (books["average_rating"].isna())
    | (books["published_year"].isna())
]  # 303 rows
##########################################
books_comp = books[
    ~(books["description"].isna())
    & ~(books["num_pages"].isna())
    & ~(books["average_rating"].isna())
    & ~(books["published_year"].isna())
]

books_comp["categories"].nunique()
count_cat = (
    books_comp["categories"]
    .value_counts()
    .reset_index()
    .sort_values("count", ascending=False)
)
