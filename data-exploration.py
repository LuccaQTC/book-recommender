import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")

print("Path to dataset files:", path)
save_dir = "images/"
########################################
books = pd.read_csv(f"{path}/books.csv")
########################################
books.describe()
books.isnull().sum()  # Verificando dados nulos
books.nunique()  # Contando valores distintos
########################################
ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)

plt.xlabel("Columns")
plt.ylabel("Missing Values")

plt.savefig("images/heat-mape-for-missing-values.png")
plt.close()
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

sns.set_theme(style="white")
plt.figure(figsize=(8, 10))
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"label": "Spearman Correlation"},
)

heatmap.set_title(label="Correlation Heat Map")
plt.savefig("images/heat-map-corr-missing-description.png")
plt.close()
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
###########################################
# Para verificar como a distribuição das categorias é discrepante
count_cat.describe()
plt.figure(figsize=(8, 10))
ax = plt.axes()
boxplot = sns.boxplot(data=count_cat.iloc[2:], ax=ax)
boxplot.set_title("BoxPlot das Categorias")

plt.savefig("images/frequency_cat.png")
plt.close()
##########################################
# Verificando as descrições muito curtas dos libros
books_comp.loc[:, "n_words_in_description"] = (
    books_comp["description"].str.split().str.len()
)
# Usar .str para acessar string methods
books_comp["n_words_in_description"].describe()
##########################################
plt.figure(figsize=(8, 10))
ax = plt.axes()
sns.histplot(
    data=books_comp.loc[
        books_comp["n_words_in_description"] <= 50, ["n_words_in_description"]
    ],
    kde=True,
    element="step",
)

plt.savefig(save_dir + "n_words_dist.png")
plt.close()
###########################################
# Vejamos as colunas com poucas palavras
with pd.option_context("display.max_colwidth", 250):
    print(
        books_comp.loc[
            books_comp["n_words_in_description"].between(20, 40), ["description"]
        ]
    )

books_comp_20_words = books_comp.loc[books_comp["n_words_in_description"] >= 20]
###########################################
books_comp_20_words.loc[:, "title_and_subtitle"] = np.where(
    books_comp_20_words["subtitle"].isna(),
    books_comp_20_words["title"],
    books_comp_20_words.loc[:, ["title", "subtitle"]]
    .astype(str)
    .agg(": ".join, axis=1),
)
###########################################
books_comp_20_words.loc[:, "tagged_description"] = (
    books_comp_20_words.loc[:, ["isbn13", "description"]]
    .astype(str)
    .agg(" ".join, axis=1)
)
###########################################
# Salvando o data frame
books_comp_20_words.drop(
    ["subtitle", "missing_descriptions", "age_of_book", "n_words_in_description"],
    axis=1,
).to_csv("clean_data/books.csv", index=False)
