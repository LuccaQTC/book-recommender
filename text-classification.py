import pandas as pd
from transformers import pipeline
import numpy as np
from tqdm import tqdm

books = pd.read_csv("clean_data/books.csv")
books["categories"].value_counts().reset_index().query("count > 50")

category_mapping = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
}

books.loc[:, "simple_cat"] = books["categories"].map(category_mapping)

# Usando modelo HF
fiction_categories = ["Fiction", "Nonfiction"]
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence = books.loc[books["simple_cat"] == "Fiction", "description"].reset_index(
    drop=True
)[0]

res = pipe(sequence, fiction_categories)
max_index = np.argmax(res["scores"])  # type: ignore
max_label = pipe(sequence, fiction_categories)["labels"][max_index]  # type: ignore


def generate_predictions(sequence, categories):
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions["scores"])  # type: ignore
    max_label = predictions["labels"][max_index]  # type: ignore
    return max_label


actual_cats = []
predicted_cats = []

for i in tqdm(range(0, 300)):
    sequence = books.loc[books["simple_cat"] == "Fiction", "description"].reset_index(
        drop=True
    )[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Fiction"]

for i in tqdm(range(0, 300)):
    sequence = books.loc[
        books["simple_cat"] == "Nonfiction", "description"
    ].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Nonfiction"]

predictions_df = pd.DataFrame(
    {"actual_cat": actual_cats, "predicted_cat": predicted_cats}
)
predictions_df.loc[:, "correct_pred"] = np.where(
    predictions_df["actual_cat"] == predictions_df["predicted_cat"], 1, 0
)
right_freq = predictions_df["correct_pred"].sum() / len(predictions_df)

isbns = []
predicted_cats = []

missing_cats = books.loc[
    books["simple_cat"].isna(), ["isbn13", "description"]
].reset_index(drop=True)
for i in tqdm(range(0, len(missing_cats))):
    sequence = missing_cats["description"][i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    isbns += [missing_cats["isbn13"][i]]

missing_pred_df = pd.DataFrame({"isbn13": isbns, "predicted_cat": predicted_cats})
books = pd.merge(books, missing_pred_df, on="isbn13", how="left")
books["simple_cat"] = np.where(
    books["simple_cat"].isna(), books["predicted_cat"], books["simple_cat"]
)
books = books.drop(columns=["predicted_cat"])

# Salvando
books.to_csv("clean_data/book_with_cat.csv")
