import pandas as pd
from transformers import pipeline

books = pd.read_csv("books_with_cat.csv")

pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)
