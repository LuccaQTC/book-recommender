from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import pandas as pd

books = pd.read_csv("clean_data/books.csv")
books["tagged_description"].to_csv(
    "docs/tagged_description.txt", lineterminator="\n", index=False, header=False
)

raw_doc = TextLoader("docs/tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_doc)

persist_directory = "./chroma_db"
embedding = OllamaEmbeddings(model="granite-embedding")

# Criando a base de dados
# db_books = Chroma.from_documents(documents, embedding=embedding, persist_directory=persist_directory, collection_name="books_collection")
# Reutilizando
db_books = Chroma(
    collection_name="books_collection",
    embedding_function=embedding,
    persist_directory=persist_directory,
)
query = "A book to teach children about nature"
docs = db_books.similarity_search(query, k=10)

res_books = books[
    books["isbn13"] == int(docs[0].page_content.split()[0].replace('"', ""))
]


def retrieve_semantic_recommendations(query: str, k: int) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=k)
    books_list = []

    for i in range(0, len(recs)):
        books_list.append(int(recs[i].page_content.split()[0].replace('"', "")))

    return books.loc[books["isbn13"].isin(books_list)].head(k)
