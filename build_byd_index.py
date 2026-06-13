import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pypdf import PdfReader

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key is None:
    print("Error: GEMINI_API_KEY not found.")
    exit()

client = genai.Client(api_key=api_key)


def get_embedding(text):
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text
        )
        return result.embeddings[0].values

    except errors.APIError as e:
        print("Embedding error:")
        print(e)
        exit()


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)

    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text is not None:
            full_text = full_text + "\n" + page_text

    return full_text


def load_pdf_files():
    data_folder = Path("data")
    documents = []

    pdf_files = list(data_folder.glob("*.pdf"))

    if len(pdf_files) == 0:
        print("Error: no PDF files found inside data folder.")
        exit()

    for file_path in pdf_files:
        print("Reading PDF:", file_path.name)

        text = extract_text_from_pdf(file_path)

        print("Characters extracted:", len(text))

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def split_text_into_chunks(text, max_words=120):
    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunk_words = words[i:i + max_words]
        chunk = " ".join(chunk_words)

        if len(chunk.strip()) > 50:
            chunks.append(chunk)

    return chunks


def make_chunks(documents):
    chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_chunks = split_text_into_chunks(text)

        for chunk in document_chunks:
            chunks.append({
                "source": source,
                "chunk": chunk
            })

    return chunks


documents = load_pdf_files()
chunks = make_chunks(documents)

index = []

print("\nBuilding BYD PDF index...")
print("Total chunks:", len(chunks))

for number, item in enumerate(chunks, start=1):
    print("Creating embedding", number, "of", len(chunks))

    text_for_embedding = "Source: " + item["source"] + "\n" + item["chunk"]

    embedding = get_embedding(text_for_embedding)

    index.append({
        "source": item["source"],
        "chunk": item["chunk"],
        "embedding": embedding
    })

    time.sleep(1)

with open("byd_index.json", "w", encoding="utf-8") as file:
    json.dump(index, file)

print("\nDone. Saved PDF-based index as byd_index.json")