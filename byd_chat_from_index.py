import os
import json
import math
from dotenv import load_dotenv
from google import genai
from google.genai import errors

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


def cosine_similarity(vector_a, vector_b):
    dot_product = 0
    length_a = 0
    length_b = 0

    for a, b in zip(vector_a, vector_b):
        dot_product = dot_product + (a * b)
        length_a = length_a + (a * a)
        length_b = length_b + (b * b)

    length_a = math.sqrt(length_a)
    length_b = math.sqrt(length_b)

    if length_a == 0 or length_b == 0:
        return 0

    return dot_product / (length_a * length_b)


def load_index():
    try:
        with open("byd_index.json", "r", encoding="utf-8") as file:
            index = json.load(file)

        return index

    except FileNotFoundError:
        print("Error: byd_index.json not found.")
        print("Run build_byd_index.py first.")
        exit()


def detect_model_sources(question):
    question_lower = question.lower()

    selected_sources = []

    if "atto 2" in question_lower or "atto2" in question_lower:
        selected_sources.append("atto_2.pdf")

    if "atto 3" in question_lower or "atto3" in question_lower:
        selected_sources.append("atto_3.pdf")

    if "shark" in question_lower or "shark 6" in question_lower:
        selected_sources.append("shark_6.pdf")

    if "seal" in question_lower and "sealion" not in question_lower:
        selected_sources.append("seal.pdf")

    return selected_sources


def keyword_bonus(question, chunk):
    question_lower = question.lower()
    chunk_lower = chunk.lower()

    bonus = 0

    important_words = [
        "price", "prices",
        "color", "colors", "colour", "colours",
        "range",
        "horsepower", "hp", "power",
        "torque",
        "awd", "fwd", "rwd", "drive",
        "battery",
        "charging", "charge",
        "speed", "top speed",
        "acceleration",
        "variant", "variants",
        "safety",
        "features",
        "warranty"
    ]

    for word in important_words:
        if word in question_lower and word in chunk_lower:
            bonus = bonus + 0.05

    return bonus


def retrieve_chunks(question, index, top_k=8, minimum_score=0.50):
    question_embedding = get_embedding(question)

    selected_sources = detect_model_sources(question)

    results = []

    for item in index:
        if len(selected_sources) > 0:
            if item["source"] not in selected_sources:
                continue

        similarity = cosine_similarity(question_embedding, item["embedding"])
        final_score = similarity + keyword_bonus(question, item["chunk"])

        if final_score >= minimum_score:
            results.append({
                "source": item["source"],
                "chunk": item["chunk"],
                "similarity": similarity,
                "final_score": final_score
            })

    results = sorted(results, key=lambda item: item["final_score"], reverse=True)

    return results[:top_k]


def build_prompt(question, retrieved_chunks):
    context = ""

    for item in retrieved_chunks:
        context = context + "Source: " + item["source"] + "\n"
        context = context + item["chunk"] + "\n\n"

    prompt = f"""
You are a helpful BYD Pakistan car assistant.

Use only the context below to answer the question.
Do not use outside knowledge.

If the answer is not in the context, say:
"I could not find this in the provided BYD Pakistan PDF data."

Rules:
- Be clear and student-friendly.
- Mention the BYD model name if relevant.
- If prices are mentioned, say they are ex-factory and may change.
- If a detail is uncertain, say it should be verified with an official BYD Pakistan dealer.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def ask_gemini(prompt):
    models_to_try = [
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-3.5-flash"
    ]

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            return response.text

        except errors.ServerError:
            print("Model busy:", model_name)
            print("Trying next model...")

        except errors.APIError as e:
            print("API error with model:", model_name)
            print(e)
            print("Trying next model...")

    return "Sorry, all Gemini models are busy right now. Please try again later."


index = load_index()

print("BYD Pakistan PDF RAG Bot is ready.")
print("Loaded saved index from byd_index.json.")
print("Type 'exit' to stop.\n")

while True:
    question = input("Ask a BYD Pakistan question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    retrieved_chunks = retrieve_chunks(question, index)

    if len(retrieved_chunks) == 0:
        print("\nSorry, I could not find this topic in your BYD Pakistan PDF data.")
        print("-" * 60)
        continue

    print("\nRetrieved sources:")
    for item in retrieved_chunks:
        similarity_percent = round(item["similarity"] * 100)
        final_percent = round(item["final_score"] * 100)

        print(
            "- Similarity "
            + str(similarity_percent)
            + "%, final score "
            + str(final_percent)
            + "% from "
            + item["source"]
        )

        print("  " + item["chunk"][:150] + "...")

    final_prompt = build_prompt(question, retrieved_chunks)

    answer = ask_gemini(final_prompt)

    print("\nAI Answer:")
    print(answer)
    print("-" * 60)