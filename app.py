import os
import json
import math
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# ----------------------------
# Page setup
# ----------------------------

st.set_page_config(
    page_title="BYD Pakistan RAG Assistant",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 BYD Pakistan RAG Assistant")
st.write("Ask questions about BYD ATTO 2, ATTO 3, Shark 6, and Seal in Pakistan.")

# ----------------------------
# API key setup
# ----------------------------

load_dotenv()

api_key = None

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

if api_key is None:
    st.error("GEMINI_API_KEY not found. Add it to .env locally or Streamlit Secrets when deployed.")
    st.stop()

client = genai.Client(api_key=api_key)

# ----------------------------
# Backend functions
# ----------------------------

def get_embedding(text):
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text
        )
        return result.embeddings[0].values

    except errors.APIError as e:
        st.error("Embedding error:")
        st.write(e)
        st.stop()


def cosine_similarity(vector_a, vector_b):
    dot_product = 0
    length_a = 0
    length_b = 0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b
        length_a += a * a
        length_b += b * b

    length_a = math.sqrt(length_a)
    length_b = math.sqrt(length_b)

    if length_a == 0 or length_b == 0:
        return 0

    return dot_product / (length_a * length_b)


@st.cache_data
def load_index():
    try:
        with open("byd_index.json", "r", encoding="utf-8") as file:
            index = json.load(file)

        return index

    except FileNotFoundError:
        st.error("byd_index.json not found. Run build_byd_index.py first.")
        st.stop()


def handle_simple_questions(question):
    question_lower = question.lower().strip()

    greetings = [
        "hello",
        "hi",
        "hey",
        "salam",
        "assalamualaikum",
        "assalamu alaikum",
        "aoa"
    ]

    how_are_you = [
        "how are you",
        "how r u",
        "how are u",
        "how you doing",
        "how are you doing",
        "how r you"
    ]

    bot_identity = [
        "who are you",
        "what are you",
        "what can you do",
        "help"
    ]

    car_count_phrases = [
        "how many cars",
        "how many car",
        "how many models",
        "how many model",
        "cars available",
        "car available",
        "models available",
        "model available",
        "available cars",
        "available car",
        "available models",
        "available model",
        "total cars",
        "total models",
        "number of cars",
        "number of models",
        "how many cars availble",
        "cars availble",
        "models availble"
    ]

    if question_lower in greetings:
        return "Hello! I am your BYD Pakistan RAG Assistant. You can ask me about BYD ATTO 2, ATTO 3, Shark 6, or Seal."

    if question_lower in how_are_you:
        return "I am doing great! I am ready to help you with BYD Pakistan car questions."

    if question_lower in bot_identity:
        return "I am a BYD Pakistan RAG Assistant. I answer questions using PDF data about BYD ATTO 2, ATTO 3, Shark 6, and Seal in Pakistan."

    for phrase in car_count_phrases:
        if phrase in question_lower:
            return """There are 4 BYD car models available in this project:

1. BYD ATTO 2
2. BYD ATTO 3
3. BYD Shark 6
4. BYD Seal"""

    return None


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
        "warranty",
        "phev",
        "hybrid",
        "electric",
        "models",
        "cars"
    ]

    for word in important_words:
        if word in question_lower and word in chunk_lower:
            bonus += 0.05

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
        context += "Source: " + item["source"] + "\n"
        context += item["chunk"] + "\n\n"

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
- If the user asks about available cars or models, answer using model names, not only variant names.

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

            return response.text, model_name

        except errors.ServerError:
            continue

        except errors.APIError:
            continue

    return "Sorry, all Gemini models are busy right now. Please try again later.", "None"


# ----------------------------
# Load saved index
# ----------------------------

index = load_index()

# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:
    st.header("Project Info")
    st.write("Frontend: Streamlit")
    st.write("Backend: Python RAG")
    st.write("LLM API: Gemini")
    st.write("Source: BYD PDF files")

    st.divider()

    st.write("Models included:")
    st.write("- BYD ATTO 2")
    st.write("- BYD ATTO 3")
    st.write("- BYD Shark 6")
    st.write("- BYD Seal")

    st.divider()

    st.write("Example questions:")
    st.write("How many cars are available?")
    st.write("What is the price of BYD ATTO 2?")
    st.write("Which BYD car is PHEV?")
    st.write("What colors are available for BYD Seal?")
    st.write("Is Shark 6 AWD or FWD?")
    st.write("Which BYD has the longest range?")

    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# Chat UI
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a BYD Pakistan question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        simple_answer = handle_simple_questions(question)

        if simple_answer is not None:
            st.write(simple_answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": simple_answer
            })

        else:
            with st.spinner("Searching BYD PDF data and generating answer..."):
                retrieved_chunks = retrieve_chunks(question, index)

                if len(retrieved_chunks) == 0:
                    answer = "Sorry, I could not find this topic in the BYD Pakistan PDF data."
                    st.write(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                else:
                    final_prompt = build_prompt(question, retrieved_chunks)

                    answer, model_used = ask_gemini(final_prompt)

                    st.write(answer)

                    st.caption("Model used: " + model_used)

                    with st.expander("View retrieved PDF sources"):
                        for item in retrieved_chunks:
                            similarity_percent = round(item["similarity"] * 100)
                            final_percent = round(item["final_score"] * 100)

                            st.write(
                                "**Source:** "
                                + item["source"]
                                + " | Similarity: "
                                + str(similarity_percent)
                                + "%"
                                + " | Final score: "
                                + str(final_percent)
                                + "%"
                            )

                            st.write(item["chunk"])
                            st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
