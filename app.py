import os
import json
import math
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# ----------------------------
# Page setup
# ----------------------------

st.set_page_config(
    page_title="BYD Pakistan RAG Assistant",
    layout="wide"
)

# ----------------------------
# Professional CSS
# ----------------------------

st.markdown(
    """
    <style>
    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hide default Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        padding: 30px 35px;
        border-radius: 18px;
        border: 1px solid #374151;
        margin-bottom: 25px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }

    .subtitle {
        font-size: 17px;
        color: #d1d5db;
        line-height: 1.6;
    }

    .tag-row {
        margin-top: 18px;
    }

    .tag {
        display: inline-block;
        background-color: #111827;
        color: #d1d5db;
        border: 1px solid #4b5563;
        padding: 6px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        border-radius: 20px;
        font-size: 13px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #374151;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff;
    }

    .sidebar-card {
        background-color: #1f2937;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #374151;
        margin-bottom: 16px;
    }

    .sidebar-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .sidebar-text {
        color: #d1d5db;
        font-size: 14px;
        line-height: 1.7;
    }

    /* Chat cards */
    .chat-card-user {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-left: 5px solid #60a5fa;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    .chat-card-assistant {
        background-color: #111827;
        border: 1px solid #374151;
        border-left: 5px solid #34d399;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    .chat-label {
        font-size: 13px;
        font-weight: 700;
        color: #9ca3af;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .chat-text {
        font-size: 16px;
        color: #f9fafb;
        line-height: 1.7;
    }

    /* Input */
    div[data-testid="stChatInput"] {
        border-radius: 16px;
    }

    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #4b5563;
        background-color: #1f2937;
        color: #ffffff;
    }

    .stButton button:hover {
        border-color: #60a5fa;
        color: #ffffff;
    }

    /* Source expander */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Header
# ----------------------------

st.markdown(
    """
    <div class="header-card">
        <div class="main-title">BYD Pakistan RAG Assistant</div>
        <div class="subtitle">
            A PDF-based retrieval-augmented generation chatbot for BYD vehicles available in Pakistan.
            Ask about prices, variants, colors, battery, range, performance, and features.
        </div>
        <div class="tag-row">
            <span class="tag">Streamlit Frontend</span>
            <span class="tag">Python Backend</span>
            <span class="tag">Gemini API</span>
            <span class="tag">PDF Knowledge Base</span>
            <span class="tag">Roman Urdu Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

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


def is_roman_urdu_question(question):
    question_lower = question.lower().strip()

    words = re.findall(r"[a-zA-Z]+", question_lower)

    roman_urdu_words = {
        "kya", "kia", "kiya",
        "hai", "he", "hain", "hein",
        "ka", "ki", "ke",
        "ap", "aap", "apke", "aapke",
        "pass", "paas",
        "haal", "hal",
        "kaise", "kese",
        "kitni", "kitnay", "kitne",
        "gaari", "gari", "gaariyan", "gariyan",
        "rang", "qeemat",
        "bataye", "batao", "btao",
        "maujood", "dastiyab",
        "kon", "kaun",
        "haan", "han",
        "nahi", "nai",
        "chahiye"
    }

    roman_urdu_phrases = [
        "kon si",
        "kaun si",
        "price kya",
        "available hain",
        "available hai",
        "kitni cars",
        "kitni gariyan",
        "kitni gaariyan",
        "kya price",
        "kia price",
        "kia haal",
        "kya haal",
        "apke pass",
        "aapke paas",
        "ke colors",
        "ke rang",
        "kya hai",
        "kia hai",
        "kya he",
        "kia he"
    ]

    for phrase in roman_urdu_phrases:
        if phrase in question_lower:
            return True

    for word in words:
        if word in roman_urdu_words:
            return True

    return False


def handle_simple_questions(question):
    question_lower = question.lower().strip()
    roman_urdu = is_roman_urdu_question(question)

    greetings = [
        "hello",
        "hi",
        "hey",
        "salam",
        "assalamualaikum",
        "assalamu alaikum",
        "aoa"
    ]

    how_are_you_phrases = [
        "how are you",
        "how r u",
        "how are u",
        "how you doing",
        "how are you doing",
        "how r you",
        "ap kaise hain",
        "ap kese hain",
        "aap kaise hain",
        "aap kese hain",
        "kaise ho",
        "kese ho",
        "kia haal hein",
        "kia haal hain",
        "kya haal hai",
        "kya haal hain",
        "kia hal hai",
        "kia hal hain"
    ]

    bot_identity_phrases = [
        "who are you",
        "what are you",
        "what can you do",
        "help",
        "tum kon ho",
        "ap kon hain",
        "aap kon hain",
        "tum kya ho",
        "ap kya ho"
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
        "models availble",
        "kitni cars",
        "kitni gariyan",
        "kitni gaariyan",
        "kitne models",
        "kitnay models",
        "kitni models"
    ]

    if question_lower in greetings:
        if roman_urdu or question_lower in ["salam", "assalamualaikum", "assalamu alaikum", "aoa"]:
            return "Assalamualaikum! Main BYD Pakistan RAG Assistant hoon. Aap BYD ATTO 2, ATTO 3, Shark 6, ya Seal ke baare mein sawal pooch sakte hain."
        return "Hello. I am your BYD Pakistan RAG Assistant. You can ask me about BYD ATTO 2, ATTO 3, Shark 6, or Seal."

    for phrase in how_are_you_phrases:
        if phrase in question_lower:
            if roman_urdu:
                return "Main theek hoon. Main BYD Pakistan ki cars ke baare mein aapki madad ke liye ready hoon."
            return "I am doing great. I am ready to help you with BYD Pakistan car questions."

    for phrase in bot_identity_phrases:
        if phrase in question_lower:
            if roman_urdu:
                return "Main BYD Pakistan RAG Assistant hoon. Main PDF data use karke BYD ATTO 2, ATTO 3, Shark 6, aur Seal ke baare mein jawab deta hoon."
            return "I am a BYD Pakistan RAG Assistant. I answer questions using PDF data about BYD ATTO 2, ATTO 3, Shark 6, and Seal in Pakistan."

    for phrase in car_count_phrases:
        if phrase in question_lower:
            if roman_urdu:
                return """Is project mein BYD ki 4 car models available hain:

1. BYD ATTO 2
2. BYD ATTO 3
3. BYD Shark 6
4. BYD Seal"""
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
        "cars",
        "qeemat",
        "rang",
        "gaari",
        "gari"
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

    if is_roman_urdu_question(question):
        language_rule = "Answer in Roman Urdu only. Do not use Urdu script. Keep BYD model names in English."
        not_found_message = "Mujhe yeh information BYD Pakistan PDF data mein nahi mili."
    else:
        language_rule = "Answer in English."
        not_found_message = "I could not find this in the provided BYD Pakistan PDF data."

    prompt = f"""
You are a helpful BYD Pakistan car assistant.

Use only the context below to answer the question.
Do not use outside knowledge.

If the answer is not in the context, say:
"{not_found_message}"

Rules:
- {language_rule}
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
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Project Overview</div>
            <div class="sidebar-text">
                This assistant answers questions about selected BYD vehicles in Pakistan using PDF-based retrieval-augmented generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Technology Stack</div>
            <div class="sidebar-text">
                Frontend: Streamlit<br>
                Backend: Python RAG<br>
                LLM API: Gemini<br>
                Source: BYD PDF files
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Models Included</div>
            <div class="sidebar-text">
                BYD ATTO 2<br>
                BYD ATTO 3<br>
                BYD Shark 6<br>
                BYD Seal
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Example Questions</div>
            <div class="sidebar-text">
                How many cars are available?<br>
                What is the price of BYD ATTO 2?<br>
                Which BYD car is PHEV?<br>
                What colors are available for BYD Seal?<br>
                BYD Seal ke colors kya hain?<br>
                Kia Shark 6 PHEV hai?
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# Chat UI
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-card-user">
                <div class="chat-label">User</div>
                <div class="chat-text">{message["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-card-assistant">
                <div class="chat-label">Assistant</div>
                <div class="chat-text">{message["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

question = st.chat_input("Ask a BYD Pakistan question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.markdown(
        f"""
        <div class="chat-card-user">
            <div class="chat-label">User</div>
            <div class="chat-text">{question}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    simple_answer = handle_simple_questions(question)

    if simple_answer is not None:
        st.markdown(
            f"""
            <div class="chat-card-assistant">
                <div class="chat-label">Assistant</div>
                <div class="chat-text">{simple_answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": simple_answer
        })

    else:
        with st.spinner("Searching BYD PDF data and generating answer..."):
            retrieved_chunks = retrieve_chunks(question, index)

            if len(retrieved_chunks) == 0:
                if is_roman_urdu_question(question):
                    answer = "Mujhe yeh information BYD Pakistan PDF data mein nahi mili."
                else:
                    answer = "Sorry, I could not find this topic in the BYD Pakistan PDF data."

                st.markdown(
                    f"""
                    <div class="chat-card-assistant">
                        <div class="chat-label">Assistant</div>
                        <div class="chat-text">{answer}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            else:
                final_prompt = build_prompt(question, retrieved_chunks)

                answer, model_used = ask_gemini(final_prompt)

                st.markdown(
                    f"""
                    <div class="chat-card-assistant">
                        <div class="chat-label">Assistant</div>
                        <div class="chat-text">{answer}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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
