import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM
import requests

# ----------------------------------------
# Function to check Ollama server
# ----------------------------------------
def check_ollama_server():
    try:
        response = requests.get("http://127.0.0.1:11434", timeout=3)
        return response.status_code == 200
    except:
        return False


# ----------------------------------------
# Function to extract text from PDF
# ----------------------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

    return text


# ----------------------------------------
# Function to generate ONLY QUESTIONS
# ----------------------------------------
def generate_questions_only(content, question_type, difficulty, num_questions, model_name):
    llm = OllamaLLM(model=model_name, temperature=0.2)

    # Low RAM safe
    content = content[:3000]

    if question_type == "Mixed":
        prompt = f"""
You are an expert AI-based exam question generator for college students.

Generate ONLY QUESTIONS based on the study material provided.

Requirements:
- Difficulty Level: {difficulty}
- Total Questions: {num_questions}
- Include a balanced mix of:
  1. MCQ
  2. True/False
  3. Short Answer
  4. Long Answer
- Questions must be:
  - Clear
  - Relevant
  - Non-repetitive
  - Based only on the study material
- DO NOT provide any answers
- DO NOT provide explanations
- ONLY output the questions in a clean format

Formatting Rules:
1. Clearly label each question type
2. For MCQ:
   - Provide exactly 4 options (A, B, C, D)
3. For True/False:
   - Only provide the statement
4. For Short Answer:
   - Only provide the question
5. For Long Answer:
   - Only provide the question

Study Material:
{content}

Now generate ONLY the questions in a clean, structured, professional format.
"""
    else:
        prompt = f"""
You are an expert AI-based exam question generator for college students.

Generate ONLY {question_type} questions based on the study material provided.

Requirements:
- Difficulty Level: {difficulty}
- Number of Questions: {num_questions}
- Questions must be:
  - Clear
  - Relevant
  - Non-repetitive
  - Based only on the study material
- DO NOT provide any answers
- DO NOT provide explanations
- ONLY output the questions

Question-Type Rules:
- If MCQ:
  - Provide exactly 4 options (A, B, C, D)
- If True/False:
  - Only provide the statement
- If Short Answer:
  - Only provide the question
- If Long Answer:
  - Only provide the question

Study Material:
{content}

Now generate ONLY the questions in a clean, structured, professional format.
"""

    response = llm.invoke(prompt)
    return response


# ----------------------------------------
# Function to generate ANSWERS separately
# ----------------------------------------
def generate_answers_only(content, questions, model_name):
    llm = OllamaLLM(model=model_name, temperature=0.2)

    content = content[:3000]
    questions = questions[:4000]

    prompt = f"""
You are an expert AI tutor for college students.

Based ONLY on the study material below, provide accurate answers for the given questions.

Rules:
- Answer ONLY the provided questions
- Keep answers clear and correct
- Match the numbering of the questions
- For MCQ: mention the correct option and short reason
- For True/False: mention True or False
- For Short Answer: concise answer
- For Long Answer: descriptive answer
- Do NOT generate new questions
- Do NOT skip any question

Study Material:
{content}

Questions:
{questions}

Now provide ONLY the answers in a clean, structured format matching the question numbers.
"""

    response = llm.invoke(prompt)
    return response


# ----------------------------------------
# Streamlit Page Config
# ----------------------------------------
st.set_page_config(page_title="AI Question Generator", layout="wide")

# ----------------------------------------
# Session State
# ----------------------------------------
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = ""

if "generated_answers" not in st.session_state:
    st.session_state.generated_answers = ""

# ----------------------------------------
# App Title
# ----------------------------------------
st.title("📘 AI-Based Question Generator for Exam Practice")
st.write("Generate exam practice questions first, then reveal answers only when needed.")

# ----------------------------------------
# Sidebar Settings
# ----------------------------------------
st.sidebar.header("⚙️ Model Settings")

model_name = "llama3.2:3b"
st.sidebar.success(f"Using model: {model_name}")

if check_ollama_server():
    st.sidebar.success("🟢 Ollama server is running")
else:
    st.sidebar.error("🔴 Ollama server is NOT running")

st.sidebar.info(
    "Recommended for your laptop:\n"
    "- llama3.2:3b → Best balance of speed + accuracy for low RAM systems"
)

# ----------------------------------------
# Input Mode
# ----------------------------------------
input_mode = st.radio("Choose Input Type:", ["Upload PDF", "Enter Text/Topic"])

content = ""

# ----------------------------------------
# PDF Upload Option
# ----------------------------------------
if input_mode == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your study material PDF", type=["pdf"])

    if uploaded_file is not None:
        content = extract_text_from_pdf(uploaded_file)

        if content.strip():
            st.success("✅ PDF uploaded and text extracted successfully!")

            with st.expander("📄 View Extracted Text"):
                preview_text = content[:2000]
                st.write(preview_text + ("..." if len(content) > 2000 else ""))
        else:
            st.warning("⚠️ No readable text found in the PDF.")

# ----------------------------------------
# Manual Text Input Option
# ----------------------------------------
elif input_mode == "Enter Text/Topic":
    content = st.text_area(
        "Enter your notes, topic, or study material here:",
        height=250,
        placeholder="Example: Artificial Intelligence is the simulation of human intelligence in machines..."
    )

# ----------------------------------------
# Question Type Selection
# ----------------------------------------
question_type = st.selectbox(
    "Select Question Type:",
    ["MCQ", "True/False", "Short Answer", "Long Answer", "Mixed"]
)

# ----------------------------------------
# Difficulty Level
# ----------------------------------------
difficulty = st.selectbox(
    "Select Difficulty Level:",
    ["Easy", "Medium", "Hard"]
)

# ----------------------------------------
# Number of Questions
# ----------------------------------------
num_questions = st.slider("Select Number of Questions:", 1, 10, 5)

# ----------------------------------------
# Generate Questions Button
# ----------------------------------------
if st.button("🚀 Generate Questions"):
    if content.strip() == "":
        st.error("❌ Please upload a PDF or enter some text/topic first.")
    elif not check_ollama_server():
        st.error("❌ Ollama server is not running.")
        st.info("💡 Tip: Start Ollama or restart it.")
    else:
        with st.spinner(f"Generating questions using {model_name}..."):
            try:
                questions = generate_questions_only(
                    content=content,
                    question_type=question_type,
                    difficulty=difficulty,
                    num_questions=num_questions,
                    model_name=model_name
                )

                st.session_state.generated_questions = questions
                st.session_state.generated_answers = ""  # reset old answers

                st.success("✅ Questions generated successfully!")

            except Exception as e:
                st.error(f"❌ Error while generating questions: {e}")

# ----------------------------------------
# Display Generated Questions
# ----------------------------------------
if st.session_state.generated_questions:
    st.subheader("📝 Generated Questions")
    st.write(st.session_state.generated_questions)

    st.download_button(
        label="📥 Download Questions",
        data=st.session_state.generated_questions,
        file_name="generated_questions.txt",
        mime="text/plain"
    )

    # ----------------------------------------
    # Show Answers Button
    # ----------------------------------------
    if st.button("👀 Show Answers"):
        with st.spinner("Generating answers..."):
            try:
                answers = generate_answers_only(
                    content=content,
                    questions=st.session_state.generated_questions,
                    model_name=model_name
                )

                st.session_state.generated_answers = answers

            except Exception as e:
                st.error(f"❌ Error while generating answers: {e}")

# ----------------------------------------
# Display Answers Only After Button Click
# ----------------------------------------
if st.session_state.generated_answers:
    st.subheader("✅ Answers")
    st.write(st.session_state.generated_answers)

    st.download_button(
        label="📥 Download Answers",
        data=st.session_state.generated_answers,
        file_name="generated_answers.txt",
        mime="text/plain"
    )

# ----------------------------------------
# Footer / Tips
# ----------------------------------------
st.markdown("---")
st.markdown("### 💡 Setup Instructions")
st.code("ollama serve", language="bash")
st.code("ollama pull llama3.2:3b", language="bash")
st.code("streamlit run app.py", language="bash")