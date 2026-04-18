import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    cache_folder="./models"
)
db = None
chat_history = []

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatOllama(
    model="phi",
    temperature=0.3
)

def process_pdf(file):
    global db, chat_history

    if file is None:
        return "Please upload a PDF first."

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    if not documents:
        return "No content found in PDF."

    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    if not docs:
        return "No readable text found in PDF."

    db = FAISS.from_documents(docs, embeddings)
    chat_history = []

    return "PDF processed successfully."

def ask_question(query):
    global db, chat_history

    if db is None:
        return "Please upload and process a PDF first."

    if not query.strip():
        return "Please enter a question."

    docs = db.similarity_search(query, k=2)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an AI assistant.

Answer the question using the context below.
If partial information is available, try to answer based on that.

Only say "I could not find the answer" if absolutely nothing relevant is present.

Context:
{context}

Question:
{query}

Answer in simple and clear language:
"""

    response = llm.invoke(prompt)
    answer = response.content

    chat_history.append((query, answer))
    return answer

def handle_question(query):
    response = ask_question(query)
    return response, ""

def reset_chat():
    global chat_history
    chat_history = []
    return "", ""

with gr.Blocks() as app:
    gr.Markdown("# Shimanshu's AI PDF Chatbot (LLM + RAG)")

    file = gr.File(label="Upload PDF", file_types=[".pdf"])
    process_btn = gr.Button("Process PDF")
    status = gr.Textbox(label="Status")

    q = gr.Textbox(label="Ask Question", placeholder="Type your question and press Enter")
    ans = gr.Textbox(label="Answer", lines=12)

    reset_btn = gr.Button("Reset Chat")

    process_btn.click(process_pdf, inputs=file, outputs=status)
    q.submit(handle_question, inputs=q, outputs=[ans, q])
    reset_btn.click(reset_chat, inputs=[], outputs=[q, ans])

app.launch(share=True)