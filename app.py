from dotenv import load_dotenv
load_dotenv()
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from groq import BadRequestError
import streamlit as st
from time import sleep
from langchain_community.vectorstores import InMemoryVectorStore
import re

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def document_process(path):

    ## Document Loading
    loader = PyPDFLoader(path)
    docs = loader.load()

    # print(len(docs))

    cleaned_docs = []

    for doc in docs:

        text = doc.page_content

        # -----------------------------
        # Remove repetitive OCR garbage
        # -----------------------------
        text = re.sub(r"CEGP\d+", " ", text)
        text = re.sub(r"\d+\.\d+\.\d+\.\d+", " ", text)
        text = re.sub(r"static-\d+", " ", text)

        text = re.sub(r"SEAT\s*No\.?:?", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"P\.?T\.?O\.?", " ", text, flags=re.IGNORECASE)

        text = re.sub(r"Total No\.? of Questions.*", " ", text)
        text = re.sub(r"Total No\.? of Pages.*", " ", text)

        text = re.sub(r"Time\s*:.*", " ", text)
        text = re.sub(r"Max\.\s*Marks.*", " ", text)

        text = re.sub(r"Instructions to the candidates:.*?(?=Q1)", " ", text, flags=re.S)

        # remove decorative symbols
        text = re.sub(r"[♦◆●■▲▼★☆◊○◘◙]+", " ", text)

        # normalize whitespace
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        doc.page_content = text.strip()

        cleaned_docs.append(doc)

    ## Document Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=120,
        separators=[
            "\nQ",
            "\nOR",
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    docs = splitter.split_documents(cleaned_docs)

    # print("Chunks :", len(docs))

    ## Embedding Generation
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    ## Vector Store Memory Database
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True


st.set_page_config(page_icon="assets/image.png",page_title="ExamSaar")
st.subheader("ExamSaar - AI Powered SPPU Exam Paper Intelligence 💻",text_alignment="center")
st.subheader("Analyze Previous Year Papers  \n• Find Repeated Questions  \n• Predict Trends  \n• Prepare Smarter",text_alignment="center")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

## Document Upload Page
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Upload Your Exam Document Here",type="pdf")

    if file:
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            pdf_path = tmp.name

        with st.spinner("Processing..."):
            document_process(pdf_path)

        # Delete the temporary PDF after processing
        os.remove(pdf_path)

        st.markdown("Document Processed Successfully...!!!✅")
        sleep(2)
        st.rerun() # this will run the code again but but it will not change the session keep that in mind


## Now build the chat ui

if st.session_state.document_uploaded and st.session_state.vector_db:
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])
    
    query = st.chat_input("Ask anything about the pdf ...")
    if query:
        st.chat_message("user").markdown(query)

        st.session_state.messages.append({"role":"user","content":query})


        query_lower = query.lower()

        if any(word in query_lower for word in [
            "important",
            "repeated",
            "repeat",
            "prediction",
            "predict",
            "trend",
            "weightage",
            "frequent"
        ]):
            k = 10
        else:
            k = 5

        similar_docs = st.session_state.vector_db.similarity_search(query, k=k)

        context = ""
        for i, doc in enumerate(similar_docs, start=1):
            page = doc.metadata.get("page", "Unknown")
            context += f"""
        ========== DOCUMENT {i} (Page {page + 1}) ==========
        {doc.page_content}
        """
        
        prompt = f"""
        You are an intelligent SPPU Exam Assistant designed to analyze previous year examination papers.

        The uploaded document may contain:
        • In-Sem Question Papers
        • End-Sem Question Papers
        • Previous Year Papers
        • Multiple Years
        • Different Subjects
        • Repeated Questions

        STRICT RULES:

        1. Answer ONLY from the supplied document context.
        2. Never use external knowledge.
        3. Never fabricate answers.
        4. If sufficient information is unavailable, reply:
        "The uploaded exam papers do not contain enough information to answer this question."
        5. If the same topic appears in multiple papers or years:
        - Merge the information.
        - Avoid unnecessary repetition.
        - Mention that it is a frequently repeated topic if applicable.
        6. If the user asks:
        - Important Questions
        - Frequently Asked Questions
        - Repeated Questions
        - Topic-wise Questions
        - Unit-wise Questions
        - Exam Prediction
        - Weightage
        analyze ONLY the uploaded papers and provide insights based on document evidence.
        7. If question numbers (Q1, Q2, etc.) are present in the context, preserve them where useful.
        8. Format answers using bullet points or numbering whenever appropriate.
        9. Be concise, accurate, and exam-focused.

        -----------------------------
        DOCUMENT CONTEXT
        -----------------------------
        {context}

        -----------------------------
        USER QUESTION
        -----------------------------
        {query}

        Generate the response now.
        """

        try:
            result = llm.invoke(prompt)
            st.chat_message("ai").markdown(result.content)
            st.session_state.messages.append({"role":"ai","content":result.content})
        except BadRequestError as e:
            print("\nLLM answer generation failed")
            print(e)


st.markdown(
    """
    <div style="text-align:center; color:#808080; font-size:13px;">
        © 2026 <b>ExamSaar</b> | AI-Powered SPPU PYQ Exam Assistant <br>
        Developed by <b>Shravan Shidruk</b> • Coders Of Pune
    </div>
    """,
    unsafe_allow_html=True,
)