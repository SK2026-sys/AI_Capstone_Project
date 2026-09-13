import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")
def agent_plan(question):
    planner_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    plan_prompt = f"""Plan how to answer this question.
    Choose one tool: DOCUMENT_RETRIEVAL or GENERAL_llM.
    Question: {question}
    Do not answer the question. Only choose the routing tool.
    Return only the tool name."""
    plan = planner_llm.invoke(plan_prompt)
    plan = plan.content[0]["text"].strip()
    return plan

model = SentenceTransformer("all-MiniLM-L6-v2")
st.title("AI Capstone Project")
uploaded_file = st.file_uploader("Upload a document", type=["pdf" , "txt", "csv", "xlsx"])
question = st.text_input("Ask a question about the document")
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        text = text.strip()
    elif file_type == "txt":
        text = uploaded_file.getvalue().decode("utf-8")
    elif file_type == "csv":
        text = uploaded_file.getvalue().decode("utf-8")
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
        text = df.to_string(index=False)
    if text:
        st.success("Document processed successfully")
        chunk_size = 500
        chunk_overlap = 50
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size -chunk_overlap)]
        chunk_data =[]
        for index, chunk in enumerate(chunks):
            chunk_data.append({"id": index, "text":chunk})
        embeddings = model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        st.write("Vectors stored in FAISS;",index.ntotal)
        st.write("Number of embeddings;",len(embeddings))

        
    else:
        st.warning("No readable text found in the document")
    if question:
        agent_action = agent_plan(question)
        st.write("Agent plan:", agent_action)
        question_embedding = model.encode([question])
        distances, indices =index.search(question_embedding, k=7)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        
        context = "\n\n".join(retrieved_chunks)
        llm =ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=api_key)
        decision_prompt = f"""
Classify this question as DOCUMENT or GENERAL
Document context: {context}
Question: {question}
Return only one word: DOCUMENT or GENERAL."""
        try:
            decision = llm.invoke(decision_prompt)
            decision = decision.content[0]["text"].strip().upper()
            if decision == "DOCUMENT":
                prompt = f"""
            Answer the user's question using only the document context below.
            Context: {context}
            Question: {question}
            Give a clear and concise answer based only on the context.
            """   
        
        
            
        
            


                response = llm.invoke(prompt)
                ai_answer = response.content[0]["text"]
                validation_prompt = f"""
                Check whether the answer is supported by the document context.
                Context: {context}
                Answer: {ai_answer}
                Reply only VALID if the answer is supported by the context. Otherwise reply INVALID.
                """
                validation = llm.invoke(validation_prompt)
                validation = validation.content[0]["text"].strip().upper()
                if validation == "VALID":
                    st.markdown(ai_answer)
                else:
                    st.warning("The generated answer could not be validated against the document.")
                
            else:
                safe_question = f"Answer safely and responsibly. Do not provide harmful or dangerous instructions. Question: {question}"
                general_response = llm.invoke(safe_question)
                st.markdown(general_response.content[0]["text"])
        except Exception as e:
            st.error(str(e))   
    else:
        st.warning("Please enter a question.")

