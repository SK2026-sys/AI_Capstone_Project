# AI Capstone Project
## Project Overview
This Priject is an AI application that allows a user to upload a PDF document
The application processes the document and allow the users to ask questions using document retrieval and an AI model.

## Architecture
- Streamlit provides the user interface for uploading documents and asking questions.
- Sentence Transformers converts document chunks and user questions into embeddings.
- FAISS stores the embeddings and perform similarity search to retrieve relevant document.
- Google Gemini generates responses based on either retrieved document context or general questions.
- The agent planner decides whether to use document retrieval or the general LLM path before generating a response.
- Validation and safety checks help reduce unsupported, unsafe, or incomplete responses.

## Workflow
1. The user uploads a supported document through the Streamlit interface.
2. The application extracts the text from the uploaded document.
3. The extracted text is divided into smaller chunks for processing.
4. Sentence Transformers converts the chunks into embeddings.
5. FAISS stores the embeddings and searches for document chunks relevant to the user question.
6. The agent planner decides whether the question should use document retrieval or the general LLM path.
7. Gemini generates the response using the selected path.
8. Validation and safety checks are applied before the final response is shown to the user.

## Limitations
- The quality of document-based answers depends on content and clarity of the uploaded document.
- The system may not retrieve the correct information if the relevant content is not captured in the retrieved chunks.
- The quality of general answers dpends on the capabilities of the Gemini model.
- The application requires an internet connection to access the Gemini model.
- AI-generated responses may still contain errors, so important information should be verified by the user.

## Deployment
- The application is deployed on Streamlit Community Cloud.
- The source code is stored in a GitHub repository and connected to the deployed application.
- The Google API key is stored securely using Streamlit Secrets and is not included in the public repository.
- The deployed application was tested successfully with document upload, retrieval and AI generated responses.