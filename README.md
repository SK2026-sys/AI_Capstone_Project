# AI Capstone Project
## Project Overview
This Project is an AI application that allows a user to upload a PDF document
The application processes the document and allows users to ask questions using document retrieval and an AI model.

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
4. Sentence Transformers convert the chunks into embeddings.
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

## System Setup
- The project was developed in Visual Studio Code using Python.
- Required Python libraries are listed in the requirements.txt file.
- Environment variables such as the Google API key are stored in a .env file for local development.
- Git and GitHub are used for version control and storing the project source code.
- Streamlit is used to run the application locally and Streamlit Community Cloud is used for public deployment.

## Agent Roles
- The agent planner analyzes the user's question and decides whether to use document retrieval or the general LLM path.
- For the document questions, the system retrieves relevant chunks from FAISS and provides them as context to Gemini.
- For general questions, the system uses Gemini to generate a response without relying on the uploaded document.
- The agent therefore performs routing, retrieval and response generation using the available tools.


## Challenges Faced
- The agent planner initially reqired adjustments so that it returned only the correct routing tool instead of answering the question itself.
- The initial Streamlit Community Cloud deployment required troubleshooting and a reboot before the application deployed successfully.