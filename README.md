# XLate AI

Welcome to **XLate AI**! This project brings the power of advanced AI to your local machine, allowing you to run Large Language Models (LLMs) completely **offline**.

XLate AI is more than just a chatbot; it comes with advanced features like **Retrieval-Augmented Generation (RAG)**, enabling it to answer questions based on your own documents, and a powerful **translation** engine. Built with Ollama for local model inference and Chainlit for a user-friendly interface, XLate AI ensures your data remains private and secure on your own hardware.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:
* **Python 3.8+**
* **pip** (Python package installer)

---

## Getting Started

Follow these steps to get your local environment set up and running.

### 1. Clone the Repository
First, clone this repository to your local machine and navigate into the project directory.
```bash
git clone <your-repository-url>
cd XLate-AI 
```
*(Note: Replace `<your-repository-url>` with your actual Git repository URL and `XLate-AI` with your directory name if different.)*

### 2. Install Python Packages
Install all the necessary Python libraries listed in the `requirements.txt` file. This command will set up all the dependencies for the project.
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
XLate AI requires Ollama to be running on your machine to serve the local language models.
* Download and install **Ollama** from the [official website](https://ollama.com/).

### 4. Download Ollama Models
Next, pull the required models from the Ollama library. This project uses `qwen2.5` for generating responses and `nomic-embed-text` for creating embeddings.

Open your terminal and run the following commands one by one:
```bash
ollama pull qwen2.5
```
```bash
ollama pull nomic-embed-text
```
*This may take some time depending on your internet connection and system specifications.*

---

## How to Run the Application

Once the setup is complete, you can start the XLate AI application using Chainlit.

Run the following command in your terminal from the project's root directory:
```bash
chainlit run script.py -w
```
The `-w` flag enables "watch mode," which automatically reloads the application whenever you save a change in the source file.

Your application should now be running! Open your web browser and navigate to the local address provided in the terminal (usually `http://localhost:8000`).
