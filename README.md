# 🤖 AI-Powered Job Ad Generator (RAG)

![Python](https://img.shields.io/badge/Python-3.9-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange) ![openai/gpt-oss-20b](https://img.shields.io/badge/Model-openai/gpt-oss-20b-purple) ![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

A Generative AI application designed for HR professionals and recruiters to instantly draft high-quality, professional job advertisements. By utilizing **RAG (Retrieval-Augmented Generation)**, this tool ensures that generated ads are grounded in real-world data scraped from top job portals like LinkedIn.

## 🚀 Project Overview

Writing the perfect job description from scratch is time-consuming. This tool solves that by combining the creativity of Large Language Models (LLMs) with the relevance of real market data.

**Key Workflow:**
1.  **User Input:** The user provides basic details (Role, Company, Location, Tech Stack).
2.  **Semantic Search:** The system queries a local **FAISS Vector Database** containing thousands of scraped job descriptions to find historically similar roles.
3.  **Context Augmentation:** These "real-world" examples are retrieved and fed into the LLM as context.
4.  **Generation:** The **Groq API (openai/gpt-oss-20b)** generates 3 distinct, professional job ad variants based on the user's requirements and the retrieved context.

## 🏗️ Architecture

The project follows a standard RAG pipeline:

1.  **Data Ingestion:** Job postings were scraped from various portals (LinkedIn, etc.).
2.  **Vector Store:** Data was cleaned, tokenized, and converted into embeddings using `Sentence-Transformers`. These embeddings are stored in a local **FAISS** index.
3.  **Retrieval:** When a user asks for a "Data Scientist" role, the app performs a similarity search in FAISS to find the top 4 most relevant existing job ads.
4.  **Generation:** A prompt is constructed (`User Input` + `Retrieved Job Context`) and sent to the **openai/gpt-oss-20b** model via Groq to produce the final output.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM Engine:** [Groq API](https://groq.com/) (running openai/gpt-oss-20b)
* **Vector Database:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Deployment:** Docker & Hugging Face Spaces

## 📂 Project Structure

```bash
job-ad-generator/
├── app.py                # Main Streamlit application logic
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration for deployment
├── faiss_index/          # Folder containing the Vector DB
│   ├── jobs.index        # The actual FAISS index file
│   └── jobs_meta.json    # Metadata (text) associated with embeddings
└── README.md             # Project documentation


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
