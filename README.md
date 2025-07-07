# 🧠 Scheme Research Application

> A Streamlit-based assistant to extract, summarize, and query government schemes using cutting-edge LLM and retrieval technologies.


## 📑 Table of Contents
```
- [About the Project](#about-the-project)
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)
- [License](#license)
- [Contact](#contact)

```

## 📖 About the Project

Many citizens find it hard to understand the eligibility, process, and benefits of government schemes due to the dense, scattered nature of information.

This project solves that by enabling users to input scheme URLs or upload PDFs, then:
- summarizes the key scheme details.
- Ask questions about the scheme using an AI chatbot.
- View or download source content for verification.

It combines document parsing, semantic search, and LLM-powered QA to deliver an intelligent, research-friendly interface.


## 🎥 Demo
```
> [https://smart-scheme-research-app.streamlit.app/]

```

## 🗂 Demo: Government Scheme Links
```
Use these sample government scheme URLs to try the app:

- 🔗 [LIC Bima Sakhi Yojana – Sarkari Yojana](https://sarkariyojana.com/lic-bima-sakhi-yojana-apply-online/)
- 🔗 [Pradhan Mantri SVANidhi Scheme – Wikipedia ](https://en.wikipedia.org/wiki/Pradhan_Mantri_SVANidhi_Scheme)
- 🔗 [ Employment Linked Incentive Scheme – Sarkari Yojana](https://sarkariyojana.com/employment-linked-incentive-scheme/#google_vignette)

📌 Tip: Paste any of these URLs in the sidebar under “URLs” and click “Process” to test.

```

## ✨ Features
```

- 🔗 Accepts multiple government scheme **URLs** or uploads **PDFs**.
- 📝 summarizes:
  - Benefits
  - Application Process
  - Eligibility
  - Required Documents
- 💬 Ask custom questions using LLM (Groq LLaMA3).
- 📂 Track source documents and download PDFs.
- 🧠 Built on LangChain + FAISS for scalable document QA.

```

## 🛠️ Tech Stack
```
- **Python**
- **Streamlit**
- **LangChain**
- **FAISS (Vector DB)**
- **HuggingFace Transformers**
- **Groq (LLM API)**

```
## 🧬 Clone the Repository
```
git clone https://github.com/vishnuvardhanreddythornala/Scheme-Research-Application.git
cd Scheme-Research-Application

```

## ⚙️ Installation

```bash

# Navigate to the project folder
cd Scheme-Research-Application

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
## 🔐 Configure API Keys
```
Create a .env file in the root directory:

[api_keys]
groq_api_key = your_groq_api_key_here
```
# Run the app
```
streamlit run main.py

```
## 🚀 Usage
```
- Paste scheme URLs or upload PDFs in the sidebar.

- Click "Process URLs" to extract and embed the content.

- Click "Generate Summary" for a structured overview.

- Ask any follow-up questions and get answers with source traceability.

- Download PDFs or visit linked sources as needed.
```

## 📁 Project Structure
```
scheme-research-tool/
├── .venv/                      # Virtual environment (not committed)
├── faiss_store_openai/         # Saved FAISS index
├── logs/                       # App logs
├── uploads/                    # Uploaded PDF files
├── main.py                     # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env			                  # API keys (not committed)
├── .gitignore                  # Ignore logs, uploads, etc.
└── README file

```
## ⚡ Limitations and Future Work
```
- 🔒 API key must be manually added to .env.
- 🔍 Summarization accuracy depends on the clarity of source documents.
- 📄 Current focus is on PDFs and URLs only.
```

## 🛠 Future Improvements:
```
- Adding audio/text-to-speech interface for accessibility.
- Integrating more robust analytics for query and feedback logs.
- Enhancing support for multilingual documents.
```

## 📞 Contact
```
Name: Thornala Vishnu Vardhan Reddy
GitHub: @vishnuvardhanreddythornala
Email: vishnuvardhanreddythornala@gmail.com
```
