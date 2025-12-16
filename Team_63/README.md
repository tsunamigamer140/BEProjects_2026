# 🚀 AIML-Based Roadmap Generator for Skill Development

<div align="center">
  <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-RAG-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Pinecone-Vector_DB-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HuggingFace-Models-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Ollama-Mistral_7B-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white" />
</div>

<div align="center">
  <h3>An intelligent, AI-powered personalized roadmap generator designed for students, professionals, and learners using RAG, Vector Databases, and LLMs.</h3>
  <p>Built using Django, Python, Pinecone, Hugging Face, Ollama (Mistral-7B), and deployed on PythonAnywhere.</p>
  <br>
  <a href="https://rmpai.pythonanywhere.com/" style="font-size:22px; font-weight:bold;">🔗 Live Demo</a>
</div>

---

## 📌 Table of Contents

* [Features](#features)
* [Tech Stack](#tech-stack)
* [AI/ML Architecture](#aiml-architecture)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [Contributors](#contributors)
* [Contact](#contact)

---

## ✨ Features

### 🧠 AI-Powered Learning

* Personalized **skill roadmaps** using LLMs
* **RAG (Retrieval Augmented Generation)** for accurate responses
* Vector search using **Pinecone**
* AI-powered **NotebookLM-like document assistant**
* Custom query answering using **Mistral 7B via Ollama**

### 🧾 User System

* User authentication (Login/Register/Guest mode)
* Premium subscription system with admin approval
* Request limits for free users

### 💬 Smart Chat System

* Persistence chat memory
* Export chats as **PDF / JSON / TXT / PPTX**
* Notebook file upload + intelligent document analysis

### 🎨 UI/UX

* Modern Bootstrap UI
* Smooth animations
* Dark mode support

### ☁ Deployment

* Backend deployed on **PythonAnywhere**
* AI/RAG backend ready for Hugging Face Space/Ollama local server

---

## 🧩 Tech Stack

### **Backend**

* Django
* Python 3
* REST API
* Pinecone (Vector DB)
* RAG Pipeline
* Ollama Mistral 7B

### **Frontend**

* HTML5, CSS3, JavaScript
* Bootstrap

### **AI / ML**

* Hugging Face embeddings
* Mistral 7B (via Ollama)
* Vector search + retrieval

### **Deployment**

* PythonAnywhere (Live Server)
* HuggingFace Space (Optional Model Hosting)

---

## 🧠 AIML Architecture

```
User → Query → Embedding Model → Pinecone Vector DB → Relevant Chunks → LLM (Mistral 7B) → Final Answer
```

Flow:

1. User asks a question
2. Convert query to embedding
3. Search Pinecone vector DB
4. Retrieve top matches
5. Feed context to Mistral-7B (RAG)
6. AI produces a structured and accurate answer

---

## 📁 Project Structure

```
AIML_Based_Roadmap_Generator_for_Skill_Development/
│
├── ai_app/                 # Main app: roadmap AI, chat, premium
├── ai_notebook/            # NotebookLM-like AI processing
├── ai_roadmap_app/         # Django project settings
├── notebook_files/         # Uploaded documents
├── static/                 # JS/CSS/Images
├── media/                  # User uploads
│
├── manage.py
├── requirements.txt
├── test.py
└── README.md
```

---

## 🛠 Installation

### Prerequisites

* Python 3.10+
* Git
* Pinecone API Key
* HuggingFace Token
* Ollama installed (for local LLM)

### Step 1: Clone

```bash
git clone https://github.com/Kowshik-bh18/AIML_Based_Roadmap_Generator_for_Skill_Development.git
cd AIML_Based_Roadmap_Generator_for_Skill_Development
```

### Step 2: Virtual Environment

```bash
python -m venv myenv
myenv/Scripts/activate
pip install -r requirements.txt
```

### Step 3: Django Setup

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

### Step 4: Configure LLM (Local)

```bash
ollama run mistral
```

### Step 5: Pinecone Setup

Add in `.env` or settings:

```
PINECONE_API_KEY=xxxx
PINECONE_INDEX_NAME=roadmap-index
```

---

## 🚀 Usage

* Visit web app
* Create account or login as guest
* Chat with Roadmap AI
* Upload documents for NotebookLM-style responses
* Upgrade to premium for unlimited usage

---

## 🤝 Contributing

Want to improve this project?

1. Fork repo
2. Create branch
3. Commit changes
4. Open Pull Request

---

## 👥 Contributors

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/Kowshik-bh18.png" width="100px;" />
        <br><b>Kowshik BH</b><br>
        <a href="https://github.com/Kowshik-bh18">🚀 Developer</a>
      </td>
      <td align="center">
        <img src="https://github.com/madhusudhan-31.png" width="100px;" />
        <br><b>Madhu Sudhan</b><br>
        <a href="https://github.com/madhusudhan-31">💻 Contributor</a>
      </td>
      <td align="center">
        <img src="https://github.com/MDGanesha.png" width="100px;" />
        <br><b>MD Ganesha</b><br>
        <a href="https://github.com/MDGanesha">🔧 Contributor</a>
      </td>
    </tr>
  </table>
</div>

---

## 📞 Contact

<div align="center">

### **Kowshik BH**

[Email](mailto:kowshikbh18@gmail.com) | [LinkedIn](https://www.linkedin.com/in/kowshikbh) | [GitHub](https://github.com/Kowshik-bh18)

</div>

---

<div align="center">
  <h3>⭐ If you like this project, consider starring the repo!</h3>
</div>
