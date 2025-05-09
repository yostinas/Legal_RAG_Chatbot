
# Legal RAG-Based Chatbot (Fully Offline)

## 🚀 Project Overview
This is a fully functional Legal RAG-Based Chatbot powered by a Local LLM (GPT-J-6B) and RAG (Retrieval-Augmented Generation).

## 📂 Project Structure
- `app.py`: Main application file.
- `requirements.txt`: Dependency file.
- `README.md`: Setup and usage instructions.
- `uploads/`: Directory for uploaded legal documents.
- `models/GPT-J-6B/`: Local model files.

## 🚀 Setup Instructions
1. Make sure you have Python 3.9+ installed.
2. Download GPT-J-6B model files and place them in the `models/GPT-J-6B/` folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the chatbot:
   ```bash
   streamlit run app.py
   ```

## ✅ Usage
- Upload legal documents (PDF or TXT).
- Enter your legal question in the text box.
- The chatbot will generate an answer based on the document content.
