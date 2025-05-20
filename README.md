# 🧠 Mental Health Journal – NLP Module

This is the **NLP module** of our Mental Health Journal project. It analyzes a user's journal entry to detect their emotional state and generate a personalized mental health tip using:

- 🤗 Hugging Face Transformers (`SamLowe/roberta-base-go_emotions`)
- 🌟 Google Gemini AI (`gemini-1.5-flash`)

---

## 📂 Files Included

- `analyzer.py` – Main script that processes user input.
- `requirements.txt` – Python dependencies for emotion detection and tip generation.
- `.env.local` – Stores the Gemini API key (**not included in Git**).
- `.gitignore` – Prevents sensitive or unnecessary files from being committed (e.g., virtual environment, `.env.local`).

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/NoobCoder143/journal-emotion-analysis.git
cd journal-emotion-analysis
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv moodenv
# Windows
moodenv\Scripts\activate
# macOS/Linux
source moodenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Your Gemini API Key

- Go to your [Google Cloud Console](https://console.cloud.google.com/apis/credentials) and create an API key for the Gemini API.
- Create a file named `.env.local` in the root directory and add:

```env
api_key=YOUR_GEMINI_API_KEY_HERE
```

> ⚠️ Do not commit this file. It is ignored via `.gitignore`.

---

## 🧪 Sample Usage

You can call the `analyze_journal_entry` function in your backend like this:

```python
from analyzer import analyze_journal_entry

user_prompt = "I'm really nervous because I have an interview tomorrow."
result = analyze_journal_entry(user_prompt)
print(result)
```

### Sample Output:

```json
{
  "emotion": "nervousness",
  "confidence": 0.91,
  "tip": "Take a deep breath and remind yourself you're prepared. You've got this!"
}
```

---

## 📌 Notes

- This module is designed to be integrated into a backend API route.
- Hugging Face model is downloaded and cached locally after first run.
- Gemini API calls require an internet connection and a valid API key.
- Avoid calling the model in a loop unnecessarily — reuse loaded instances if integrating.

---


