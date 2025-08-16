---

# Clinical-Instruct API

This repository contains a **Flask-based API** that serves a fine-tuned GPT-2 (355M) language model for clinical reasoning tasks. The fine-tuning was inspired by concepts from *Sebastian Rachka’s* book **“How to Build Large Language Models from Scratch”** and applied to the dataset from the [Zindi Kenya Clinical Reasoning Challenge](https://zindi.africa/competitions/kenya-clinical-reasoning-challenge/data).

## Project Overview

This project started as a way to put into practice some of the ideas I learned while studying how instruction fine-tuning works at a low level. I decided to revisit a clinical reasoning dataset from a past hackathon and experiment with applying GPT-2 (355M) to it. The model was trained and wrapped into a simple Flask API so that it could be queried directly or integrated into other applications.

During testing, I evaluated the responses using Microsoft Phi-3.5-mini-instruct, and the fine-tuned model reached an automated conversational benchmark score of **60.47**. While not state-of-the-art, the results showed that the approach worked reasonably well and provided a solid hands-on learning experience. To make interaction easier, I also built a sample dashboard that connects to the API and lets users try out the model through a simple interface.

---

## Setup Instructions

### Requirements

* Python (≥3.8)
* pip (Python package manager)

### Steps

1. **Navigate to your desired project directory**

   ```bash
   cd path/to/your/project
   ```

2. **Create a virtual environment and activate it**

   * On Windows (Command Prompt):

     ```bash
     python -m venv myenv
     myenv\Scripts\activate.bat
     ```
   * On macOS/Linux:

     ```bash
     python -m venv myenv
     source myenv/bin/activate
     ```

3. **Clone this repository**

   ```bash
   git clone https://github.com/BetikuOluwatobi/clinical-instruct-api.git
   ```

4. **Navigate into the repository**

   ```bash
   cd clinical-instruct-api
   ```

5. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

6. **Prepare model weights**

   * Inside the repo, navigate to the `static` directory.
   * Create a folder named `weights`.
   * Download the model weights from your [Google Drive](https://drive.google.com/drive/u/1/my-drive) and place them into the `weights` directory.
   * ⚠️ If you rename the file, update the **name variable** in `app.py`.

---

## Running the API

Start the Flask server on port `3000` (or any port of your choice):

```bash
flask --app app run --port 3000
```

The server will output the localhost address, e.g.:

```
http://127.0.0.1:3000/
```

Visiting that address will display the **API documentation page**.

---

## Testing the API

### 1. Test via Browser

You can test the API directly in your browser:

```bash
http://127.0.0.1:3000/instruct?prompt={your_text_prompt}&max_num_tokens={num_of_tokens_to_generate}&temperature=1&top_k=5
```

Example:

```
http://127.0.0.1:3000/instruct?prompt=Explain+the+symptoms+of+malaria&max_num_tokens=200&temperature=1&top_k=5
```

---

### 2. Test via `curl`

```bash
curl "http://127.0.0.1:3000/instruct?prompt=Explain+the+symptoms+of+malaria&max_num_tokens=200&temperature=1&top_k=5"
```

---

### 3. Test via Python `requests`

```python
import requests

url = "http://127.0.0.1:3000/instruct"
params = {
    "prompt": "Explain the symptoms of malaria",
    "max_num_tokens": 200,
    "temperature": 1,
    "top_k": 5
}

response = requests.get(url, params=params)
print(response.json())
```

---

## Dashboard Integration

A simple dashboard app is available for visualizing and interacting with the API:
👉 [Clinical-Instruct Dashboard](https://github.com/BetikuOluwatobi/clinical-instruct-dashboard)

This dashboard queries the API on port `3000` by default.

---

## Notes

* Model inference on CPU is **slow** (≈2 minutes per response for 100 tokens).
* For best performance, run on a machine with GPU support.
* You can extend this API with your frontend/dashboard or integrate it into other systems.

---
