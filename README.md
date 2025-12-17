# 📌 Sentiment Analysis using Deep Learning (LSTM / GRU)

A complete **Natural Language Processing (NLP)** project that performs **sentiment analysis on IMDb movie reviews** using **Deep Learning (LSTM & GRU)** and compares results with a **traditional Machine Learning baseline (Logistic Regression)**.

This project is designed to demonstrate **end‑to‑end NLP workflow**, clean project structuring, and strong theoretical understanding—making it suitable for **GitHub portfolios, CVs, and Master’s SOPs**.

---

## 🚀 Project Overview

**Goal:**
Classify IMDb movie reviews as **Positive** or **Negative** using deep learning models.

**Key Highlights:**

* Text preprocessing & tokenization
* Word embeddings
* Sequential deep learning models (LSTM & GRU)
* Comparison with Logistic Regression (ML baseline)
* Clear explanation of *why deep learning outperforms traditional ML for NLP*

---

## 🧠 Learning Outcomes

Through this project, I learned:

* How to preprocess raw text data for NLP tasks
* Tokenization, padding, and vocabulary management
* Understanding word embeddings
* Training LSTM and GRU models for sequence learning
* Handling vanishing gradient problems
* Comparing deep learning with traditional ML approaches

---

## 📂 Dataset

* **IMDb Movie Reviews Dataset**
* Binary sentiment classification:

  * `positive` → 1
  * `negative` → 0

Dataset structure:

```
data/raw/imdb_reviews.csv
```

Columns:

* `review` – movie review text
* `sentiment` – sentiment label

---

## 🛠 Tech Stack

* **Programming Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **NLP Tools:** NLTK
* **Machine Learning:** Scikit‑learn
* **Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure

```
text-classification-nlp/
│
├── data/
│   ├── raw/
│   │   └── imdb_reviews.csv
│
├── notebooks/
│   └── exploratory.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── baseline_ml.py
│
├── results/
│   └── models/
│       ├── lstm_model.h5
│       └── gru_model.h5
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/text-classification-nlp.git
cd text-classification-nlp
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Project

```bash
python main.py
```

Models will be saved in:

```
results/models/
```

---

## 📊 Model Comparison

| Model               | Description                    | Strength                               |
| ------------------- | ------------------------------ | -------------------------------------- |
| Logistic Regression | TF‑IDF based ML baseline       | Fast, interpretable                    |
| LSTM                | Long Short‑Term Memory network | Handles long‑term dependencies         |
| GRU                 | Gated Recurrent Unit           | Faster & efficient alternative to LSTM |

---

## 🧠 Why LSTM / GRU over Traditional ML?

Traditional ML models (e.g., Logistic Regression):

* Treat text as independent features
* Lose word order and context

LSTM / GRU models:

* Preserve **sequential information**
* Learn **long‑term dependencies** in text
* Capture sentiment patterns spread across sentences

---

## ⚠️ Vanishing Gradient Problem (Explained)

* In standard RNNs, gradients shrink during backpropagation
* This prevents learning long‑range dependencies

**LSTM & GRU solve this using gates:**

* Control what information to remember or forget
* Enable stable gradient flow over long sequences

---

## 📓 Exploratory Notebook

The `exploratory.ipynb` notebook includes:

* Class distribution analysis
* Word clouds for positive & negative reviews
* Sequence length visualization
* Quick LSTM training
* Accuracy & loss plots

---

## 🏁 Conclusion

This project demonstrates:

* Strong understanding of NLP fundamentals
* Practical deep learning skills
* Clean software engineering practices
* Readiness for advanced studies or applied research in AI

---

⭐ This project is intended for academic demonstration and postgraduate applications.

