# ✉️ Spam vs Ham Classification using BOW & TF‑IDF

A Natural Language Processing (NLP) project focused on classifying **SMS or email messages** into **Spam (unwanted)** or **Ham (legitimate)** using **Bag‑of‑Words (BOW)** and **TF‑IDF** feature extraction techniques combined with machine learning classification models.

---

## 📌 Project Overview

This project implements a complete **end‑to‑end text classification pipeline** that includes:

* Collecting a labelled **spam/ham message dataset**
* Performing **text preprocessing and normalization**
* Engineering textual features using **BOW and TF‑IDF vectorization**
* Training and evaluating **machine learning classification models**
* Extracting **linguistic insights** that differentiate spam from legitimate messages

The primary objective is to **accurately detect unwanted messages** while understanding the **key textual indicators of spam**.

---

## 🧰 Tech Stack

**Language:** Python
**Libraries:** pandas, numpy, scikit‑learn, matplotlib, seaborn, NLTK / spaCy
**Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1️⃣ Data Collection & Pre‑processing

* Load labelled dataset containing **message text** and target label (**spam/ham**)
* Clean and normalize text:

  * Lowercasing
  * Removing punctuation and stop‑words
  * Optional **stemming or lemmatization**
* Split dataset into **training and testing sets** (e.g., 80/20) with stratification

### 2️⃣ Feature Engineering – BOW & TF‑IDF

Vectorize textual data using:

* **Bag‑of‑Words (CountVectorizer)**
* **TF‑IDF (TfidfVectorizer)**

Additional steps:

* Limit vocabulary size (e.g., top 5,000 words)
* Remove rare or noisy terms
* Compare performance between **BOW vs TF‑IDF** feature representations

### 3️⃣ Modeling

Baseline machine learning classifiers:

* **Logistic Regression**
* **Multinomial Naive Bayes**
* *(Optional)* Tree‑based or ensemble methods such as **Random Forest**

Models are trained separately on **BOW and TF‑IDF features** to evaluate representation impact.

### 4️⃣ Evaluation

Performance measured using:

* Accuracy
* Precision
* Recall
* F1‑Score
* Confusion Matrix

Special focus on **recall for the spam class**, since undetected spam has higher real‑world cost.

### 5️⃣ Insights & Application

* Identify **top spam‑indicative words** (e.g., *free, win, offer, click*)
* Distinguish linguistic patterns common in **ham messages**
* Demonstrate how **feature representation affects classification quality**
* Provide guidance for **spam filtering systems or alert mechanisms**

---

## 📁 Project Structure

```
Spam-Ham-Classification-BOW-TFIDF/
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   └── spam_ham_classification.ipynb
│── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   └── evaluate.py
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* **TF‑IDF features** slightly outperformed raw BOW by reducing noise and highlighting discriminative terms
* Words like **“free”, “win”, “offer”, “now”** strongly signaled spam messages
* **Logistic Regression and Naive Bayes** provided strong, efficient performance for this task
* Proper preprocessing (**stop‑word removal, lemmatization**) improved stability across feature sets

---

## 🚀 Future Improvements

* Transition to **word embeddings or transformer‑based models** for contextual understanding
* Expand to **multi‑language spam detection** and diverse communication formats
* Deploy as a **web/mobile application** with confidence scoring
* Integrate into **real‑time messaging systems** with user feedback loops
* Add **model explainability** using LIME or SHAP for transparency

---

## 🎯 Learning Outcomes

* Hands‑on experience with **text preprocessing and vectorization techniques**
* Understanding of **machine learning for NLP classification tasks**
* Insight into **real‑world spam detection system design**

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository and submit a pull request.

---

## ⭐ Support

If you found this project useful, consider **starring the repository** and sharing feedback.
