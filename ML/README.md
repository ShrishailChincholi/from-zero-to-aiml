## 📌 What is Machine Learning?
Machine Learning is a subset of Artificial Intelligence that allows systems to **learn from data and improve performance without being explicitly programmed**.

---

## 📚 Types of Machine Learning

### 1️⃣ Supervised Learning
- Works with labeled data
- Used for prediction and classification

**Examples:**
- House price prediction
- Spam email detection

**Algorithms:**
- Linear Regression
- Logistic Regression
- Decision Tree
- KNN

---

### 2️⃣ Unsupervised Learning
- Works with unlabeled data
- Finds hidden patterns

**Examples:**
- Customer segmentation
- Data grouping

**Algorithms:**
- K-Means Clustering
- Hierarchical Clustering
- PCA

---

### 3️⃣ Reinforcement Learning
- Learns using rewards and penalties
- Based on trial and error

**Examples:**
- Game AI
- Self-driving cars

---

## ❓ Types of Machine Learning Problems

- **Classification** – Predict categories (Spam / Not Spam)
- **Regression** – Predict numerical values (Price, Salary)
- **Clustering** – Group similar data
- **Recommendation Systems** – Suggest products or movies
- **Anomaly Detection** – Detect fraud or unusual behavior

---

## Day 2
## 📊 Types of Data

In Data Science and Machine Learning, data is mainly divided into **Structured** and **Unstructured** data.

---

### 1️⃣ Structured Data
Structured data is **well-organized** and stored in a fixed format such as **rows and columns**.

**Characteristics:**
- Easy to store and analyze
- Stored in databases and spreadsheets
- Follows a predefined schema

**Examples:**
- Excel files
- SQL tables
- CSV files
- Customer records (ID, Name, Age, Salary)

**Usage:**
- Data analysis
- Machine learning models
- Business reports

---

### 2️⃣ Unstructured Data
Unstructured data does **not have a fixed format** and is difficult to organize.

**Characteristics:**
- No predefined structure
- Large volume of data
- Harder to analyze

**Examples:**
- Text documents
- Images
- Audio files
- Videos
- Social media posts
- Emails

**Usage:**
- Natural Language Processing (NLP)
- Computer Vision
- Speech Recognition

---

### 🔁 Semi-Structured Data
Semi-structured data is a mix of structured and unstructured data.

**Examples:**
- JSON files
- XML files
- HTML web data
- Log files

---

## 🧠 Summary Table
| Data Type | Structure | Examples |
|---------|----------|----------|
| Structured | Fixed format | CSV, SQL, Excel |
| Unstructured | No format | Images, Videos, Text |
| Semi-Structured | Partial structure | JSON, XML |

## 📏 Model Evaluation

Model evaluation is the process of **measuring how well a Machine Learning model performs** on unseen (new) data.

It helps us understand:
- How accurate the model is
- Whether the model is overfitting or underfitting
- Which model performs better

---

## 🧪 Types of Evaluation

### 1️⃣ Classification Evaluation
Used when the output is a **category** (Yes/No, Spam/Not Spam).

**Common Metrics:**
- **Accuracy** – Overall correctness of the model
- **Precision** – How many predicted positives are correct
- **Recall** – How many actual positives are correctly predicted
- **F1-Score** – Balance between precision and recall
- **Confusion Matrix** – Shows correct and incorrect predictions

**Examples:**
- Spam detection
- Disease prediction

---

### 2️⃣ Regression Evaluation
Used when the output is a **numerical value**.

**Common Metrics:**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score (Coefficient of Determination)**

**Examples:**
- House price prediction
- Salary prediction

---

### 3️⃣ Clustering Evaluation
Used for **unsupervised learning** models.

**Common Metrics:**
- **Silhouette Score**
- **Davies–Bouldin Index**
- **Inertia**

**Examples:**
- Customer segmentation
- Market analysis

---

### 4️⃣ Train-Test Evaluation
Used to check model performance on unseen data.

**Methods:**
- Train-Test Split
- Cross-Validation (K-Fold)

---

## 📊 Evaluation Summary
| Problem Type | Evaluation Metrics |
|-------------|------------------|
| Classification | Accuracy, Precision, Recall, F1 |
| Regression | MAE, MSE, RMSE, R² |
| Clustering | Silhouette, Davies-Bouldin |
| General | Cross-Validation |

---

## 🎯 Why Evaluation is Important
- Improves model performance
- Prevents overfitting
- Helps in model comparison
- Ensures reliability

