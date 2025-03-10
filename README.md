
# 🏥 Stroke Risk Predictor

## 📌 Overview
The **Stroke Risk Predictor** is a machine learning-based web application designed to predict the risk of stroke based on user input. It helps in assessing stroke probability using key health indicators.

## ⚙️ Features
- ✅ Predicts stroke risk using a trained **machine learning model**.
- ✅ Takes inputs such as **age, hypertension, heart disease, BMI, smoking status, etc.**
- ✅ Provides a **user-friendly web interface** using **Streamlit**.
- ✅ **Fast and lightweight** model deployment.

## 🌍 Live Demo
Try out the **Stroke Risk Predictor** live:  
🔗 [Live Demo](https://strokeriskpredictor.netlify.app/) 

---

## 🛠️ Technologies Used
- **Python** 🐍
- **Scikit-learn** 🤖
- **Pandas & NumPy** 📊
- **Django** 🌐
- **Streamlit** 🎨
- **Pickle for model serialization** 📁

---

## 📂 File Structure
```
/stroke-risk-predictor
│── /model
│   ├── stroke_model_5MB.pkl.gz  # Trained ML Model
│── /webapp
│   ├── app.py  # Streamlit Frontend
│   ├── requirements.txt  # Dependencies
│── README.md
```

---

## 🚀 Installation & Setup
### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-repo/stroke-risk-predictor.git
cd stroke-risk-predictor
```
### 2️⃣ Create a virtual environment (optional but recommended)  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the web application  
```bash
streamlit run webapp/app.py
```

---

## 🎯 Usage
1. Open the web application in your browser.
2. Enter **health details** such as age, smoking status, BMI, etc.
3. Click **Predict** to check the risk of stroke.
4. The model will return a **probability score** indicating stroke risk.

---

## 📜 Code Examples

### 1️⃣ Loading the ML Model  
```python
import pickle
import gzip

# Load the compressed model
with gzip.open("model/stroke_model_5MB.pkl.gz", "rb") as file:
    model = pickle.load(file)
```

### 2️⃣ Making Predictions  
```python
import numpy as np

# Example input: [Age, Hypertension, Heart Disease, BMI, Smoking Status]
sample_input = np.array([[45, 1, 0, 28.5, 2]])

# Predict stroke risk
prediction = model.predict(sample_input)
print(f"Predicted Stroke Risk: {prediction[0]}")
```

### 3️⃣ Streamlit Frontend Example  
```python
import streamlit as st
import numpy as np

st.title("🏥 Stroke Risk Predictor")

age = st.number_input("Age", min_value=0, max_value=120, value=25)
hypertension = st.radio("Hypertension", [0, 1])
heart_disease = st.radio("Heart Disease", [0, 1])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
smoking_status = st.selectbox("Smoking Status", [0, 1, 2])

if st.button("Predict"):
    input_data = np.array([[age, hypertension, heart_disease, bmi, smoking_status]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Stroke Risk: {prediction[0]}")
```

---

## 📌 Acknowledgements
- **Scikit-learn** for providing efficient ML tools.
- **Streamlit** for an interactive UI.
- **Django** for backend support.
- Open-source datasets used for training the model.

