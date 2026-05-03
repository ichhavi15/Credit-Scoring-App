import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import cv2
import numpy as np
import os
if "chat" not in st.session_state:
    st.session_state.chat = []
from gtts import gTTS
import tempfile

def speak(text):
    tts = gTTS(
        text=text,
        lang='en',
        tld='co.in',   # 🔥 Indian English accent (better sound)
        slow=False     # fast speaking
    )
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name
import tempfile
def speak(text):
    tts = gTTS(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

from PIL import Image
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit([[10000, 50000, 600],[20000, 100000, 750]], [0, 1])
st.set_page_config(layout="wide")  
st.set_page_config(page_title="Credit Scoring App")
if 'history' not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# ✅ ADD THIS
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117 !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0E1117 !important;
    }

    input, select, textarea {
        background-color: #262730 !important;
        color: white !important;
    }

    .stButton>button {
        background-color: #262730 !important;
        color: white !important;
        border-radius: 8px;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
    }

    .card {
        background: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e3f2fd, #ffffff);
        color: black;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    input, select, textarea {
        border-radius: 8px;
    }

    div[data-testid="stMetric"] {
        background: rgba(0,0,0,0.05);
        padding: 15px;
        border-radius: 10px;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def bank_rules_check(income, loan_amount, cibil, self_emp):
    reasons = []
    if cibil < 500:
        reasons.append("Very low credit score")
    if cibil < 600:
        reasons.append("Low CIBIL Score")

    if income > 0 and loan_amount > income * 5:
        reasons.append("Loan too high vs income")

    if income < 15000:
        reasons.append("Income too low")

    if self_emp == 1 and income < 25000:
        reasons.append("Unstable income (self-employed)")

    emi = loan_amount / 12 if loan_amount > 0 else 0
    if income > 0 and emi > income * 0.4:
        reasons.append("High EMI burden")

    return reasons
import speech_recognition as sr

def get_voice_input():
    r = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("🎤 Speak now...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=5)
        
        text = r.recognize_google(audio)
        st.success(f"You said: {text}")
        return text.lower()

    except sr.UnknownValueError:
        st.error("❌ Could not understand audio")
    except sr.RequestError:
        st.error("❌ API unavailable")
    except Exception as e:
        st.error("❌ Voice input error")
    
    return ""



# =========================
# LOAD DATA
# =========================
try:
    data = pd.read_csv("loan_data.csv")
    data.columns = data.columns.str.strip()
except:
    st.error("❌ File not found: loan_data.csv")

# =========================
# TARGET CLEANING
# =========================
data['loan_status'] = data['loan_status'].apply(
    lambda x: 1 if pd.notnull(x) and str(x).lower().strip() in 
    ['approved', 'yes', 'y', '1', 'true'] else 0
)

# =========================
# OTHER CLEANING
# =========================
# education
data['education'] = data['education'].astype(str).str.strip().str.lower().map({
    'graduate': 1,
    'not graduate': 0
})

# self employed
data['self_employed'] = data['self_employed'].astype(str).str.strip().str.lower().map({
    'yes': 1,
    'no': 0
})

# dependents
data['no_of_dependents'] = pd.to_numeric(data['no_of_dependents'], errors='coerce')

# handle missing
data.fillna(0, inplace=True)

if 'loan_status' in data.columns:
    X = data.drop(columns=['loan_id', 'loan_status'], errors='ignore')
    y = data['loan_status']

else:
    st.error("❌ 'loan_status' column not found")
# =========================
#Features & tareget
# =========================
if 'loan_status' in data.columns:
    X = data.drop(columns=['loan_id', 'loan_status'], errors='ignore')
    y = data['loan_status']

else:
    st.error("❌ 'loan_status' column not found")

# =========================
# =========================
# FRAUD FUNCTION
# =========================
def detect_fraud(income, loan_amount, cibil):
    if income == 0 and loan_amount > 0:
        return "⚠ No income but loan requested"
    if cibil < 300:
        return "⚠ Very low CIBIL score"
    if income > 0 and loan_amount > income * 10:
        return "⚠ Loan too high vs income"
    if loan_amount > 1000000:
        return "⚠ Very high loan amount"
    return None

# =========================
# UI
# =========================
st.markdown("<h1 style='text-align: center;'>💳 AI-Based Credit Scoring System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Smart Loan Approval using AI + Banking Rules</p>", unsafe_allow_html=True)

# session_state init
if "income" not in st.session_state:
    st.session_state.income = 0
if "loan_amount" not in st.session_state:
    st.session_state.loan_amount = 0
if "cibil" not in st.session_state:
    st.session_state.cibil = 0

st.subheader("📥 Enter Loan Details")

st.subheader("📄 Upload Documents")

uploaded_file = st.file_uploader("Upload your document/image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file)
    
    if st.session_state.result == 1:
        st.success("📄 Documents look valid for approval")
    else:
        st.warning("📄 Please upload better supporting documents")
        # 🔥 OCR
        pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        st.subheader("📄 Extracted Text")
        st.write(text)

        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            detected_income = max(map(int, numbers))
            st.success(f"💰 Detected Income: ₹{detected_income}")

    # auto fill
            st.session_state.income = detected_income
    st.subheader("🕵️ Document Verification")
    if "salary" in text.lower() or "income" in text.lower():
        st.success("✅ Looks like a valid financial document")
    else:
        st.error("🚨 Suspicious document - No salary info found")

        import cv2
import numpy as np
from PIL import Image

def detect_face(uploaded_file):
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return len(faces)
    import re

def verify_aadhaar(text):
    aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    match = re.search(aadhaar_pattern, text)

    if match:
        return match.group()
    else:
        return None


col1, col2 = st.columns(2)

with col1:
    income = int(st.number_input("💰 Income", value=st.session_state.income))
    loan_amount = int(st.number_input("🏦 Loan Amount", value=st.session_state.loan_amount))

with col2:
    if "cibil" not in st.session_state:
        st.session_state.cibil = 300
    
    cibil = st.number_input(
    "📊 CIBIL Score",
    min_value=300,
    max_value=900,
    value=max(300, st.session_state.cibil)
    )

# update session
st.session_state.income = income
st.session_state.loan_amount = loan_amount
st.session_state.cibil = cibil

if st.button("🎤 Use Voice Input"):
    voice_text = get_voice_input()

    try:
        import re
        numbers = re.findall(r'\d+', voice_text)

        if len(numbers) >= 3:
            st.session_state.income = int(numbers[0])
            st.session_state.loan_amount = int(numbers[1])
            st.session_state.cibil = max(300, int(numbers[2]))

            st.success("✅ Voice input applied (Smart Mode)")
            st.rerun()   # 👈 IMPORTANT

        else:
            st.error("❌ बोलो properly: income loan cibil")

    except Exception as e:
        st.error("❌ Voice processing error")

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.selectbox("Self Employed", ["Yes", "No"])

education = 1 if education == "Graduate" else 0
self_emp = 1 if self_emp == "Yes" else 0
# =========================
#Button click 
# =========================
st.markdown("---")
st.subheader("🔍 Loan Prediction")

# 💰 EMI Calculator
st.subheader("💰 EMI Calculator")

loan_term = st.slider("Loan Term (Months)", 6, 60, 12)
interest_rate = 10  # yearly %

monthly_rate = interest_rate / 12 / 100
emi = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_term) / ((1 + monthly_rate)**loan_term - 1)

emi = loan_amount / loan_term
st.write(f"Monthly EMI: ₹{int(emi)}")

# 🔥 EMI Risk Warning
if emi > income * 0.4:
    st.warning("⚠ EMI is too high compared to your income")
else:
    st.success("✅ EMI is manageable")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "💡 Advisor",
    "🛡 Fraud",
    "📜 History"
])

# session state for result
if "result" not in st.session_state:
    st.session_state.result = None

# -------**********------

if st.button("🚀 Predict Loan Status"):
    input_data = [[income, loan_amount, cibil]]
    result = model.predict(input_data)[0]
    st.session_state.result = result
    reasons = bank_rules_check(income, loan_amount, cibil, self_emp)
    if len(reasons) > 0:
        final_result = 0   # Reject
    else:
        final_result = 1   # Approve
    st.session_state.result = final_result

    # ✅ ADD THIS (VERY IMPORTANT)
    result_label = "Approved" if final_result == 1 else "Rejected"
    st.session_state.history.append({
    "Income": income,
    "Loan": loan_amount,
    "CIBIL": cibil,
    "Result": result_label
    })
    user_msg = f"I want a loan of ₹{loan_amount} with income ₹{income} and CIBIL {cibil}"
    if final_result == 1:
        bot_msg = f"✅ Loan approved! Your profile looks strong 💯"
    else:
        bot_msg = "❌ Sorry, your loan is rejected."
    st.session_state.chat.append(("user", user_msg))
    st.session_state.chat.append(("bot", bot_msg))
    if "result" in st.session_state:
        name = "User"
    if st.session_state.result == 1:
        st.success("✅ Loan Approved")
        text = (f"Congratulations! Your loan has been approved.")
    else:
        st.error("❌ Loan Rejected")
        text = (f"Sorry, your loan is rejected due to high risk factors.")
    audio_file = speak(text)
    import base64
    audio_file = speak(text)
    audio_bytes = open(audio_file, "rb").read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f"""
        <audio id="audio" autoplay style="display:none;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <script>
        var audio = document.getElementById("audio");
        audio.playbackRate = 1.25;  // 🔥 perfect speed
        </script>
        """,
        unsafe_allow_html=True)

with tab1:
    if 'result' in st.session_state:
        st.subheader("📊 Prediction Result")
        st.success("✔ Prediction completed successfully")
        st.subheader("💬 AI Assistant")
        for role, msg in st.session_state.chat:
            if role == "user":
                st.markdown(f"""
                <div style='text-align:right; background:#DCF8C6; padding:10px; border-radius:10px; margin:5px'>
                🧑 {msg}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align:left; background:#F1F0F0; padding:10px; border-radius:10px; margin:5px'>
                🤖 {msg}
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("Click Predict to see result")

    st.subheader("📊 Loan Data Insights")
    # bar chart
    fig, ax = plt.subplots(figsize=(3,2))
    data['loan_status'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Loan Approval Distribution")
    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Count")
    st.pyplot(fig,use_container_width=False)
    #pie chart
    fig2, ax2 = plt.subplots(figsize=(3,2))
    data['loan_status'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=["Approved", "Rejected"],
    ax=ax2
    )
    ax2.set_title("Loan Approval Ratio")
    ax2.set_ylabel('')
    st.pyplot(fig2,use_container_width=False)

confidence = st.session_state.confidence
confidence_str = f"{confidence:.2f}"

result_label = "N/A"
if st.session_state.result is not None:
    if st.session_state.result == 1:
        result_label = "Approved"
    else:
        result_label = "Rejected"

result_text = (
    f"Income: {income}\n"
    f"Loan: {loan_amount}\n"
    f"CIBIL: {cibil}\n\n"       
    f"Confidence: {confidence_str}%\n"
    f"Result: {result_label}"
)

st.download_button("📥 Download Report",
data=result_text,
file_name="loan_report.txt")

# reasons
reasons = []

if cibil < 600:
    reasons.append("Low CIBIL score")

if income < 10000:
    reasons.append("Low income")

if reasons:
    st.warning("⚠ Risk Factors:")
    for r in reasons:
        st.write("•", r)
        # =========================
# 💡 AI LOAN ADVISOR
# =========================
with tab2:
    if 'result' in st.session_state:
        st.subheader("💡 AI Loan Advisor")
        advice = []
        if cibil < 650:
            advice.append("Improve your CIBIL score above 700")

        if income > 0 and loan_amount > income * 4:
            advice.append("Reduce loan amount for better approval chances")

        if income < 20000:
            advice.append("Increase income proof or add co-applicant")

        if st.session_state.result == 0:
            advice.append("Try applying after improving your financial profile")
        else:
            advice = []   # clear advice if approved

# SHOW ADVICE
        if advice:
            for a in advice:
                st.write("👉", a)
        else:
            st.success("🎉 Your profile looks strong! No suggestions needed.")
with tab3:
    st.subheader("🛡 Fraud Detection System")
    fraud_msg = detect_fraud(income, loan_amount, cibil)
    #fraud check
    if fraud_msg:
        st.error(fraud_msg)
    else:
        st.success("No fraud detected") 
     # ===== FRAUD SCORE =====   
        fraud_score = 0
        if cibil < 500:
            fraud_score += 1
        if income > 0 and loan_amount > income * 5:
            fraud_score += 1
        if income < 10000:
            fraud_score += 1
            st.subheader("🎯 Fraud Risk Score")
            st.write(f"Fraud Score: {fraud_score}/3")
            st.progress(int((fraud_score / 3) * 100))
        if fraud_score == 0:
            st.success("🟢 Low Risk")
        elif fraud_score == 1:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        # 🔥 ADVANCED FRAUD ALERT
        if fraud_score >= 2:
            st.error("🚨 High Fraud Risk – Bank may investigate this application")
            st.warning("⚠ Please verify your income and loan details carefully")

    # ===== MODEL PREDICTION =====
    try:
        proba = model.predict_proba(input_data)
        confidence = float(max(proba[0])) * 100
    except:
        confidence = 75.0
   
    st.session_state.confidence = confidence    

    st.info(f"Confidence: {confidence:.2f}%") 
    st.subheader("📊 Confidence Level")
    st.progress(int(confidence))
    st.write(f"🎯 Model Confidence: {confidence:.2f}%")

    st.write(f"Model Used: LogisticRegression ")            
            
    # =========================
# STEP 3: FINAL DECISION
# =========================
# save history
with tab4:
    st.subheader("📜 Prediction History")
    if len(st.session_state.history) > 0:
        import pandas as pd
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.info("No history yet")

    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History Cleared!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with ❤️ using AI | Credit Scoring System</p>",
unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: grey;'>Built by Chhavi Agrawal 💻</p>",
unsafe_allow_html=True)