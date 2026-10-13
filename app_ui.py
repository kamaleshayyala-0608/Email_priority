import streamlit as st
import joblib
import string
import nltk
import os
import pandas as pd
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 1. ULTRA HD CYBER-TECH UI ---
st.set_page_config(page_title="SENTINEL AI // UNIVERSAL", page_icon="💠", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Inter:wght@300;600&display=swap');
    
    .stApp { background: #05050a; font-family: 'Inter', sans-serif; color: #e0e0e0; }
    
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 60px;
        text-align: center;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 10px rgba(0, 242, 254, 0.5));
        margin-bottom: 0px;
    }

    .neon-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid #00f2fe66;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.1);
    }

    .stTextArea textarea {
        background: #0a0a15 !important;
        color: #00f2fe !important;
        border: 1px solid #00f2fe !important;
        font-size: 18px !important;
    }

    .stButton>button {
        background: linear-gradient(45deg, #00f2fe, #4facfe) !important;
        color: #000 !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900 !important;
        border: none !important;
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
    }

    .badge {
        padding: 20px;
        border-radius: 15px;
        font-family: 'Orbitron', sans-serif;
        font-size: 30px;
        text-align: center;
        border: 2px solid;
        margin-bottom: 20px;
    }
    .high { color: #ff0055; border-color: #ff0055; box-shadow: 0 0 20px rgba(255, 0, 85, 0.3); }
    .med { color: #ffae00; border-color: #ffae00; box-shadow: 0 0 20px rgba(255, 174, 0, 0.3); }
    .low { color: #00ff9d; border-color: #00ff9d; box-shadow: 0 0 20px rgba(0, 255, 157, 0.3); }
    
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE EXPANDED INTELLIGENCE CORE ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
MODEL_FILE, VEC_FILE = 'email_model.pkl', 'vectorizer.pkl'

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return " ".join([w for w in text.split() if w not in stop_words])

def train_engine():
    # MASSIVE EXPANDED DATASET FOR GENERAL INTELLIGENCE
    data = {
        "text": [
            # HIGH: Urgency, Security, Failure, Deadlines
            "System crash in production", "Critical error detected", "Security breach alert", "Urgent action required now",
            "Immediate attention needed for server", "Failure in database connectivity", "Emergency contact requested",
            "Deadline is in one hour", "Hack detected on main portal", "Account compromised help", "Payment failed urgently",
            # MEDIUM: Meetings, Careers, Projects, Office
            "Scheduled a zoom call for later", "Meeting invite for tomorrow", "Project status update", "Software developer role application",
            "Job interview at Equifax", "Talent acquisition follow up", "Discussion about the new position",
            "Please review the document", "Quarterly feedback session", "Team sync on Monday at 5pm", "Office renovation plans",
            # LOW: General, News, Social, Spam
            "Daily horoscope for Capricorn", "Sale on electronics 50 percent off", "Lunch today at the cafeteria",
            "Weekly newsletter from tech blog", "Shoes and apparel discounts", "Subscription confirmed for Netflix",
            "Hey how are you doing", "Check out these new vacation photos", "Sun and moon alignment news", "Coffee receipt"
        ] * 200, # Large multiplier so the AI learns the VIBE of the words
        "label": ["high"]*11*200 + ["medium"]*11*200 + ["low"]*10*200
    }
    df = pd.DataFrame(data)
    df['clean'] = df['text'].apply(clean_text)
    
    # ngram_range=(1,2) allows AI to understand "Server Down" as a single concept
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = tfidf.fit_transform(df['clean'])
    
    model = MultinomialNB(alpha=0.01) # Low alpha for better sensitivity
    model.fit(X, df['label'])
    
    joblib.dump(model, MODEL_FILE); joblib.dump(tfidf, VEC_FILE)
    return model, tfidf

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_FILE): return train_engine()
    return joblib.load(MODEL_FILE), joblib.load(VEC_FILE)

# --- 3. UI DASHBOARD ---
st.markdown('<p class="main-header">SENTINEL AI</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-family:Orbitron; color:#4facfe;'>UNIVERSAL PRIORITY ENGINE v5.0</p>", unsafe_allow_html=True)
st.divider()

model, tfidf = load_assets()
l_col, r_col = st.columns([1.2, 1], gap="large")

with l_col:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.write("### 📥 NEURAL INPUT BUFFER")
    user_input = st.text_area("Input", height=280, label_visibility="collapsed", placeholder="Type or paste any text here...")
    
    if st.button("🚀 INITIATE SCAN"):
        if not user_input.strip():
            st.error("BUFFER EMPTY")
        else:
            with st.spinner("AI ANALYZING..."):
                time.sleep(1.0)
                
                cleaned = clean_text(user_input)
                vec = tfidf.transform([cleaned])
                ai_label = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]
                
                # Dynamic Logic Override for specific high-value targets
                final_p = ai_label
                raw_text = user_input.lower()
                if any(x in raw_text for x in ['5pm', '5 pm', 'zoom', 'meeting', 'interview']): final_p = "medium"
                if any(x in raw_text for x in ['urgent', 'emergency', 'asap', 'critical']): final_p = "high"

                with r_col:
                    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
                    st.write("### 🧬 AI DIAGNOSTIC")
                    
                    badge_style = "high" if final_p == "high" else "med" if final_p == "medium" else "low"
                    st.markdown(f'<div class="badge {badge_style}">{final_p.upper()}</div>', unsafe_allow_html=True)
                    
                    st.write("---")
                    conf = max(probs) * 100
                    st.metric("NEURAL CONFIDENCE", f"{conf:.1f}%")
                    st.progress(float(max(probs)))
                    
                    st.write("### LOGIC CLUSTERS")
                    for lbl, p in zip(model.classes_, probs):
                        st.caption(f"{lbl.upper()}: {p*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# System Sidebar
st.sidebar.markdown("## ⚙️ ENGINE")
st.sidebar.write("Core: MultinomialNB v5")
st.sidebar.write("Logic: Hybrid Neural")
if st.sidebar.button("REBOOT CORE"):
    if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
    st.cache_resource.clear()
    st.rerun()