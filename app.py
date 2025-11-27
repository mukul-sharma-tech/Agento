import os
import streamlit as st
import datetime
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pymongo
import bcrypt
import random
import string
import time

# --- IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO

# --- URL MAPPINGS ---
PAGE_TO_URL = {
    "employee_workspace": "agento",
    "ai_call_mode": "call-mode",
    "admin_dashboard": "dashboard",
    "user_profile": "profile",
    "auth": "login"
}
URL_TO_PAGE = {v: k for k, v in PAGE_TO_URL.items()}


# --- CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
load_dotenv()

st.set_page_config(page_title="AGENTO: Enterprise", layout="wide", initial_sidebar_state="expanded")

# --- SESSION STATE INIT ---
if "page" not in st.session_state: st.session_state.page = "auth"
if "user" not in st.session_state: st.session_state.user = None
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "theme" not in st.session_state: st.session_state.theme = "dark" # Default theme

# --- STYLING ---
def get_theme_css():
    themes = {
        "dark": {
            "--background-color": "#161B22",
            "--bg-gradient-start": "#0D1117",
            "--bg-gradient-end": "#161B22",
            "--primary-text-color": "#E6EDF3",
            "--secondary-text-color": "#8B949E",
            "--accent-color": "#58A6FF",
            "--accent-color-hover": "#79C0FF",
            "--card-background-color": "rgba(33, 39, 48, 0.7)",
            "--border-color": "rgba(139, 148, 158, 0.3)",
            "--glow-color": "rgba(88, 166, 255, 0.5)"
        },
    }
    theme = themes[st.session_state.theme]
    
    return f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

    <style>
        @keyframes gradientAnimation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        :root {{
            --background-color: {theme['--background-color']};
            --bg-gradient-start: {theme['--bg-gradient-start']};
            --bg-gradient-end: {theme['--bg-gradient-end']};
            --primary-text-color: {theme['--primary-text-color']};
            --secondary-text-color: {theme['--secondary-text-color']};
            --accent-color: {theme['--accent-color']};
            --accent-color-hover: {theme['--accent-color-hover']};
            --card-background-color: {theme['--card-background-color']};
            --border-color: {theme['--border-color']};
            --glow-color: {theme['--glow-color']};
        }}

        html, body, [class*="st-"] {{
            font-family: 'Poppins', sans-serif;
            color: var(--primary-text-color);
        }}

        .main {{
            background: linear-gradient(-45deg, var(--bg-gradient-start), var(--bg-gradient-end));
            background-size: 400% 400%;
            animation: gradientAnimation 15s ease infinite;
        }}

        h1, h2, h3 {{
            font-family: 'Poppins', sans-serif;
            color: var(--primary-text-color);
            font-weight: 600;
        }}

        [data-testid="stSidebar"] {{
            background-color: var(--card-background-color);
            backdrop-filter: blur(10px);
            border-right: 1px solid var(--border-color);
        }}

        .stButton > button {{
            border: 1px solid var(--accent-color);
            background-color: transparent;
            color: var(--accent-color) !important;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
            border-radius: 8px;
        }}
        .stButton > button p, .stButton > button div {{
            color: var(--accent-color) !important;
            transition: all 0.3s ease-in-out;
        }}

        .stButton>button:hover {{
            background-color: var(--accent-color);
            border-color: var(--accent-color);
            color: white !important;
            transform: translateY(-2px);
            box-shadow: 0 0 15px var(--glow-color);
        }}
        .stButton > button:hover p, .stButton > button:hover div {{
            color: white !important;
        }}

        .stTextInput > div > div > input, .stSelectbox > div > div {{
            background-color: var(--card-background-color);
            color: var(--primary-text-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            backdrop-filter: blur(5px);
            transition: all 0.2s ease-in-out;
        }}
        .stTextInput > div > div > input:focus, .stSelectbox > div > div:focus-within {{
            border-color: var(--accent-color);
            box-shadow: 0 0 8px var(--glow-color);
        }}

        [data-testid="stChatMessage"] {{
            background: var(--card-background-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            backdrop-filter: blur(5px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }}
        [data-testid="stChatMessage"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}

        [data-testid="stMetric"] {{
            background-color: var(--card-background-color);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(5px);
            transition: all 0.3s ease;
        }}
        [data-testid="stMetric"]:hover {{
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-color: var(--accent-color);
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--secondary-text-color);
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: var(--secondary-text-color);
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: var(--accent-color);
            border-bottom: 3px solid var(--accent-color);
        }}
        
        i.bi {{
            color: var(--primary-text-color);
        }}
    </style>
    """

# --- DATABASE (FIXED) ---
@st.cache_resource
def get_db():
    try:
        uri = os.getenv("MONGO_URI")
        if uri: 
            client = pymongo.MongoClient(uri)
            return client["jarvis_saas"]
        return None
    except: 
        return None

# --- ROUTING HELPER ---
def navigate_to(page):
    url_path = PAGE_TO_URL.get(page, page)
    st.query_params["page"] = url_path
    st.session_state.page = page
    st.rerun()

# --- AUTH LOGIC ---
def generate_id(name):
    prefix = name[:3].upper()
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{prefix}_{suffix}"

def register_company(db, name, email, password):
    if db["users"].find_one({"email": email}): return False, "Email exists."
    cid = generate_id(name)
    db["companies"].insert_one({"company_id": cid, "name": name, "created_at": datetime.datetime.now(), "admin": email})
    db["users"].insert_one({
        "email": email, 
        "password": bcrypt.hashpw(password.encode(), bcrypt.gensalt()), 
        "role": "admin", "company_id": cid, "company_name": name
    })
    return True, cid

def join_company(db, email, password, cid):
    company = db["companies"].find_one({"company_id": cid})
    if not company: return False, "Invalid ID."
    if db["users"].find_one({"email": email}): return False, "Email exists."
    db["users"].insert_one({
        "email": email, 
        "password": bcrypt.hashpw(password.encode(), bcrypt.gensalt()), 
        "role": "employee", "company_id": cid, "company_name": company["name"]
    })
    return True, "Joined!"

def login(db, email, password):
    user = db["users"].find_one({"email": email})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        st.session_state.user = user
        st.session_state.auth_status = True
        return True
    return False

# --- RAG LOGIC ---
def ingest_file(db, pdf, category, user):
    text = "".join([p.extract_text() for p in PdfReader(pdf).pages])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(text)
    
    # Save Metadata & Text
    db["documents"].insert_one({
        "company_id": user["company_id"],
        "filename": pdf.name,
        "category": category,
        "uploaded_by": user["email"],
        "upload_date": datetime.datetime.now(),
        "full_text": text
    })
    
    # Save Vectors
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    path = f"faiss_indexes/index_{user['company_id']}"
    if os.path.exists(path):
        v = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        v.add_texts(chunks)
    else:
        v = FAISS.from_texts(chunks, embeddings)
    v.save_local(path)

def chat_with_jarvis(user, query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    path = f"faiss_indexes/index_{user['company_id']}"
    if not os.path.exists(path): return "⚠️ No knowledge base found for your company."
    
    db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    template = """You are AGENTO. Answer based ONLY on the context provided.
    Context: {context}
    Question: {question}
    Answer:"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template) | model | StrOutputParser()
    )
    return chain.invoke(query)

# ==========================================
# 共通 SIDEBAR
# ==========================================
def render_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.title(f"🏢 {user['company_name']}")
        st.markdown(f"**Agent:** {user['email']}")
        st.markdown(f"**Workspace ID:** `{user['company_id']}`")
        st.markdown(f"**Role:** {user['role'].capitalize()}")
        st.markdown("---")

        # Define all possible pages and their icons/labels
        pages = {
            "employee_workspace": {"label": "Go to Chat", "icon": "💬"},
            "ai_call_mode": {"label": "AI Call Mode", "icon": "📞"},
            "admin_dashboard": {"label": "Admin Dashboard", "icon": "📊", "admin_only": True},
            "user_profile": {"label": "User Profile", "icon": "👤"}
        }

        for page_id, page_info in pages.items():
            # Skip admin pages for non-admin users
            if page_info.get("admin_only") and user['role'] != 'admin':
                continue
            
            # Don't show a button for the current page
            if st.session_state.get('page') != page_id:
                if st.button(page_info["label"], use_container_width=True):
                    navigate_to(page_id)

        st.markdown("---")

        # Logout button at the bottom
        if st.button("Logout", use_container_width=True):
            # Clear session state completely on logout
            for key in st.session_state.keys():
                del st.session_state[key]
            # Set page to auth to trigger rerun to login page
            st.session_state.page = "auth"
            st.query_params.clear()
            st.rerun()


# ==========================================
# 📄 PAGE 0: LANDING (for logged-in users)
# ==========================================
def page_landing():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center; margin-top: 50px;'>Welcome Back!</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<p style='text-align: center;'>You are already logged in. You can proceed to your workspace or log out.</p>", unsafe_allow_html=True)
        
        if st.button("Go to your AI Agent", use_container_width=True):
            navigate_to("employee_workspace")
            
        if st.button("Logout", use_container_width=True):
            st.session_state.auth_status = False
            st.session_state.page = "auth"
            st.rerun()

# ==========================================
# 📄 PAGE 1: AUTHENTICATION PORTAL
# ==========================================
def page_auth(db):
    st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'><i class='bi bi-shield-lock'></i> AGENTO: Secure Access</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register, tab_join = st.tabs(["Login", "Register Company", "Join Team"])
        
        with tab_login:
            email = st.text_input("Email", key="l_email")
            pwd = st.text_input("Password", type="password", key="l_pwd")
            if st.button("Login"):
                if login(db, email, pwd):
                    if st.session_state.user["role"] == "admin":
                        navigate_to("admin_dashboard")
                    else:
                        navigate_to("employee_workspace")
                else:
                    st.error("Invalid Credentials")

        with tab_register:
            c_name = st.text_input("Company Name")
            a_email = st.text_input("Admin Email")
            a_pwd = st.text_input("Admin Password", type="password")
            if st.button("Register Company"):
                success, msg = register_company(db, c_name, a_email, a_pwd)
                if success: st.success(f"Success! ID: {msg}"); time.sleep(2); st.rerun()
                else: st.error(msg)

        with tab_join:
            j_id = st.text_input("Workspace ID")
            j_email = st.text_input("Your Email")
            j_pwd = st.text_input("Password", type="password")
            if st.button("Join Team"):
                success, msg = join_company(db, j_email, j_pwd, j_id)
                if success: st.success(msg)
                else: st.error(msg)

# ==========================================
# 📄 PAGE 2: ADMIN DASHBOARD
# ==========================================
def page_admin_dashboard(db):
    user = st.session_state.user
    render_sidebar()
    st.title("Company Command Center")
    
    # --- Stats Row ---
    col1, col2, col3 = st.columns(3)
    doc_count = db["documents"].count_documents({"company_id": user["company_id"]})
    user_count = db["users"].count_documents({"company_id": user["company_id"]})
    col1.metric("Total Documents", doc_count)
    col2.metric("Team Members", user_count)
    col3.metric("System Status", "Online")
    
    st.markdown("---")

    # --- Upload Section ---
    col_upload, col_list = st.columns([1, 2])
    
    with col_upload:
        st.subheader("Upload New Knowledge")
        predefined_cats = ["HR", "Engineering", "Sales", "Legal", "General"]
        selected_option = st.selectbox("Category", predefined_cats + ["➕ Create New"])

        final_category = selected_option
        if selected_option == "➕ Create New":
            new_cat = st.text_input("New Category Name:")
            if new_cat: final_category = new_cat.strip()

        files = st.file_uploader("Select PDF", type="pdf", accept_multiple_files=True)
        if st.button("Index Documents"):
            if files and final_category and final_category != "➕ Create New":
                with st.spinner("Processing..."):
                    for f in files: ingest_file(db, f, final_category, user)
                st.success("Done!")
                time.sleep(1)
                st.rerun()

    with col_list:
        st.subheader("Recent Uploads")
        # List Documents
        docs = list(db["documents"].find(
            {"company_id": user["company_id"]}, 
            {"_id":0, "filename":1, "category":1, "upload_date":1, "uploaded_by":1}
        ).sort("upload_date", -1).limit(10))
        
        if docs:
            st.dataframe(pd.DataFrame(docs), use_container_width=True)
        else:
            st.info("No documents yet.")

# ==========================================
# 📄 PAGE 3: EMPLOYEE WORKSPACE
# ==========================================
def page_employee_workspace(db):
    user = st.session_state.user
    render_sidebar()

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("<h1 style='padding-top: 20px;'>AGENTO AI Assistant</h1>", unsafe_allow_html=True)
    with col2:
        st.image("public/globe.gif")

    
    # Chat Logic
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("How can I help you?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = chat_with_jarvis(user, prompt)
        
        with st.chat_message("assistant"): st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Log Chat
        db["chat_history"].insert_one({
            "company_id": user["company_id"],
            "user": user["email"],
            "query": prompt,
            "response": response,
            "timestamp": datetime.datetime.now()
        })

# --- HELPER: TEXT TO SPEECH (The AI's Voice) ---
def speak_text(text):
    """Converts text to audio bytes for Streamlit to play"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp
    except Exception as e:
        st.error(f"Audio Error: {e}")
        return None

# --- HELPER: SPEECH TO TEXT (The AI's Ears) ---
def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        # Save bytes to a temporary wav file (required for SpeechRecognition)
        with open("temp_input.wav", "wb") as f:
            f.write(audio_bytes.read())
            
        with sr.AudioFile("temp_input.wav") as source:
            audio_data = r.record(source)
            # Using Google's free speech API
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return None
    except Exception as e:
        return None

# ==========================================
# 📞 PAGE: AI CALL MODE (Immersive)
# ==========================================
def page_ai_call_mode(db):
    user = st.session_state.user
    render_sidebar()
    
    # --- CSS FOR IMMERSIVE LAYOUT ---
    st.markdown("""
    <style>
        .main {
            background: #000000 !important;
        }
        .block-container { padding-top: 2rem; }
        audio { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # --- MAIN VISUALS (THE AVATAR) ---
    # Using columns to perfectly center the big GIF
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h2 style='text-align: center; color: #00FFCC; margin-bottom: 0px;'>AGENTO</h2>", unsafe_allow_html=True)
        # The AI Avatar (Center Stage)
        st.image("public/face.gif", use_container_width=True)
        
        # Status Indicator
        if "processing_voice" not in st.session_state:
            st.session_state.processing_voice = False
            
        status_text = "🟢 LISTENING..." if not st.session_state.processing_voice else "🟣 PROCESSING..."
        st.markdown(f"<p style='text-align: center; color: #888; letter-spacing: 2px;'>{status_text}</p>", unsafe_allow_html=True)

    # --- VOICE INTERACTION LOOP ---
    st.write("---") 
    _, c2, _ = st.columns([2,1,2])
    with c2:
        audio_value = st.audio_input("Tap to Speak", label_visibility="collapsed")

    if audio_value:
        st.session_state.processing_voice = True
        
        # 1. TRANSCRIBE (Listen)
        user_text = transcribe_audio(audio_value)
        
        if user_text:
            # Show what user said (Subtitles)
            # st.toast(f"👤 You: {user_text}", icon="🎤")
            
            # 2. THINK (RAG Process)
            ai_response = chat_with_jarvis(user, user_text)
            
            # 3. SPEAK (TTS)
            audio_fp = speak_text(ai_response)
            
            if audio_fp:
                # Autoplay the response
                st.audio(audio_fp, format='audio/mp3', start_time=0, autoplay=True)
                # st.toast(f"🤖 Agent: {ai_response[:50]}...", icon="🧠")
                
                # Log to DB
                db["chat_history"].insert_one({
                    "company_id": user["company_id"],
                    "user": user["email"],
                    "query": user_text,
                    "response": ai_response,
                    "timestamp": datetime.datetime.now(),
                    "mode": "voice_call"
                })
        else:
            st.warning("Could not understand audio. Please speak clearly.")
            
        st.session_state.processing_voice = False


def page_user_profile(db):
    render_sidebar()
    st.markdown("<h1 style='text-align: center;'>User Profile</h1>", unsafe_allow_html=True)

# ==========================================
# 🚦 MAIN ROUTER (FIXED)
# ==========================================
def main():
    db = get_db()
    if db is None:
        st.error("Database Connection Failed. Please check your MONGO_URI in .env file.")
        return

    st.markdown(get_theme_css(), unsafe_allow_html=True)

    # --- ROUTING LOGIC ---
    query_params = st.query_params.to_dict()
    page_in_url = query_params.get("page")

    if not st.session_state.auth_status:
        # If not logged in, always show auth page.
        page_auth(db)
    else:
        # If logged in...
        if page_in_url:
            page_to_render = URL_TO_PAGE.get(page_in_url)
        else:
            # If at the root URL (no page in query param), show the landing page.
            page_to_render = 'landing'
        
        # Fallback for invalid page names in URL
        if not page_to_render:
            page_to_render = 'landing'

        # Render the determined page
        if page_to_render == 'landing':
            page_landing()
        elif page_to_render == 'auth': # Logged-in user trying to access login page
            page_landing()
        elif page_to_render == 'admin_dashboard' and st.session_state.user.get('role') == 'admin':
            page_admin_dashboard(db)
        elif page_to_render == 'employee_workspace':
            page_employee_workspace(db)
        elif page_to_render == 'ai_call_mode':
            page_ai_call_mode(db)
        elif page_to_render == 'user_profile':
            page_user_profile(db)
        else:
            # Default/fallback page for authenticated users
            page_landing()

if __name__ == "__main__":
    main()
