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
from io import BytesIO
import re # regex for parsing mermaid
import streamlit.components.v1 as components # For rendering diagram

# --- IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS
import speech_recognition as sr
import base64

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
if "theme" not in st.session_state: st.session_state.theme = "dark" 

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
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        @keyframes gradientAnimation {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
        :root {{ --background-color: {theme['--background-color']}; --primary-text-color: {theme['--primary-text-color']}; --accent-color: {theme['--accent-color']}; --card-background-color: {theme['--card-background-color']}; }}
        html, body, [class*="st-"] {{ font-family: 'Poppins', sans-serif; color: var(--primary-text-color); }}
        .main {{ background: linear-gradient(-45deg, {theme['--bg-gradient-start']}, {theme['--bg-gradient-end']}); background-size: 400% 400%; animation: gradientAnimation 15s ease infinite; }}
        [data-testid="stSidebar"] {{ background-color: var(--card-background-color); backdrop-filter: blur(10px); border-right: 1px solid {theme['--border-color']}; }}
        .stButton > button {{ border: 1px solid var(--accent-color); background-color: transparent; color: var(--accent-color) !important; border-radius: 8px; }}
        .stButton > button:hover {{ background-color: var(--accent-color); color: white !important; box-shadow: 0 0 15px {theme['--glow-color']}; }}
        .stTextInput > div > div > input, .stSelectbox > div > div {{ background-color: var(--card-background-color); color: var(--primary-text-color); border: 1px solid {theme['--border-color']}; border-radius: 8px; }}
        [data-testid="stChatMessage"] {{ background: var(--card-background-color); border: 1px solid {theme['--border-color']}; border-radius: 12px; }}
    </style>
    """

# --- DATABASE ---
@st.cache_resource
def get_db():
    try:
        uri = os.getenv("MONGO_URI")
        if uri: return pymongo.MongoClient(uri)["jarvis_saas"]
        return None
    except: return None

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

# --- MERMAID RENDERER (The New Feature) ---
# def render_mermaid(code):
#     """
#     Renders Mermaid code using a simple HTML wrapper
#     """
#     html_code = f"""
#     <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
#     <script>
#         mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
#     </script>
#     <div class="mermaid">
#         {code}
#     </div>
#     """
#     return components.html(html_code, height=400, scrolling=True)
def render_mermaid(code):
    """
    Renders Mermaid code with error suppression.
    If syntax is wrong, it fails silently (shows nothing) instead of a red error box.
    """
    html_code = f"""
    <div id="mermaid-container" style="width: 100%; overflow-x: auto;">
        <div class="mermaid">
            {code}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            suppressErrorRendering: true, // <--- THE FIX: Hides red error box on failure
        }});
        
        // Optional: Catch parsing errors explicitly if needed
        mermaid.parseError = function(err, hash) {{
            console.log('Mermaid syntax error ignored');
            document.getElementById('mermaid-container').style.display = 'none';
        }};
    </script>
    """
    return components.html(html_code, height=400, scrolling=True)
# def parse_and_display_response(response_text):
#     """
#     Detects if mermaid code exists, splits it, and renders text + diagram.
#     """
#     # Regex to find ```mermaid ... ``` blocks
#     pattern = r"```mermaid(.*?)```"
#     matches = re.split(pattern, response_text, flags=re.DOTALL)
    
#     if len(matches) > 1:
#         # We have diagrams!
#         for i, part in enumerate(matches):
#             if i % 2 == 0:
#                 # Text Part
#                 if part.strip(): st.markdown(part)
#             else:
#                 # Diagram Part
#                 st.caption("📊 Process Visualization")
#                 render_mermaid(part.strip())
#     else:
#         # Normal Text
#         st.markdown(response_text)
def parse_and_display_response(response_text):
    # Regex to find ```mermaid ... ``` blocks (case insensitive)
    pattern = r"```mermaid(.*?)```"
    matches = re.split(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
    
    if len(matches) > 1:
        for i, part in enumerate(matches):
            if i % 2 == 0:
                # Text Part
                if part.strip(): st.markdown(part)
            else:
                # Diagram Part
                # Clean up the code to remove any lingering backticks or whitespace
                clean_code = part.strip().replace("`", "")
                if clean_code:
                    st.caption("📊 Process Flow")
                    render_mermaid(clean_code)
    else:
        # Normal Text
        st.markdown(response_text)
# --- RAG LOGIC ---
def ingest_file(db, pdf, category, user):
    text = "".join([p.extract_text() for p in PdfReader(pdf).pages])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(text)
    
    db["documents"].insert_one({
        "company_id": user["company_id"],
        "filename": pdf.name,
        "category": category,
        "uploaded_by": user["email"],
        "upload_date": datetime.datetime.now(),
        "full_text": text
    })
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = f"collection_{user['company_id']}"
    
    try:
        QdrantVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name,
            force_recreate=False
        )
    except Exception as e: st.error(f"Qdrant Error: {e}")

# def chat_with_jarvis(user, query):
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     collection_name = f"collection_{user['company_id']}"
    
#     try:
#         vector_store = QdrantVectorStore.from_existing_collection(
#             embedding=embeddings,
#             collection_name=collection_name,
#             url=os.getenv("QDRANT_URL"),
#             api_key=os.getenv("QDRANT_API_KEY"),
#         )
#         retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
#         # --- MODIFIED PROMPT FOR MERMAID ---
#         template = """You are AGENTO. Answer based on the context provided.
        
#         IMPORTANT: If the answer involves a process, workflow, sequence of steps, or a decision tree:
#         1. Explain it in text first.
#         2. Then, provide a valid Mermaid.js diagram code block.
#         3. Use syntax: ```mermaid graph TD; A-->B; ... ```
        
#         Context: {context}
#         Question: {question}
#         Answer:"""
        
#         model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
#         chain = (
#             {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
#              "question": RunnablePassthrough()}
#             | ChatPromptTemplate.from_template(template) | model | StrOutputParser()
#         )
#         return chain.invoke(query)
#     except Exception:
#         return "⚠️ I don't have any knowledge for your company yet."
def chat_with_jarvis(user, query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = f"collection_{user['company_id']}"
    
    try:
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # --- PROFESSIONAL CORPORATE PROMPT ---
        template = """You are AGENTO, a professional internal knowledge assistant for {company_name}.
        Your goal is to assist employees by retrieving accurate information from the company's knowledge base.

        GUIDELINES:
        1. **Tone:** Professional, concise, and direct. Avoid casual slang or excessive emojis.
        2. **Accuracy:** Answer ONLY based on the Context provided below. If the answer is not in the context, state: "I cannot find this information in the company documents."
        3. **Formatting:** Use clear headings, bullet points, and bold text for readability.

        VISUALIZATION RULES (Strict):
        1. If the answer describes a workflow, hierarchy, or step-by-step process, generate a Mermaid.js diagram.
        2. **Syntax Safety:** - Do NOT use brackets () or [] inside node labels as they break syntax. 
           - Use simple IDs like A, B, C. 
           - Example: A[Start] --> B[Process]
        3. Place the mermaid code block at the very end of your response.

        Context: {context}
        
        Employee Question: {question}
        
        Professional Answer:"""
        
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        
        # Pass company name into prompt for personalization
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                "question": RunnablePassthrough(),
                "company_name": lambda x: user['company_name']
            }
            | prompt_template 
            | model 
            | StrOutputParser()
        )
        return chain.invoke(query)
    except Exception as e:
        # st.error(e) # Uncomment to debug
        return "⚠️ I am unable to access the company knowledge base at this moment."
# --- AUDIO HELPERS ---
def speak_text(text):
    try:
        # Strip code blocks before speaking to avoid reading weird symbols
        clean_text = re.sub(r"```.*?```", " I have displayed the diagram below.", text, flags=re.DOTALL)
        tts = gTTS(text=clean_text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp
    except: return None

def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        with open("temp_input.wav", "wb") as f: f.write(audio_bytes.read())
        with sr.AudioFile("temp_input.wav") as source:
            return r.recognize_google(r.record(source))
    except: return None

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    md = f"""<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
    st.markdown(md, unsafe_allow_html=True)

# --- SIDEBAR ---
def render_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.title(f"🏢 {user['company_name']}")
        st.markdown(f"**Agent:** {user['email']}")
        st.markdown("---")
        pages = {
            "employee_workspace": {"label": "Go to Chat", "icon": "💬"},
            "ai_call_mode": {"label": "AI Call Mode", "icon": "📞"},
            "admin_dashboard": {"label": "Admin Dashboard", "icon": "📊", "admin_only": True},
            "user_profile": {"label": "User Profile", "icon": "👤"}
        }
        for page_id, page_info in pages.items():
            if page_info.get("admin_only") and user['role'] != 'admin': continue
            if st.session_state.get('page') != page_id:
                if st.button(page_info["label"], use_container_width=True): navigate_to(page_id)
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for key in st.session_state.keys(): del st.session_state[key]
            st.session_state.page = "auth"
            st.query_params.clear()
            st.rerun()

# --- PAGE FUNCTIONS ---
def page_landing():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center; margin-top: 50px;'>Welcome Back!</h1>", unsafe_allow_html=True)
        if st.button("Go to Workspace", use_container_width=True): navigate_to("employee_workspace")
        if st.button("Logout", use_container_width=True):
            st.session_state.auth_status = False; st.session_state.page = "auth"; st.rerun()

def page_auth(db):
    st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'><i class='bi bi-shield-lock'></i> AGENTO: Secure Access</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register, tab_join = st.tabs(["Login", "Register Company", "Join Team"])
        with tab_login:
            email = st.text_input("Email", key="l_email")
            pwd = st.text_input("Password", type="password", key="l_pwd")
            if st.button("Login"):
                if login(db, email, pwd): navigate_to("admin_dashboard" if st.session_state.user["role"] == "admin" else "employee_workspace")
                else: st.error("Invalid Credentials")
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

def page_admin_dashboard(db):
    user = st.session_state.user
    render_sidebar()
    st.title("Company Command Center")
    col1, col2, col3 = st.columns(3)
    doc_count = db["documents"].count_documents({"company_id": user["company_id"]})
    user_count = db["users"].count_documents({"company_id": user["company_id"]})
    col1.metric("Total Documents", doc_count); col2.metric("Team Members", user_count); col3.metric("System Status", "Online")
    st.markdown("---")
    col_upload, col_list = st.columns([1, 2])
    with col_upload:
        st.subheader("Upload Knowledge")
        predefined_cats = ["HR", "Engineering", "Sales", "Legal", "General"]
        selected = st.selectbox("Category", predefined_cats + ["➕ Create New"])
        cat = st.text_input("New Category:").strip() if selected == "➕ Create New" else selected
        files = st.file_uploader("Select PDF", type="pdf", accept_multiple_files=True)
        if st.button("Index Documents"):
            if files and cat:
                with st.spinner("Pushing to Qdrant..."):
                    for f in files: ingest_file(db, f, cat, user)
                st.success("Indexed!"); time.sleep(1); st.rerun()
    with col_list:
        st.subheader("Recent Uploads")
        docs = list(db["documents"].find({"company_id": user["company_id"]}, {"_id":0, "filename":1, "category":1, "upload_date":1}).sort("upload_date", -1).limit(10))
        if docs: st.dataframe(pd.DataFrame(docs), use_container_width=True)

def page_employee_workspace(db):
    user = st.session_state.user
    render_sidebar()
    col1, col2 = st.columns([5, 1])
    with col1: st.markdown("<h1 style='padding-top: 20px;'>AGENTO AI Assistant</h1>", unsafe_allow_html=True)
    with col2: st.image("public/globe.gif")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    
    # Render History with Mermaid Parsing
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            # Use the parser here too so history has diagrams
            if msg["role"] == "assistant":
                parse_and_display_response(msg["content"])
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("How can I help you?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking & Visualizing..."):
            response = chat_with_jarvis(user, prompt)
        
        with st.chat_message("assistant"): 
            parse_and_display_response(response) # Render Diagram Here
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        db["chat_history"].insert_one({"company_id": user["company_id"], "user": user["email"], "query": prompt, "response": response, "timestamp": datetime.datetime.now()})

def page_ai_call_mode(db):
    user = st.session_state.user
    render_sidebar()
    st.markdown("""<style>.main { background: #000000 !important; } .block-container { padding-top: 2rem; } audio { display: none !important; }</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #00FFCC; margin-bottom: 0px;'>AGENTO</h2>", unsafe_allow_html=True)
        st.image("public/face.gif", use_container_width=True)
        status = "🟢 LISTENING..." if not st.session_state.get("processing_voice") else "🟣 PROCESSING..."
        st.markdown(f"<p style='text-align: center; color: #888; letter-spacing: 2px;'>{status}</p>", unsafe_allow_html=True)
    st.write("---") 
    _, c2, _ = st.columns([2,1,2])
    with c2: audio_value = st.audio_input("Tap to Speak", label_visibility="collapsed")
    if audio_value:
        st.session_state.processing_voice = True
        user_text = transcribe_audio(audio_value)
        if user_text:
            ai_response = chat_with_jarvis(user, user_text)
            audio_fp = speak_text(ai_response)
            if audio_fp:
                audio_fp.seek(0)
                autoplay_audio(audio_fp)
                db["chat_history"].insert_one({"company_id": user["company_id"], "user": user["email"], "query": user_text, "response": ai_response, "timestamp": datetime.datetime.now(), "mode": "voice_call"})
        else: st.warning("Could not understand audio.")
        st.session_state.processing_voice = False

def page_user_profile(db):
    render_sidebar()
    st.markdown("<h1 style='text-align: center;'>User Profile</h1>", unsafe_allow_html=True)

# --- MAIN ROUTER ---
def main():
    db = get_db()
    if db is None: st.error("Database Connection Failed."); return
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    query_params = st.query_params.to_dict()
    page_in_url = query_params.get("page")
    
    if not st.session_state.auth_status: page_auth(db)
    else:
        page_to_render = URL_TO_PAGE.get(page_in_url, 'landing')
        if page_to_render == 'landing': page_landing()
        elif page_to_render == 'auth': page_landing()
        elif page_to_render == 'admin_dashboard' and st.session_state.user.get('role') == 'admin': page_admin_dashboard(db)
        elif page_to_render == 'employee_workspace': page_employee_workspace(db)
        elif page_to_render == 'ai_call_mode': page_ai_call_mode(db)
        elif page_to_render == 'user_profile': page_user_profile(db)
        else: page_landing()

if __name__ == "__main__":
    main()
