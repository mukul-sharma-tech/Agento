# import os
# import streamlit as st
# import datetime
# import pandas as pd
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import pymongo
# import bcrypt
# import random
# import string
# import time
# from io import BytesIO
# import re 
# import streamlit.components.v1 as components
# from mutagen.mp3 import MP3
# import base64

# # --- IMPORTS ---
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings 
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from gtts import gTTS
# import speech_recognition as sr

# # --- URL MAPPINGS ---
# PAGE_TO_URL = {
#     "employee_workspace": "agento",
#     "ai_call_mode": "call-mode",
#     "admin_dashboard": "dashboard",
#     "user_profile": "profile",
#     "auth": "login"
# }
# URL_TO_PAGE = {v: k for k, v in PAGE_TO_URL.items()}

# # --- CONFIGURATION ---
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# load_dotenv()

# st.set_page_config(page_title="AGENTO: Enterprise", layout="wide", initial_sidebar_state="expanded")

# # --- SESSION STATE INIT ---
# if "page" not in st.session_state: st.session_state.page = "auth"
# if "user" not in st.session_state: st.session_state.user = None
# if "auth_status" not in st.session_state: st.session_state.auth_status = False
# if "theme" not in st.session_state: st.session_state.theme = "dark" 
# if "popup_diagram" not in st.session_state: st.session_state.popup_diagram = None

# # --- STYLING ---
# def get_theme_css():
#     themes = {
#         "dark": {
#             "--background-color": "#161B22",
#             "--bg-gradient-start": "#0D1117",
#             "--bg-gradient-end": "#161B22",
#             "--primary-text-color": "#E6EDF3",
#             "--secondary-text-color": "#8B949E",
#             "--accent-color": "#58A6FF",
#             "--accent-color-hover": "#79C0FF",
#             "--card-background-color": "rgba(33, 39, 48, 0.7)",
#             "--border-color": "rgba(139, 148, 158, 0.3)",
#             "--glow-color": "rgba(88, 166, 255, 0.5)"
#         },
#     }
#     theme = themes[st.session_state.theme]
    
#     return f"""
#     <link rel="preconnect" href="https://fonts.googleapis.com">
#     <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
#     <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
#     <style>
#         @keyframes gradientAnimation {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
#         :root {{ --background-color: {theme['--background-color']}; --primary-text-color: {theme['--primary-text-color']}; --accent-color: {theme['--accent-color']}; --card-background-color: {theme['--card-background-color']}; }}
#         html, body, [class*="st-"] {{ font-family: 'Poppins', sans-serif; color: var(--primary-text-color); }}
#         .main {{ background: linear-gradient(-45deg, {theme['--bg-gradient-start']}, {theme['--bg-gradient-end']}); background-size: 400% 400%; animation: gradientAnimation 15s ease infinite; }}
        
#         /* SIDEBAR STYLING */
#         [data-testid="stSidebar"] {{ 
#             background-color: var(--card-background-color); 
#             backdrop-filter: blur(10px); 
#             border-right: 1px solid {theme['--border-color']}; 
#         }}
        
#         .stButton > button {{ border: 1px solid var(--accent-color); background-color: transparent; color: var(--accent-color) !important; border-radius: 8px; }}
#         .stButton > button:hover {{ background-color: var(--accent-color); color: white !important; box-shadow: 0 0 15px {theme['--glow-color']}; }}
#         .stTextInput > div > div > input, .stSelectbox > div > div {{ background-color: var(--card-background-color); color: var(--primary-text-color); border: 1px solid {theme['--border-color']}; border-radius: 8px; }}
#         [data-testid="stChatMessage"] {{ background: var(--card-background-color); border: 1px solid {theme['--border-color']}; border-radius: 12px; }}
#     </style>
#     """

# # --- DATABASE ---
# @st.cache_resource
# def get_db():
#     try:
#         uri = os.getenv("MONGO_URI")
#         if uri: return pymongo.MongoClient(uri)["jarvis_saas"]
#         return None
#     except: return None

# # --- ROUTING HELPER ---
# def navigate_to(page):
#     url_path = PAGE_TO_URL.get(page, page)
#     st.query_params["page"] = url_path
#     st.session_state.page = page
#     st.rerun()

# # --- AUTH LOGIC ---
# def generate_id(name):
#     prefix = name[:3].upper()
#     suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
#     return f"{prefix}_{suffix}"

# def register_company(db, name, email, password):
#     if db["users"].find_one({"email": email}): return False, "Email exists."
#     cid = generate_id(name)
#     db["companies"].insert_one({"company_id": cid, "name": name, "created_at": datetime.datetime.now(), "admin": email})
#     db["users"].insert_one({
#         "email": email, 
#         "password": bcrypt.hashpw(password.encode(), bcrypt.gensalt()), 
#         "role": "admin", "company_id": cid, "company_name": name
#     })
#     return True, cid

# def join_company(db, email, password, cid):
#     company = db["companies"].find_one({"company_id": cid})
#     if not company: return False, "Invalid ID."
#     if db["users"].find_one({"email": email}): return False, "Email exists."
#     db["users"].insert_one({
#         "email": email, 
#         "password": bcrypt.hashpw(password.encode(), bcrypt.gensalt()), 
#         "role": "employee", "company_id": cid, "company_name": company["name"]
#     })
#     return True, "Joined!"

# def login(db, email, password):
#     user = db["users"].find_one({"email": email})
#     if user and bcrypt.checkpw(password.encode(), user["password"]):
#         st.session_state.user = user
#         st.session_state.auth_status = True
#         return True
#     return False

# # --- MERMAID RENDERER ---
# def render_mermaid(code, height=400):
#     html_code = f"""
#     <div id="mermaid-container" style="width: 100%; overflow-x: auto;">
#         <div class="mermaid">
#             {code}
#         </div>
#     </div>
#     <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
#     <script>
#         mermaid.initialize({{
#             startOnLoad: true,
#             theme: 'dark',
#             securityLevel: 'loose',
#             suppressErrorRendering: true,
#         }});
#     </script>
#     """
#     return components.html(html_code, height=height, scrolling=True)

# def parse_and_display_response(response_text):
#     pattern = r"```mermaid(.*?)```"
#     matches = re.split(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
#     if len(matches) > 1:
#         for i, part in enumerate(matches):
#             if i % 2 == 0:
#                 if part.strip(): st.markdown(part)
#             else:
#                 clean_code = part.strip().replace("`", "")
#                 if clean_code:
#                     st.caption("📊 Process Flow")
#                     render_mermaid(clean_code)
#     else:
#         st.markdown(response_text)

# # --- RAG LOGIC ---
# def ingest_file(db, pdf, category, user):
#     text = "".join([p.extract_text() for p in PdfReader(pdf).pages])
#     chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(text)
    
#     db["documents"].insert_one({
#         "company_id": user["company_id"],
#         "filename": pdf.name,
#         "category": category,
#         "uploaded_by": user["email"],
#         "upload_date": datetime.datetime.now(),
#         "full_text": text
#     })
    
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     collection_name = f"collection_{user['company_id']}"
    
#     try:
#         QdrantVectorStore.from_texts(
#             texts=chunks,
#             embedding=embeddings,
#             url=os.getenv("QDRANT_URL"),
#             api_key=os.getenv("QDRANT_API_KEY"),
#             collection_name=collection_name,
#             force_recreate=False
#         )
#     except Exception as e: st.error(f"Qdrant Error: {e}")

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
        
#         template = """You are AGENTO, a professional internal knowledge assistant for {company_name}.
#         GUIDELINES:
#         1. **Tone:** Professional, concise, and direct.
#         2. **Accuracy:** Answer ONLY based on the Context.
#         VISUALIZATION RULES:
#         1. If the answer describes a workflow/process, generate a Mermaid.js diagram.
#         2. **Syntax Safety:** No brackets () in node labels. Use A[Start] --> B[Process].
#         3. Place mermaid code at the end.

#         Context: {context}
#         Question: {question}
#         Professional Answer:"""
        
#         model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
#         prompt_template = ChatPromptTemplate.from_template(template)
#         chain = (
#             {
#                 "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
#                 "question": RunnablePassthrough(),
#                 "company_name": lambda x: user['company_name']
#             }
#             | prompt_template 
#             | model 
#             | StrOutputParser()
#         )
#         return chain.invoke(query)
#     except Exception:
#         return "⚠️ I am unable to access the company knowledge base at this moment."

# # --- AUDIO HELPERS ---
# def speak_text(text):
#     try:
#         clean_text = re.sub(r"```.*?```", " I have displayed the diagram.", text, flags=re.DOTALL)
#         tts = gTTS(text=clean_text, lang='en', slow=False)
#         fp = BytesIO()
#         tts.write_to_fp(fp)
#         return fp
#     except: return None

# def transcribe_audio(audio_bytes):
#     r = sr.Recognizer()
#     try:
#         with open("temp_input.wav", "wb") as f: f.write(audio_bytes.read())
#         with sr.AudioFile("temp_input.wav") as source:
#             return r.recognize_google(r.record(source))
#     except: return None

# def autoplay_audio(audio_bytes):
#     b64 = base64.b64encode(audio_bytes.read()).decode()
#     md = f"""<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
#     st.markdown(md, unsafe_allow_html=True)

# # --- SIDEBAR ---
# def render_sidebar():
#     user = st.session_state.user
#     with st.sidebar:
#         st.title(f"🏢 {user['company_name']}")
#         st.markdown(f"**Agent:** {user['email']}")
#         st.markdown("---")
#         pages = {
#             "employee_workspace": {"label": "Go to Chat", "icon": "💬"},
#             "ai_call_mode": {"label": "AI Call Mode", "icon": "📞"},
#             "admin_dashboard": {"label": "Admin Dashboard", "icon": "📊", "admin_only": True},
#             "user_profile": {"label": "User Profile", "icon": "👤"}
#         }
#         for page_id, page_info in pages.items():
#             if page_info.get("admin_only") and user['role'] != 'admin': continue
#             if st.session_state.get('page') != page_id:
#                 if st.button(page_info["label"], use_container_width=True): navigate_to(page_id)
#         st.markdown("---")
#         if st.button("Logout", use_container_width=True):
#             for key in st.session_state.keys(): del st.session_state[key]
#             st.session_state.page = "auth"
#             st.query_params.clear()
#             st.rerun()

# # --- PAGES ---
# def page_landing():
#     col1, col2, col3 = st.columns([1,2,1])
#     with col2:
#         st.markdown("<h1 style='text-align: center; margin-top: 50px;'>Welcome Back!</h1>", unsafe_allow_html=True)
#         if st.button("Go to Workspace", use_container_width=True): navigate_to("employee_workspace")
#         if st.button("Logout", use_container_width=True):
#             st.session_state.auth_status = False; st.session_state.page = "auth"; st.rerun()

# def page_auth(db):
#     st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'><i class='bi bi-shield-lock'></i> AGENTO: Secure Access</h1>", unsafe_allow_html=True)
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         tab_login, tab_register, tab_join = st.tabs(["Login", "Register Company", "Join Team"])
#         with tab_login:
#             email = st.text_input("Email", key="l_email")
#             pwd = st.text_input("Password", type="password", key="l_pwd")
#             if st.button("Login"):
#                 if login(db, email, pwd): navigate_to("admin_dashboard" if st.session_state.user["role"] == "admin" else "employee_workspace")
#                 else: st.error("Invalid Credentials")
#         with tab_register:
#             c_name = st.text_input("Company Name")
#             a_email = st.text_input("Admin Email")
#             a_pwd = st.text_input("Admin Password", type="password")
#             if st.button("Register Company"):
#                 success, msg = register_company(db, c_name, a_email, a_pwd)
#                 if success: st.success(f"Success! ID: {msg}"); time.sleep(2); st.rerun()
#                 else: st.error(msg)
#         with tab_join:
#             j_id = st.text_input("Workspace ID")
#             j_email = st.text_input("Your Email")
#             j_pwd = st.text_input("Password", type="password")
#             if st.button("Join Team"):
#                 success, msg = join_company(db, j_email, j_pwd, j_id)
#                 if success: st.success(msg)
#                 else: st.error(msg)

# def page_admin_dashboard(db):
#     user = st.session_state.user
#     render_sidebar()
#     st.title("Company Command Center")
#     col1, col2, col3 = st.columns(3)
#     doc_count = db["documents"].count_documents({"company_id": user["company_id"]})
#     user_count = db["users"].count_documents({"company_id": user["company_id"]})
#     col1.metric("Total Documents", doc_count); col2.metric("Team Members", user_count); col3.metric("System Status", "Online")
#     st.markdown("---")
#     col_upload, col_list = st.columns([1, 2])
#     with col_upload:
#         st.subheader("Upload Knowledge")
#         predefined_cats = ["HR", "Engineering", "Sales", "Legal", "General"]
#         selected = st.selectbox("Category", predefined_cats + ["➕ Create New"])
#         cat = st.text_input("New Category:").strip() if selected == "➕ Create New" else selected
#         files = st.file_uploader("Select PDF", type="pdf", accept_multiple_files=True)
#         if st.button("Index Documents"):
#             if files and cat:
#                 with st.spinner("Pushing to Qdrant..."):
#                     for f in files: ingest_file(db, f, cat, user)
#                 st.success("Indexed!"); time.sleep(1); st.rerun()
#     with col_list:
#         st.subheader("Recent Uploads")
#         docs = list(db["documents"].find({"company_id": user["company_id"]}, {"_id":0, "filename":1, "category":1, "upload_date":1}).sort("upload_date", -1).limit(10))
#         if docs: st.dataframe(pd.DataFrame(docs), use_container_width=True)

# def page_employee_workspace(db):
#     user = st.session_state.user
#     render_sidebar()
#     col1, col2 = st.columns([5, 1])
#     with col1: st.markdown("<h1 style='padding-top: 20px;'>AGENTO AI Assistant</h1>", unsafe_allow_html=True)
#     with col2: st.image("public/globe.gif")
#     if "messages" not in st.session_state: st.session_state.messages = []
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]): 
#             if msg["role"] == "assistant": parse_and_display_response(msg["content"])
#             else: st.markdown(msg["content"])
#     if prompt := st.chat_input("How can I help you?"):
#         st.chat_message("user").markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.spinner("Thinking..."):
#             response = chat_with_jarvis(user, prompt)
#         with st.chat_message("assistant"): parse_and_display_response(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         db["chat_history"].insert_one({"company_id": user["company_id"], "user": user["email"], "query": prompt, "response": response, "timestamp": datetime.datetime.now()})
# # ==============
# # ============================
# # 📞 PAGE: AI CALL MODE (Fixed Sidebar CSS)
# # ==========================================
# def page_ai_call_mode(db):
#     user = st.session_state.user
#     render_sidebar()
    
#     if "audio_key" not in st.session_state: st.session_state.audio_key = 0
#     if "active_diagram" not in st.session_state: st.session_state.active_diagram = None

#     # --- ADVANCED CSS (Modal) ---
#     st.markdown("""
#     <style>
#         .main { background: #000000 !important; } 
#         .block-container { padding-top: 2rem; } 
#         audio { display: none !important; }
        
#         /* 1. THE MODAL CONTAINER */
#         div[data-testid="stVerticalBlock"]:has(div.modal-marker) {
#             position: fixed;
#             top: 50%; left: 50%; transform: translate(-50%, -50%);
#             width: 70vw; max-height: 80vh;
#             background-color: #0D1117; border: 1px solid #58A6FF;
#             border-radius: 12px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.9);
#             z-index: 99999; padding: 20px; overflow-y: auto;
#         }

#         /* 2. CLOSE BUTTON STYLING (SCOPED STRICTLY TO MODAL) */
#         /* This prevents it from breaking the Sidebar buttons */
#         div[data-testid="stVerticalBlock"]:has(div.modal-marker) button {
#             float: right; 
#             border: none; 
#             background: transparent; 
#             color: #ff4b4b; 
#             font-size: 20px;
#         }
#         div[data-testid="stVerticalBlock"]:has(div.modal-marker) button:hover {
#             color: #ff0000;
#             background: rgba(255, 0, 0, 0.1);
#             box-shadow: none;
#         }
#     </style>
#     """, unsafe_allow_html=True)

#     # --- MODAL LOGIC ---
#     modal_placeholder = st.empty()

#     if st.session_state.active_diagram:
#         with modal_placeholder.container():
#             # The Marker allows CSS to find this specific box
#             st.markdown('<div class="modal-marker"></div>', unsafe_allow_html=True)
            
#             c_head, c_close = st.columns([9, 1])
#             with c_head: st.markdown("### 🧠 Workflow Visualization")
#             with c_close: 
#                 if st.button("✕", key="close_modal"):
#                     st.session_state.active_diagram = None
#                     st.rerun()
#             st.markdown("---")
#             render_mermaid(st.session_state.active_diagram, height=450)

#     # --- MAIN SCREEN ---
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.markdown("<h2 style='text-align: center; color: #00FFCC; margin-bottom: 0px;'>AGENTO</h2>", unsafe_allow_html=True)
#         st.image("public/face.gif", use_container_width=True)
#         status = "🟢 LISTENING..." if not st.session_state.get("processing_voice") else "🟣 PROCESSING..."
#         st.markdown(f"<p style='text-align: center; color: #888; letter-spacing: 2px;'>{status}</p>", unsafe_allow_html=True)
    
#     st.write("---") 
#     _, c2, _ = st.columns([2,1,2])
#     with c2: 
#         audio_value = st.audio_input("Tap to Speak", label_visibility="collapsed", key=f"audio_input_{st.session_state.audio_key}")

#     # --- PROCESSING LOOP ---
#     if audio_value:
#         st.session_state.processing_voice = True
#         user_text = transcribe_audio(audio_value)
        
#         if user_text:
#             ai_response = chat_with_jarvis(user, user_text)
            
#             # --- INSTANT DIAGRAM POPUP ---
#             pattern = r"```mermaid(.*?)```"
#             matches = re.split(pattern, ai_response, flags=re.DOTALL | re.IGNORECASE)
            
#             if len(matches) > 1:
#                 diagram_code = matches[1].strip()
#                 st.session_state.active_diagram = diagram_code
                
#                 with modal_placeholder.container():
#                     st.markdown('<div class="modal-marker"></div>', unsafe_allow_html=True)
#                     c_head, c_close = st.columns([9, 1])
#                     with c_head: st.markdown("### 🧠 Workflow Visualization")
#                     with c_close: st.button("✕", disabled=True) 
#                     st.markdown("---")
#                     render_mermaid(diagram_code, height=450)
            
#             # --- AUDIO ---
#             audio_fp = speak_text(ai_response)
#             if audio_fp:
#                 audio_fp.seek(0); audio_meta = MP3(audio_fp); duration = audio_meta.info.length 
#                 audio_fp.seek(0); autoplay_audio(audio_fp)
#                 db["chat_history"].insert_one({"company_id": user["company_id"], "user": user["email"], "query": user_text, "response": ai_response, "timestamp": datetime.datetime.now(), "mode": "voice_call"})
#                 time.sleep(duration + 1)
#                 st.session_state.audio_key += 1
#                 st.rerun()
#         else: st.warning("Could not understand audio.")
#         st.session_state.processing_voice = False
# # ==========================================
# # 👤 PAGE: USER PROFILE (Dynamic)
# # ==========================================
# def page_user_profile(db):
#     user = st.session_state.user
#     render_sidebar()
    
#     st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>User Profile</h1>", unsafe_allow_html=True)
    
#     # Fetch fresh company data (incase stats changed)
#     company_data = db["companies"].find_one({"company_id": user["company_id"]})
    
#     # --- LAYOUT GRID ---
#     col1, col2 = st.columns([1, 1], gap="large")
    
#     # --- LEFT COLUMN: PERSONAL INFO (Everyone sees this) ---
#     with col1:
#         st.markdown("### 👤 Personal Identity")
#         st.markdown(f"""
#         <div style="
#             background-color: rgba(22, 27, 34, 0.8); 
#             padding: 25px; 
#             border-radius: 15px; 
#             border: 1px solid #30363d;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         ">
#             <p style="font-size: 1.1rem; margin-bottom: 10px;"><strong>Email:</strong> <span style="color: #58A6FF;">{user['email']}</span></p>
#             <p style="font-size: 1.1rem; margin-bottom: 10px;"><strong>Role:</strong> {user['role'].upper()}</p>
#             <p style="font-size: 1.1rem; margin-bottom: 0px;"><strong>Status:</strong> <span style="color: #238636;">● Active</span></p>
#         </div>
#         """, unsafe_allow_html=True)

#     # --- RIGHT COLUMN: COMPANY INFO (Admin gets extra details) ---
#     with col2:
#         if user['role'] == 'admin':
#             st.markdown("### 🏢 Company Registry")
            
#             # Format Date
#             reg_date = company_data.get('created_at', datetime.datetime.now()).strftime("%B %d, %Y")
            
#             st.markdown(f"""
#             <div style="
#                 background-color: rgba(22, 27, 34, 0.8); 
#                 padding: 25px; 
#                 border-radius: 15px; 
#                 border: 1px solid #58A6FF; /* Blue border for Admin importance */
#                 box-shadow: 0 0 20px rgba(88, 166, 255, 0.1);
#             ">
#                 <p style="font-size: 1.2rem; font-weight: bold; color: #E6EDF3;">{company_data['name']}</p>
#                 <hr style="border-color: #30363d; margin: 10px 0;">
#                 <p style="margin-bottom: 8px;"><strong>Registered On:</strong> {reg_date}</p>
#                 <p style="margin-bottom: 8px;"><strong>Admin:</strong> {company_data['admin']}</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             st.write("") # Spacer
#             st.markdown("#### 🔑 Workspace Access Key")
#             st.caption("Share this ID with your employees so they can join your workspace:")
#             # Use st.code so it's easy to copy-paste
#             st.code(user['company_id'], language="text")
            
#         else:
#             # EMPLOYEE VIEW
#             st.markdown("### 🏢 Organization")
#             st.markdown(f"""
#             <div style="
#                 background-color: rgba(22, 27, 34, 0.8); 
#                 padding: 25px; 
#                 border-radius: 15px; 
#                 border: 1px solid #30363d;
#             ">
#                 <p style="font-size: 1.1rem;">You are a member of:</p>
#                 <h2 style="color: #58A6FF; margin: 0;">{user['company_name']}</h2>
#                 <p style="margin-top: 10px; color: #888;">Workspace ID: {user['company_id']}</p>
#             </div>
#             """, unsafe_allow_html=True)

#     # --- BOTTOM SECTION: ACCOUNT ACTIONS ---
#     st.write("---")
#     c_left, c_right = st.columns([6, 1])
#     with c_right:
#         if st.button("🔒 Secure Logout", type="primary", use_container_width=True):
#             for key in st.session_state.keys(): del st.session_state[key]
#             st.session_state.page = "auth"
#             st.rerun()
# # --- MAIN ROUTER ---
# def main():
#     db = get_db()
#     if db is None: st.error("Database Connection Failed."); return
#     st.markdown(get_theme_css(), unsafe_allow_html=True)
#     query_params = st.query_params.to_dict()
#     page_in_url = query_params.get("page")
    
#     if not st.session_state.auth_status: page_auth(db)
#     else:
#         page_to_render = URL_TO_PAGE.get(page_in_url, 'landing')
#         if page_to_render == 'landing': page_landing()
#         elif page_to_render == 'auth': page_landing()
#         elif page_to_render == 'admin_dashboard' and st.session_state.user.get('role') == 'admin': page_admin_dashboard(db)
#         elif page_to_render == 'employee_workspace': page_employee_workspace(db)
#         elif page_to_render == 'ai_call_mode': page_ai_call_mode(db)
#         elif page_to_render == 'user_profile': page_user_profile(db)
#         else: page_landing()

# if __name__ == "__main__":
#     main()




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
import re 
import streamlit.components.v1 as components
from mutagen.mp3 import MP3
import base64
from urllib.parse import quote_plus
import urllib.parse
import pymongo.errors

# --- IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS
import speech_recognition as sr

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

# Azure Settings
DB_NAME = "jarvis_saas"
VECTOR_COLLECTION_NAME = "vector_store"
INDEX_NAME = "vectorSearchIndex"

st.set_page_config(page_title="AGENTO: Enterprise", layout="wide", initial_sidebar_state="expanded")

# --- SESSION STATE INIT ---
if "page" not in st.session_state: st.session_state.page = "auth"
if "user" not in st.session_state: st.session_state.user = None
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "theme" not in st.session_state: st.session_state.theme = "dark" 
if "popup_diagram" not in st.session_state: st.session_state.popup_diagram = None
if "audio_key" not in st.session_state: st.session_state.audio_key = 0
if "active_diagram" not in st.session_state: st.session_state.active_diagram = None

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
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
        @keyframes gradientAnimation {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
        :root {{ --background-color: {theme['--background-color']}; --primary-text-color: {theme['--primary-text-color']}; --accent-color: {theme['--accent-color']}; --card-background-color: {theme['--card-background-color']}; }}
        html, body, [class*="st-"]:not(.material-icons):not(.material-symbols-outlined) {{
            font-family: 'Poppins', sans-serif;
            color: var(--primary-text-color);
        }}
        .main {{ background: linear-gradient(-45deg, {theme['--bg-gradient-start']}, {theme['--bg-gradient-end']}); background-size: 400% 400%; animation: gradientAnimation 15s ease infinite; }}
        
        /* SIDEBAR STYLING */
        [data-testid="stSidebar"] {{ 
            background-color: var(--card-background-color); 
            backdrop-filter: blur(10px); 
            border-right: 1px solid {theme['--border-color']}; 
        }}
        
        .stButton > button {{ border: 1px solid var(--accent-color); background-color: transparent; color: var(--accent-color) !important; border-radius: 8px; }}
        .stButton > button:hover {{ background-color: var(--accent-color); color: white !important; box-shadow: 0 0 15px {theme['--glow-color']}; }}
        .stTextInput > div > div > input, .stSelectbox > div > div {{ background-color: var(--card-background-color); color: var(--primary-text-color); border: 1px solid {theme['--border-color']}; border-radius: 8px; }}
        [data-testid="stChatMessage"] {{ background: var(--card-background-color); border: 1px solid {theme['--border-color']}; border-radius: 12px; }}
    </style>
    """

# --- DATABASE CONNECTION LOGIC ---
def sanitize_mongo_uri(uri: str):
    """Percent-encodes username/password in the URI if special chars exist."""
    if not uri: return uri, False
    try:
        parsed = urllib.parse.urlparse(uri)
    except Exception: return uri, False

    if parsed.username or parsed.password:
        hostpart = parsed.netloc.split('@')[-1]
        user = parsed.username or ""
        pwd = parsed.password or ""
        quoted_user = quote_plus(user)
        quoted_pwd = quote_plus(pwd)
        if quoted_user != user or quoted_pwd != pwd:
            new_netloc = f"{quoted_user}:{quoted_pwd}@{hostpart}"
            new_uri = urllib.parse.urlunparse((parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
            return new_uri, True
    return uri, False

def test_mongo_connection(uri: str):
    """Validates MongoDB connection."""
    if not uri: return False, "MONGO_URI is not set"
    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        client[DB_NAME].list_collection_names()
        return True, "Connected"
    except Exception as e:
        return False, f"Connection Failed: {e}"

@st.cache_resource
def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri: return None
    
    ok, msg = test_mongo_connection(uri)
    if not ok:
        return None
        
    try:
        sanitized_uri, changed = sanitize_mongo_uri(uri)
        client = pymongo.MongoClient(sanitized_uri if changed else uri)
        return client[DB_NAME]
    except Exception as e:
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

# --- MERMAID RENDERER ---
def render_mermaid(code, height=400):
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
            suppressErrorRendering: true,
        }});
    </script>
    """
    return components.html(html_code, height=height, scrolling=True)

def parse_and_display_response(response_text):
    pattern = r"```mermaid(.*?)```"
    matches = re.split(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
    if len(matches) > 1:
        for i, part in enumerate(matches):
            if i % 2 == 0:
                if part.strip(): st.markdown(part)
            else:
                clean_code = part.strip().replace("`", "")
                if clean_code:
                    st.caption("📊 Process Flow")
                    render_mermaid(clean_code)
    else:
        st.markdown(response_text)

# --- AZURE COSMOS DB VECTOR SEARCH LOGIC ---

def create_vector_index():
    """Create vector search index in Azure Cosmos DB."""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return False, "MONGO_URI is not set in environment"

    try:
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        collection = client[DB_NAME][VECTOR_COLLECTION_NAME]

        # Drop existing index if it exists
        try:
            if INDEX_NAME in collection.index_information():
                collection.drop_index(INDEX_NAME)
                print(f"🗑️  Dropped existing index: {INDEX_NAME}")
        except:
            pass

        # Create vector search index
        collection.create_index(
            [("vectorContent", "cosmosSearch")],
            name=INDEX_NAME,
            cosmosSearchOptions={
                "kind": "vector-ivf",
                "numLists": 1,
                "similarity": "COS",
                "dimensions": 384
            }
        )
        return True, f"Vector index '{INDEX_NAME}' created successfully."
    except Exception as e:
        return False, f"Failed to create index: {e}"

def direct_vector_search(user, query, k=4):
    """Perform vector search directly using MongoDB aggregation - WORKING VERSION"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        mongo_uri = os.getenv("MONGO_URI")
        client = pymongo.MongoClient(mongo_uri)
        collection = client[DB_NAME][VECTOR_COLLECTION_NAME]
        
        # Generate query embedding
        query_vector = embeddings.embed_query(query)
        
        # WORKING PIPELINE: Search first, filter after
        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_vector,
                        "path": "vectorContent",
                        "k": 10  # Get more results to filter from
                    }
                }
            },
            {
                "$match": {
                    "metadata.company_id": user["company_id"]
                }
            },
            {
                "$limit": k
            }
        ]
        
        # Execute search
        results = list(collection.aggregate(pipeline))
        
        if not results:
            return "No relevant documents found for your company."
        
        # Extract text content
        contexts = [r.get("textContent", "") for r in results]
        return "\n\n".join(contexts)
        
    except Exception as e:
        return f"Search error: {str(e)[:200]}"

def ingest_file(db, pdf, category, user):
    """Ingest PDF file into Azure Cosmos DB with vector embeddings"""
    text = "".join([p.extract_text() for p in PdfReader(pdf).pages])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(text)
    
    # 1. Save Metadata to Azure Mongo
    db["documents"].insert_one({
        "company_id": user["company_id"],
        "filename": pdf.name,
        "category": category,
        "uploaded_by": user["email"],
        "upload_date": datetime.datetime.now(),
        "full_text": text
    })
    
    # 2. Save Vectors to Azure Cosmos Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        mongo_uri = os.getenv("MONGO_URI")
        client = pymongo.MongoClient(mongo_uri)
        collection = client[DB_NAME][VECTOR_COLLECTION_NAME]
        
        for chunk in chunks:
            # Generate embedding
            embedding = embeddings.embed_query(chunk)
            
            # Create document in the correct format
            doc = {
                "metadata": {
                    "company_id": user["company_id"],
                    "category": category,
                    "filename": pdf.name,
                    "uploaded_by": user["email"]
                },
                "textContent": chunk,
                "vectorContent": embedding
            }
            
            # Insert into vector collection
            collection.insert_one(doc)
            
        st.success(f"✅ Indexed {len(chunks)} chunks for {pdf.name}")
        
    except Exception as e:
        st.error(f"Azure Vector Error: {e}")

def chat_with_jarvis(user, query):
    """WORKING RAG function for Azure Cosmos DB with proper filtering"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        mongo_uri = os.getenv("MONGO_URI")
        client = pymongo.MongoClient(mongo_uri)
        collection = client[DB_NAME][VECTOR_COLLECTION_NAME]
        
        # Generate query embedding
        query_vector = embeddings.embed_query(query)
        
        # WORKING PIPELINE: Search first, filter after
        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_vector,
                        "path": "vectorContent",
                        "k": 10  # Get enough to filter from
                    }
                }
            },
            {
                "$match": {
                    "metadata.company_id": user["company_id"]
                }
            },
            {
                "$limit": 4
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        if not results:
            return "I couldn't find relevant information in your company's documents. Please try uploading more documents or asking a different question."
        
        # Prepare context
        contexts = [r.get("textContent", "") for r in results]
        context = "\n\n".join(contexts)
        
        # Generate response with Mermaid diagram support
        template = """You are AGENTO, a professional internal knowledge assistant for {company_name}.

GUIDELINES:
1. **Tone:** Professional, concise, and direct.
2. **Accuracy:** Answer ONLY based on the Context.

VISUALIZATION RULES:
1. If the answer describes a workflow/process, generate a Mermaid.js diagram.
2. **Syntax Safety:** No brackets () in node labels. Use A[Start] --> B[Process].
3. Place mermaid code at the end.

Context: {context}

Question: {question}

Professional Answer:"""
        
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        prompt_template = ChatPromptTemplate.from_template(template)
        
        chain = (
            {
                "context": lambda x: context,
                "question": RunnablePassthrough(),
                "company_name": lambda x: user['company_name']
            }
            | prompt_template
            | model
            | StrOutputParser()
        )
        
        return chain.invoke(query)
        
    except Exception as e:
        err = str(e)
        print(f"Error in chat_with_jarvis: {err}")
        
        # Fallback to simple direct search
        try:
            context = direct_vector_search(user, query, k=4)
            if "Search error" in context:
                return f"⚠️ I'm having trouble accessing the knowledge base. Please try again later."
            
            # Simple fallback response
            simple_template = """Answer based on this context:
            
{context}

Question: {question}

Answer:"""
            
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            prompt_template = ChatPromptTemplate.from_template(simple_template)
            
            chain = (
                {"context": lambda x: context, "question": lambda x: query}
                | prompt_template
                | model
                | StrOutputParser()
            )
            
            return chain.invoke({"context": context, "question": query})
            
        except Exception as e2:
            return f"⚠️ System error: {str(e2)[:200]}"

# --- AUDIO HELPERS ---
def speak_text(text):
    try:
        clean_text = re.sub(r"```.*?```", " I have displayed the diagram.", text, flags=re.DOTALL)
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

# --- PAGES ---
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

    # Vector Index Management Section
    st.subheader("Azure Cosmos DB Vector Index")
    
    if st.button("Create/Verify Vector Index"):
        with st.spinner("Creating/checking vector index..."):
            ok, msg = create_vector_index()
        if ok:
            st.success(msg)
        else:
            st.error(msg)
    
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
                with st.spinner("Pushing to Azure Cosmos DB..."):
                    for f in files: ingest_file(db, f, cat, user)
                st.success("Indexed!"); time.sleep(1); st.rerun()
    with col_list:
        st.subheader("Recent Uploads")
        docs = list(db["documents"].find({"company_id": user["company_id"]}, {"_id":0, "filename":1, "category":1, "upload_date":1}).sort("upload_date", -1).limit(10))
        if docs: st.dataframe(pd.DataFrame(docs), use_container_width=True)

# def page_employee_workspace(db):
#     user = st.session_state.user
#     render_sidebar()
#     col1, col2 = st.columns([5, 1])
#     with col1: st.markdown("<h1 style='padding-top: 20px;'>AGENTO AI Assistant</h1>", unsafe_allow_html=True)
#     with col2: st.image("public/globe.gif")
#     if "messages" not in st.session_state: st.session_state.messages = []
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]): 
#             if msg["role"] == "assistant": parse_and_display_response(msg["content"])
#             else: st.markdown(msg["content"])
#     if prompt := st.chat_input("How can I help you?"):
#         st.chat_message("user").markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.spinner("Thinking..."):
#             response = chat_with_jarvis(user, prompt)
#         with st.chat_message("assistant"): parse_and_display_response(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         db["chat_history"].insert_one({"company_id": user["company_id"], "user": user["email"], "query": prompt, "response": response, "timestamp": datetime.datetime.now()})

def page_employee_workspace(db):
    user = st.session_state.user
    render_sidebar()
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("<div class='hero-title'>AGENTO AI Assistant</div>", unsafe_allow_html=True)
    with col2:
        st.image("public/globe.gif")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # FIX: Use proper emojis instead of shortcodes
    for msg in st.session_state.messages:
        # Use different avatars for user vs assistant
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant":
                parse_and_display_response(msg["content"])
            else:
                st.markdown(msg["content"])
    
    if prompt := st.chat_input("How can I help you?"):
        # User message with avatar
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = chat_with_jarvis(user, prompt)
        
        # Assistant message with avatar
        with st.chat_message("assistant", avatar="🤖"):
            parse_and_display_response(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        db["chat_history"].insert_one({
            "company_id": user["company_id"],
            "user": user["email"],
            "query": prompt,
            "response": response,
            "timestamp": datetime.datetime.now()
        })


# ==========================================
# 📞 PAGE: AI CALL MODE (Fixed Layout)
# ==========================================
def page_ai_call_mode(db):
    user = st.session_state.user
    render_sidebar()
    
    if "audio_key" not in st.session_state: st.session_state.audio_key = 0
    if "active_diagram" not in st.session_state: st.session_state.active_diagram = None
    if "last_processed_audio" not in st.session_state: st.session_state.last_processed_audio = b""

    st.markdown("""
    <style>
        .main { background: #000000 !important; } 
        .block-container { padding-top: 2rem; } 
        audio { display: none !important; }
        
        div[data-testid="column"] { display: flex; align-items: center; justify-content: center; }
        .stAudioInput { width: 100%; }
        div[data-testid="column"] button { border-radius: 50%; width: 50px; height: 50px; padding: 0; display: flex; align-items: center; justify-content: center; }
        div[data-testid="stVerticalBlock"]:has(div.modal-marker) { position: fixed; top: 50% !important; left: 50% !important; transform: translate(-50%, -50%) !important; width: 600px; max-width: 90vw; max-height: 80vh; background-color: #0D1117; border: 1px solid #58A6FF; border-radius: 12px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.9); z-index: 999999; padding: 20px; overflow-y: auto; margin: 0 !important; }
        div[data-testid="stVerticalBlock"]:has(div.modal-marker) button { float: right; border: none; background: transparent; color: #ff4b4b; font-size: 20px; border-radius: 5px; width: auto; height: auto; }
    </style>
    """, unsafe_allow_html=True)

    modal_placeholder = st.empty()
    if st.session_state.active_diagram:
        with modal_placeholder.container():
            st.markdown('<div class="modal-marker"></div>', unsafe_allow_html=True)
            c_head, c_close = st.columns([9, 1])
            with c_head: st.markdown("### 🧠 Workflow Visualization")
            with c_close: 
                if st.button("✕", key="close_modal_btn"):
                    st.session_state.active_diagram = None
                    st.rerun()
            st.markdown("---")
            render_mermaid(st.session_state.active_diagram, height=400)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #00FFCC; margin-bottom: 0px;'>AGENTO</h2>", unsafe_allow_html=True)
        st.image("public/face.gif", use_container_width=True)
        status_placeholder = st.empty()
        status_text = "🟢 LISTENING..." if not st.session_state.get("processing_voice") else "🟣 PROCESSING..."
        status_placeholder.markdown(f"<p style='text-align: center; color: #888; letter-spacing: 2px;'>{status_text}</p>", unsafe_allow_html=True)
    st.write("---") 
    c_spacer_l, c_mic, c_btn, c_spacer_r = st.columns([1, 3, 0.5, 1])
    with c_mic: audio_value = st.audio_input("Tap to Speak", label_visibility="collapsed")
    with c_btn:
        has_diagram = st.session_state.active_diagram is not None
        if st.button("📊", disabled=not has_diagram, help="Show Diagram"):
            st.session_state.active_diagram = st.session_state.active_diagram
            st.rerun()

    if audio_value:
        current_audio_bytes = audio_value.getvalue()
        if current_audio_bytes != st.session_state.last_processed_audio:
            st.session_state.processing_voice = True
            st.session_state.last_processed_audio = current_audio_bytes
            status_placeholder.markdown(f"<p style='text-align: center; color: #888; letter-spacing: 2px;'>🟣 PROCESSING...</p>", unsafe_allow_html=True)
            user_text = transcribe_audio(audio_value)
            if user_text:
                ai_response = chat_with_jarvis(user, user_text)
                pattern = r"```mermaid(.*?)```"
                matches = re.split(pattern, ai_response, flags=re.DOTALL | re.IGNORECASE)
                if len(matches) > 1:
                    diagram_code = matches[1].strip()
                    st.session_state.active_diagram = diagram_code
                    with modal_placeholder.container():
                        st.markdown('<div class="modal-marker"></div>', unsafe_allow_html=True)
                        c_head, c_close = st.columns([9, 1])
                        with c_head: st.markdown("### 🧠 Workflow Visualization")
                        with c_close: st.button("✕", disabled=True) 
                        st.markdown("---")
                        render_mermaid(diagram_code, height=400)
                audio_fp = speak_text(ai_response)
                if audio_fp:
                    audio_fp.seek(0); autoplay_audio(audio_fp)
                    db["chat_history"].insert_one({"company_id": user["company_id"], "user": user["email"], "query": user_text, "response": ai_response, "timestamp": datetime.datetime.now(), "mode": "voice_call"})
            else: st.warning("Could not understand audio.")
            st.session_state.processing_voice = False
            status_placeholder.markdown(f"<p style='text-align: center; color: #888; letter-spacing: 2px;'>🟢 LISTENING...</p>", unsafe_allow_html=True)

def page_user_profile(db):
    user = st.session_state.user
    render_sidebar()
    st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>User Profile</h1>", unsafe_allow_html=True)
    company_data = db["companies"].find_one({"company_id": user["company_id"]})
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("### 👤 Personal Identity")
        st.markdown(f"""<div style="background-color: rgba(22, 27, 34, 0.8); padding: 25px; border-radius: 15px; border: 1px solid #30363d;"><p style="font-size: 1.1rem; margin-bottom: 10px;"><strong>Email:</strong> <span style="color: #58A6FF;">{user['email']}</span></p><p style="font-size: 1.1rem; margin-bottom: 10px;"><strong>Role:</strong> {user['role'].upper()}</p><p style="font-size: 1.1rem; margin-bottom: 0px;"><strong>Status:</strong> <span style="color: #238636;">● Active</span></p></div>""", unsafe_allow_html=True)
    with col2:
        if user['role'] == 'admin':
            st.markdown("### 🏢 Company Registry")
            reg_date = company_data.get('created_at', datetime.datetime.now()).strftime("%B %d, %Y")
            st.markdown(f"""<div style="background-color: rgba(22, 27, 34, 0.8); padding: 25px; border-radius: 15px; border: 1px solid #58A6FF; box-shadow: 0 0 20px rgba(88, 166, 255, 0.1);"><p style="font-size: 1.2rem; font-weight: bold; color: #E6EDF3;">{company_data['name']}</p><hr style="border-color: #30363d; margin: 10px 0;"><p style="margin-bottom: 8px;"><strong>Registered On:</strong> {reg_date}</p><p style="margin-bottom: 8px;"><strong>Admin:</strong> {company_data['admin']}</p></div>""", unsafe_allow_html=True)
            st.write(""); st.markdown("#### 🔑 Workspace Access Key"); st.caption("Share this ID with your employees:"); st.code(user['company_id'], language="text")
        else:
            st.markdown("### 🏢 Organization")
            st.markdown(f"""<div style="background-color: rgba(22, 27, 34, 0.8); padding: 25px; border-radius: 15px; border: 1px solid #30363d;"><p style="font-size: 1.1rem;">You are a member of:</p><h2 style="color: #58A6FF; margin: 0;">{user['company_name']}</h2><p style="margin-top: 10px; color: #888;">Workspace ID: {user['company_id']}</p></div>""", unsafe_allow_html=True)
    st.write("---")
    c_left, c_right = st.columns([6, 1])
    with c_right:
        if st.button("🔒 Secure Logout", type="primary", use_container_width=True):
            for key in st.session_state.keys(): del st.session_state[key]
            st.session_state.page = "auth"; st.rerun()

# --- MAIN ROUTER ---
def main():
    db = get_db()
    if db is None:
        st.error("❌ Database Connection Failed. Please check .env settings.")
        return
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    query_params = st.query_params.to_dict()
    page_in_url = query_params.get("page")
    
    if not st.session_state.auth_status:
        page_auth(db)
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
