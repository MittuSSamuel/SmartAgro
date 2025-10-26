import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io, os, json, base64
import time

# Optional ML imports (only used if a model is present)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False

# Optional Firebase (firestore) imports
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except Exception as e:
    FIREBASE_AVAILABLE = False

# --- Config ---
st.set_page_config(page_title="SmartAgro — Prototype", layout="wide", initial_sidebar_state="expanded")

# CSS Styling
def local_css(css_text: str):
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

css = """
/* Basic card look */
.block-container{padding-top:1rem;}
.card {background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); border-radius:12px; padding:16px; box-shadow: 0 4px 14px rgba(30,41,59,0.06);}
.header {display:flex; align-items:center; gap:12px;}
.kpi {background:#0ea5a4; color:white; padding:10px; border-radius:8px; font-weight:600;}
.small-muted {color:#6b7280; font-size:0.95rem;}
.center {text-align:center;}
"""

local_css(css)

# --- Helpers ---
def pil_to_np(img):
    return np.array(img.convert("RGB"))

def detect_spots_and_yellow(np_img):
    # same heuristics as before but with safer checks
    try:
        import cv2
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        lower_y = np.array([15,40,40])
        upper_y = np.array([40,255,255])
        mask_y = cv2.inRange(hsv, lower_y, upper_y)
        pct_yellow = mask_y.sum()/(255*np_img.shape[0]*np_img.shape[1])
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        _,th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum([cv2.contourArea(c) for c in contours])
        spot_score = min(1.0, total_area/(np_img.shape[0]*np_img.shape[1]*0.05))
        return float(pct_yellow), float(spot_score)
    except Exception:
        arr = np_img.astype('float32')/255.0
        r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        yellow_mask = (r>0.6) & (g>0.5) & (b<0.5)
        pct_yellow = yellow_mask.mean()
        lum = 0.2126*r + 0.7152*g + 0.0722*b
        spot_score = np.mean(lum<0.3)
        return float(pct_yellow), float(spot_score)

def heuristic_classify(img: Image.Image):
    np_img = pil_to_np(img.resize((400,400)))
    pct_yellow, spot_score = detect_spots_and_yellow(np_img)
    if spot_score > 0.08:
        label = "Leaf Blight (possible fungal/bacterial)"
        confidence = min(0.95, 0.6 + spot_score)
    elif pct_yellow > 0.12:
        label = "Nitrogen Deficiency (yellowing)"
        confidence = min(0.9, 0.55 + pct_yellow)
    else:
        label = "Likely Healthy"
        confidence = max(0.5, 1 - (pct_yellow+spot_score))
    region = "Leaf"
    return {"label": label, "confidence": round(confidence*100,1), "pct_yellow": round(pct_yellow,3), "spot_score": round(spot_score,3), "region": region}

# Optional model loader
MODEL_DIR = "model"
model = None
if TF_AVAILABLE and os.path.isdir(MODEL_DIR):
    try:
        model = tf.keras.models.load_model(MODEL_DIR)
        st.sidebar.success("Loaded SavedModel from ./model")
    except Exception as e:
        st.sidebar.warning("SavedModel found but failed to load. Falling back to heuristics.")

def model_predict(img: Image.Image):
    # expects model to accept 224x224 RGB input, normalize to [0,1]
    try:
        img_resized = img.resize((224,224))
        arr = np.array(img_resized).astype("float32")/255.0
        x = np.expand_dims(arr, axis=0)
        preds = model.predict(x)
        # If model outputs probabilities and class names in config, map them.
        # We provide fallback mapping for common case of binary or multi-class.
        if preds.ndim==2:
            idx = int(np.argmax(preds[0]))
            prob = float(np.max(preds[0]))
            # Attempt to load class names file
            classes_path = os.path.join(MODEL_DIR, "classes.json")
            if os.path.exists(classes_path):
                with open(classes_path,"r") as f:
                    classes = json.load(f)
                label = classes[idx]
            else:
                label = f"Class {idx}"
            return {"label": label, "confidence": round(prob*100,1), "model_used": True}
        else:
            return {"label":"Model output (uninterpreted)","confidence":0,"model_used":True}
    except Exception as e:
        return {"error": str(e)}

# Firestore integration scaffolding
FIREBASE_CONFIG_PATH = "firebase_config.json"
db = None
if FIREBASE_AVAILABLE and os.path.exists(FIREBASE_CONFIG_PATH):
    try:
        cred = credentials.Certificate(FIREBASE_CONFIG_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        st.sidebar.success("Connected to Firestore (firebase_config.json)")
    except Exception as e:
        st.sidebar.error("Failed to init Firebase Admin SDK: " + str(e))

# Simple local JSON "DB" for community posts
DB_PATH = "community.json"
if not os.path.exists(DB_PATH):
    with open(DB_PATH,"w") as f:
        json.dump([{"farmer":"Ravi","crop":"Tomato","alert":"Detected leaf blight in East plot. Apply fungicide."}], f, indent=2)

def load_posts_local():
    try:
        with open(DB_PATH,"r") as f:
            return json.load(f)
    except:
        return []

def add_post_local(farmer, crop, alert):
    posts = load_posts_local()
    posts.insert(0, {"farmer":farmer,"crop":crop,"alert":alert, "ts": int(time.time())})
    with open(DB_PATH,"w") as f:
        json.dump(posts, f, indent=2)

def add_post_firestore(farmer, crop, alert):
    if db is None:
        return False, "Firestore not configured"
    doc = {"farmer":farmer,"crop":crop,"alert":alert, "ts": firestore.SERVER_TIMESTAMP}
    db.collection("community").add(doc)
    return True, "Posted to Firestore"

# --- UI ---
st.title("🌾 SmartAgro — Prototype")
st.markdown("**AI + IoT + Community** — crop health monitoring demo. (Hackathon-ready Streamlit UI)")
st.sidebar.header("Prototype Controls")
use_model = st.sidebar.checkbox("Use SavedModel if available", value=True)
use_firestore = st.sidebar.checkbox("Use Firestore (requires firebase_config.json)", value=False)

# Top KPIs
k1, k2, k3 = st.columns([1,1,1])
with k1:
    st.markdown('<div class="card center"><div class="kpi">Live Risk</div><div class="small-muted">Calculated from sensors & weather</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="card center"><div class="kpi">Last Detection</div><div class="small-muted">Most recent image analysis</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="card center"><div class="kpi">Community Alerts</div><div class="small-muted">Nearby farmer reports</div></div>', unsafe_allow_html=True)

st.markdown("---")

tabs = st.tabs(["Detection", "Risk Prediction", "Voice Assistant", "Community", "Dashboard", "Dev / Ops"])

with tabs[0]:
    st.header("1. Crop Disease Detection")
    st.write("Upload an image (leaf/plant) captured by rover/drone/phone. If a SavedModel is provided in `./model`, it will be used.")
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], accept_multiple_files=False, key="upl")
    col1,col2 = st.columns([2,1])
    with col1:
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img_disp = ImageOps.fit(img, (600,400))
            st.image(img_disp, caption="Uploaded image", use_column_width=True)
        else:
            st.info("No image uploaded. You can use the sample images folder (not included) or upload your own.")
    with col2:
        if uploaded:
            detector_choice = "model" if (use_model and model is not None) else "heuristic"
            st.markdown(f"**Detection method:** `{detector_choice}`")
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    if detector_choice=="model":
                        pred = model_predict(img)
                        if pred.get("error"):
                            st.error("Model error: " + pred["error"])
                            pred = heuristic_classify(img)
                            st.warning("Falling back to heuristic result")
                        else:
                            st.success(f"Model detected: **{pred['label']}** — Confidence: **{pred['confidence']}%**")
                            st.write("Note: Provide classes.json in model/ to map indices to labels.")
                    else:
                        pred = heuristic_classify(img)
                        st.success(f"Detected: **{pred['label']}** — Confidence: **{pred['confidence']}%**")
                        st.write("Heuristics:", {"yellow_pct":pred["pct_yellow"], "spot_score":pred["spot_score"]})
                    st.info("Recommendation:")
                    # recommendation logic
                    if "Blight" in pred["label"] or "blight" in pred["label"].lower():
                        st.write("- Apply recommended fungicide; avoid overhead irrigation; improve drainage.")
                    elif "Nitrogen" in pred["label"]:
                        st.write("- Apply nitrogen-rich fertilizer and test soil pH.")
                    elif "Healthy" in pred["label"]:
                        st.write("- No action needed. Continue monitoring.")
                    else:
                        st.write("- Monitor and consult agronomist.")
                    # Post option
                    if st.checkbox("Save alert to community feed"):
                        farmer_name = st.text_input("Your name to post as", value="FarmerUser", key="postname")
                        crop_name = st.text_input("Crop", value="Tomato", key="postcrop")
                        if st.button("Post Alert to Feed"):
                            if use_firestore and FIREBASE_AVAILABLE and os.path.exists(FIREBASE_CONFIG_PATH):
                                ok,msg = add_post_firestore(farmer_name, crop_name, f"{pred['label']}")
                                if ok:
                                    st.success(msg)
                                else:
                                    st.error(msg)
                            else:
                                add_post_local(farmer_name, crop_name, f"{pred['label']}")
                                st.success("Posted locally to community feed")

with tabs[1]:
    st.header("2. Disease Risk Prediction (Rule-based / ML placeholder)")
    st.write("---")
    st.subheader('OpenWeatherMap Integration (optional)')
    st.write('Enter your OpenWeatherMap API key (or leave blank to use manual inputs).')
    owm_key = st.text_input('OpenWeatherMap API Key', value='', type='password')
    owm_city = st.text_input('City name for weather (e.g., Chennai)', value='Chennai')
    if owm_key:
        try:
            import requests
            resp = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={owm_city}&appid={owm_key}&units=metric", timeout=8)
            if resp.status_code==200:
                w = resp.json()
                temp_c = int(round(w['main']['temp']))
                humidity = int(round(w['main']['humidity']))
                st.success(f"Loaded weather for {owm_city}: {temp_c}°C, {humidity}% humidity")
            else:
                st.error('OpenWeatherMap error: ' + str(resp.status_code) + ' - ' + resp.text[:150])
        except Exception as e:
            st.error('Failed to fetch OWM: ' + str(e))
    st.write('You can also adjust sensor sliders below manually.')
    st.write("Provide sensor values (or use defaults). The app estimates risk using simple rules. For a production model, replace with a trained classifier.")
    col1,col2,col3 = st.columns(3)
    with col1:
        soil_moisture = st.slider("Soil moisture (%)", 0, 100, 30, key="sm")
    with col2:
        temp_c = st.slider("Temperature (°C)", -5, 50, 32, key="t")
    with col3:
        humidity = st.slider("Humidity (%)", 0, 100, 75, key="h")
    risk = "Low"
    reasons = []
    if humidity > 70 and temp_c > 28:
        risk = "High"
        reasons.append("Warm and humid — favorable for fungal growth")
    if soil_moisture < 15:
        reasons.append("Low soil moisture — plant stress")
        if risk != "High":
            risk = "Moderate"
    st.metric("Predicted Disease Risk", risk)
    st.write("Reasons:", reasons or ["No immediate high-risk indicators"])
    with st.expander("Suggested preventive actions"):
        if risk=="High":
            st.write("- Inspect crops in next 24 hours\n- Apply preventive fungicide if applicable\n- Improve drainage and avoid overhead irrigation")
        elif risk=="Moderate":
            st.write("- Monitor fields closely\n- Adjust irrigation to reduce plant stress")
        else:
            st.write("- Continue routine monitoring")

with tabs[2]:
    st.header("3. Voice Assistant (Text-to-Speech demo)")
    st.write("Type a question and choose language. This demo uses gTTS to play audio answers.")
    query = st.text_input("Ask SmartAgro", "What is the status of my tomato field?")
    lang = st.selectbox("Language", ("en","ta"))
    if st.button("Speak Answer"):
        posts = load_posts_local()
        last_alert = posts[0]["alert"] if posts else "No recent alerts in your area."
        answer = f"Status: Risk level is {risk}. Recent alert: {last_alert}"
        try:
            from gtts import gTTS
            tts = gTTS(text=answer, lang=lang)
            bio = io.BytesIO()
            tts.write_to_fp(bio)
            bio.seek(0)
            st.audio(bio.read(), format="audio/mp3")
        except Exception as e:
            st.error("gTTS audio failed: " + str(e))
    st.write('---')
    st.subheader('Record voice (in browser)')
    st.write('You can record audio in the browser using the recorder below, then upload it to use speech-to-text or save for records.')
    recorder_html = """
    <script>
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));
    async function startRecorder(){
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      let chunks = [];
      mediaRecorder.ondataavailable = e => chunks.push(e.data);
      mediaRecorder.onstop = async e => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
          const base64data = reader.result;
          const el = document.getElementById('rec-output');
          el.value = base64data;
        }
      };
      document.getElementById('rec-status').innerText = 'Recording...';
      mediaRecorder.start();
      await sleep(4000);
      mediaRecorder.stop();
      document.getElementById('rec-status').innerText = 'Recording complete. Click Upload to send to Streamlit.';
    }
    </script>
    <div>
      <button onclick="startRecorder()">Record 4s</button>
      <span id='rec-status'>Idle</span>
      <input id='rec-output' type='hidden'></input>
    </div>
    """
    import streamlit.components.v1 as components
    components.html(recorder_html, height=140)
    st.write('After recording, copy the value of the hidden field `rec-output` into a text file and upload it below, or use the native Upload button to submit your recorded file.')
    st.write('Browser recording is limited in this demo. Alternatively, upload an audio file (wav/mp3).')
    uploaded_audio = st.file_uploader('Upload recorded audio (wav/mp3/webm)', type=['wav','mp3','webm'])
    if uploaded_audio is not None:
        st.audio(uploaded_audio)
        st.write('You uploaded:', uploaded_audio.name)
        st.write('Speech-to-text (scaffolding):')
        st.write('If you have Google Cloud Speech credentials or another STT API, configure it and we can transcribe uploaded audio. For now, the app will attempt local transcription using SpeechRecognition (if installed).')
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with open('temp_audio', 'wb') as f:
                f.write(uploaded_audio.read())
            audio_file = sr.AudioFile('temp_audio')
            with audio_file as source:
                audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data)
                st.success('Transcription (Google Web API): ' + text)
            except Exception as e:
                st.error('Local transcription failed: ' + str(e))
        except Exception as e:
            st.info('SpeechRecognition library not available or transcription failed: ' + str(e))

with tabs[3]:
    st.header("4. Farmer Community Feed")
    st.write("Share updates and view nearby alerts. Toggle Firestore in sidebar to post to cloud (requires firebase_config.json).")
    if use_firestore and FIREBASE_AVAILABLE and db is not None:
        st.success("Firestore connected: community posts will be saved to cloud")
    with st.form("postform", clear_on_submit=True):
        name = st.text_input("Your name", "Ravi")
        crop = st.text_input("Crop", "Tomato")
        alert = st.text_area("Alert / Update", "Detected leaf blight in east plot.")
        submitted = st.form_submit_button("Post")
        if submitted:
            if use_firestore and FIREBASE_AVAILABLE and db is not None:
                ok,msg = add_post_firestore(name,crop,alert)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                add_post_local(name,crop,alert)
                st.success("Posted locally!")
    st.subheader("Recent posts")
    posts = load_posts_local()
    for p in posts[:50]:
        ts = p.get("ts")
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
        st.markdown(f"**{p['farmer']}** — *{p['crop']}*  \n> {p['alert']}  \n**{ts_str}**")

with tabs[4]:
    st.header("5. Combined Dashboard")
    left, right = st.columns([2,1])
    with left:
        st.subheader("Recent Detection & Risk")
        if uploaded:
            st.image(img_disp, width=450)
            st.write(pred if 'pred' in locals() else "No detection result in this session")
        else:
            st.info("No image uploaded in this session.")
        st.subheader("Mock Weather / Soil Snapshot")
        st.write(f"Temperature: {temp_c} °C  \nHumidity: {humidity}%  \nSoil moisture: {soil_moisture}%")
    with right:
        st.subheader("Nearby Alerts (summary)")
        posts = load_posts_local()[:10]
        st.write(f"Recent {len(posts)} posts")
        for p in posts[:5]:
            st.write(f"- **{p['farmer']}**: {p['alert']}")

with tabs[5]:
    st.header("Dev / Ops & Integration Notes")
    st.markdown("""
    **SavedModel integration**:
    - Place a TensorFlow SavedModel directory under `./model/` (this is the folder expected by the app).
    - If the model outputs class probabilities, add a `classes.json` mapping file in the model folder to map indices to readable labels.
    - Example `classes.json`: `["Healthy","Leaf Blight","Nitrogen Deficiency"]`

    **Firestore integration**:
    - Create a Firebase project and download the service account JSON. Save it as `firebase_config.json` in the project root.
    - Install `firebase-admin` and enable the Firestore API for the service account.
    - Toggle 'Use Firestore' in the sidebar to post cloud-backed community messages.

    **Styling & UI**:
    - The app includes basic CSS for a card-like look. You can further customize by editing the `css` variable in the app.

    **Notes & next steps**:
    - I added a SavedModel loader and Firestore scaffolding; because we cannot distribute heavy ML models, the app falls back to heuristics when a model is not present.
    - If you'd like, I can prepare a small transfer-learning notebook to fine-tune MobileNetV2 on PlantVillage and produce a `./model` folder you can drop into this app.
    """)

st.caption("Prototype built for hackathon demo. Not a substitute for expert agronomic advice.")