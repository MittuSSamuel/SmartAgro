# SmartAgro - Streamlit Prototype

This is a lightweight prototype for the SmartAgro system designed for hackathon demo purposes.
Features included:
- Image-based heuristic disease detection (rule-based, demo)
- Simple disease risk prediction (rule-based)
- Text-to-speech "voice assistant" demo (gTTS)
- Local community feed stored in `community.json`
- Single-file Streamlit app: `app.py`

## How to run (locally)
1. Create a python virtual environment (recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Open the URL shown by Streamlit (usually http://localhost:8501).

Notes:
- The image analysis is heuristic-based (color + spot detection) to simulate AI results without heavy ML models.
- For a production/strong demo, replace `classify_plant_image` with a trained CNN model (TensorFlow/PyTorch) and use a proper backend.
- community.json is used as a simple local store. For a cloud-backed demo, integrate Firebase Firestore.

## Updated
- Improved Streamlit styling and layout
- Optional TensorFlow SavedModel loader (place model in ./model)
- Firebase Firestore scaffolding (place firebase_config.json)

## New Features Added
- OpenWeatherMap integration: enter API key in sidebar to auto-fill weather sensors.
- Browser audio recorder HTML snippet (demo) + upload support.
- Speech-to-text scaffolding using SpeechRecognition (Google Web API) if available.
- Transfer-learning training script `train_mobilenet.py` to fine-tune MobileNetV2 and export a SavedModel to `./model/`.

See DEV_NOTES.txt for details.
