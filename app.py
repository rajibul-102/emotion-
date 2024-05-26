from flask import Flask, render_template, request, jsonify, Response
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load the pre-trained emotion detection model and the face detector
emotion_model_path = 'emotion_detection_model.h5'  # Replace with the path to your emotion model
emotion_model = load_model(emotion_model_path)
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

def analyze_emotion(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        emotion = "Happy"
    elif compound <= -0.05:
        emotion = "Sad"
    else:
        emotion = "Neutral"
        
    if "love" in text.lower() or "romantic" in text.lower():
        emotion = "Romantic"
    elif "angry" in text.lower() or "mad" in text.lower() or "furious" in text.lower():
        emotion = "Angry"
    elif "helpless" in text.lower() or "hopeless" in text.lower():
        emotion = "Helpless"
        
    return emotion, scores

def analyze_face_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3)
    emotions = []
    
    if len(faces) == 0:
        emotions.append(("Null 404 No Face Detected", None))
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+w, x:x+h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = np.array(roi_gray).astype('float32')
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255

            emotion_prediction = emotion_model.predict(image_pixels)
            max_index = np.argmax(emotion_prediction[0])
            predicted_emotion = emotion_dict[max_index]
            emotions.append((predicted_emotion, (x, y, w, h)))
        
    return emotions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data['text']
    emotion, scores = analyze_emotion(text)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "timestamp": timestamp,
        "text": text,
        "emotion": emotion,
        "scores": scores
    }
    return jsonify(result)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            face_emotions = analyze_face_emotion(frame)
            for (emotion_label, face_coords) in face_emotions:
                if face_coords is not None:
                    x, y, w, h = face_coords
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)
