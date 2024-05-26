import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load the pre-trained emotion detection model and the face detector
emotion_model_path = r'd:\download (chrome)\emotion_detection_model.h5'  # Replace with the path to your emotion model
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
    print("Number of faces detected:", len(faces))  # Debugging line
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

def track_emotions():
    emotions_over_time = []
    
    cap = cv2.VideoCapture(0)
    
    while True:
        text = input("Enter your thoughts (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        
        emotion, scores = analyze_emotion(text)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue
        
        face_emotions = analyze_face_emotion(frame)
        
        emotions_over_time.append({
            "timestamp": timestamp,
            "text": text,
            "emotion": emotion,
            "scores": scores,
            "face_emotions": [fe[0] for fe in face_emotions]
        })
        
        print(f"Text Emotion: {emotion}, Scores: {scores}, Face Emotions: {[fe[0] for fe in face_emotions]}")
        
        # Display the image with emotions
        for (emotion_label, face_coords) in face_emotions:
            if face_coords is not None:
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Emotion Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    # Save emotions to a file
    with open('emotions.json', 'w') as f:
        json.dump(emotions_over_time, f, indent=4)
    
    # Plotting emotions over time
    plot_emotions(emotions_over_time)

def plot_emotions(emotions_over_time):
    timestamps = [entry['timestamp'] for entry in emotions_over_time]
    happy_scores = [1 if entry['emotion'] == 'Happy' else 0 for entry in emotions_over_time]
    sad_scores = [1 if entry['emotion'] == 'Sad' else 0 for entry in emotions_over_time]
    angry_scores = [1 if entry['emotion'] == 'Angry' else 0 for entry in emotions_over_time]
    romantic_scores = [1 if entry['emotion'] == 'Romantic' else 0 for entry in emotions_over_time]
    helpless_scores = [1 if entry['emotion'] == 'Helpless' else 0 for entry in emotions_over_time]
    other_scores = [1 if entry['emotion'] == 'Neutral' else 0 for entry in emotions_over_time]
    
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, happy_scores, label='Happy', color='green')
    plt.plot(timestamps, sad_scores, label='Sad', color='blue')
    plt.plot(timestamps, angry_scores, label='Angry', color='red')
    plt.plot(timestamps, romantic_scores, label='Romantic', color='pink')
    plt.plot(timestamps, helpless_scores, label='Helpless', color='purple')
    plt.plot(timestamps, other_scores, label='Other', color='grey')
    
    plt.xlabel('Time')
    plt.ylabel('Emotion Frequency')
    plt.title('Emotion Tracking Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    track_emotions()
