import cv2
import numpy as np
import os
import google.generativeai as genai
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

difficulty_level = 2

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Extract regions for emotion detection
        mouth_region = roi_gray[int(0.7 * h):h, :]
        eye_region = roi_gray[:int(0.3 * h), :]

        # Calculate intensity for heuristics
        mouth_intensity = np.mean(mouth_region)
        eye_intensity = np.mean(eye_region)

        # Heuristic rules for emotions
        if mouth_intensity > 130 and eye_intensity < 80:
            #emotion = "Angry"
            difficulty_level = 1
        elif mouth_intensity > 120:
            #emotion = "Happy"
            difficulty_level = 3
        elif mouth_intensity < 80 and eye_intensity > 100:
            #emotion = "Sad"
            difficulty_level = 1
        elif 85 < mouth_intensity < 120 and 85 < eye_intensity < 120:
            #emotion = "Confused"
            difficulty_level = 1
        else:
            #emotion = "Neutral"
            difficulty_level = 2

    def create_prompt(difficulty, qtype, topic):
      if difficulty == 1:
        prompt = "Give me an easy"
      if difficulty == 2:
        prompt = "Give me a medium difficulty"
      if difficulty == 1:
        prompt = "Give me a hard"
      prompt = prompt + qtype + " question based on " + topic
      return prompt
    #Question bot
    genai.configure(api_key = "AIzaSyCfOvUOrHl-y_frEUDII4SF3CNgNTu5jTQ")
    model = genai.GenerativeModel()
    history = []
    qtype = "Theory"
    topic = "Statistics"
    prompt = create_prompt(difficulty_level,qtype,topic)
    while True:
      chat_session = model.start_chat(
      history = history
      )
    response = chat_session.send_message(prompt)
    model_response = response.text
    Q = model_response.split('?')[0].split(':')[1] + '?'
    optA = model_response.split('.')[0].split('?')[1]
    keys = ['Q','a','b','c','d',"correct"]
    values = [Q,model_response.split('.')[0].split('?')[1]]
    values.extend(model_response.split('.')[1:4])
    values.append(model_response.split(':')[2].split('.')[0])
    kv = dict(zip(keys, values))

    @app.route
    def get_data():
      return jsonify(kv),200, {"Content-Type": "application/json"};
    app.run(host = "0.0.0.0",port = 5000)

    if keyboard.is_pressed('q'):
      break

cap.release()
cv2.destroyAllWindows()
