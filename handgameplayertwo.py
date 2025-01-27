import ssl
import cv2
import mediapipe as mp
from pytube import YouTube
import os
from pynput.keyboard import Key, Controller
import time
from collections import Counter

ssl._create_default_https_context = ssl._create_unverified_context
keyboard = Controller()

def download_youtube_video(url, output_path='video.mp4'):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    stream.download(filename=output_path)
    return output_path

def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    if thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return 'UP'
    elif thumb_tip.y > index_tip.y and index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return 'Z'
    elif thumb_tip.x > index_tip.x:
        return 'LEFT'
    elif thumb_tip.x < index_tip.x:
        return 'RIGHT'
    elif thumb_tip.y < index_tip.y and middle_tip.y < ring_tip.y and pinky_tip.y < ring_tip.y:
        return 'DOWN'
    elif thumb_tip.y > index_tip.y and middle_tip.y > ring_tip.y and pinky_tip.y > ring_tip.y:
        return 'X'
    elif index_tip.y < middle_tip.y < ring_tip.y and pinky_tip.y > ring_tip.y:
        return 'W'
    else:
        return 'RETURN'

key_mapping = {
    'UP': 'i',
    'DOWN': 'k',
    'LEFT': 'j',
    'RIGHT': 'l',
    'Z': 'n',
    'X': 'm',
    'W': 'b',
    'RETURN': Key.enter
}

def process_video_with_mediapipe(video_path):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=4,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    gesture_counter = Counter()

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 360))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture = detect_gesture(hand_landmarks)
                gesture_counter[gesture] += 1

                most_common_gestures = [gesture for gesture, _ in gesture_counter.most_common()]
                keys_to_press = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'Z', 'X', 'W', 'RETURN']

                for i, key in enumerate(keys_to_press):
                    if i < len(most_common_gestures):
                        mapped_key = key_mapping[most_common_gestures[i]]
                        print(f'Pressing {mapped_key} for gesture {most_common_gestures[i]}')
                        keyboard.press(mapped_key)
                        time.sleep(0.1)
                        keyboard.release(mapped_key)

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    youtube_url = 'https://www.youtube.com/watch?v=8lIgbwCzUCQ'
    video_path = download_youtube_video(youtube_url)
    process_video_with_mediapipe(video_path)

    os.remove(video_path)
