import mediapipe as mp
import cv2
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '1' for INFO, '2' for WARNING, '3' for ERROR

def crop_hand(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        h, w, _ = image.shape
        hand = results.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in hand.landmark]
        y_coords = [lm.y for lm in hand.landmark]
        xmin, xmax = int(min(x_coords)*w), int(max(x_coords)*w)
        ymin, ymax = int(min(y_coords)*h), int(max(y_coords)*h)
        # add padding
        pad = 20
        xmin, xmax = max(0, xmin-pad), min(w, xmax+pad)
        ymin, ymax = max(0, ymin-pad), min(h, ymax+pad)
        cropped = image[ymin:ymax, xmin:xmax]
        return cropped

image = cv2.imread("test_image.jpg")
cropped = crop_hand(image)

if cropped is None:
    print("❌ No hand detected.")
    sys.exit(1)

# ✅ Resize and normalize the cropped hand image
cropped_resized = cv2.resize(cropped, (128, 128))  # match training image size
cropped_resized = cropped_resized / 255.0  # normalize to 0–1
cropped_input = np.expand_dims(cropped_resized, axis=0)  # shape: (1, 128, 128, 3)
cropped = crop_hand(image)

if cropped is None:
    print("❌ No hand detected.")


# ✅ Resize and normalize the cropped hand image
cropped_resized = cv2.resize(cropped, (128, 128))  # match training image size
cropped_resized = cropped_resized / 255.0  # normalize to 0–1
cropped_input = np.expand_dims(cropped_resized, axis=0)  # shape: (1, 128, 128, 3)

exit()



