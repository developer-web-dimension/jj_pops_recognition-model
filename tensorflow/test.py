# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model

# model = load_model("jimjam_classifier.h5")

# def predict_jimjam(img_path):
#     img = image.load_img(img_path, target_size=(224,224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     preds = model.predict(x)[0]
#     classes = ["jim jam pops", "jim jam", "not jim jam"]

#     print("Prediction:", classes[np.argmax(preds)], "| Confidence:", np.max(preds))

# predict_jimjam("assest/nothing/6.jpg")


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("jimjam_classifier.h5")

# Class names
classes = ["jim jam pops", "jim jam", "nothing"]

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Start webcam
cap = cv2.VideoCapture(1)

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected!")
        break

    # Preprocess frame
    x = preprocess_frame(frame)

    # Predict
    preds = model.predict(x, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]
    label = f"{classes[class_id]} ({confidence:.2f})"

    # Show label on screen
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("JimJam Live Classifier", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
