import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
CLASS_NAMES = ['Nothing', 'jimjam', 'jimjam_pops']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(path="jimjam_classifier.pth"):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 3)
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded")
    return model

model = load_model()

def predict_frame(frame):
    img = Image.fromarray(frame).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][pred].item()

    return CLASS_NAMES[pred.item()], prob
def live_video():
    cap = cv2.VideoCapture(1)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prediction
        label, prob = predict_frame(frame)

        # Draw result
        text = f"{label}: {prob:.2f}"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # Show video
        cv2.imshow("Live Classification", frame)

        # Quit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_video()