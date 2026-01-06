import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io, base64


from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# ------------------- FLASK SETUP -------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ------------------- MODEL LOADING -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model.to(device).eval()
    print("Model Loaded")
    return model

model = load_model()

# ------------------- WEBSOCKET HANDLER -------------------
@socketio.on("frame")
def handle_frame(data):
    try:
        # Remove base64 header → decode
        image_data = data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            prob = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(prob, 1)

        result = {
            "label": CLASS_NAMES[pred.item()],
            "confidence": float(conf.item())
        }

        emit("prediction", result)

    except Exception as e:
        emit("error", str(e))


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
