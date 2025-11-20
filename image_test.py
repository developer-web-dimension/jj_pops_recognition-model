import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

CLASS_NAMES = ['Nothing', 'jimjam', 'jimjam_pops']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(weights_path="jimjam_classifier.pth"):
    model = models.mobilenet_v2(weights=None)  # no pretrained weights
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 3)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

model = load_model()

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)

    class_name = CLASS_NAMES[pred.item()]
    probability = torch.softmax(outputs, dim=1)[0][pred].item()

    print(f"\nImage: {image_path}")
    print(f"Prediction: {class_name}  ({probability:.4f})")

    return class_name, probability

if __name__ == "__main__":
    test_path = r"assest/jimjam/3456.jpg"   
    predict_image(test_path)