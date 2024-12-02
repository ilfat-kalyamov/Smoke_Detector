import torch
from torchvision import transforms
from PIL import Image
from modules.ai.config import load_model

# Определите устройство
device, model = load_model()
label_list = ['Не курит', 'Курит']

predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model, transforms, device):
    # Открыть и преобразовать изображение
    image = Image.open(image_path).convert('RGB')
    image = transforms(image).unsqueeze(0)  # Добавить батч размерности

    # Переместить изображение на устройство
    image = image.to(device)

    # Сделать предсказание
    with torch.no_grad():
        outputs = model(image)
        print(outputs)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
    
    predicted_label = label_list[predicted.item()]

    return predicted_label

