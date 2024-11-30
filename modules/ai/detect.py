import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Определите устройство
device = torch.device('cpu')

# Загрузите модель
model = models.swin_v2_b(weights='DEFAULT')
model.head = nn.Linear(in_features=model.head.in_features, out_features=2)  # Предполагается, что у вас два класса
model.load_state_dict(torch.load('checkpoints/best.pth'))
model.to(device)
model.eval()

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

    return predicted.item()

# Путь к вашему изображению
image_path1 = 'test_images/image1.png'
image_path2 = 'test_images/image2.png'
image_path3 = 'test_images/image3.png'
image_path4 = 'test_images/image4.png'

# Сделать предсказание
predicted_class_id1 = predict_image(image_path1, model, predict_transforms, device)
predicted_class_id2 = predict_image(image_path2, model, predict_transforms, device)
predicted_class_id3 = predict_image(image_path3, model, predict_transforms, device)
predicted_class_id4 = predict_image(image_path4, model, predict_transforms, device)

# Определить метку класса
label_list = ['notsmoking', 'smoking']  # Замените на свои метки, если они отличаются
predicted_label1 = label_list[predicted_class_id1]
predicted_label2 = label_list[predicted_class_id2]
predicted_label3 = label_list[predicted_class_id3]
predicted_label4 = label_list[predicted_class_id4]

print(f'Предсказанный класс для изображения 1: {predicted_label1}')
print(f'Предсказанный класс для изображения 2: {predicted_label2}')
print(f'Предсказанный класс для изображения 3: {predicted_label3}')
print(f'Предсказанный класс для изображения 4: {predicted_label4}')

