import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os

# Определите устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры
BATCH_SIZE = 6
label_list = ['notsmoking', 'smoking']
class_size = len(label_list)

# Определение Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms_):
        self.df = dataframe
        self.transforms_ = transforms_
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['path']
        img = Image.open(img_path).convert("RGB")
        transformed_img = self.transforms_(img)
        class_id = self.df.iloc[index]['class_id']
        return transformed_img, class_id

def load_test_data(test_path):
    test_img_list = glob(test_path + '\\*.jpg')
    test_df = pd.DataFrame({"path": [], "label": [], "class_id": []})

    for img in test_img_list:
        file_name = os.path.splitext(img)[0].split("\\")[-1]
        if file_name.startswith(label_list[0]):
            new_data = pd.DataFrame({"path": img, "label": label_list[0], "class_id": 0}, index=[0])
            test_df = pd.concat([test_df, new_data], ignore_index=True)
        elif file_name.startswith(label_list[1]):
            new_data = pd.DataFrame({"path": img, "label": label_list[1], "class_id": 1}, index=[0])
            test_df = pd.concat([test_df, new_data], ignore_index=True)

    return test_df

def evaluate_test_set(dataloader, model):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data_, target_ in dataloader:
            data_ = data_.to(device)
            outputs = model(data_)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target_.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    classification_rep = classification_report(all_labels, all_preds, target_names=label_list)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, classification_rep, conf_matrix

def main():
    test_path = 'data\\Testing'
    test_df = load_test_data(test_path)

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = MyDataset(test_df, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = models.swin_v2_b(weights=None)
    model.head = nn.Linear(in_features=model.head.in_features, out_features=class_size)
    model.load_state_dict(torch.load('checkpoints/best.pth'))
    model.to(device)
    model.eval()

    test_accuracy, test_classification_report, test_confusion_matrix = evaluate_test_set(test_loader, model)

    print(f'Точность на тестовом наборе: {test_accuracy:.4f}')
    print('Отчет классификации:')
    print(test_classification_report)
    print('Матрица неточностей:')
    print(test_confusion_matrix)

    sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.title('Матрица неточностей на тестовом наборе')
    plt.show()

if __name__ == "__main__":
    main()
