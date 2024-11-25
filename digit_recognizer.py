import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import torch.optim
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
# Custom Dataset
class CustomMNISTDataset(Dataset):
    def __init__(self, name, transform=ToTensor(), label_name="label", is_labeled=True):
        self.data = pd.read_csv(name)  
        self.transform = transform
        self.label_name = label_name
        self.is_labeled = is_labeled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixel_data = self.data.iloc[idx].drop(self.label_name, errors='ignore').values.astype('float32')
        scaled_pixel = (pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min())
        image = scaled_pixel.reshape(28, 28)
        if self.transform:
            image = self.transform(image)

        if self.is_labeled:
            label = int(self.data.iloc[idx][self.label_name])
            return image, label
        return image

# Load Datasets
train_dataset = CustomMNISTDataset(name="train.csv", is_labeled=True)
test_dataset = CustomMNISTDataset(name="test.csv", is_labeled=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CNN Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x

# Training
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        if len(images.shape) != 4:
            images = images.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Validation
model.eval()
y_pred = []
y_true = []
correct = 0
if test_dataset.is_labeled:

    with torch.no_grad():
        for images, labels in test_loader:
            if len(images.shape) != 4:
                images = images.unsqueeze(1)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy())
        accuracy = accuracy_score(y_true,y_pred)
        F1_score = f1_score(y_true,y_pred,average="weighted")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"F1-Score: {F1_score:.2f}%")
else:
    with torch.no_grad():
        for images in test_loader:
            if len(images.shape) != 4:
                images = images.unsqueeze(1)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            y_pred.extend(predictions.cpu().numpy().tolist())
    #simulations if labels were provided:
    nlabels = np.random.choice(10,len(y_pred)) 
    accuracy = accuracy_score(nlabels,y_pred)
    F1_score = f1_score(nlabels,y_pred,average="weighted")
    print(f"Accuracy if no labels: {accuracy:.2f}%")
    print(f"F1-Score if no labels: {F1_score:.2f}%")


print(f"Predictions (first 10): {y_pred[:10]}")
