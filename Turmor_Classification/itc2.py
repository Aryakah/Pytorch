import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2 
import torch.nn.functional as F
from tqdm import tqdm
import PyQt5



class mCNN (nn.Module):
    def __init__(self):
        super(mCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64) 
        self.fc2 = nn.Linear(64, 1) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  
        return x

#

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")




model = mCNN().to(device)

criterion = nn.BCELoss()  
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)



tumor = []
path_t = "Turmor_Classification/Data/archive/yes/Y*"
for f in glob.iglob(path_t):
    img = cv2.imread(f)
    img = cv2.resize(img,(128,128))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tumor.append(img)

path_h = "Turmor_Classification/Data/archive/no/*"
healthy = []
for f in glob.iglob(path_h):
    img = cv2.imread(f)
    img = cv2.resize(img,(128,128))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    healthy.append(img)

healthy = np.array(healthy)
tumor = np.array (tumor)

ALL = np.concatenate ((healthy,tumor))

def plot_random(h,t,num = 5):
    h_image = healthy[np.random.choice(healthy.shape[0],5,False)]
    t_image = tumor[np.random.choice(tumor.shape[0],5,False)]
    

    plt.figure (figsize=(16,9))
    for i in range (num):
        plt.subplot(3,num,i+1)
        plt.title('healthy')
        plt.imshow(h_image[i])

    plt.figure (figsize=(16,9))
    for i in range (num):
        plt.subplot(3,num,i+1)
        plt.title('tumor')
        plt.imshow(t_image[i])

    plt.show()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),         

])

root = 'Turmor_Classification/Data/archive'
mdataset = datasets.ImageFolder(root, transform=transform)


train_size = int(0.8 * len(mdataset))  
test_size = len(mdataset) - train_size 

train_dataset, test_dataset = random_split(mdataset, [train_size, test_size])


batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)



num_epochs = 10

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()  
            
            
            outputs = model(images)
            
            loss = criterion(outputs.squeeze(), labels)  
            
        
            optimizer.zero_grad()
           
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))  
            pbar.update(1)  
            
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
    
 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            predicted = (outputs.squeeze() > 0.5).float()  # Threshold at 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")



