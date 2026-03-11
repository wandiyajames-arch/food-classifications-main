import torch
import sys
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from model_baseline import simple_cnn
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate import evaluate
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "train")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# dataset
dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

num_classes = len(dataset.classes)

# split train / validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# model
model = simple_cnn(num_classes=num_classes).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0

for epoch in range(10):

    # TRAIN
    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs,1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = train_correct / train_total

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

    # save best model
    # Remplacer la section de sauvegarde par ceci :
    if val_acc > best_acc:
        best_acc = val_acc
        save_dir = os.path.join(BASE_DIR, "models") # Utilise le chemin absolu défini en haut
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, "baseline_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Modèle sauvegardé : {save_path}")

print("Training finished")
print("Best validation accuracy:", best_acc)