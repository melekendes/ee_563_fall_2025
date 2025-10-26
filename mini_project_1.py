import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time, copy, os
import matplotlib.pyplot as plt
import pandas as pd
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "hymenoptera_data","hymenoptera_data")

print("Using DATA_DIR:", DATA_DIR)
print("Contents of DATA_DIR:", os.listdir(DATA_DIR)) #for control 

BATCH_SIZE = 8 #every step we use 8 image
EPOCHS = 3   #all data 3 times try to model
LR_VALUES = [0.0001, 0.001, 0.01]
MOMENTUM_VALUES = [0.5, 0.7, 0.9]  
BACKBONES = ["resnet18", "resnet34", "resnet50"]
OPTIMIZERS = ["SGD", "Adam"]
LOSSES = ["CrossEntropy", "NLLLoss"]
DEVICES = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
DEVICES = ["cuda"]

#loading data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']} #for classification

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)
               for x in ['train', 'val']} #for batches

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} #number of train and val sets

class_names = image_datasets['train'].classes #ant and bees

#train function
def train_model(model, criterion, optimizer, num_epochs, device):
    since = time.time() #for calculate total time
    best_acc = 0.0 
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    #we use dict as history for save the lsot and accuracy data
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} on {device}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # forward pass
    
                    if isinstance(criterion, nn.NLLLoss):
                        outputs = torch.log_softmax(outputs, dim=1)
    
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase] #loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase] #accuracy
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {history['val_loss'][-1]:.4f}, "
              f"Val Acc: {best_acc:.4f}")

    time_elapsed = time.time() - since
    model.load_state_dict(best_model_wts)
    return model, history, time_elapsed

#define model, use resnets models
def get_model(backbone, num_classes, device):
    model = getattr(models, backbone)(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

results = []

for device in DEVICES:
    if device == "cuda" and not torch.cuda.is_available():
        continue
    for backbone in BACKBONES:
        for lr in LR_VALUES:
            for mom in MOMENTUM_VALUES:
                for opt_type in OPTIMIZERS:
                    for loss_type in LOSSES:
                        print(f"\n=== {backbone} | lr={lr} | mom={mom} | {opt_type} | {loss_type} | {device} ===")
                        model = get_model(backbone, len(class_names), device)

                        if loss_type == "CrossEntropy":
                            criterion = nn.CrossEntropyLoss()
                        else:
                            criterion = nn.NLLLoss()

                        if opt_type == "SGD":
                            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
                        else:
                            optimizer = optim.Adam(model.parameters(), lr=lr)

                        trained_model, hist, t_elapsed = train_model(model, criterion, optimizer, EPOCHS, device)
                        best_acc = max(hist["val_acc"])

                        results.append({
                            "Backbone": backbone,
                            "LR": lr,
                            "Momentum": mom,
                            "Optimizer": opt_type,
                            "Loss": loss_type,
                            "Device": device,
                            "Accuracy": best_acc,
                            "Time(s)": t_elapsed
                        })

                        #plot every graffic
                        plt.figure()
                        plt.plot(hist["train_acc"], label="Train Acc")
                        plt.plot(hist["val_acc"], label="Val Acc")
                        plt.plot(hist["train_loss"], label="Train Loss")
                        plt.plot(hist["val_loss"], label="Val Loss")
                        plt.legend()
                        plt.title(f"{backbone} - lr={lr}, mom={mom}, {device}")
                        plt.xlabel("Epoch")
                        plt.ylabel("Value")
                        plt.savefig(f"plot_{backbone}_{lr}_{mom}_{opt_type}_{loss_type}_{device}.png")
                        plt.close()

# save results
df = pd.DataFrame(results)
df.to_csv("results_summary.csv", index=False) 
#df.to_csv("results_gpu_summary.csv", index=False)
#df.to_csv("results_cpu_summary.csv", index=False)
print("\nAll experiments done. Results saved to results_summary.csv")

# random 4 images func.
def visualize_predictions(model, device, num_images=4):
    model.eval()
    inputs, classes = next(iter(dataloaders['val']))
    inputs, classes = inputs.to(device), classes.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(10, 8))
    for i in range(num_images):
        idx = random.randint(0, inputs.size(0)-1)
        ax = plt.subplot(2, 2, i+1)
        ax.axis('off')
        ax.set_title(f"Pred: {class_names[preds[idx]]} | True: {class_names[classes[idx]]}")
        img = inputs[idx].cpu().permute(1, 2, 0)
        plt.imshow(img)
    plt.savefig("sample_predictions.png")
    plt.close()
    print("Sample predictions saved to sample_predictions.png")


visualize_predictions(trained_model, device)
