import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

import sys
sys.path.append('..')
from dataset import create_data_loaders
from transforms import get_val_transforms
from model import create_model

# Test settings
IMG_SIZE = 224
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model,test,num_classes):
    """
     Evaluate test performance
    :param model:
    :param test:
    :param num_classes:
    :return:
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images,labels in tqdm(test,desc="Evaluating..."):
            images,labels = images.to(DEVICE),labels.to(DEVICE)
            outputs = model(images)
            _,predictions = torch.max(outputs,1)

            all_predictions.extend(predictions.cpu().numpy()) # Ensure predictions are on CPU before converting to a NumPy array (NumPy does not support GPU tensors)
            all_labels.extend(labels.cpu().numpy())
    # Evaluate metrics
    accuracy = accuracy_score(all_labels,all_predictions)
    precision = precision_score(all_labels,all_predictions,average='macro',zero_division=0)
    recall = recall_score(all_labels,all_predictions,average='macro',zero_division=0)
    f1 = f1_score(all_labels,all_predictions,average='macro',zero_division=0)
    cm = confusion_matrix(all_labels,all_predictions,labels=np.arange(num_classes))

    return accuracy,precision,recall,f1,cm

def run_evaluation(model_name="mobilenet_v2",model_path="../../models/mobilenet_v2_best_model.pt"):

    #DataLoader
    data_path = "../../data/processed/01_final_dataset"
    _,_,test_loader = create_data_loaders(
        data_path=data_path,
        train_transform=None,
        val_transform=get_val_transforms(IMG_SIZE),
        batch_size=BATCH_SIZE,
        num_workers=4,
        class_weights=False
    )

    # Number of Classes
    with open('../../data/processed/01_metadata/class_names.json','r') as f:
        class_to_idx = json.load(f)
        NUM_CLASSES = len(class_to_idx)

    # Create model and load weights
    model = create_model(num_classes=NUM_CLASSES,model_name=model_name,fine_tune=True)
    model.load_state_dict(torch.load(model_path,map_location=DEVICE)) # map_location ensures the model weights are loaded onto the specified device
    model.to(DEVICE)

    # Evaluate the model
    accuracy, precision, recall, f1, cm = evaluate_model(model, test_loader, NUM_CLASSES)

    print("\n--- RESULTs ---")
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")

    # Confusion Matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save graphic
    cm_path = Path(f"../../outputs/reports/{model_name}_confusion_matrix.png")
    cm_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(cm_path)
    print(f"\nConfusion Matrix saved to {cm_path}")

if __name__ == "__main__":
    run_evaluation()