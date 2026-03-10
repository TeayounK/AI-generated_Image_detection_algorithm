# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
import random
from PIL import Image

# Imports for preprocessing and model components
from preprocess import preprocess
from preprocess.smash import smash
from preprocess.reconstruct import reconstruct
from model.filters import SRMFilters
from model.fingerprint import FingerprintExtractor
from model.cnn import PatchCraftCNN



def training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    
    """
    Training loop for PatchCraft: Loads batches, preprocesses (Smash, Reconstruct, Filters, Fingerprint),
    trains the CNN on features, with optional validation.
    
    Args:
        model: The PatchCraftCNN instance.
        train_loader: DataLoader with raw images.
        val_loader: Optional validation loader.
        epochs: Number of epochs.
        lr: Learning rate for Adam.
        device: 'cuda' or 'cpu'.

    """


    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    srm = SRMFilters().to(device).eval()
    extractor = FingerprintExtractor().to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()  # [B] for BCE

            features = preprocess(images, srm, extractor, device)

            # Forward pass
            preds = model(features)
            loss = criterion(preds, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += ((preds > 0).float() == labels).sum().item()
            train_total += images.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")

        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).float()

                    features = preprocess(images, srm, extractor, device)
                    preds = model(features)
                    loss = criterion(preds, labels)

                    val_loss += loss.item() * images.size(0)
                    val_correct += ((preds > 0).float() == labels).sum().item()
                    val_total += images.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            print(f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

    return model