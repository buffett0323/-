# Train the model for level 2 with data split: 60:20:20
import torch
from params import device


# Training without multiprocess
def training(data_loader, model, criterion, optimizer, l2_lambda=0.001):
    model.train()
    train_loss = 0.0
    for batch in data_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets.flatten())

        # Apply L2 regularization through weight decay & Adjust the L2 regularization strength
        if l2_lambda > 0:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss


# Training
def training_hybrid(data_loader, model, criterion, optimizer, l2_lambda=0.001):
    model.train()
    train_loss = 0.0
    for sequences, static_features, labels in data_loader:
        sequences, static_features, labels = sequences.to(device), static_features.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(sequences, static_features)
        loss = criterion(outputs, labels.flatten())
        
        # Apply L2 regularization through weight decay & Adjust the L2 regularization strength
        if l2_lambda > 0:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss
