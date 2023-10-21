import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import pandas as pd
from iou import iou_metric

def train_with_iou(dataloader_train, dataloader_validation, model, n_epochs, optimizer_name = "sgd", loss_fn=nn.MSELoss(), device=torch.device('cpu'), lr=0.001, print_values=True):
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")
    

    values = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Validation Loss', 'Train IoU', 'Validation IoU'])
    weight_information = {}
    model.train()
    for epoch in range(n_epochs):
        
        train_loss = 0.0
        train_iou = 0.0
        for data, target in dataloader_train:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()   
            iou_train_batch = iou_metric(output, target).item() 
            train_iou += iou_train_batch
            
            
        train_loss /= len(dataloader_train)
        train_iou /= len(dataloader_train)

        #model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for data_val, target_val in dataloader_validation:
                output_val = model(data_val)
                loss_val = loss_fn(output_val, target_val)
                val_loss += loss_val.item()
                iou_val_batch = iou_metric(output_val, target_val).item() 
                val_iou += iou_val_batch
                
                
        val_iou /= len(dataloader_validation)   
        val_loss /= len(dataloader_validation)
                
        
        # Save different weights 
        weight_information[str("Epoch Number").replace("Number", str(epoch+1))] = model.state_dict()       
        
        # Save the values
        values = pd.concat([values, pd.DataFrame({'Epoch': epoch+1, 'Train Loss': train_loss, 'Validation Loss': val_loss, 'Train IoU': train_iou, 'Validation IoU': val_iou}, index=[0])], ignore_index=True)
        
        if print_values:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Validation Loss: {val_loss:.5f}, Train IoU: {train_iou:.5f}, Validation IoU: {val_iou:.5f}")
        
    return values, weight_information