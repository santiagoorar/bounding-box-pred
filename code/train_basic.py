import torch
import torch.nn as nn



def simple_train(dataloader, model, n_epochs, optimizer=None, loss_fn=nn.MSELoss(), device=torch.device('cpu')):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), 0.001)
    final_l = 50  # one high number to start
    model.train()
    for epoch in range(n_epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if loss.item() < final_l:  # the current model is better
                best_model_information = model.state_dict()           
    
        print('Epoch {}, loss: {}'.format(epoch+1, loss.item()))

    if best_model_information is not None:
        model.load_state_dict(best_model_information)
    