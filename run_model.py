import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

def draw_figure(plt,lines,labels=None, loc='best'):    
    if labels is None:
        for line in lines:
            plt.plot(range(len(line)), line)
    else:
        for line, label in zip(lines,labels):
            plt.plot(range(len(line)), line, label = label)
        
        plt.legend(loc='best')
    plt.show()

def run(model, dataloaders, criterion = nn.CrossEntropyLoss(),\
        optimizer = None, scheduler = None,\
        num_epochs = 10, print_freq = 1, model_path = None, best_acc = 0.):
    
    # Record list
    loss_list = []
    acc_train_list = []
    acc_test_list = []
    
    # Move data do gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Setup optimizer
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(),lr=1e-3, weight_decay=5e-4, momentum=0.9)
    
    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            ################
            pred_list = []
            ################
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for x, y in dataloaders[phase]:
                x = x.to(device)
                y = y.to(device)
        
                optimizer.zero_grad()
            
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Run the net & Update
                    outputs = model(x)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, y)
                
                # Run backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += preds.eq(y.data).cpu().sum().item()
                pred_list = pred_list + preds.tolist() 
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            
            if epoch % print_freq == 0:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                print('{} pred_list.value_counts() = \n{}'.format(phase, pd.Series(pred_list).value_counts()))
            
            # Step scheduler
            if phase == 'train' and scheduler is not None:
                scheduler.step(epoch_loss)

            if phase == 'train':
                acc_train_list.append(epoch_acc)
            else:
                acc_test_list.append(epoch_acc)
                # Save the best model
                if epoch_acc > best_acc and model_path is not None:         
                    print('epoch acc = ',epoch_acc,', best_acc = ',best_acc)
                    best_acc = epoch_acc

                    print('Store model : ', model_path)
                    torch.save(model, model_path)

        loss_list.append(epoch_loss)
            
    return loss_list, acc_train_list, acc_test_list, best_acc


