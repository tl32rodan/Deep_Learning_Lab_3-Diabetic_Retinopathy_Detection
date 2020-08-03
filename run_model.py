import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
        num_epochs = 10, print_freq = 1):
    
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
                running_corrects += torch.sum(preds == y.data)
                
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            
            if epoch % print_freq == 0:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step(loss)

            if phase == 'train':
                acc_train_list.append(epoch_acc)
            else:
                acc_test_list.append(epoch_acc)

        loss_list.append(epoch_loss)
            
    return loss_list, acc_train_list, acc_test_list


