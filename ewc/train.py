import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from data import PermutedMNIST
from models import Network
from utils import parse_args
from ewc import EWC

import random
import ipdb


def main(args):
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Transform, Dataset and Dataloaders
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_datasets, test_datasets = {}, {}
    train_loaders, test_loaders = {}, {}
    permute_idx = [i for i in range(28*28)]
    for i in range(args.num_tasks):
        train_datasets[i] = PermutedMNIST(transform=transform, train=True, permute_idx=permute_idx)
        train_loaders[i] = torch.utils.data.DataLoader(train_datasets[i], batch_size=args.batch_size, shuffle=True)
        test_datasets[i] = PermutedMNIST(transform=transform, train=False, permute_idx=permute_idx)
        test_loaders[i] = torch.utils.data.DataLoader(test_datasets[i], batch_size=args.batch_size, shuffle=True)
        random.shuffle(permute_idx)

    # Model, Optimizer, Criterion, ewc_class if needed
    model = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    ewc_class = None

    # Recoders
    train_losses, train_accs, test_losses, test_accs = {}, {}, {}, {}

    # Train Proper
    for i in range(args.num_tasks):
        print("Currently Training on Task {}".format(i + 1))
        curr_task = i
        # Initialize per task recorders
        train_losses[i], train_accs[i], test_losses[i], test_accs[i] = [], [], [], []
        NUM_EPOCHS = args.num_epochs_per_task
        for epoch in range(NUM_EPOCHS):
            # Train
            train_loss, train_acc = train(
                model, 
                optimizer, 
                criterion,
                train_loaders[i],
                device,
                args.ewc_train,
                ewc_class,
                args.ewc_weight,
                curr_task
            )
            print("[Train Epoch {:>5}/{}]  loss: {:>0.4f} | acc: {:>0.4f}".format(epoch, NUM_EPOCHS, train_loss, train_acc))
            # Record Loss
            train_losses[i].append(train_loss)    
            train_accs[i].append(train_acc)

            # Test
            for j in range(i+1):
                test_loss, test_acc = test(
                    model,
                    criterion,
                    test_loaders[j],
                    device
                )
                print("[ Test Epoch {:>5}/{}]  loss: {:>0.4f} | acc: {:>0.4f}  [Task {}]".format(epoch, NUM_EPOCHS, test_loss, test_acc, j))

                test_losses[j].append(test_loss)
                test_accs[j].append(test_acc) 

        if (args.ewc_train):
            # Consolidate
            ewc_class = EWC(model, train_loaders, curr_task, device) 

    # Save Losses and Accuracies
    suffixes = "{}_{}_{}_{}_{}".format(
        str(args.num_tasks),
        str(args.num_epochs_per_task),
        str(args.ewc_weight),
        "ewc" if args.ewc_train else "std", 
        args.custom_suffix)
    torch.save(train_losses, "train_losses_{}.txt".format(suffixes))
    torch.save(train_accs, "train_accs_{}.txt".format(suffixes))
    torch.save(test_losses, "test_losses_{}.txt".format(suffixes))
    torch.save(test_accs, "test_accs_{}.txt".format(suffixes))


def train(model, optimizer, criterion, loader, device, ewc_train, ewc_class, ewc_weight, curr_task):
    model.train()

    train_loss, corrects, num_samples = 0, 0, 0
    for i, (x,y) in enumerate(loader):
        batch_size = x.shape[0]
        x, y = x.view(batch_size, -1).to(device), y.to(device)

        # Forward Pass
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        # Add EWC Loss if needed
        ewc_loss = torch.tensor(0)
        if ((ewc_train) and (curr_task != 0)):
            ewc_loss = ewc_weight * ewc_class.ewc_loss(model)
            loss += ewc_loss

        loss.backward()
        optimizer.step()

        # Record Loss
        train_loss += batch_size * loss.item()
        num_samples += batch_size

        # Record Accuracy
        pred = out.max(1)[1].view(-1)
        corrects += torch.sum(pred == y) 

    return train_loss/num_samples, corrects/num_samples

def test(model, criterion, loader, device):
    model.eval()

    test_loss, corrects, num_samples = 0, 0, 0
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            batch_size = x.shape[0]
            x, y = x.view(batch_size, -1).to(device), y.to(device)

            # Forward Pass
            out = model(x)
            loss = criterion(out, y)

            # Record Loss
            # During we don't need to know the EWC loss
            test_loss += batch_size * loss.item()
            num_samples += batch_size

            # Record Accuracy
            pred = out.max(1)[1].view(-1)
            corrects += torch.sum(pred == y) 

    return test_loss/num_samples, corrects.item()/num_samples

if __name__ == '__main__':
    args = parse_args()
    main(args)

