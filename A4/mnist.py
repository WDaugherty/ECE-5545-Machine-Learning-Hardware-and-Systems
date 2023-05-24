from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from src.matmul import svd
import matplotlib.pyplot as plt
import time
import numpy as np 
import copy



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_fc.pt")

    compression_ratios = []
    accuracies = []
    runtimes = []

    for rank_fc1 in range(model.fc1.weight.size(1) // 75, model.fc1.weight.size(1) // 2, 30):
        for rank_fc2 in range(model.fc2.weight.size(1) // 75, model.fc2.weight.size(1) // 2, 30):
            model_copy = copy.deepcopy(model)
            compress_model(model_copy, rank_fc1, rank_fc2)
            start_time = time.time()
            correct = test(model_copy, device, test_loader)
            end_time = time.time()
            runtime = end_time - start_time
            runtimes.append(runtime)
            
            compression_ratio = (model.fc1.weight.numel() + model.fc2.weight.numel()) / (rank_fc1 * (model.fc1.weight.size(1) + model.fc1.weight.size(0)) + rank_fc2 * (model.fc2.weight.size(1) + model.fc2.weight.size(0)))
            compression_ratios.append(compression_ratio)
            
            accuracy = 100. * correct / len(test_loader.dataset)
            accuracies.append(accuracy)
            print(f"Compression ratio: {compression_ratio}, Runtime: {runtime}, Accuracy: {accuracy}")

    return compression_ratios, accuracies, runtimes

def compress_model(model, rank_fc1=None, rank_fc2=None):
    if rank_fc1 is not None:
        model.fc1.weight.data = svd(model.fc1.weight.data, torch.eye(model.fc1.weight.data.size(1)), rank_fc1)
    
    if rank_fc2 is not None:
        model.fc2.weight.data = svd(model.fc2.weight.data, torch.eye(model.fc2.weight.data.size(1)), rank_fc2)
    
    return model

def evaluate_compressed_model(model, device, test_loader):
    start_time = time.time()
    correct = test(model, device, test_loader)
    end_time = time.time()
    runtime = end_time - start_time
    return correct, runtime

def test_compression(model, device, test_loader, ranks_fc1, ranks_fc2):
    compression_ratios = []
    accuracies = []
    runtimes = []
    
    for rank_fc1 in ranks_fc1:
        for rank_fc2 in ranks_fc2:
            model_copy = compress_model(copy.deepcopy(model), rank_fc1, rank_fc2)
            
            correct, runtime = evaluate_compressed_model(model_copy, device, test_loader)
            
            compression_ratio = (model.fc1.weight.numel() + model.fc2.weight.numel()) / (rank_fc1 * (model.fc1.weight.size(1) + model.fc1.weight.size(0)) + rank_fc2 * (model.fc2.weight.size(1) + model.fc2.weight.size(0)))
            compression_ratios.append(compression_ratio)
            
            accuracy = 100. * correct / len(test_loader.dataset)
            accuracies.append(accuracy)
            runtimes.append(runtime)
            print(f"Compression ratio: {compression_ratio}, Runtime: {runtime}, Accuracy: {accuracy}")
    
    return compression_ratios, accuracies, runtimes



if __name__ == '__main__':
    compression_ratios, accuracies, runtimes = main()
    np.savez("arr.npz", compression_ratios=compression_ratios, accuracies=accuracies, runtimes=runtimes)
 
