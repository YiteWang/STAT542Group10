import time
import torch

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=20, scheduler=None):
    model.train()
    total = 0
    correct1 = 0
    correct5 = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        _, pred = output.topk(1, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct1 += correct[:,:1].sum().item()
        # correct5 += correct[:,:5].sum().item()
        train_loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    # accuracy5 = 100. * correct5 / len(dataloader.dataset)
    end_time = time.time()
    print('Epoch: {}, Train: Top 1 Accuracy: {}/{} ({:.2f}%), Time Used: {} mins'.format(
             epoch, correct1, len(dataloader.dataset), accuracy1, (end_time-start_time)/60.0))
    return total / len(dataloader.dataset)

def model_eval(model, loss, dataloader, device, name='valid'):
    model.eval()
    total = 0
    correct1 = 0
    # correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(1, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            # correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    # accuracy5 = 100. * correct5 / len(dataloader.dataset)
    # if verbose:
    print('{} Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            name, average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1