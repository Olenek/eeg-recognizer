import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset

from src.utils import RESOLUTION, SEQ_LEN, IMG_CHANNELS


class MyDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        if self.labels is not None:
            y = self.labels[index]
            return x, y
        else:
            return x


def train_epoch(model, dataloader, criterion, optimizer, device='cuda'):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        data, target = data.float().to(device), target.float().to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(dataloader)
    return train_loss


def validate(model, dataloader, device='cuda'):
    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.float().to(device), target.float().to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
    val_loss = running_loss / len(dataloader)
    return val_loss


def predict(model, dataloader):
    model.eval()
    prediction_list = []
    true_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            output = torch.clamp(model(data.to('cuda')), min=-1, max=1)
            prediction_list.append(output.cpu())
            true_list.append(target.cpu())

    return torch.concat(true_list, dim=0), torch.concatenate(prediction_list, dim=0)


def evaluate(true, preds):
    size = len(true)
    mse = np.sum(np.square((true - preds).numpy()), axis=0) / size

    hits = (torch.where(torch.logical_and(true > 0, preds > 0), 1, 0) + torch.where(
        torch.logical_and(true < 0, preds < 0), 1, 0)).numpy()
    val_acc, ars_acc = np.sum(hits, axis=0) / size

    return mse, val_acc, ars_acc


from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader


def train_subject(idx, X, y, model):
    _x, _y = X[idx].reshape((-1, SEQ_LEN, IMG_CHANNELS, RESOLUTION, RESOLUTION)), y[idx].reshape((-1, y.shape[-1]))
    _xt, _xv, _yt, _yv = train_test_split(_x, _y, train_size=0.9, shuffle=False)
    train_dl = DataLoader(MyDataset(_xt, _yt), batch_size=4, shuffle=True)
    val_dl = DataLoader(MyDataset(_xv, _yv), batch_size=4, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    best_model_state = model.state_dict()
    prev_loss = 0
    iter_no_change = 0
    model.to('cuda')
    for epoch in range(100):
        running_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_dl):
            optimizer.zero_grad()
            data, target = data.float().to('cuda'), target.float().to('cuda')
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(train_dl)

        epoch_val_loss = validate(model, val_dl)
        if epoch_val_loss < prev_loss or epoch == 0:
            prev_loss = epoch_val_loss
            iter_no_change = 0
            best_model_state = model.state_dict()
        else:
            iter_no_change += 1

        if iter_no_change > 10:
            model.load_state_dict(best_model_state)
            break

    return model, epoch_val_loss


def cv_subject(idx, X, y, model, folds=40):
    _x, _y = X[idx], y[idx]
    kf = KFold(n_splits=folds)
    сv_loss = []
    cv_accuracies = []
    model.to('cuda')

    model.train()
    for i, (train_index, test_index) in enumerate(kf.split(_x)):
        prev_loss = 0
        iter_no_change = 0
        best_model_state = model.state_dict()

        train_dl = DataLoader(MyDataset(_x[train_index], _y[train_index]), batch_size=1, shuffle=True)
        val_dl = DataLoader(MyDataset(_x[test_index], _y[test_index]), batch_size=1, shuffle=False)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.01)

        for epoch in range(1000):
            model.train()
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_dl):
                optimizer.zero_grad()
                data, target = data.float().to('cuda'), target.float().to('cuda')
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_train_loss = running_loss / len(train_dl)

            true, preds = predict(model, val_dl)
            epoch_val_loss, val_acc, ars_acc = evaluate(true, preds)

            if np.mean(epoch_val_loss) < prev_loss or epoch == 0:
                prev_loss = np.mean(epoch_val_loss)
                iter_no_change = 0
                best_model_state = model.state_dict()
            else:
                iter_no_change += 1

            if iter_no_change > 5:
                model.load_state_dict(best_model_state)
                break

        сv_loss.append(epoch_val_loss)
        cv_accuracies.append([val_acc, ars_acc])


    return сv_loss, cv_accuracies


def finetune_evaluate(idx, X, y, model, trial_num):
    tx, ty = X[:, :trial_num], y[:, :trial_num]
    vx, vy = X[:, 20:], y[:, 20:]

    tx, ty = tx[idx].reshape((-1, SEQ_LEN, IMG_CHANNELS, RESOLUTION, RESOLUTION)), ty[idx].reshape((-1, y.shape[-1]))
    vx, vy = vx[idx].reshape((-1, SEQ_LEN, IMG_CHANNELS, RESOLUTION, RESOLUTION)), vy[idx].reshape((-1, y.shape[-1]))

    val_loss = []
    val_accuracies = []
    model.train()

    iter_no_change = 0
    best_model_state = model.state_dict()
    best_loss = 0
    best_metrics = []

    train_dl = DataLoader(MyDataset(tx, ty), batch_size=4, shuffle=True)
    val_dl = DataLoader(MyDataset(vx, vy), batch_size=4, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    model.to('cuda')

    for epoch in range(10):
        model.train()
        epoch_train_loss = train_epoch(model, train_dl, optimizer, criterion)

        epoch_val_loss, val_acc, ars_acc = evaluate(*predict(model, val_dl))

        if np.mean(epoch_val_loss) < best_loss or epoch == 0:
            best_loss = np.mean(epoch_val_loss)
            best_metrics = epoch_val_loss, val_acc, ars_acc
            iter_no_change = 0
            best_model_state = model.state_dict()
        else:
            iter_no_change += 1

        if iter_no_change > 5:
            model.load_state_dict(best_model_state)
            break

    return best_metrics
