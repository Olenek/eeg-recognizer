import numpy as np
from torch import optim

from src.prediction_models import *
from torch.utils.data import DataLoader

from src.utils import NumpyArrayDataset


def to_classification(y_reg):
    np.dot(y_reg > 4.5, [1, 2])

def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        data, target = data.float(), target.float()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(dataloader)
    return train_loss


# Define the validation function
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.float(), target.float()
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
    val_loss = running_loss / len(dataloader)
    return val_loss


npzfile = np.load('dataset.npz')

t_y, v_y = npzfile['train_y'],  npzfile['val_y']


batch_size = 32
tdl = DataLoader(NumpyArrayDataset(npzfile['train_x'], npzfile['train_y']), batch_size=batch_size, shuffle=True)
vdl = DataLoader(NumpyArrayDataset(npzfile['val_x'], npzfile['val_y']), batch_size=batch_size, shuffle=False)

model = Net()
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Train the model
train_loss = []
val_loss = []
for epoch in range(100):
    epoch_train_loss = train(model, tdl, optimizer, criterion)
    epoch_val_loss = validate(model, vdl, criterion)
    print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.3f}, Val Loss: {epoch_val_loss:.3f}')
    train_loss.append(epoch_train_loss)
    val_loss.append(epoch_val_loss)
