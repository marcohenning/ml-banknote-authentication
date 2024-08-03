import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
from model import Model


torch.manual_seed(1337)
model = Model()

dataset = 'https://raw.githubusercontent.com/Kuntal-G/Machine-Learning/master/R-machine-learning/data/banknote-authentication.csv'
df = pd.read_csv(dataset)

X = df.drop('class', axis=1)
y = df['class']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
losses = []
losses_table = PrettyTable()
losses_table.field_names = ['Epoch', 'Loss']

for i in range(epochs):
    y_prediction = model.forward(X_train)
    loss = criterion(y_prediction, y_train)
    losses.append(loss.detach().numpy())
    if i == 0 or (i + 1) % 10 == 0:
        losses_table.add_row([i + 1, loss.detach().numpy()])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(losses_table)

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_prediction = model.forward(data)
        if y_prediction.argmax().item() == y_test[i]:
            correct += 1

print(f'Test Accuracy: {correct / len(X_test) * 100}% ({correct}/{len(X_test)})')

plt.get_current_fig_manager().set_window_title('Training')
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
