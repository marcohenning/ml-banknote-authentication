import torch
import pandas as pd
from sklearn.model_selection import train_test_split
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
