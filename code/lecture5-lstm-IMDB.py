import random
from torchtext.legacy import data
from torchtext.legacy import datasets
# from torchtext.vocab import Vectors, GloVe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

SEED = 1234
#
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
TEXT = data.Field(fix_length=50) #sequential=True,fix_length=50 tokenize=tokenize, lower=True,include_lengths=True, batch_first=True, fix_length=50
# TEXT = data.Field(tokenize='spacy',tokenizer_language = 'en',
#                   include_lengths = True) #python -m spacy download en_core_web_sm
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(vars(train_data.examples[0]))
train_data, valid_data = train_data.split(random_state=random.seed(SEED),split_ratio=0.25)
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')
TEXT.build_vocab(train_data, max_size=100)
LABEL.build_vocab(train_data)
print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

BATCH_SIZE = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cpu'#

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 10
OUTPUT_DIM = len(TEXT.vocab)
b = next(iter(train_iterator))
print(b)
# ttext = torch.tensor(b.text, dtype=int)
# embedding = nn.Embedding(INPUT_DIM, EMBEDDING_DIM)
# eb = embedding(ttext)

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, BATCH_SIZE, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        dim = output.size()
        output = output.view(-1, output.shape[2])
        output1 = F.log_softmax(output, dim=1)

        if BATCH_SIZE == dim[1]:
            output1 = output1.view(-1, OUTPUT_DIM, BATCH_SIZE)
        else:
            output1 = output1.view(dim[1], OUTPUT_DIM, -1)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        return output1

model = RNN(INPUT_DIM, EMBEDDING_DIM,BATCH_SIZE,OUTPUT_DIM)

optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.NLLLoss()
model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion, BATCH_SIZE):
    epoch_loss = 0
    epoch_acc = 0
    epoch_label_count = 0
    loss = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text)

        dim = predictions.size()
        if dim[2] != BATCH_SIZE:
            BATCH_SIZE = dim[2]

        pad = torch.tensor([1] * BATCH_SIZE, device="cuda:0").view(BATCH_SIZE, -1)
        _, preds = torch.max(predictions, 1)
        labels = batch.text.view(-1, BATCH_SIZE)
        labels = labels[1:]
        pad = torch.tensor([1] * BATCH_SIZE, device=device).view(-1, BATCH_SIZE)
        labels = torch.cat((labels, pad), 0)
        loss = criterion(predictions, labels)
        acc = torch.sum(preds == labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_label_count += labels.numel()

    return epoch_loss / len(iterator), (epoch_acc / epoch_label_count)

N_EPOCHS = 10
train_losses = []
valid_losses = []
for epoch in range(N_EPOCHS):
    train_loss,train_acc = train(model, train_iterator, optimizer, criterion,BATCH_SIZE)
    train_losses.append(train_loss)
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% ')

plt.plot(train_losses)
plt.title("Training Performane by EPOCHS - 1000 words 25% training size")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()