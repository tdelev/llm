import torch
import matplotlib.pyplot as pl
import torch.nn.functional as F
import random

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i + 1:s for i,s in enumerate(chars)}
itos[0] = '.'
BLOCK_SIZE = 3 # context legnth: how many characters we take to predict the next one?
VOCAB_SIZE = 27 # the vocabulary size
HIDDEN_LAYER_SIZE = 200 # the number of neurons in the hidden layer
DIMENSIONS = 10 # the dimensionality of the character embedding vectors
CD = DIMENSIONS * BLOCK_SIZE

def build_dataset(words):
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * BLOCK_SIZE
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '---->', itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
C = torch.randn((VOCAB_SIZE, DIMENSIONS))
W1 = torch.randn((CD, HIDDEN_LAYER_SIZE)) * (5/3)/(CD ** 0.5)
b1 = torch.randn(HIDDEN_LAYER_SIZE) * 0.01
W2 = torch.randn((HIDDEN_LAYER_SIZE, VOCAB_SIZE)) * 0.01
b2 = torch.randn(VOCAB_SIZE) * 0
bngain = torch.ones((1, HIDDEN_LAYER_SIZE))
bnbias = torch.zeros((1, HIDDEN_LAYER_SIZE))
bnmean_running = torch.zeros((1, HIDDEN_LAYER_SIZE))
bnstd_running = torch.ones((1, HIDDEN_LAYER_SIZE))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
total_params = sum(p.nelement() for p in parameters)
print('Total parameters: ', total_params)
for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
lri = []
lossi = []
STEPS = 200_000
BATCH_SIZE = 32

# with torch.no_grad():
#     emb = C[Xtr]
#     embcat = emb.view(emb.shape[0], -1)
#     hlpreact = embcat @ W1 + b1
#     bnmean = hlpreact.mean(0, keepdim=True)
#     bnstd = hlpreact.std(0, keepdim=True)


@torch.no_grad()
def eval(X, Y):
    emb = C[X]
    embcat = emb.view(-1, CD) # concatenate the vectors
    hlpreact = embcat @ W1 + b1  # hidden layer pre-activation
    # measure the mean/std over the entire training set
    hlpreact = bngain * (hlpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hlpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    return loss

# def train(X, Y):
for i in range(STEPS):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))

    # forward pass
    emb = C[Xtr[ix]] # embed the characters into vectors
    embcat = emb.view(-1, CD) # concatenate the vectors
    hlpreact = embcat @ W1 + b1  # hidden layer pre-activation
    bnmeani = hlpreact.mean(0, keepdim=True)
    bnstdi = hlpreact.std(0, keepdim=True)
    hlpreact = bngain * (hlpreact - bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    
    h = torch.tanh(hlpreact) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Ytr[ix]) # loss function
    # print('Loss: ', loss.item())
    lossi.append(loss.log10().item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    # lr = lrs[i]
    lr = 0.1 if i < STEPS * .5 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10_000 == 0:
        print(f'{i:7d}/{STEPS:7d}: {loss.item():.4f}')
    # lri.append(lre[i])
    # lossi.append(loss.item())

print('Loss: ', loss.item())

# train(Xtr, Ytr)

print('Validation loss: ', eval(Xdev, Ydev))
print('Test loss: ', eval(Xte, Yte))

def sample():
    g = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(20):
        out = []
        context = [0] * BLOCK_SIZE
        while True:
            emb = C[torch.tensor([context])] # (1, BLOCK_SIZE, d)
            h = torch.tanh(emb.view(-1, CD) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            else:
                out.append(ix)
        print(''.join(itos[i] for i in out))

pl.plot(torch.arange(STEPS).tolist(), lossi)
pl.show()
