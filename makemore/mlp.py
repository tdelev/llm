import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i + 1:s for i,s in enumerate(chars)}
itos[0] = '.'
g = torch.Generator().manual_seed(2147483647)

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + [] if self.bias is None else [self.bias]


class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # paramters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def paramters(self):
        return [self.gamma, self.beta]


class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

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

C = torch.randn((VOCAB_SIZE, DIMENSIONS), generator=g)
layers = [
    Linear(CD, HIDDEN_LAYER_SIZE), Tanh(),
    Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE), Tanh(),
    Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE), Tanh(),
    Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE), Tanh(),
    Linear(HIDDEN_LAYER_SIZE, VOCAB_SIZE),
]

with torch.no_grad():
    # last layer: make it less confident
    layers[-1].weight *= 0.1
    # all layers: apply gain
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

# W1 = torch.randn((CD, HIDDEN_LAYER_SIZE)) * (5/3)/(CD ** 0.5)
# b1 = torch.randn(HIDDEN_LAYER_SIZE) * 0.01
# W2 = torch.randn((HIDDEN_LAYER_SIZE, VOCAB_SIZE)) * 0.01
# b2 = torch.randn(VOCAB_SIZE) * 0
# bngain = torch.ones((1, HIDDEN_LAYER_SIZE))
# bnbias = torch.zeros((1, HIDDEN_LAYER_SIZE))
# bnmean_running = torch.zeros((1, HIDDEN_LAYER_SIZE))
# bnstd_running = torch.ones((1, HIDDEN_LAYER_SIZE))

# parameters = [C, W1, b1, W2, b2, bngain, bnbias]
parameters = [C] + [p for layer in layers for p in layer.parameters()]
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


# @torch.no_grad()
# def eval(X, Y):
#     emb = C[X]
#     embcat = emb.view(-1, CD) # concatenate the vectors
#     hlpreact = embcat @ W1 + b1  # hidden layer pre-activation
#     # measure the mean/std over the entire training set
#     hlpreact = bngain * (hlpreact - bnmean_running) / bnstd_running + bnbias
#     h = torch.tanh(hlpreact)
#     logits = h @ W2 + b2
#     loss = F.cross_entropy(logits, Y)
#     return loss

for i in range(STEPS):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb] # embed the characters into vectors
    x = emb.view(-1, CD) # concatenate the vectors
    for layer in layers:
        x = layer(x)

    loss = F.cross_entropy(x, Yb) # loss function
    # print('Loss: ', loss.item())
    lossi.append(loss.log10().item())

    # backward pass
    for layer in layers:
        layer.out.retain_grad()
        
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
    break

print('Loss: ', loss.item())

# train(Xtr, Ytr)

# print('Validation loss: ', eval(Xdev, Ydev))
# print('Test loss: ', eval(Xte, Yte))

def sample():
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

# plt.plot(torch.arange(STEPS).tolist(), lossi)
# plt.show()


# visualize historgrams

plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('activation distribution')

plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad
        print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('gradient distribution')
plt.show()
