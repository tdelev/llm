import torch
import matplotlib.pyplot as pl
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

print(words[:10])

chars = sorted(list(set(''.join(words))))
# print(chars)

stoi = {s:i + 1 for i,s in enumerate(chars)}
stoi['.'] = 0
# stoi['<S>'] = 26
# stoi['<E>'] = 27
itos = {i + 1:s for i,s in enumerate(chars)}
# itos[26] = '<S>'
# itos[27] = '<E>'
itos[0] = '.'
# b = {}
N = torch.zeros(27, 27, dtype=torch.int32)
for w in words:
    wl = ['.'] + list(w) + ['.']
    for c1, c2 in zip(wl, wl[1:]):
        i1 = stoi[c1]
        i2 = stoi[c2]
        N[i1, i2] += 1
        # b[bigram] = b.get(bigram, 0) + 1
print(N)
P = N.float() / N.sum(1, keepdim=True)
# print(P)
# print(P[0].sum())
#
# # print(sorted(b.items(), key= lambda x: x[1]))
# print(N)
# pl.imshow(N)
# pl.figure(figsize=(16, 16))
# pl.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         pl.text(j, i, chstr, ha='center', va='bottom', color='gray')
#         pl.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
#
# pl.show()

g = torch.Generator().manual_seed(2147483647)

log_likelihood = 0.0
n = 0

# for w in words:
#     wl = ['.'] + list(w) + ['.']
#     for c1, c2 in zip(wl, wl[1:]):
#         i1 = stoi[c1]
#         i2 = stoi[c2]
#         prob = P[i1, i2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         n += 1

# print(f'll={log_likelihood}')
# print(f'nll={-log_likelihood}')
# print(f'avg nll={-log_likelihood/n}')

# create the training set of bigrams (x, y)

xs, ys = [], []

for w in words:
    wl = ['.'] + list(w) + ['.']
    for c1, c2 in zip(wl, wl[1:]):
        xs.append(stoi[c1])
        ys.append(stoi[c2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('Number of elements: ', num)
W = torch.randn((27, 27), generator=g, requires_grad=True)
for i in range(100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W # log-counts
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    print(f'loss={loss}')

    # backward pass
    W.grad = None
    loss.backward() # backpropagation
    W.data += -50 * W.grad # update weights

# print(W.data.exp())


g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        else:
            out.append(itos[ix])
    print(''.join(out))

print('NN model')
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W # log-counts
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        else:
            out.append(itos[ix])
    print(''.join(out))
