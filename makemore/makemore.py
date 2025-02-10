import torch
import matplotlib.pyplot as pl

words = open('names.txt', 'r').read().splitlines()

print(words[:10])

chars = sorted(list(set(''.join(words))))
print(chars)

stoi = {s:i + 1 for i,s in enumerate(chars)}
stoi['.'] = 0
# stoi['<S>'] = 26
# stoi['<E>'] = 27
itos = {i + 1:s for i,s in enumerate(chars)}
# itos[26] = '<S>'
# itos[27] = '<E>'
itos[0] = '.'
print(stoi)
# print(itos)
# b = {}
N = torch.zeros(28, 28, dtype=torch.int32)
for w in words:
    wl = ['.'] + list(w) + ['.']
    for c1, c2 in zip(wl, wl[1:]):
        index1 = stoi[c1]
        index2 = stoi[c2]
        N[index1, index2] += 1
        # b[bigram] = b.get(bigram, 0) + 1
        # print(c1, c2)
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
p = torch.rand(3, generator=g)
p = p / p.sum()
print(p)
m = torch.multinomial(p, num_samples=20, replacement=True, generator=g)
print(m)
