import torch.nn.functional as F
import torch
sentence = torch.tensor(
    [0,  # can
     7,  # you
     1,  # help
     2,  # me
     5,  # to
     6,  # translate
     4,  # this
     3   # sentence
     ])
print(sentence)
# imagine we have an embedding layer with a size of 16 and a dictionary of 10 words
torch.manual_seed(123)
embed = torch.nn.Embedding(10, 16)
embedded_sentence = embed(sentence).detach()
print(embedded_sentence.shape)

# computing w(i,j) as the dot product between ith and jth word embedding
w = torch.empty(8, 8)
for i, x_i in enumerate(embedded_sentence):
    for j, x_j, in enumerate(embedded_sentence):
        w[i, j] = torch.dot(x_i, x_j)
w_effecient = torch.empty(8, 8)
# another way to acheive the same thing this loop does is with matmul
w_effecient = embedded_sentence.matmul(embedded_sentence.T)

print(w, '\n vs \n', w_effecient)

# attention can be obtained by "softmaxing" the wij matrix
attention_weights = F.softmax(w, dim=1)
print(attention_weights.shape)
print(attention_weights)
print(attention_weights.sum(dim=1))
