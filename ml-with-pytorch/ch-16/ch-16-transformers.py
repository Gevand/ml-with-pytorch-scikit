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

x_2 = embedded_sentence[1,:]
context_vec_2 = torch.zeros(x_2.shape)
for j in range(8):
    x_j = embedded_sentence[j,:]
    context_vec_2 += attention_weights[1, j] * x_j

print(context_vec_2)


torch.manual_seed(123)
d = embedded_sentence.shape[1]
U_query = torch.rand(d,d)
U_key = torch.rand(d,d)
U_value = torch.rand(d,d)

x_2 = embedded_sentence[1]

query_2 = U_query.matmul(x_2)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)

keys = U_key.matmul(embedded_sentence.T).T
print(torch.allclose(key_2, keys[1]))

values = U_value.matmul(embedded_sentence.T).T
print(torch.allclose(value_2,values[1]))

w_2_3 = query_2.dot(keys[2])
print(w_2_3)

w_2 = query_2.matmul(keys.T)
print(torch.allclose(w_2_3, w_2[2]))

attention_weights_2 = F.softmax(w_2 / d**0.5, dim=0)
print(attention_weights_2)

context_vec_2 = attention_weights_2.matmul(values)
print(context_vec_2)

torch.manual_seed(123)
d = embedded_sentence.shape[1]
one_U_query = torch.rand(d,d)

h = 8
multihead_U_query = torch.rand(h,d,d)
multihead_U_key = torch.rand(h,d,d)
multihead_U_value = torch.rand(h,d,d)

multihead_query_2 = multihead_U_query.matmul(x_2)
print(multihead_query_2.shape)

multihead_key_2 = multihead_U_key.matmul(x_2)
multihead_value_2 = multihead_U_value.matmul(x_2)
print(multihead_key_2[2])

stacked_inputs = embedded_sentence.T.repeat(8,1,1)
print(stacked_inputs.shape)

multihead_keys = torch.bmm(multihead_U_key, stacked_inputs)
print(multihead_keys.shape)

multihead_keys = multihead_keys.permute(0, 2, 1)
print(multihead_keys.shape)

print(multihead_keys[2,1])

multihead_values = torch.matmul(multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0,2,1)

multihead_z_2 = torch.rand(8, 16)
linear = torch.nn.Linear(8*16, 16)
context_vector_2 = linear(multihead_z_2.flatten())
print(context_vector_2.shape)