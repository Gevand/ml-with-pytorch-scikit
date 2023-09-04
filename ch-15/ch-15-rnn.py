import torch
import torch.nn as nn

torch.manual_seed(1)

rnn_layer = nn.RNN(input_size=5, hidden_size=2, num_layers=1, batch_first=True)

w_xh = rnn_layer.weight_ih_l0
w_hh = rnn_layer.weight_hh_l0
b_xh = rnn_layer.bias_ih_l0
b_hh = rnn_layer.bias_hh_l0

print('w_xh shape ', w_xh.shape)
print('w_hh shape ', w_hh.shape)
print('b_xh shape ', b_xh.shape)
print('b_hh shape ', b_hh.shape)

x_seq = torch.tensor([[1.0] * 5, [2.0] * 5, [3.0] * 5]).float()
output, hn = rnn_layer(torch.reshape(x_seq, (1, 3, 5)))
out_man = []
for t in range(3):
    xt = torch.reshape(x_seq[t], (1, 5))
    print(f'Time step {t} =>')
    print(' Input : ', xt.numpy())
    ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_hh
    print(' Hidden :', ht.detach().numpy())

    if t > 0:
        prev_h = out_man[t-1]
    else:
        prev_h = torch.zeros((ht.shape))
    
    ot = ht + torch.matmul(prev_h, torch.transpose(w_hh, 0 , 1)) + b_hh
    ot = torch.tanh(ot)
    out_man.append(ot)
    print(' Output (manual) :', ot.detach().numpy())
    print(' RNN output :', output[:, t].detach().numpy())
    print(' ----------------------------------')
