import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = True
        self.x2h = nn.Linear(input_size, 3*hidden_size, bias = self.bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias= self.bias)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(h)

        # gate_x = gate_x.squeeze()
        # gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (h - newgate)
        return hy


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_len, _ = inputs.shape
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h = self.cell(x_t, h)  # h를 업데이트
        
        return h  # 마지막 타임스텝의 hidden state만 반환

        