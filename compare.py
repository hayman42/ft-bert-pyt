import torch
import torch.nn as nn
import torch.nn.quantized as quant
from quantize import QuantizedLinear, quantize, dequantize


def mae(x, y):
    return torch.abs(x - y).mean()


def mse(x, y):
    return ((x-y)*(x-y)).mean()


def q_dq(x, n_bits=8):
    a, b = x.max(), x.min()
    n_nums = (1 << n_bits)-1
    s = (a-b)/n_nums
    z = ((-b) / s).round().to(torch.int)
    dq = quant.DeQuantize()(quant.Quantize(s, z, torch.quint8)(x))
    return dq


def compare_Linear_vs_QuantizedLinear(x, hidden):
    linear = nn.Linear(hidden, hidden+1)
    ql = QuantizedLinear(hidden, hidden+1)
    ql.bias = linear.bias
    print(linear.weight)
    q, s, z = quantize(linear.weight)
    print(q)
    ql.weight = nn.Parameter(q)
    ql.scale = nn.Parameter(s)
    ql.zero_point = nn.Parameter(z)

    y = linear(x)

    q_y = ql._forward(x)

    print("compare_Linear_vs_QuantizedLinear")
    print("mae:")
    print(mae(y, q_y))
    print("mse:")
    print(mse(y, q_y))
    print()
    print()


def compare_x_vs_quantized(x):
    q, s, z = quantize(x)
    dq = dequantize(q, s, z)

    print("compare_x_vs_quantized")
    print("mae:")
    print(mae(x, dq))
    print("mse:")
    print(mse(x, dq))
    print()
    print()


def compare_x_vs_torch_quantized(x):
    dq = q_dq(x)

    print("compare_x_vs_torch_quantized")
    print("mae:")
    print(mae(x, dq))
    print("mse:")
    print(mse(x, dq))
    print()
    print()


def compare_QLinear_vs_QuantLinear(x, h):
    qx = torch.quantize_per_tensor(x, 1.0, 0, torch.quint8)
    quantl = quant.Linear(h, h)(qx)
    print(quantl)


if __name__ == "__main__":
    h = 10
    x = torch.randn(h, h)
    compare_Linear_vs_QuantizedLinear(x, h)
    # compare_x_vs_quantized(x)
    # compare_x_vs_torch_quantized(x)
