import torch
from torch import nn
from torch.nn import functional as F


def quantize(tensor, bits):
    s = (1 << bits) - 1

    # norm = torch.norm(tensor)
    norm = tensor.abs().max()

    sign_array = torch.sign(tensor).to(dtype=torch.int8)

    l_array = torch.abs(tensor) / norm * s
    l_array_floored = l_array.to(dtype=torch.int)
    prob_array = l_array - l_array_floored
    prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

    mask = torch.bernoulli(prob_array)
    xi_array = l_array_floored + mask
    xi_array = xi_array.to(dtype=torch.int32)

    sign_xi_array = (sign_array * xi_array).to(dtype=torch.int8)
    norm = norm / s

    return norm, sign_xi_array


def dequantize(norm, sign_xi_array, bits):
    s = (1 << bits) - 1

    weights = norm * sign_xi_array

    return weights


class FakeLinearQuantizationFunction(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""

    @staticmethod
    def forward(ctx, input, bits=7):
        norm, quantized_weight = quantize(input, bits)
        return dequantize(norm, quantized_weight, bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


_fake_quantize = FakeLinearQuantizationFunction.apply


class QuantizedLinear(nn.Linear):
    """Linear layer with quantization aware training capability"""

    def __init__(self, *args, weight_bits=8, warmup_step=0, **kwargs):
        super().__init__(*args, **kwargs)

        if weight_bits < 2:
            raise ValueError(f"weight_bits={weight_bits} must be higher than 1 ")

        self.weight_bits = weight_bits
        self.warmup_step = warmup_step
        self.accumulation_bits = 32
        
        self._fake_quantized_weight = None
        if kwargs.get("bias", True):
            self.register_buffer("quantized_bias", None)
            self.register_buffer("bias_norm", None)

        self.register_buffer("_step", torch.zeros(1))

        # buffers for inference
        self.register_buffer("quantized_weight", None)
        self.register_buffer("weight_norm", None)

    def training_quantized_forward(self, input):
        """Fake quantizes weights. This function should only be used while training"""
        assert self.training, "should only be called when training"

        self._fake_quantized_weight = _fake_quantize(self.weight, self.weight_bits)
        out = F.linear(input, self._fake_quantized_weight, self.bias)

        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. This function should only be used while doing inference"""
        assert not self.training, "should only be called when not training"

        weight = self.weight_norm * self.quantized_weight

        if self.bias is not None:
            bias = self.bias_norm * self.quantized_bias

        out = F.linear(input, weight, bias)

        return out

    def _eval(self):
        self.weight_norm, self.quantized_weight = quantize(self.weight, self.weight_bits)

        if self.bias is not None:
            self.bias_norm, self.quantized_bias = quantize(self.bias, self.accumulation_bits)

    def forward(self, input):
        if self.training:
            if self._step > self.warmup_step:
                out = self.training_quantized_forward(input)
            else:
                out = super().forward(input)
            self._step += 1
        else:
            out = self.inference_quantized_forward(input)
        return out


def quantizer(model, quantization_bits=7, binarize_all_linear=False):
    for name, layer in model.named_children():
        if 'generator' in name:
            continue

        if type(layer) == nn.Linear and binarize_all_linear:
            model.__dict__['_modules'][name] = QuantizedLinear(layer.in_features, layer.out_features, weight_bits=quantization_bits)
        elif type(layer) == nn.Linear:
            if name in ['0', '1', '2']:
                model.__dict__['_modules'][name] = QuantizedLinear(layer.in_features, layer.out_features, weight_bits=quantization_bits)
        else:
            layer_types = [type(l) for l in layer.modules()]

            if nn.Linear in layer_types:
                quantizer(layer, quantization_bits, binarize_all_linear)

    return model
