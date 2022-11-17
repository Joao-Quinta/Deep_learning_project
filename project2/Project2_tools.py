import torch
import math




def generate__set(n):
    input = torch.rand(n, 2)
    radius = 1/math.sqrt(2*math.pi)# 0 if outside the disk centered at (0.5, 0.5)  of radius 1/√2π, and 1 inside
    target = (input.sub(0.5).pow(2).sum(1).sqrt() < radius).long()
    return input, target


class ReLU(object):
    def __init__(self):
        self.prev_input = 0

    

    def forward(self, input):
        self.prev_input = input
        input[input<0.0] = 0.0
        return input

    def backward(self, input):#derivative / gradiant_input
        self.prev_input[self.prev_input<0.0] = 0.0
        self.prev_input[self.prev_input>0.0] = 1.0
        return self.prev_input.mul(input)

    def param(self):
        return []


class Tanh(object):

    def __init__(self):
        self.prev_input = 0

    def forward(self, input):
        self.prev_input = input
        return torch.tanh(input)

    def backward(self, input):#derivative / gradiant_input
        return (1.0-torch.tanh(self.prev_input)**2).mul(input)

    def param(self):
        return []

class Linear(object):

    def __init__(self, dim_x, dim_y, bias=0):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.W_x = torch.rand(dim_y, dim_x).sub(bias)
        self.b = torch.rand(dim_y).sub(bias) #bias
        self.W_x_grad = torch.zeros(dim_y, dim_x)
        self.b_grad = torch.zeros(dim_y)
        self.prev_input = torch.zeros(dim_y)

    def SGD(self, eta):
        self.W_x = self.W_x - eta*self.W_x_grad
        self.b = self.b - eta*self.b_grad


    def zero_grad(self):
        self.W_grad = torch.zeros(self.dim_y, self.dim_x)
        self.b_grad = torch.zeros(self.dim_y)

    def forward(self, input):
        self.prev_input = input
        return self.W_x.mv(input.T)+self.b

    def backward(self, input):
        self.b_grad += input
        self.W_x_grad += input.outer(self.prev_input.T)

        return self.W_x.T.mv(input)

    def param(self):
        return [(self.W_x, self.W_x_grad), (self.b, self.b_grad)]


