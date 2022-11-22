import torch
import math




def generate_set(n):# 0 if outside the disk centered at (0.5, 0.5)  of radius 1/√2π, and 1 inside
    input = torch.rand(n, 2)
    radius = 1/math.sqrt(2*math.pi)
    target = (input.sub(0.5).pow(2).sum(1).sqrt() < radius).long()
    return input, target


class ReLU(object): #output = R(x), 0 if input = 0 else output = input
    def __init__(self):
        self.prev_input = 0


    def forward(self, input): # max(0,input)
        self.prev_input = input
        input[input<0.0] = 0.0
        return input

    def backward(self, gradwrtoutput):#derivative / gradiant_input , ax(0,input) is 0 if input <= 0, else it is 1
        self.prev_input[self.prev_input<0.0] = 0.0
        self.prev_input[self.prev_input>0.0] = 1.0
        return self.prev_input.mul(gradwrtoutput)

    def param(self):
        return []


class Tanh(object):

    def __init__(self):
        self.prev_input = 0

    def forward(self, input):
        self.prev_input = input
        return torch.tanh(input)

    def backward(self, gradwrtoutput):#derivative / gradiant_input
        return (1.0-torch.tanh(self.prev_input)**2).mul(gradwrtoutput)

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

    def Stock_Grad_Descent(self, LR):#LR = learning rate
        self.W_x = self.W_x - LR*self.W_x_grad
        self.b = self.b - LR*self.b_grad


    def zero_grad(self):
        self.W_grad = torch.zeros(self.dim_y, self.dim_x)
        self.b_grad = torch.zeros(self.dim_y)

    def forward(self, input):
        self.prev_input = input #formula x*w + b
        return self.W_x.mv(input.T)+self.b

    def backward(self, gradwrtoutput):
        self.b_grad += gradwrtoutput
        self.W_x_grad += gradwrtoutput.outer(self.prev_input.T)

        return self.W_x.T.mv(gradwrtoutput)

    def param(self):
        return [(self.W_x, self.W_x_grad), (self.b, self.b_grad)]


