import torch
import math




def generate_set(n):# 0 if outside the disk centered at (0.5, 0.5)  of radius 1/√2π, and 1 inside
    input = torch.rand(n, 2)
    target = (input.sub(0.5).pow(2).sum(1).sqrt() < 1/math.sqrt(2*math.pi)).long()
    target[target == 0] = -1
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

    def Stock_Grad_Descent(self, LR):  # lr = LEARNING RATE
        return []

    def zero_grad(self):
        return []

    def param(self):
        return []


class Tanh(object): # use tanh func to send input in next layer

    def __init__(self):
        self.prev_input = 0

    def forward(self, input): #tanh()
        self.prev_input = input
        return torch.tanh(input)

    def backward(self, gradwrtoutput):#derivative / gradiant_input
        return (1.0-torch.tanh(self.prev_input)**2).mul(gradwrtoutput)


    def Stock_Grad_Descent(self, LR):  # lr = LEARNING RATE
        return []

    def zero_grad(self):
        return []

    def param(self):
        return []






class Linear(object):
#can change the dimenton with the input/output in the layer
#nodes are weightrd and biased
#they allow to make prediction with a given output
    def __init__(self, dim_x, dim_y, bias=0):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.W_x = torch.rand(dim_y, dim_x).sub(bias)#weigth randomly at init
        self.b = torch.rand(dim_y).sub(bias) #same for bias
        self.W_x_grad = torch.zeros(dim_y, dim_x)
        self.b_grad = torch.zeros(dim_y)
        self.prev_input = torch.zeros(dim_y)#keep last result




    def zero_grad(self):#recompute it using new parameters
        self.W_x_grad = torch.zeros(self.dim_y, self.dim_x)
        self.b_grad = torch.zeros(self.dim_y)

    def forward(self, input):
        # chapter 3.6 slide 9
        self.prev_input = input #formula x*w + b
        return self.W_x.mv(input.T)+self.b

    def backward(self, gradwrtoutput):
        # chapter 3.6 slide 9
        self.b_grad += gradwrtoutput
        self.W_x_grad += gradwrtoutput.outer(self.prev_input.T)

        #chapter 3.6 slide 8
        return self.W_x.T.mv(gradwrtoutput)#w^T * (dl/dx)


    def Stock_Grad_Descent(self, LR):  # LR = learning rate
        self.W_x = self.W_x - LR * self.W_x_grad
        self.b = self.b - LR * self.b_grad


    def param(self):
        return [(self.W_x, self.W_x_grad), (self.b, self.b_grad)]


class Sequential(object):

    def __init__(self, *argv):
        self.arg = argv



    def forward(self, input):
        re = input
        for i in self.arg:
            re = i.forward(re)
        return re

    def backward(self, gradwrtoutput):
        re = gradwrtoutput
        for i in reversed(self.arg):
            re = i.backward(re)
        return re

    def zero_grad(self):#recompute it using new parameters
        for i in self.arg:
            i.zero_grad()

    def Stock_Grad_Descent(self,LR):#lr = LEARNING RATE
        for i in self.arg:
            i.Stock_Grad_Descent(LR)

    def param(self):
        for i in reversed(self.arg):
            print(i.param())


class MSE(object):

    def __init__(self):
        self.prev_input = 0
        self.prev_target = 0
        self.loss = 0

    def forward(self, input, target):
        # = (x-y)²
        self.loss = torch.pow(torch.sub(input, target),2)
        self.prev_input = input
        self.prev_target = target
        return self.loss

    def backward(self):
        # dL/dy = 2(x-y)
        return 2*torch.sub(self.prev_input, self.prev_target) #dL/dy = 2(x-y)

    def param(self):
        return []