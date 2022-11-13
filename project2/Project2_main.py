class ReLU(object):
    def __init__(self):
        self.prev_input = 0

    def stepSGD(self,eta):
        pass

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


