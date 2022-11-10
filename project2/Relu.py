class ReLUModule(object):
    def __init__(self):
        self.last_input = 0

    def stepSGD(self,eta):
        pass

    def forward(self, input):
        self.last_input = input
        input[input<0.0] = 0.0
        return input

    def backward(self, gradwrtoutput):#derivative
        self.last_input[self.last_input<0.0] = 0.0
        self.last_input[self.last_input>0.0] = 1.0
        return self.last_input.mul(gradwrtoutput)



    def param(self):
        return []