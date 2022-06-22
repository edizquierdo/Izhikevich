import numpy as np

class IzhikevichNetwork:

    def __init__(self, size):
        self.size = size                        # number of neurons in the network
        self.Voltage = np.zeros(size) - 65           # neuron activation vector
        self.u = np.zeros(size)           # neuron activation vector
        self.a = np.ones(size)                  # a
        self.b = np.zeros(size)                 # b
        self.c = np.zeros(size)                 # c
        self.d = np.zeros(size)                 # d
        self.Weight = np.zeros((size,size))     # weight matrix
        self.Input = np.zeros(size)             # neuron output vector

    def randomizeParameters(self):
        self.Weight = np.random.uniform(-10,10,size=(self.size,self.size))
        self.a = np.random.uniform(0.02,0.1,size=self.size)
        self.b = np.random.uniform(0.2,0.27,size=self.size)
        self.c = np.random.uniform(-65,-50,size=self.size)
        self.d = np.random.uniform(0.05,8,size=self.size)

    def step(self,dt):
        for i in range(self.size):
            self.Input[i] = 0
            for j in range(self.size):
                if self.Voltage[i] >= self.c[i]:
                    self.Input[i] += self.Weight[j][i]
        for i in range(self.size):
            dVdt = (0.04 * (self.Voltage[i]**2)) + (5 * self.Voltage[i]) + 140.0 - self.u[i] + self.Input[i]
            dudt = self.a[i] * ((self.b[i] * self.Voltage[i]) - self.u[i])
            self.Voltage[i] += dt * dVdt
            self.u[i] += dt * dudt
            if self.Voltage[i] >= 30.0:
                self.Voltage[i] = self.c[i]
                self.u[i] = self.u[i] + self.d[i]
