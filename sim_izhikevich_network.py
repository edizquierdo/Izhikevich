import izhikevich as iz
import matplotlib.pyplot as plt
import numpy as np

# Global Parameters 
size = 100
duration = 60
stepsize = 0.005
        
time = np.arange(0.0,duration,stepsize)

nn = iz.IzhikevichNetwork(size)
nn.randomizeParameters()

outputs = np.zeros((len(time),size))

# Run simulation
step = 0
for t in time:
    nn.step(stepsize)
    outputs[step] = nn.Voltage
    step += 1

# Plot activity
plt.plot(time,outputs)
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Neural activity")
plt.show()

