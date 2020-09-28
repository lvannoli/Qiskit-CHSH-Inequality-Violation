# Qiskit-CHSH-Inequality-Violation
Qiskit code to simulate CHSH inequality violation

Scheme of Aspect's Experiment

![Alt text](images/Aspect_epr.png?raw=true "Aspect's Experiment")

```python
%matplotlib inline
# Importing standard Qiskit libraries and configuring account from qiskit import QuantumCircuit, execute, Aer, IBMQ, BasicAer, QuantumRegister, ClassicalRegister
from qiskit.providers.ibmq import least_busy
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import *
from qiskit.visualization import plot_state_city, plot_bloch_multivector 
from qiskit.visualization import plot_state_paulivec, plot_state_hinton 
from qiskit.quantum_info.analysis import average_data
from qiskit_textbook.tools import array_to_latex
# Loading your IBM Q account(s)
provider = IBMQ.load_account()
from math import pi,sqrt
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
```
Chose the simulator backend, the device backend and the number of shots

```python
import qiskit
qiskit.__qiskit_version__
# use simulator to learn more about entangled quantum states where possible 
sim_backend = BasicAer.get_backend('qasm_simulator')
sim_shots = 8192
# use device to test entanglement
device_shots = 1024
device_backend = least_busy(IBMQ.get_provider().backends(operational=True,simulator=False))
print(the backend is " + device_backend.name())
```

```
the backend is ibmq_london
```
Creating the circuits for the Z and the X basis measurements
```python
# Creating registers
q = QuantumRegister(2)
c = ClassicalRegister(2)
# quantum circuit to make an entangled bell state
bell = QuantumCircuit(q, c)
bell.h(q[0])
bell.cx(q[0], q[1])
# quantum circuit to measure q in the Z basis
measureZZ = QuantumCircuit(q, c) 
measureZZ.measure(q[0], c[0]) 
measureZZ.measure(q[1], c[1]) 
bellZZ = bell+measureZZ
# quantum circuit to measure q in the X basis
measureXX = QuantumCircuit(q, c) 
measureXX.h(q[0]) measureXX.h(q[1]) 
measureXX.measure(q[0], c[0]) 
measureXX.measure(q[1], c[1]) 
bellXX = bell+measureXX
# quantum circuit to measure ZX
measureZX = QuantumCircuit(q, c) 
measureZX.h(q[0]) 
measureZX.measure(q[0], c[0]) 
measureZX.measure(q[1], c[1]) 
bellZX = bell+measureZX
  # quantum circuit to measure XZ
measureXZ = QuantumCircuit(q, c) 
measureXZ.h(q[1]) 
measureXZ.measure(q[0], c[0]) 
measureXZ.measure(q[1], c[1]) 
bellXZ = bell+measureXZ

circuits = [bellZZ,bellXX,bellZX,bellXZ]
```
```
observable_correlated ={'00': 1, '01': -1, '10': -1, '11': 1}
```

Defining the function to compute the value of CHSH inequality
```
CHSH = lambda x : x[0]+x[1]+x[2]-x[3]
measure = [measureZZ, measureZX, measureXX, measureXZ]
```

Instantiate the simulation circuits. The measurments are done rotating the first qbit to obtain the dependence of the CHSH inequality value from the angular rotation of the polarizers.

```python
# Theory
sim_chsh_circuits = [] 
sim_x = []
sim_steps = 30
for step in range(sim_steps):
  theta = 2.0*np.pi*step/30 
  bell_middle = QuantumCircuit(q,c) 
  bell_middle.ry(theta,q[0])
  for m in measure: 
    sim_chsh_circuits.append(bell+bell_middle+m)
  sim_x.append(theta) 
#print(sim_x)
```

```python
sim_chsh_circuits[0].draw(output='mpl') 
sim_chsh_circuits[1].draw(output='mpl') 
sim_chsh_circuits[2].draw(output='mpl') 
sim_chsh_circuits[3].draw(output='mpl')
```

![Alt text](images/ZZRy_circ.png?raw=true "ZZ Circuit")
![Alt text](images/ZXRy_circ.png?raw=true "ZX Circuit")
![Alt text](images/XZRy_circ.png?raw=true "XZ Circuit")
![Alt text](images/XXRy_circ.png?raw=true "XX Circuit")

Getting the results of the simulation
```python
job = execute(sim_chsh_circuits, backend=sim_backend, shots=sim_shots)
result = job.result()
```
Evaluating the simulation CHSH inequality value
```python
sim_chsh = [] circ = 0
for x in range(len(sim_x)): 
  temp_chsh = []
  for m in range(len(measure)): 
    temp_chsh.append(average_data(result.get_counts( sim_chsh_circuits[circ].name),observable_correlated)) 
    circ += 1
  sim_chsh.append(CHSH(temp_chsh)) 
#print("sim_chsh = ",sim_chsh)
```

Same procedure for the "Experiment" section. In this part we are using a real Quantum Computer Backend
```python
# Experiment
real_chsh_circuits = [] real_x = []
real_steps = 16
for step in range(real_steps):
  theta = 2.0*np.pi*step/16 
  bell_middle = QuantumCircuit(q,c) 
  bell_middle.ry(theta,q[0])
  for m in measure:
    real_chsh_circuits.append(bell+bell_middle+m) 
  real_x.append(theta)
```

```python
job = execute(real_chsh_circuits, backend=device_backend, shots=device_shots)
job_monitor(job)
```

```python
result = job.result()
```

Evaluating of the CHSH inequality value
```python
real_chsh = [] 
circ = 0
angle = [] 
data = []
for x in range(len(real_x)):
  temp_chsh = []
  #print("x = ",x, "real_x = ", real_x[x]) 
  for m in range(len(measure)):
    angle.append(real_x[x]) 
    data.append(result.get_counts(real_chsh_circuits[circ])) 
    temp_chsh.append(average_data(result.get_counts( real_chsh_circuits[circ].name),observable_correlated)) 
    circ += 1
  real_chsh.append(CHSH(temp_chsh)) 
  #print(CHSH(temp_chsh))
```
Now, we evaluate the value of C (CHSH function) at the maximum violation angle and its statistic error
``python
c = np.zeros(4)
err_q = np.zeros(4)
circ = 0
data_index = []
for i,t in enumerate(angle):
  if t == pi/4:
    data_index.append(i)
    c[circ] += (data[i]["00"]+data[i]["11"]- data[i]["01"]-data[i]["10"])/device_shots 
    err_q[circ]+=data[i]["00"]/device_shots*(1-data[i]["00"]/device_shots)/device_shots 
    err_q[circ]+=data[i]["01"]/device_shots* (1-data[i]["01"]/device_shots)/device_shots 
    err_q[circ]+=data[i]["11"]/device_shots* (1-data[i]["11"]/device_shots)/device_shots 
    err_q[circ]+=data[i]["10"]/device_shots* (1-data[i]["10"]/device_shots)/device_shots
#print(err_q)
C = c[0]+c[1]+c[2]-c[3] 
errC = 0
for r in err_q:
  errC += r*r
errC = sqrt(errC) 
print("C = ",C,"+-",errC)
```
Showing the histograms of measurments result of the experiment mesurments of the quantum circuits
```python
plot_histogram(data[data_index[0]], color='midnightblue', title="MeasureZZ") 
plot_histogram(data[data_index[1]], color='midnightblue', title="MeasureZX") 
plot_histogram(data[data_index[2]], color='midnightblue', title="MeasureXX") 
plot_histogram(data[data_index[3]], color='midnightblue', title="MeasureXZ")
```
![Alt text](images/fig_measureZZ.png?raw=true "ZZ Measure")
![Alt text](images/fig_MeasureZX.png?raw=true "ZX Measure")
![Alt text](images/fig_measureXZ.png?raw=true "XZ Measure")
![Alt text](images/fig_MeasureXX.png?raw=true "XX Measure")

Drawing the curves of CHSH value vs angles comparing simulation (blue dots) with the experiment (red line)
```python
from matplotlib.ticker import FuncFormatter, MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
plt.plot(sim_x, sim_chsh, 'r-', real_x, real_chsh, 'bo') 
red_patch = mpatches.Patch(color='red', label='Simulation') 
blue_patch = mpatches.Patch(color='blue',label='Real') 
plt.legend(handles=[red_patch,blue_patch]) 
plt.plot([0, 2*np.pi], [2, 2], 'b-') 
plt.plot([0, 2*np.pi], [-2, -2], 'b-') 
plt.grid()
plt.ylabel('CHSH', fontsize=20) 
plt.xlabel(r'$\theta$', fontsize=20) 
ax = plt.gca()
ax.grid(True)
ax.set_aspect(1.0)
ax.axhline(0, color='black', lw=2) 
ax.axvline(np.pi/4, color='green',lw=1) 
ax.axvline(5*np.pi/4, color='green',lw=1)
ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0' ))
ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
plt.show()
```
![Alt text](images/fig_CHSHvsAngle.png?raw=true "CHSH values vs Angle")
