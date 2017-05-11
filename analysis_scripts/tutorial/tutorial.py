"""
Tutorial for Joint Quantum State and Measurement Tomography
AK 2016
"""

from partial_tomography import *
import time
import numpy as np
import datetime
t = time.time()
N = 2   # number of qubits
dim = 2**N  # dimension of density matrices

################## BUILD HISTOGRAMS FROM DATA ######################
# Ion trappers, make a list of counts using countsFromExperimentClass()
# in general a list of counts or an existing histogram (array) will do
hists = []         # list of ALL histograms
input_state = []   # list of input states (labels for unknown states)
unitaries = []     # analysis unitaries

# Dark, Dark Reference
dark = np.zeros((dim, dim))+0j
dark[0, 0] = 1.0
c = np.loadtxt('hist0',delimiter=',')
h = Hist(c)
hists.append(h)
input_state.append(dark)
unitaries.append(np.eye(dim))

# Dark, Bright Reference
db =  np.zeros((dim, dim))+0j
db[1,1] = 1.0
c = np.loadtxt('hist1',delimiter=',')
h = Hist(c)
hists.append(h)
input_state.append(db)
unitaries.append(np.eye(dim))

# Bright, Dark Reference
bd =  np.zeros((dim, dim))+0j
bd[2,2] = 1.0
c = np.loadtxt('hist2',delimiter=',')
h = Hist(c)
hists.append(h)
input_state.append(bd)
unitaries.append(np.eye(dim))

# Bright, Bright Reference
bright = np.zeros((dim, dim))+0j
bright[-1, -1] = 1.0
c = np.loadtxt('hist3',delimiter=',')
h = Hist(c)
hists.append(h)
input_state.append(bright)
unitaries.append(np.eye(dim))

# Population Data
c = np.loadtxt('hist4',delimiter=',')
h = Hist(c)
hists.append(h)
input_state.append(1)   # label "1" for first unknown density matrix
unitaries.append(np.eye(dim))

# Parity Data
phases = [0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345, 
          3.141592653589793, 3.9269908169872414, 4.71238898038469, 
          5.497787143782138]
for i in range(5, 13):
    c = np.loadtxt('hist' + str(i),delimiter=',')
    h = Hist(c)
    hists.append(h)
    input_state.append(1)   # label "1" for first unknown density matrix
    unitaries.append(pulseSequence([np.pi/2], phi=[phases[i-5]], N=N))
    

################## RELEVANT PARAMETERS AND TARGET STATES ######################
num_sub = dim
statemap = np.eye(dim)
P_j = np.zeros((num_sub, dim, dim)) # Projectors defining subspaces
for j in range(num_sub):
    P_j[j, :, :] = np.diag(statemap[:, j])

# Target unknown density matrix
state_vector = (1/np.sqrt(2))*np.array([1,0,0,1])
target_rho = np.outer(state_vector, state_vector)

binnum = [2,2] # number of bins for each axis


########################### DO ANALYSIS #####################################
SE_name = 'tutorial_save'  # use a useful name
SE = AnalysisSet(hists, input_state, unitaries, P_j, trainFrac=0.1,
                    name=SE_name, measure=fidelity, targets=[target_rho])
# measure() is a function that measures inferred rhos with respecet to targets
SE.autobin(binnum)   # autobin


print('Running Tomography...')
inferred_fidelities = SE.tomography()
## Note SE.estRho is a list of inferred density matrices (not just one!)
print('Estimated Fidelity = ' + str(SE.fidelity))

bootstrap_iters = 20
print('Running Bootstrap...')
bootstrap_fids = SE.bootstrap(method='parametric', iters=bootstrap_iters)
SE.bootstrap_analysis()
#print('Bootstrap Standard Deviation: ' + str(np.std(bootstrap_fids)))
#
#SE.save(SE_name)  # Note: SE autosaves after every important computation
#                  # i.e. tomographyML(), bootstrap(), sensitivity_analysis()
#
print('Elapsed Time: ' + str(time.time()- t) + ' seconds')
