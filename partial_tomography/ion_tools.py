"""Functions for performing analysis on ion trap data

Author = Adam Keith
 
Assumes |down> = |1>, |up> = |0> 
        |dark>        |bright>    
        
However, ACK doesn't always use this convention because it's not intuitive to
him. You see, it makes more sense that logical 0 is "darker" than logical 1
and so on. So sometimes he uses these functions in an unexpected way ONLY
when simulating his own things. Otherwise, these are the correct conventions 
for ion trappers, as long as they remember their labeling correctly!
"""

import numpy as np


def fiducial(dim=4):
    """Make fiducial state used in ion traps. First state in binary
    ordering is prepared (up=0). Called the "bright state" """
    rho = np.zeros((dim, dim))
    rho = rho+0j
    rho[0, 0] = 1.0
    return rho


def makeRho(dim=4, prob=1):
    """Make bell state density matrix + maximally mixed state of dimension dim

    rho = prob*(bell state density matrix) + (1-prob)*(maximally mixed state)
    """
    if dim == 4:
        # Standard 2 qubit bell state
        state_vector = (1/np.sqrt(2))*np.array([1, 0, 0, 1])
    elif dim == 8:
        # W State for three ions
        state_vector = (1/np.sqrt(3))*np.array([0, 1, 1, 0, 1, 0, 0, 0])
    bell = np.outer(state_vector, state_vector)
    rho = prob*bell + (1-prob)*np.identity(dim)/dim
    return rho + 0j


def pulseSequence(theta, phi, N=2):
    """Make N qubit unitary as a product of rotation unitaries
    for theta along axis phi in the xy plane for each qubit (symmetric)

    Input:
           theta  -- angle to rotate
           phi    -- list of phase angles of rotation unitaries
                     IF theta,phi are arrays (with equal length), make
                     Unitary a product of rotation matrices with those params
           N      -- Number of qubits
    """
    theta = np.array(theta)
    phi = np.array(phi)
    assert theta.size == phi.size
    U = np.eye(2**N)
    for i in range(theta.size):
        t = theta[i]
        p = phi[i]
        R = nist_rotation(t, p)
        temp = np.kron(R, R)
        for i in range(N-2):
            temp = np.kron(temp, R)
        U = temp.dot(U)
    return U


def pulseSequence_asymmetric(theta, phi, N=2):
    """Make N qubit unitary as a product of rotation unitaries
    for theta along axis phi in the xy plane for each qubit

    Input:
           theta  -- angles to rotate (length equal to N)
           phi    -- list of phase angles of rotation unitaries
                     for each qubit
           N      -- Number of qubits
    """
    theta = np.array(theta)
    phi = np.array(phi)
    assert theta.size == phi.size
    temp = 1
    for i in range(N):
        t = theta[i]
        p = phi[i]
        R = nist_rotation(t, p)
        temp = np.kron(temp, R)
    return temp


def nist_rotation(theta=np.pi/2, phi=0):
    """Construct NIST standard rotation matrix"""
    # Assumes |down> = |1>=|dark>, |up> = |0>=|bright>
    R = np.array([[np.cos(1/2*theta),
                  -1j*np.exp(-1j*phi)*np.sin(1/2*theta)],
                  [-1j*np.exp(1j*phi)*np.sin(1/2*theta),
                  np.cos(1/2*theta)]])
    return R


def scanUs(phases=10, N=2, theta=np.pi/2):
    """Make N qubit rotation unitaries for theta along axis
    phi in the xy plane

    Input: phase  -- list of phase angles of rotation unitaries
                     IF phase is an integer, make
                     phases = np.linspace(0, 2*np.pi, phases)
           N      -- Number of qubits
           theta  -- angle to rotate
    """
    if isinstance(phases, int):
        phases = np.linspace(0, 2*np.pi, phases)
    Us = []
    for phi in phases:
        R = nist_rotation(theta, phi)
        temp = np.kron(R, R)
        for i in range(N-2):
            temp = np.kron(temp, R)
        Us.append(temp)
    return Us


def makeUs(phases=10, N=2, theta=np.pi/2):
    """Make Unitaries that describe measurements of rho

    Assume first measurement is identity and all other measurements are pi/2
    rotations along axis phi. See scanUs for use.
    """
    Us = []
    temp = np.kron(np.eye(2), np.eye(2))
    for i in range(N-2):
        temp = np.kron(temp, np.eye(2))
    Us.append(temp)
    Us.extend(scanUs(phases, N, theta))
    return Us


def mixCriteria(rho):
    """Calculate trace(rho^2)"""
    return np.trace(rho.dot(rho))


def symStateMap(Nq=2):
    """Make state map for symmetric qubits. That is, all the states with the
    same Hamming weight are indistinguishable from each other

    Input:
        Nq -- number of qubits

    Output:
        f          -- the statemap
        purestates -- the number of subspaces
    """
    num_subs = Nq+1
    f = np.zeros((2**Nq, num_subs))
    for i in range(2**Nq):
        for j in range(num_subs):
            # find all states for same hamming weight j
            if (bin(i).count("1") == j):
                f[i, j] = 1
    return f, num_subs


def pre_bin(counts, real_binbounds):
    """Bin arbitrary measurement list and convert to integer bins for Hist()"""
    hist = np.histogram(counts, real_binbounds)
    real_binbounds
    binbounds = [[i for i in range(len(real_binbounds[a]))]
                 for a in len(real_binbounds)]
    return (hist, binbounds)


def countsFromExperimentClass(experiment, pmtlabel):
    """Reset Histogram from an ExperimentClass object"""
    # Create array of counts
    counts = []
    for i in experiment.triallist:  # loop through trials
        tempcounts = []
        try:
            for p in pmtlabel:  # for each pmtlabel (for multiion hist)
                try:
                    # Get count associated with measurement
                    # Assumes only 1 measurement of type pmtlabel
                    tempcounts.append(i.detresults[i.detnames.index(p)])
                except ValueError:  # label doesn't exist in this trial
                    continue
        except TypeError:
            try:
                # Assumes only 1 measurement of type pmtlabel
                tempcounts.append(i.detresults[i.detnames.index(pmtlabel)])
            except ValueError:  # label doesn't exist in this trial
                continue
        # Check for negative counts?
        if np.any(np.array(tempcounts) < 0):
            print(tempcounts)
        counts.append(tempcounts)  # this might be wrong (for multi)
    return counts


def bright_fiducial_error(eps_array=2*np.logspace(-5, -3, 50),
                          numrefs=10, N=2, **kwargs):
    """Return series of input states for all reference histograms scanned
    over some kind of error model (in this case, symmetric error on
    1 qubit bright state preparation)

    Input:
        eps_array       -- array of single qubit error on
                           bright state preparation
    Output:
        modified_states -- list of lists of modified density matrices for
                           each reference histogram
    """
    modified_states = []
    for eps in eps_array:
        one_qubit_error = np.diag([eps, 1-eps])
        error = np.kron(one_qubit_error, one_qubit_error)
        for i in range(N-2):
            error = np.kron(error, one_qubit_error)
        states_for_eps = []
        for i in range(numrefs):
            states_for_eps.append(error)
        modified_states.append(states_for_eps)
    return modified_states


