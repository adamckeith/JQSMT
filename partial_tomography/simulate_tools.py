"""Functions for simulating analysis on qutrits

Author = Adam Keith
 
Assumes |down> = |0>, |up> = |1> 
        |dark>        |bright>    

NOTE THIS IS DIFFERENT THAN ion_tools.py
"""
import numpy as np
from histogram import Hist


def generateExpectedCounts(states, background, countsPerState):
    """Generate linear list of expected counts for each pure state

    Input:
        states          -- Number of states
        background      -- background counts (only for darkest state)
        countsPerState  -- number of counts per state
    """
    expectedCounts = np.arange(0, states*countsPerState, countsPerState)
    expectedCounts[0] = background
    return expectedCounts


def fiducial(dim=4):
    """Make fiducial state. Assumes last state in binary
    ordering is prepared. Called the "bright state" """
    rho = np.zeros((dim, dim))
    rho = rho+0j
    rho[-1, -1] = 1.0
    return rho


def makeRho(dim=4, prob=1):
    """Make bell state density matrix + maximally mixed state of dimension dim

    rho = prob*(bell state density matrix) + (1-prob)*(maximally mixed state)
    """
    if dim == 2:
        state_vector = (1/np.sqrt(2))*np.array([1, 1])
    if dim == 4:
        # Standard 2 qubit bell state
        state_vector = (1/np.sqrt(2))*np.array([1, 0, 0, 1])
        #state_vector = (1/np.sqrt(2))*np.array([0, 1, 1, 0])
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
                     First theta is right most unitary.
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
        temp = R
        #temp = np.kron(R, R)
        #for i in range(N-2):
        for i in range(N-1):
            temp = np.kron(temp, R)
        U = temp.dot(U)
    return U


def nist_rotation(theta=np.pi/2, phi=0):
    """Construct NIST standard rotation matrix"""
    # Assumes |0>=|dark>, |1>=|bright>
    R = np.array([[np.cos(1/2*theta),
                  -1j*np.exp(-1j*phi)*np.sin(1/2*theta)],
                  [-1j*np.exp(1j*phi)*np.sin(1/2*theta),
                  np.cos(1/2*theta)]])
    return R


def rotation_error(theta_eps=0):
    """Small rotation between two states. Just rotate around phi=0 axis, X?"""
    R = np.array([[np.cos(1/2*theta_eps), -1j*np.sin(1/2*theta_eps)],
                  [-1j*np.sin(1/2*theta_eps), np.cos(1/2*theta_eps)]])
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

def one_qubit(fid=0.99, **kwargs):
    rho = []
    Us = []
    input_state = []
    N = 1
    dim = 2**N
    statemap, num_sub = symStateMap(N)
    
    # reference pulse parameters
    ref_phases = np.pi*np.array([0, 1/4, 1/2, 3/4, 1, 5/4, 3/2, 7/4])
    numrefs = ref_phases.size
    fid_state = fiducial(dim=dim)

    # References
    rho.append(fid_state)
    Us.append(np.eye(dim))
    input_state.append(rho[-1])
    for i in range(numrefs):
        rho.append(fid_state)
        ref_theta_list = np.array([np.pi/2, np.pi, np.pi/2])
        ref_phase_list = np.array([0, 0, ref_phases[i]])
        Us.append(pulseSequence(ref_theta_list, phi=ref_phase_list, N=N))
        input_state.append(rho[-1])
    
    # Non-references
    numdata = 2
    data_phases = np.linspace(0, 2*np.pi, numdata)
    #rho.append(makeRho(dim=dim, prob=fid))
    #Us.append(np.eye(dim))
    #input_state.append(1)
    for i in range(numdata):
        rho.append(makeRho(dim=dim, prob=fid))
        Us.append(pulseSequence([np.pi/2], phi=[data_phases[i]], N=N))
        input_state.append(1)
    return (rho, Us, input_state, statemap)
    

def two_qubit_sym_bell_state(fid=0.99, **kwargs):
    """ Simulate symmetric qubit bell state

    Input:
        fid    -- probability of bell state (fid*bell+(1-fid)*mixed)
    Output:
        rho    -- list of density matrice to simulate (logical basis)
        Us     -- list of unitaries
        state_label
        statemap
    """
    rho = []
    Us = []
    input_state = []
    N = 2
    dim = 2**N
    statemap, num_sub = symStateMap(N)
    P_j = np.zeros((num_sub, dim, dim))
    for j in range(num_sub):
        P_j[j, :, :] = np.diag(statemap[:, j])

    # reference pulse parameters
    #ref_phases = np.pi*np.array([0, 1/4, 1/2, 3/4, 1])#, 5/4, 3/2, 7/4])
    ref_phases = np.pi*np.array([0, 1/2, 1])#, 5/4, 3/2, 7/4])
    #ref_phases = np.pi*np.array([0, 1])#, 5/4, 3/2, 7/4])
    numrefs = ref_phases.size

    # add in fiducial error
    eps = 0
    error = np.diag([eps**2, eps*(1-eps), eps*(1-eps),
                     eps**2-2*eps])
    fid_state = fiducial(dim=dim) + error

    # References
    for i in range(numrefs):
        rho.append(fid_state)
        ref_theta_list = np.array([np.pi/2, np.pi, np.pi/2])
        ref_phase_list = np.array([0, 0, ref_phases[i]])
        Us.append(pulseSequence(ref_theta_list, phi=ref_phase_list, N=N))
        input_state.append(rho[-1])

    # Non-references
    numdata = 2
    data_phases = np.array([0, np.pi/2])
    #data_phases = np.linspace(0, 2*np.pi, numdata)
    rho.append(makeRho(dim=dim, prob=fid))
    Us.append(np.eye(dim))
    input_state.append(1)
    for i in range(numdata):
        rho.append(makeRho(dim=dim, prob=fid))
        Us.append(pulseSequence([np.pi/2], phi=[data_phases[i]], N=N))
        input_state.append(1)
        
#    # Non-references
#    state_vector = (1/np.sqrt(2))*np.array([0, 1, 1, 0])        
#    bell = np.outer(state_vector, state_vector)
#    bell = fid*bell + (1-fid)*np.identity(dim)/dim
#    numdata = 2
#    data_phases = np.array([0, np.pi/2])
#    #data_phases = np.linspace(0, 2*np.pi, numdata)
#    rho.append(bell)
#    Us.append(np.eye(dim))
#    input_state.append(2)
#    for i in range(numdata):
#        rho.append(bell)
#        Us.append(pulseSequence([np.pi/2], phi=[data_phases[i]], N=N))
#        input_state.append(2)

    return (rho, Us, input_state, P_j)


def two_qubit_asym_bell_state(fid=0.99, **kwargs):
    """ Simulate 2D histograms given a density matrix and unitaries

    Input:
        rho    -- list of density matrice to simulate (logical basis)
        Us     -- list of unitaries
        trials -- number of trials for each hist (can be a list)
        mu     -- average counts for each poisson distribution
        fid    -- probability of bell state (fid*bell+(1-fid)*mixed)
    """
    rho = []
    Us = []
    input_state = []
    N = 2
    dim = 2**N
    statemap = np.eye(dim)
    num_sub = dim

    # reference pulse parameters
    #ref_phases = np.pi*np.array([0, 1/4, 1/2, 3/4, 1, 5/4, 3/2])
    ref_phases = np.pi*np.array([0, 1/2, 1])#, 5/4, 3/2, 7/4])
    numrefs = ref_phases.size

    # add in fiducial error
    eps = 0
    error = np.diag([eps**2, eps*(1-eps), eps*(1-eps),
                     eps**2-2*eps])
    fid_state = fiducial(dim=dim) + error

    # References
#    ref_theta = np.pi*np.array([0,1])
#    for i in range(2):
#        for j in range(2):
#            rho.append(fid_state)
#            Us.append(pulseSequence_asymmetric([[ref_theta[i]], [ref_theta[j]]],
#                                           [[0], [0]], N=N))
#            input_state.append(rho[-1])
        
     #numrefs = 4 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
    for i in range(numrefs):
        rho.append(fid_state)
        ref_theta_list = np.array([np.pi/2, np.pi, np.pi/2])
        ref_phase_list = np.array([0, 0, ref_phases[i]])
        Us.append(pulseSequence(ref_theta_list, phi=ref_phase_list, N=N))
       # Us.append(pulseSequence_asymmetric([ref_theta_list, ref_theta_list],
       #                                    [ref_phase_list, ref_phase_list],
       #                                     N=N))
        #Us.append(pulseSequence_asymmetric(np.array([2*np.pi*np.random.rand(), 2*np.pi*np.random.rand()]), phi=np.array([2*np.pi*np.random.rand(), 2*np.pi*np.random.rand()]), N=N))
      #  Us.append(pulseSequence_asymmetric([2*np.pi*np.random.rand(3), 2*np.pi*np.random.rand(3)],
      #                                     [2*np.pi*np.random.rand(3), 2*np.pi*np.random.rand(3)], N=N))
        input_state.append(rho[-1])

    # Non-references
    numdata = 7
    data_phases = np.linspace(0, 2*np.pi, numdata)
    rho.append(makeRho(dim=dim, prob=fid))
    Us.append(np.eye(dim))
    input_state.append(1)
    for i in range(numdata):
        rho.append(makeRho(dim=dim, prob=fid))
        #Us.append(pulseSequence_asymmetric([[np.pi/2], [np.pi/2]], 
         #                       [[data_phases[i]],[data_phases[i]]], N=N))
        Us.append(pulseSequence([np.pi/2], phi=[data_phases[i]], N=N))
        input_state.append(1)
    return (rho, Us, input_state, statemap)


#def pulseSequence_asymmetric(theta, phi, N=2):
#    """Make N qubit unitary as a product of rotation unitaries
#    for theta along axis phi in the xy plane for each qubit
#
#    Input:
#           theta  -- angles to rotate (length equal to N)
#           phi    -- list of phase angles of rotation unitaries
#                     for each qubit
#           N      -- Number of qubits
#    """
#    theta = np.array(theta)
#    phi = np.array(phi)
#    assert theta.size == phi.size
#    temp = 1
#    for i in range(N):
#        t = theta[i]
#        p = phi[i]
#        R = nist_rotation(t, p)
#        temp = np.kron(temp, R)
#    return temp
    
def pulseSequence_asymmetric(theta, phi, N=2):
    """Make N qubit unitary as a product of rotation unitaries
    for theta along axis phi in the xy plane for each qubit

    Input:
           theta  -- angles to rotate for each qubit
           phi    -- list of phase angles of rotation unitaries
                     for each qubit
           N      -- Number of qubits
    """
    theta = np.array(theta)
    phi = np.array(phi)
    pulses = phi.shape[1]
    U = np.eye(2**N)
    for i in range(pulses):
        temp = 1
        for j in range(N):
            t = theta[j,i]
            p = phi[j,i]
            R = nist_rotation(t, p)
            temp = np.kron(temp, R)
        U = temp.dot(U)
    return U


def one_dimensional_poisson_hists(rho, Us, P_j, trials, mu=None,
                                  **kwargs):
    Us = np.array(Us)
    rho = np.array(rho)
    dim = rho.shape[-1]
    P_j = np.array(P_j)
    assert Us.shape[0] == rho.shape[0]
    num_sub = P_j.shape[0]  # number of subspaces
    numU = Us.shape[0]  # number of histograms

    P_ij = np.zeros((numU, num_sub, dim, dim))+0j
    for i in range(numU):
        for j in range(num_sub):
            P_ij[i, j, :, :] = Us[i].dot(P_j[j]).dot(Us[i].conj().T)
                                                
    pops = np.real(np.trace(np.einsum('iak,ijkb->ijab', rho, P_ij), 
                            axis1=-1, axis2=-2))

    trials = trials*np.ones(numU)
    if mu is None:
        mu = generateExpectedCounts(num_sub, 2, 35)

    hists = []
    for k in range(numU):
        #measrho = Us[k].dot(rho[k]).dot(Us[k].conj().T)
        hists.append(Hist())
        #diag = np.real(np.diag(measrho))
        #pops = np.sum(statemap.T*diag, axis=1)
        hists[-1].simPoisson(trials=trials[k], pops=pops[k], mu=mu)
    return hists


def n_dimensional_poisson_hists(rho, Us, statemap, trials, mu=None, **kwargs):
    """Multidimensional measurement outcomes. Only for 2 qubits right now"""
    Us = np.array(Us)
    rho = np.array(rho)
    statemap = np.array(statemap)
    assert Us.shape[0] == rho.shape[0]
    num_sub = statemap.shape[1]  # number of subspaces

    numU = Us.shape[0]  # number of histograms
    trials = trials*np.ones(numU)
    if mu is None:
        mu = generateExpectedCounts(num_sub, 2, 20)
        mu = np.tile(mu, (2, 1))

    hists = []
    for k in range(numU):
        measrho = Us[k].dot(rho[k]).dot(Us[k].conj().T)
        hists.append(Hist())
        diag = np.real(np.diag(measrho))
        pops = np.sum(statemap.T*diag, axis=1)
        hists[-1].simulate_multi(nspecies=2, trials=trials[k],
                       statepops=pops, mu=[[2, 20], [2, 20]])
    return hists


def qutrit_unitaries(phases=10, N=2, theta=np.pi/2):
    """Make N symmetric qutrit rotation unitaries on two of the three states.
    The third state is always the identity.

    Input: phase  -- list of phase angles of rotation unitaries
                     IF phase is an integer, make
                     phases = np.linspace(0, 2*np.pi, phases)
           N      -- Number of qubits
           theta  -- angle to rotate
    
    Assumes |1>=|bright>, |dark> = |0>,|2>"""
    if isinstance(phases, int):
        phases = np.linspace(0, 2*np.pi, phases)
    Us = []
    for phi in phases:
        R = 1j*np.zeros((3, 3))  # qutrit rotation
        R[0:2, 0:2] = nist_rotation(theta, phi)
        R[-1, -1] = 1
        temp = R
        for i in range(N-1):
            temp = np.kron(temp, R)
        Us.append(temp)
    return Us


def qutrit_rotation_error(theta_eps=0, N=2):
    """Make erorr unitary between bright state and 3rd trapped dark state
    Assumes |1>=|bright>, |dark> = |0>,|2>"""
    R = 1j*np.zeros((3, 3))  # qutrit rotation
    R[1:3, 1:3] = rotation_error(theta_eps)
    R[0, 0] = 1
    temp = R
    for i in range(N-1):
        temp = np.kron(temp, R)
    return temp


def two_qubit_matrix_to_trapped(U):
    """Map two qubit matrices (density matrices or unitaries)
    to a larger hilbert space with trapped states (qutrits)

    Assumes |1>=|bright>, |dark> = |0>,|2>"""
    dim = 4  # for two qubit unitaries
    new_block = np.insert(np.insert(U, 2, np.zeros(dim), axis=1),
                          2, np.zeros(dim+1), axis=0)
    new_U = np.zeros((3**2, 3**2))+0j  # expand to two qutrits
    new_U[0:5, 0:5] = new_block
    return new_U


def trapped_state_map(Nq=2):
    """Make state map for symmetric qutrits. Really, qubit + 1 trapped state.

    Input:
        Nq -- number of qutrits

    Output:
        f          -- the statemap
        num_subs   -- the number of subpsaces
    """
    
    num_subs = Nq+1
    num_pure = 3**Nq
    f = np.zeros((num_pure, num_subs))
    f[:,0] = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1])
    f[:,1] = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    f[:,2] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    return f, num_subs

#def trapped_state_map(Nq=2):
#    """Make state map for symmetric qutrits. Really, qubit + 1 trapped state.
#
#    Input:
#        Nq -- number of qutrits
#
#    Output:
#        f          -- the statemap
#        num_subs   -- the number of subpsaces
#    """
#    num_subs = Nq+1
#    num_pure = 3**Nq
#    f = np.zeros((num_pure, num_subs))
#    for i in range(num_pure):
#        for j in range(num_subs):
#            # find all states for same hamming weight j
#            if np.count_nonzero([int(num) for num in base(i, 3)]) == j:
#                f[i, j] = 1
#    return f, num_subs


def one_bright_fiducial_error(eps_array=2*np.logspace(-5, -3, 50), **kwargs):
    """Three reference sensitivity error. Dark, mixed (1 bright), bright
    references (for Ting Rei's error analysis)"""
    dim = 4
    modified_states = []
    for eps in eps_array:
        states_for_eps = []
        dark = np.zeros((dim, dim))
        dark = dark+0j
        dark[0, 0] = 1.0
        states_for_eps.append(dark)
        bright = fiducial(dim)
        states_for_eps.append(bright)
        one_qubit_error = np.diag([0.5-eps, 0.5+eps])
        error = np.kron(one_qubit_error, one_qubit_error)
        states_for_eps.append(error)
        modified_states.append(states_for_eps)
    return modified_states
    

def transfer_error(eps_array=2*np.logspace(-5, -3, 50), **kwargs):
    """Three reference sensitivity error. Dark, mixed (1 bright), bright
    references (for Ting Rei's error analysis)"""
    dim = 4
    modified_states = []
    for eps in eps_array:
        states_for_eps = []
        dark = np.zeros((dim, dim))
        dark = dark+0j
        dark[0, 0] = 1.0
        states_for_eps.append(dark)
        bright = fiducial(dim)
        states_for_eps.append(bright)
        one_qubit_error = np.diag([0.5-eps, 0.5+eps])
        error = np.kron(one_qubit_error, one_qubit_error)
        states_for_eps.append(error)
        modified_states.append(states_for_eps)
    return modified_states
    Us.append(qutrit_rotation_error(theta_eps=theta_eps))


def base(decimal, base):
    """Convert decimal to string in base base"""
    list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    answer = ""
    while decimal != 0:
        answer += list[int(decimal % base)]
        decimal /= base
    return answer[::-1]

 
def mixed_ref_fidelity_sensitivity(fid=0.99, **kwargs):
    """ Simulate 1D histograms for 2 qubit with 3 references"""
    sim_rhos = []
    Us = []
    input_state = []
    N = 2
    dim = 2**N
    statemap, num_sub = symStateMap(N)

    # Simulate states
    
    # dark reference
    dark = np.zeros((dim, dim))
    dark = dark+0j
    dark[0, 0] = 1.0
    sim_rhos.append(dark)
    Us.append(np.eye(dim))
    input_state.append(dark)
        
    # bright
    sim_rhos.append(fiducial(dim))
    Us.append(np.eye(dim))
    input_state.append(fiducial(dim))
    
    # 1 ion bright
    #eps = 1e-3
    eps = 0
    one_qubit_error = np.diag([0.5-eps, 0.5+eps])
    error = np.kron(one_qubit_error, one_qubit_error)
    for i in range(N-2):  
        error = np.kron(error, one_qubit_error)
    sim_rhos.append(error)
    Us.append(np.eye(dim))
    input_state.append(error)

    # Data, perfect
    numdata = 8
    data_phases = np.linspace(0, 2*np.pi, numdata)
    sim_rhos.append(makeRho(dim=dim, prob=fid))
    Us.append(np.eye(dim))
    input_state.append(2)
    for i in range(numdata):
        sim_rhos.append(makeRho(dim=dim, prob=fid))
        Us.append(pulseSequence([np.pi/2], phi=[data_phases[i]], N=N))
        input_state.append(2)
    return (sim_rhos, Us, input_state, statemap)


def mixed_ref_fidelity_sensitivity_trapped(fid=0.99, **kwargs):
    """ Simulate 1D histograms for 2 qutrits with 3 references"""
    BELL = makeRho(dim=4)
    BELL = two_qubit_matrix_to_trapped(BELL)
    sim_rhos = []
    Us = []
    input_state = []
    N = 2
    dim = 3**N
    statemap, purestates = trapped_state_map(N)
    theta_eps=0  # rotation error from trapped state 
    # Simulate states
    
    # dark reference
    dark = np.zeros((dim, dim))
    dark = dark+0j
    dark[0, 0] = 1.0
    sim_rhos.append(dark)
    input_state.append(dark)
    Us.append(qutrit_rotation_error(theta_eps=theta_eps))
    #Us.append(np.eye(dim))
    
    # bright
    sim_rhos.append(two_qubit_matrix_to_trapped(fiducial(2**N)))
    input_state.append(two_qubit_matrix_to_trapped(fiducial(2**N)))
    #Us.append(np.eye(dim))
    Us.append(qutrit_rotation_error(theta_eps=theta_eps))
    
    # 1 ion bright
    eps = 0
    one_qubit_error = np.diag([0.5-eps, 0.5+eps])
    error = np.kron(one_qubit_error, one_qubit_error)
    sim_rhos.append(two_qubit_matrix_to_trapped(error))
    input_state.append(two_qubit_matrix_to_trapped(error))
    #Us.append(np.eye(dim))
    Us.append(qutrit_rotation_error(theta_eps=theta_eps))

    theta_eps= 0.03 # rotation error from trapped state 
    #theta_eps= 0 # rotation error from trapped state 
    # Data, perfect
    numdata = 8
    data_phases = np.linspace(0, 2*np.pi, numdata)
    sim_rhos.append(two_qubit_matrix_to_trapped(makeRho(dim=2**N, prob=fid)))
    #Us.append(np.eye(dim))
    Us.append(qutrit_rotation_error(theta_eps=theta_eps))
    input_state.append(1)
    for i in range(numdata):
        sim_rhos.append(two_qubit_matrix_to_trapped(makeRho(dim=2**N, prob=fid)))
        U = qutrit_unitaries([data_phases[i]], N=2, theta=np.pi/2)[0]
        U = qutrit_rotation_error(theta_eps=theta_eps).dot(U)
        Us.append(U)
        input_state.append(1)
    return (sim_rhos, Us, input_state, statemap)

