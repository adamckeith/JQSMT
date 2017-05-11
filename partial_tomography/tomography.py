"""Tomography functions. Performs PQSTMLE but does not require other PQSTMLE
classes

Author = Adam Keith

Most of this code translated from Scott Glancy's Matlab tomography project.
Some features from that code are missing here.

Problems:
Python solvers can't handle Nans or Inf in gradient? Start with mixed q
Print warnings for max iterations
Failure at large number of bins
Large memory use because each experiment $i$ has its own rho
"""
import copy
import numpy as np
from scipy import optimize as opt

class Tomographer(object):
    """Performs PQSTMLE"""

    def __init__(self, initial_q, initial_rho, hists, state_label, P_ij):
        """Initialize tomography

        Input:
            initial_q    - Initial POVM mixture coefficients or
                            subspace distributions
            initial_rho  - Initial density matrices
            hists        - List of 1D histograms (first index labels setting)
            unitaries    - Unitaries that rotate PI projectors
            state_label  - Distinguishes between references and data histograms
                         - ASSUMES 0 is for references
            state_map    - Maps computational basis states to subspaces
        """
        self.dim = P_ij.shape[-1]            # dimension of density matrix
        self.num_hists = P_ij.shape[0]       # number of histograms
        self.num_subs = P_ij.shape[1]        # number of subspaces
        self.P_ij = P_ij
        self.bins = hists[0].size            # number of bins (assumes all
                                             # histograms binned equally)
        self.hists = np.array(hists)         # Histograms
        self.state_label = state_label       # label of histogram type
        self.select_subsets()                # indices for each inferred rho
        
        self.model_indep_loglike()    # loglike with empirical frequencies
        self.iterations_q = 0         # Current number of q iterations
        self.iterations_rho = []      # number of rho iterations
        self.max_iters_rho = 10000        # maximum number of iterations
        self.max_iters_q = 10000     # maximum number of iterations
        self.stop_q = 0.25           # stopping condition on q
        self.stop_rho = 0.3           # stopping condition on rho
        self.stop_full = self.stop_q + self.stop_rho
        self.rho = initial_rho        # Current estimates of density matrix
        self.q = initial_q            # Current estimate of POVM mixture coeffs
        self.initial_q = initial_q
        #self.q = np.ones((self.num_subs, self.bins))/self.bins
        self.update_pops()
        self.update_POVM()
        self.make_R_RrhoR()

    def copy(self):
        """Deep copy a tomography object."""
        return copy.deepcopy(self)

    def update_POVM(self):
        """Calculate POVMs"""
        self.F = np.sum(self.q[np.newaxis, ..., np.newaxis, np.newaxis] *
                        self.P_ij[:, :, np.newaxis, ...], 1)
        # Force hermiticity?
        # Add assertions that POVM is physical?

    def check_POVM(self):
        """Assert that POVM is physical"""

        # Assert POVM is hermitian
        assert np.all(np.transpose(self.F.conj(), axes=(0, 1, 3, 2)) == self.F)

        # Assert POVM is positive semidefinite
        assert np.all(np.array([np.linalg.eigvalsh(
                      self.F.reshape((-1, self.dim, self.dim))[i, ...])
                      for i in range(self.bins*self.num_hists)]) >
                      (-100*np.finfo(float).eps))

        # Assert each POVM sums to identity
        assert np.all(np.around(np.sum(self.F, 1), decimals=12) ==
                      np.eye(self.dim))

    def update_pops(self):
        """Calculate populations for all histograms"""
        self.pops = np.real(np.trace(np.einsum('iak,ijkb->ijab', self.rho,
                                               self.P_ij), axis1=-1, axis2=-2))

    def estimate_q(self):
        """Advance by an expectation maximization iteration"""
        self.iterations_q += 1
        x0 = self.q.flatten()
        # Box constraints - all elements are between 0 and 1
        bounds = [(0, 1) for i in range(self.q.size)]

        # Construct constraints (sum of probabilities add to 1)
        constraints = []

        for i in range(self.num_subs):
            def f(x, ind=i):
                temp = x.reshape((self.num_subs, self.bins))
                return np.sum(temp[ind])-1
            constraints.append({'type': 'eq', 'fun': f})

        res = opt.minimize(Tomographer.loglikelihood_q, x0, (self,),
                           method='SLSQP', jac=True,
                           bounds=bounds,
                           constraints=constraints,
                           tol=1.0e-9, options={'maxiter': 1000,
                                                'disp': False})

        self.q = res.x.reshape((self.num_subs, self.bins))
        self.q = self.q/np.sum(self.q, 1)[:, np.newaxis]
        self.update_POVM()
        return self.loglikelihood()

    def loglikelihood_q(x, this):
        """Compute -loglikelihood for PQSTMLE model (returns positive value
        for solver to minimize)"""
        x = x.reshape((this.num_subs, this.bins))
        prob = this.pops.dot(x)   # probability of observing counts for hist
        L = this.hists*np.log(prob)
        #L = this.hists*np.log(this.pops.dot(x))
        #L = np.sum(L[np.isfinite(L)])  # drop terms that have no counts
        L = np.sum(L[np.nonzero(this.hists)])  # drop terms that have no counts
        #HIC = this.hists/this.pops.dot(x)
        HIC = this.hists/prob
        HIC[this.hists == 0] = 0  # erase infs and nans if histogram is 0
        #HIC[~np.isfinite(HIC)] = 0  # erase infs and nans if histogram is 0
        grad = np.transpose(this.pops).dot(HIC)  # p_ji * H_ic = grad(q_jc)
        grad = grad.flatten()
        #grad[~np.isfinite(grad)] = 0
        return (-(L - this.loglike_model_indepedent), -grad)
        #return -(L - this.loglike_model_indepedent)        

    def estimate_rho(self):
        """Advance by an R rho R iteration"""
        self.iterations_rho[-1] += 1
        self.make_R_RrhoR()
        for k in range(self.num_rhos):
            rho = self.R[k].dot(
                  self.rho[self.data_ind[k][0]]).dot(self.R[k])
            rho = rho/np.trace(rho)
            rho = self.fix_rho(rho)

            # update all "rhos" for this density matrix
            for i in self.data_ind[k]:
                self.rho[i] = rho
        self.update_pops()
        return self.loglikelihood()

    def full_tomography(self):
        """Alternates between iterations estimating q and rho"""
        self.loglikelihood_list = []
        while self.stopping_criteria_full() > self.stop_full and \
                self.iterations_rho+self.iterations_q < \
                self.max_iters_rho*self.max_iters_q:
            #print(self.final_stop)
            self.loglikelihood_list.append(self.estimate_q())
            #print(self.loglikelihood_list[-1])
            self.loglikelihood_list.append(self.estimate_rho())
            #print(self.loglikelihood_list[-1])
        print('Is loglikelihood list monotonic? ', monotonic(self.loglikelihood_list))
        return np.array(self.loglikelihood_list)

    def iterative_tomography(self):
        """Alternates between fully estimating q and rho"""
        self.loglikelihood_list = []
        cond = True
        if self.num_rhos == 0:
            self.loglikelihood_list.append(self.estimate_q())
            self.est_rho_final = 0 
            return self.stopping_criteria_q()
        while self.stopping_criteria_q() > self.stop_q and \
                self.iterations_q < self.max_iters_q:
            self.loglikelihood_list.append(self.estimate_q())
            self.iterations_rho.append(0)
            while cond and self.iterations_rho[-1] < self.max_iters_rho:
                self.loglikelihood_list.append(self.estimate_rho())
                if self.stopping_criteria_rho() < self.stop_rho:
                    cond = False
            cond = True
        self.final_stop = self.stopping_criteria_full()
        #print(self.loglikelihood_list[-1], monotonic(self.loglikelihood_list),
        #      self.final_stop, self.iterations_q, self.iterations_rho)
        #print(self.stopping_criteria_rho(), self.stopping_criteria_q())
        if self.final_stop >= self.stop_full:
            print('Warning: total stopping criteria not met', 
                  self.iterations_q,
                  np.max(self.iterations_rho))
        # print a warning for max iterations
        #print(len(loglikelihood_list))
        #print('Is loglikelihood list monotonic? ', monotonic(self.loglikelihood_list))
        # Update parameters to make sure everything is consistent
        self.update_pops()
        self.update_POVM()
        self.est_rho_final = self.rho[[self.data_ind[k][0]
                                    for k in range(self.num_rhos)]]
        return self.final_stop

    def model_indep_loglike(self):
        """Compute loglikelihood for empirical frequencies"""
        np.seterr(all='ignore')
        L = self.hists*np.log(self.hists/np.sum(self.hists, 1)[:, np.newaxis])
        L = np.sum(L[np.nonzero(self.hists)])  # drop terms that have no counts
        self.loglike_model_indepedent = L
        
    def loglikelihood(self):
        """Compute loglikelihood for PQSTMLE model"""
        L = self.hists*np.log(self.pops.dot(self.q))
        #print(np.nonzero(self.hists))
        L = np.sum(L[np.nonzero(self.hists)])  # drop terms that have no counts
        return L - self.loglike_model_indepedent

    def select_subsets(self):
        """Find POVMs and histograms for unique density matrices to be
        inferred"""
        data_labels = np.unique(self.state_label)  # labels for groups of hists
        data_labels = data_labels[data_labels!=0]  # only want data labels
        self.num_rhos = data_labels.size           # number of rho to estimate
        self.data_ind = [np.where(self.state_label == data_labels[k])[0]
                         for k in range(self.num_rhos)]
        self.data_measurements_total = [np.sum(self.hists[self.data_ind[k]])
                                        for k in range(self.num_rhos)]

    def make_R_RrhoR(self):
        """Make R matrix for R rho R for each rho to be estimated"""
        R = []
        HIC = self.hists/self.pops.dot(self.q)
#        HIC[~np.isfinite(HIC)] = 0  # erase infs and nans if histogram is 0
        HIC[self.hists == 0] = 0  # erase infs and nans if histogram is 0
        for ind in self.data_ind:  # for each rho to estimate
            R.append(np.sum(HIC[ind][..., np.newaxis, np.newaxis]*
                     self.F[ind], (0, 1)))
        self.R = np.array(R)

    def stopping_criteria_full(self):
        self.final_stop = self.stopping_criteria_rho() + \
                          self.stopping_criteria_q()
        return self.final_stop
        
    def stopping_criteria_rho(self):
        """Stopping criteria for R rho R algorithm"""
        # This is too good for some reason?
        stop = []
        for k in range(self.num_rhos):
            stop.append(np.real(np.max(np.linalg.eigvals(self.R[k])) - \
                                self.data_measurements_total[k]))
        return np.max(stop)
        
    def stopping_criteria_q(self):
        """Linear program to estimate possible improvement when optimizing q"""
        stop = []
        Aeq = np.zeros((self.num_subs, self.q.size))
        for j in range(self.num_subs):
            Aeq[j,j*self.bins:(j+1)*self.bins] = np.ones(self.bins)
        beq = np.ones(self.num_subs)
        bounds = [(0, 1) for i in range(self.q.size)]
        L, grad = Tomographer.loglikelihood_q(self.q.flatten(), self)
        #grad = np.nan_to_num(grad)
        try:
            res = opt.linprog(grad, A_ub=None, b_ub=None, A_eq=Aeq, b_eq=beq,
                          bounds=bounds, method='simplex', callback=None,
                          options=None)
            stop = -grad.dot(res.x-self.q.flatten())
        except:
            print('Linear program failed?')
            stop = self.stop_q+1
        return stop
    
    def fix_rho(self, rho):
        """Make sure rho is positive hermitian"""
        rho = (rho.conj().T + rho)/2  # make rho hermitian
        
        # Force positivity
        try:
            min_eig = np.min(np.linalg.eigvalsh(rho))
        except:
            print(rho)
        if min_eig < 0:
            rho = rho - min_eig*np.eye(self.dim)
            rho = rho/(1-min_eig*self.dim)

        # Force trace 1
        rho = rho/np.trace(rho)
        return rho

def monotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)
#    def make_R_EM(self):
#        """Make R matrix for R rho R for each rho to be estimated"""
#        R = []
#        max_imag_tr = []
#        for ind in self.state_ind:  # for each rho to estimate
#            num = list(map(np.multiply,
#                           self.F[ind].reshape((-1, self.dim, self.dim)),
#                           self.hists[ind].flatten()))
#            den = list(map(np.trace,
#                           self.F[ind].reshape((-1, self.dim, self.dim)).dot(
#                           self.rho[ind[0]])))
#            max_imag_tr.append(np.max(np.imag(den)))
#            temp_R = np.sum(list(map(np.divide,num,den)), axis=0)
#            #temp_R = (temp_R.conj().T + temp_R)/2  # force hermiticity in R
#            R.append(temp_R)
#        self.R = np.array(R)

#    def checkPOVMS(self, B_c):
#        """Check POVMS to make sure they are physical"""
#        
#        self.F[i, c, :, :] 
#        all_good = True
#        bins = np.prod(self.bins)  # number of elements of flattened hist
#        for k in range(len(B_c)):
#            sumB = 1j*np.zeros((self.dim, self.dim))
#            for c in range(bins):
#                if not np.all(B_c[k][c] ==
#                              B_c[k][c].conj().T):
#                    print('POVM element is not hermitian', k, c)
#                    all_good = False
#
#                # Check if POVMS are positive semidefinite
#                vals = np.linalg.eigvalsh(B_c[k][c])
#                if np.any(vals < -100*np.finfo(float).eps):
#                    print('POVM element is not positive semidefinite', k, c)
#                    print(np.min(vals))
#                    all_good = False
#                sumB += B_c[k][c]
#
#            # Check if POVMS sum to 1 for each measurement
#            if not np.all(np.around(sumB, decimals=12) == np.eye(self.dim)):
#                print('POVM elements do not sum to identity', k)
#                print(sumB)
#                #print(sumB-np.eye(self.dim))
#                all_good = False
#        return all_good