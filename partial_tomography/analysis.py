"""Classes for collections of Hist() objects. Used to manipulate histograms
for partial quantum state tomography using maximum likelihood estimation.
(PQSTMLE)

Author = Adam Keith

Open Design Issues
    Requires Matlab 2015 engine or later
        use build a conda package from Matlab's setup.py
        Matlab requires YALMIP and OptiToolbox
    Everything is in camelCase (sorry didn't know about python convention until
        too late) ... starting to mix in underscore case.
        sorry for inconsistency
    Autosaves analysis object after tomography and/or bootstrap
    This code works for an arbitrary number of density matrices to estimate
        so remember that if you are only estimating one, its a list of one
        density matrix, i.e. self.estRho[0] is a numpy array with
        shape (dim, dim)

Future Changes
    A lot of redundant information in Tomographer
    Generalize Unitary measurements to general quantum operations
    Remove default functions for simulate()
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.linalg as linalg
#import matlab.engine
from scipy.stats import percentileofscore
import pickle
import datetime
import os
#import picos as pic
#import cvxopt as cvx
from histogram import *        # Import Hist() class and helper functions
from simulate_tools import *   # This is only to define default behavior of 
                               # simulate() and sensitivity_analysis()
from tomography import *       # Tomography code for PQSTMLE

# This doesn't work
#global rev_number = filter(str.isdigit, "$Revision$")   # SVN revision number
global rev_number   # SVN revision number
rev_number = filter(str.isdigit, "$Revision$")   # SVN revision number


class HistList(object):
    """A collection of Hist() objects

    This manages histograms having same binbounds (among other attributes
    that need to be the same)"""

    def __init__(self, hists=None, binflag=True):
        """Construct a HistList object from a list of histograms

        Input:
            hists     -- list of Hist() objects
            binflag   -- flag to automatically bin all histograms
                         in integer spacing (don't want to do this for
                         parameteric resampling)
        """
        if hists is not None:         # Hists given, do things
            self.hists = np.array(hists)  # list of Hist objects
            self.num_hists = len(hists)    # number of Hist objects
            self.confirmEqSpecies()
            if binflag is True:
                # Force all histograms to same binning
                HistList.rebin(self)

    def confirmEqSpecies(self):
        """Confirm  histograms have same number of nspecies. """
        temp = self.hists[0].nspecies
        for r in self.hists:
            if r.nspecies != temp:
                raise TypeError("Histograms are not all the "
                                "same number of species")
                return
        self.nspecies = temp    # Number of distinguishable species

    def findMaxCount(self):
        """Find largest maxcounts and smallest mincounts of all
        hists along each dimension."""
        maxcounts = []
        mincounts = []
        for r in self.hists:
            r.findMaxCount()  # update maxcounts of each histogram
            maxcounts.append(r.maxcounts)
            mincounts.append(r.mincounts)
        self.maxcounts = np.amax(maxcounts, 0)
        self.mincounts = np.amin(mincounts, 0)

    def convertListOfHistsToMatrix(self):
        """Convert hists from list of Hist objects and stack them
        vertically as numpy array"""
        if not np.array(self.hists).size:  # empty list
            return np.array([])
        hist_stack = [x.hist1D() for x in self.hists]
        hist_stack = np.vstack(hist_stack)
        return hist_stack

    def rebin(self, newbinbounds=None, bin_index=None):
        """Rebin histograms in the same way

        Input:
            newbinbounds  -- new bin boundaries to bin all reference hists
                                See Hist().rebin()
                             None -> bin all refs in each dimension by integers
                             from each dimension's mincount to maxcount
            bin_index     -- indices of self.hists that you would like to bin
                             default is all of the histograms
        """
        # Bin histogram along each axis from self.mincount to self.maxcount+2
        # mincount and maxcount in that dimension
        if newbinbounds is None:
            self.findMaxCount()
            newbinbounds = []
            for i in range(self.nspecies):
                newbinbounds.append(np.arange(self.mincounts[i],
                                              self.maxcounts[i]+2))
        if bin_index is None:
            bin_index = range(self.num_hists)

        # Do the binning
        for i in bin_index:
            self.hists[i].rebin(newbinbounds)
        self.binbounds = self.hists[0].binbounds  # pass up the binbounds
        self.mbins = self.hists[0].bins           # number of bins in each dim
        self.bins = np.prod(self.mbins)           # number of bins flattened
        return self.binbounds


class Experiment(HistList):
    """A collection of histograms that constitute an experiment with parameters
    that generated those histograms"""

    def __init__(self, hists=None, input_state=None, unitaries=None,
                 P_j=None, binflag=True, name=None):
        """Construct a reference object from a list of histograms and
        nominal populations

        Input:
            hists       -- list of Hist() objects
            input_state -- list of input density matrices (as numpy array)
                           before unitary was applied. If state is unknown,
                           use a natural number to label all histograms that
                           have the same initial state (0 is reserved for
                           references). Your labels will be changed to whole
                           numbers in ascending order starting with 1 (this
                           makes it difficult to manually add new histograms)
            unitaries   -- list of unitaries applied to input states
                           This is assumed to be known always (for now)
            P_j         -- list of subspace projector matrices
            binflag     -- flag to automatically bin all reference histograms
                           in integer spacing (don't want to do this for
                           parameteric resampling)
            name        -- string describing what this experiment is
        Assumes all input_states and unitaries have the same dimension and
            are square
        Assumes all histograms have the same number of distinguishable species
        Assumes all histograms with known input state are references
        Assumes hists correspond with input state and unitaries
        References are used for arbitrary number of states that need to be
        estimated
        This does not preserve the order of the input 

        Handle no reference histograms somehow
        """
        super().__init__(hists, binflag)
        # Don't do anything else because there isn't anything else to do
        if hists is None:
            return

        self.name = name         # name this experiment something useful
        if self.name is None:    # default is today's date
            self.name = str(datetime.date.today())

        self.input_state = input_state            # Preparation state, as list
        self.unitaries = np.array(unitaries)      # Analysis Unitaries
        self.P_j = np.array(P_j)                  # subspace projectors
        self.num_subs = self.P_j.shape[0]    # Number of subspaces
        self.dim = self.unitaries.shape[-1]       # Dimension of density matrix
        self.measurement_projectors()             # apply unitaries to projecto
        self.est_q = None                         # estimate of subspaces dists
        assert self.num_hists == len(input_state) == self.unitaries.shape[0]
        # label each histogram as reference or data (can distinguish different
        # input states for data). Reference have label 0
        self.state_label = copy.deepcopy(self.input_state)
        for i in range(self.num_hists):
            # I don't understand why this works syntactically
            # apparently requires self.input_state to be list of arrayNOT array
            if np.array(self.input_state[i]).size > 1:
                self.state_label[i] = 0
            elif self.state_label[i] == 0:
                raise ValueError("State Label 0 is reserved for reference "
                                 "histograms")
        self.input_state = np.array(self.input_state)
        self.state_label = np.array(self.state_label)
        self.distinguish_hists()
        if binflag:
            self.rebin()  # merging removes binning
            
    def measurement_projectors(self):
        """Make measurement projectors with measurement settings defined by
        unitaries"""
        # Rotated projectors (measurement with settings)
        self.P_ij = np.zeros((self.num_hists, self.num_subs,
                              self.dim, self.dim))+0j
        for i in range(self.num_hists):
            for j in range(self.num_subs):
                self.P_ij[i, j, :, :] = self.unitaries[i].dot(self.P_j[j]).dot(
                                                    self.unitaries[i].conj().T)
                                                    
    def reference_populations(self):
        """Calculate populations for reference histograms"""
        rho = np.array([self.input_state[self.ref_ind][k]
                        for k in range(self.num_refs)])
        self.pops = np.real(np.trace(np.einsum('iak,ijkb->ijab', rho,
                                               self.P_ij[self.ref_ind]),
                                               axis1=-1, axis2=-2))
    def distinguish_hists(self):
        """Distinguish between reference histograms and histograms from unknown
        initial states (data) and calculate nominal populations for reference
        histograms"""
        ### References ###
        self.ref_ind = np.where(self.state_label == 0)[0]
        self.num_refs = self.ref_ind.size
        if self.num_refs == 0:
            raise ValueError('No reference histograms == no tomography :(')
            
        # Calculate nominal populations for references
        self.reference_populations()

        ### Non-references (data) ### 

        # Relabel state_labels so that data labels are in ascending order    
        labels = np.unique(self.state_label)
        temp = np.arange(labels.size)
        # now, subtracting 1 from state_label will give index on rhos IF DATA
        self.state_label = np.array([temp[labels == self.state_label[i]][0] 
                            for i in range(self.num_hists)])

        # Data indices regardless of unknown density matrix
        self.data_ind = np.nonzero(self.state_label)[0]
        data_labels, num = np.unique(self.state_label, return_counts=True)
        self.num_data_each = num[data_labels != 0]  # number of hists for each 
        self.num_data = np.sum(self.num_data_each)  # total data hists
        data_labels = data_labels[data_labels != 0]  # only data labels
        self.num_rhos = data_labels.size       # number of rho to estimate

        # For each unique rho, index all data histograms (Matlab, phase out)
        data_state_label = self.state_label[self.data_ind]
        self.data_ind_data = [np.where(data_state_label == data_labels[k])[0]
                              for k in range(self.num_rhos)]
#        # For each unique rho, index all histograms
#        self.data_ind_all = [np.where(self.state_label ==
#                             data_labels[k])[0]
#                             for k in range(self.num_rhos)]                               

    def rebin(self, newbinbounds=None, bin_index=None):
        HistList.rebin(self, newbinbounds=newbinbounds, bin_index=bin_index)
        if self.bins < self.num_subs:
            print('Warning: Fewer bins than distinguishable subspaces')
        # arbitrary rebinning may not be possible with estimated distributions
        # so quick reestimate the subspace distributions
        self.estimate_q_quick()
        return self.binbounds

    def autobin(self, nbins=None, strat='subspace'):
        """Bin reference histograms automagically.

        Use mutual information between reference states and observed counts
        with target state equal to the collection of all observed counts from
        references.

        For multi ion binning, bin each marginal distribution as a single ion
        then use binning along each dimension to form multidimensional
        histogram

        Unexpected behavior for 1 or fewer bins (fix this)
        Cannot auto bin for more bins than natural binning

        Input:
            nbins  -- number of bins final binning should have
                      (in each dimension)
            strat  -- 'subspace' or 'references' different strategies
                      to bin. 'subspace' estimates subspaces distributions,
                      and builds mixed state using them. 'references' builds
                      mixed state by adding all references together
        Output:
            mutinforatio -- ratio of mutual information of new binning to
                            no binning along each axis.
        """
        assert(strat is 'subspace' or strat is 'references')
        if self.num_refs == 0:
            return

        if nbins is None:
            if self.nspecies == 1:
                nbins = self.num_subs
            else:
                # this ensures that no matter how many levels each species has
                # we will have at least that many bins.
                nbins = int(np.ceil(np.log2(self.num_subs)))

        if isinstance(nbins, int):
            nbins = nbins*np.ones(self.nspecies, dtype=int)

        # return to no binning (well, uniform integer binning)
        self.rebin(bin_index=self.ref_ind)
        # rebinning estimates subspace distributions
        # if this failed, change strat to use reference histograms
        if self.est_q is None and strat is 'subspace':
            print('Since Q cannot be estimated, change binning strategy to '
                  'use reference histograms')
            strat = 'references'
        binboundslist = []  # only for self.nspecies > 1
        pc, pr, p_cgr, states = self.binning_strategy(strat)
        mutinforatio = []           # ratio of mutual info of unbinned to
                                    # binned along each dimension
        # Compute mutual info without any binning
        pc = pc.reshape(self.mbins)  # same shape as underyling histograms
        for n in range(self.nspecies):
            pc_n = marginal(pc, n)  # marginal probability for counts in mixed
            # marginal probability for counts given subspace
            p_cgr_n = np.array([marginal(p_cgr[i].reshape(self.mbins), n)
                                for i in range(states)])
            mutinforatio.append(mutualInfo(pc_n, pr, p_cgr_n))
            # for each dimension, bin marginals
            lb = self.mincounts[n]
            ub = self.maxcounts[n]+1
            bindivchoices = np.arange(lb+1, ub)
            binbounds = np.array([lb, ub])  # one bin to start off
            for bincount in range(2, nbins[n]+1):
                # Remove binbounds already used
                notused = np.in1d(bindivchoices, binbounds, invert=True)
                bindivchoices = bindivchoices[notused]
                # Add a new bin and calculate new mutual information
                infos = []
                for bindivider in bindivchoices:
                    tempbnds = np.append(binbounds, bindivider)
                    tempbnds = np.sort(tempbnds)
                    # Bin distributions
                    count_prob_bin = binDists(pc_n, tempbnds)
                    count_prob_givenR_bin = binDists(p_cgr_n, tempbnds)
                    infos.append(mutualInfo(count_prob_bin, pr,
                                            count_prob_givenR_bin))

                # find index of maximum mutual info
                ind = np.argmax(infos)
                newbindiv = bindivchoices[ind]  # corresponding divider
                binbounds = np.append(binbounds, newbindiv)  # permanent bins
                binbounds = np.sort(binbounds)
            binboundslist.append(binbounds)  # save binning on this dimension
            mutinforatio[-1] = np.amax(infos)/mutinforatio[-1]

        # Do the binning!
        self.rebin(binboundslist)
        return mutinforatio

    def binning_strategy(self, strat='subspace'):
        """Create probability distributions for binning depending on
        particular strategy

        Input:
            strat  -- 'subspace' or 'references' different strategies
                      to bin. 'subspace' estimates subspaces distributions,
                      and builds mixed state using them. 'references' builds
                      mixed state by adding all references together

        Assumes all reference histograms are binned the same."""
        self.bin_strat = strat
        if strat is 'references':
            # probability of ref
            pr = [self.hists[k].trials for k in self.ref_ind]
            # add all reference histograms and divide by total trials
            pc = np.zeros(self.mbins)
            for k in self.ref_ind:
                pc += self.hists[k].hist
            pc = pc/np.sum(pr)
            pr = pr/np.sum(pr)
            # probability of counts given reference histogram
            p_cgr = np.array([self.hists[k].hist/self.hists[k].trials
                              for k in self.ref_ind])
            states = self.num_refs
        elif strat is 'subspace':
            pc = self.makeState()    # probability of observing count, 1D mixed
            # probability of subspace
            pr = np.ones(self.num_subs)/self.num_subs
            p_cgr = self.est_q   # probability of counts given subspace
            states = self.num_subs
        return (pc, pr, p_cgr, states)

    def mutual_info_ratio(self):
        """Calculate mutual information ratio of unbinned histograms
        to histograms binned by autobin() according to strategy <strat>"""
        # Assumes autobin() has previously been called
        autobin_bounds = self.binbounds  # save autobins
        self.rebin()  # rebin to no bins to calculate original information
        pc, pr, p_cgr, states = self.binning_strategy(self.bin_strat)
        mutinforatio = []
        pc = pc.reshape(self.mbins)  # same shape as underyling histograms
        for n in range(self.nspecies):
            pc_n = marginal(pc, n)  # marginal probability for counts in mixed
            # marginal probability for counts given subspace
            p_cgr_n = np.array([marginal(p_cgr[i].reshape(self.mbins), n)
                                for i in range(states)])
            mutinforatio.append(mutualInfo(pc_n, pr, p_cgr_n))
            count_prob_bin = binDists(pc_n, autobin_bounds[n])
            count_prob_givenR_bin = binDists(p_cgr_n, autobin_bounds[n])
            autobin_info = mutualInfo(count_prob_bin, pr,
                                      count_prob_givenR_bin)
            mutinforatio[-1] = autobin_info/mutinforatio[-1]
        self.rebin(autobin_bounds)  # back to autobinbounds
        return mutinforatio

    def optBins(self):
        """Vary bin boundaries for a given binning to maximize mutual info"""
        pass

    def makeState(self, p_target=None):
        """Make 1D state distribution as a linear combination of
        estimated pure states.

        Input:
            p_target -- Desired populations. If None, equally mixed state
                        is generated
        Output:
            q_target -- Generated state probabilities for each count
        """
        if p_target is None:
            p_target = np.ones(self.num_subs)/self.num_subs

        # Use linear combination of estimated subspaces distributions to
        # build arbitrary state
        if self.est_q is None:
            self.estimate_q_quick()

        q_target = np.zeros(self.est_q.shape[1])
        for i in range(self.num_subs):
            q_target += p_target[i]*self.est_q[i]
        return q_target

    def estimate_q_quick(self):
        """Estimate subspace distributions from linear combination of
        reference relative frequencies (histogram/trials)"""
        H = self.convertListOfHistsToMatrix()[self.ref_ind]
        H = H/np.sum(H, 1)[:, np.newaxis]
        try:
            q = np.linalg.inv(np.transpose(self.pops).dot(
                self.pops)).dot(np.transpose(self.pops)).dot(H)
        except np.linalg.LinAlgError:  # probably indistinguishable pure states
            print('Warning: P is singular, cannot estimate Q by inversion.')
            return None
        #q[q < 0] = 1e-9  # "zero" out negative probabilities (make small)
        q[q < 0] = 0  # "zero" out negative probabilities (make small)
        q += 1e-9  # "zero" out negative probabilities (make small)
        q = q/np.sum(q, 1)[:, np.newaxis]
        self.est_q = q
        return self.est_q


class AnalysisSet(Experiment):
    """A collection of histograms that constitute an experiment with a
    maximum likelihood algorithm protocol which can estimate
    the parameters that generated those histograms

    seed      -- seed for the RNG
    measure   -- function that 'measures' inferred density matrices
                    (only takes two arguments, rho, sigma density matrices)
                 First argument is target density matrix, second argument
                 is estimated density matrix
    targets   -- list of target density matrices, same size as unknown density
                 matrices
    autosave  -- if True, AnalysisSet will save itself after tomography,
                 bootstrap, and senstivity analysis
    """

    def __init__(self, hists=None, input_state=None, unitaries=None,
                 P_j=None, trainFrac=0.1, binflag=True,
                 name=None, seed=None, measure=None, targets=None,
                 autosave=True):
        self.eng = None             # Matlab Engine
        self.set_seed(seed)         # Set Seed (regardless if any hists)
        self.autosave = autosave    # Auto save flag
        self.fidelity = []          # inferred expectations
        if measure is None:
            measure = fidelity      # Use fidelity function defined below
        self.measure = measure      # function to measure density matrices
        self.targets = targets      # list of target density matrices

        super().__init__(hists, input_state, unitaries, P_j, binflag,
                         name)
        if hists is not None:
            self.bootstrap_log = []     # Log of bootstrap resamples
            self.estRho = None          # Estimated density matrices for data
            self.trainFrac = trainFrac  # approximate training fraction
            self.trainRef = None        # Training set Experiment object
            self.trainingSample(self.trainFrac)  # Calculate training set
            if self.targets is not None:
                self.targets = np.array(self.targets)
                assert(len(self.targets) == self.num_rhos or
                       self.num_rhos == 0)

    def set_seed(self, seed=None):
        if seed is None:  # if none, assign it so it will have a value to save
            seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)        # Seed the RNG
        self.seed = seed            # save the seed used

    def trainingSample(self, trainFrac=0.1):
        """Sample from references without replacement to generate training data

        Training removes any previous binning.
        """
        self.trainFrac = trainFrac  # approximate fraction
        # This should not be run if no traning fraction
        if (self.trainFrac <= 0 or self.trainFrac >= 1):
            return None

        training_refs = []
        for i in range(self.num_refs):
            training_refs.append(Hist())
            ref_ind = self.ref_ind[i]
            trainTrial = int(np.floor(self.trainFrac*self.hists[ref_ind].trials))
            counts = makeCounts(self.hists[ref_ind].rawhist,
                                self.hists[ref_ind].rawhistbins)
            # random integers without replacement as indices to counts
            train_ind = np.random.choice(self.hists[ref_ind].trials,
                                         trainTrial, replace=False)
            hist_ind = np.setdiff1d(np.arange(
                                    self.hists[ref_ind].trials),
                                    train_ind)
            # Make both histograms (remake original with less counts)
            orig_hist = self.hists[ref_ind].hist       # Get original histogram
            orig_bins = self.hists[ref_ind].binbounds
            training_refs[i].makeFromCounts(counts[train_ind])
            self.hists[ref_ind].makeFromCounts(counts[hist_ind])
            training_refs[i].rebin(orig_bins)
            self.hists[ref_ind].rebin(orig_bins)
            assert np.all(orig_hist ==
                          self.hists[ref_ind].hist + training_refs[i].hist)

        # Return to binning for remaining reference histograms
        self.rebin()

        # Make training samples an experiment object
        input_state = [np.array(self.input_state[self.ref_ind[k]]) for k in 
                        range(self.num_refs)]
        self.trainRef = Experiment(training_refs, input_state,
                                   self.unitaries[self.ref_ind], self.P_j)

    def autobin(self, nbins=None):
        """Bin reference histograms automagically. See Experiment.autobin()

        If trainFrac is > 0, use training data to find bins
        training sample may cause fringe effects with large number of bins

        Input:
            nbins  -- number of bins final binning should have
                      (in each dimension)
        Output:
            mutinforatio -- ratio of mutual information of new binning to
                            no binning along each axis."""
        if self.trainFrac > 0:
            # use training data to bin
            self.trainRef.autobin(nbins)
            self.bin_strat = self.trainRef.bin_strat
            # Training data may not have outer bounds correct, manually fix
            for i in range(self.nspecies):
                self.trainRef.binbounds[i][0] = self.mincounts[i]
                self.trainRef.binbounds[i][-1] = self.maxcounts[i]+1
                # Bin everything in the same way
            self.rebin(self.trainRef.binbounds)
        else:
            super().autobin(nbins)
        return self.mutual_info_ratio()

    def copy(self):
        """Deep copy an Analysis object. Copies share matlab engines"""
        temp_eng = self.eng
        self.eng = None
        SE = copy.deepcopy(self)  # deepcopy can't copy matlab engines
        SE.eng = temp_eng
        self.eng = temp_eng
        return SE

    def sensitivity_analysis(self, error_model=None, **error_kwargs):
        """Perform sensitivity analysis on reference preparation. This adds
        errors on the reference fiducial states and reruns the analysis.

        Input:
            error_model         -- error model function that returns:
                modified_states -- list of lists of modified density matrices
                                   for each reference histogram
            error_kwargs        -- arguments to be passed to error_model()
        error_model() should always have **kwargs argument in case.
        Returns a list of inferred density matrices for each series
        of modified input density matrices from error_model.
        Assumes histograms were not merged
        """
        self.sensitivity_rhos = []   # reinferred density matrices for each
                                     # error step
        self.sensitivity_fids = []   # measure according to self.measure()
                                     # for each error step and density matrix
        # Compute modified input states
        if error_model is None:
            error_model = bright_fiducial_error
        modified_states = error_model(**error_kwargs)

        # Run the sensitivity analysis
        SE_list = []
        SA = self.copy()
        for states in modified_states:
            SA.input_state[SA.ref_ind] = states
            SA.distinguish_hists()  # recalculate nominal populations
            #SA.tomographyML()       # rerun tomography
            #SA.tomography()       # rerun tomography
            log_log, fids, stability_list = SA.tomography_stability(20)
            self.sensitivity_rhos.append(SA.estRho)
            if self.targets is not None:
                self.sensitivity_fids.append([self.measure(self.targets[i],
                                              SA.estRho[i])
                                              for i in range(self.num_rhos)])
            SE_list.append(SA.copy())
        return np.array(SE_list)        
#        # Measure targets
#        if self.targets is not None:
#            for rhos in self.sensitivity_rhos:
#                fids = []
#                for i in range(self.num_rhos):
#                    fids.append(self.measure(self.targets[i], rhos[i]))
#                self.sensitivity_fids.append(fids)
#        self.auto_save()   # autosave
#        return (self.sensitivity_fids, self.sensitivity_rhos)

    def startMatlabEng(self):
        """Start (or restart) Matlab Engine"""
        if self.eng is None:
            print('Starting Matlab Engine...          ', end="")
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(
                self.eng.genpath(os.path.dirname(os.path.realpath(__file__))))
            print('[[DONE]]')
        return self.eng

    def killMatlabEng(self):
        print('Killing Matlab Engine')
        self.eng.quit()
        self.eng = None

    def plotPureDists(self, show=True):
        """Plot distribution of each pure state on top of each other"""
        if self.est_q is None:
            self.estimate_q_quick()
        plt.plot(np.transpose(self.est_q))
        if show:
            plt.show()
            plt.draw()

    def nonParametricResample(self):
        """Resample all histograms. Return new AnalysisSet object"""
        new_hists = []
        for i in range(self.num_hists):
            new_hists.append(Hist())
            new_hists[-1].makeFromHist(self.hists[i].resample(),
                                       self.hists[i].rawhistbins)
        resampledSuper = AnalysisSet(new_hists, self.input_state,
                                     self.unitaries, self.P_j,
                                     trainFrac=0, binflag=False,
                                     measure=self.measure,
                                     targets=self.targets, autosave=False)
        # share matlab engine if it exists for expecation SDP
        resampledSuper.eng = self.startMatlabEng() \
                             if self.eng is not None else None
        return resampledSuper

    def parametricResample(self):
        """Resample all histograms using inferred binned measurable
        state distributions and inferred density matrix

        Note: this does not preserve the number of distinguishable ions nor
        can these histograms be rebinned to more bins.

        Return new AnalysisSet object
        """
        new_hists = []
        for i in range(self.num_hists):
            # use populations estimated from tomography to calculate
            # probability of observing of each "count"
            prob = self.tom.pops[i].dot(self.est_q)
            new_hists.append(Hist())
            hist = np.random.multinomial(self.hists[i].trials, prob)
            new_hists[-1].makeFromHist(hist, self.hists[i].binbounds)
            new_hists[-1].maxcounts = self.hists[i].maxcounts
            new_hists[-1].mincounts = self.hists[i].mincounts
        resampledSuper = AnalysisSet(new_hists, self.input_state,
                                     self.unitaries, self.P_j,
                                     trainFrac=0, binflag=False,
                                     measure=self.measure,
                                     targets=self.targets, autosave=False)
        # share matlab engine if it exists for expecation SDP
        resampledSuper.eng = self.startMatlabEng() \
                             if self.eng is not None else None
        resampledSuper.binbounds = self.binbounds
        resampledSuper.mbins = self.mbins
        resampledSuper.bins = self.bins
        return resampledSuper

    def bootstrap(self, method='parametric', iters=10):
        """Bootstrap to estimate variances for estimators

        Input:
            method    -- 'nonparametric' or 'parametric'
            iters     --  number of resamples

        Output:
            est_q         -- list of estimated reference distributions
            estRhos       -- list of estimated density matrices for each
                             resample
            bootstrap_log -- tomographer list for each resample

        Bootstrap assumes targets for calling object same as bootstrap
        resamples"""
        self.bootstrap_log = []   # Bootstrap Log
        self.bootstrap_fidelities = []
        self.boot_iters = iters
        rho_start = None
        for i in range(self.boot_iters):
            if method == 'nonparametric':
                resampledSuper = self.nonParametricResample()
                resampledSuper.rebin(self.binbounds)  # use prebinned bins
            elif method == 'parametric':
                resampledSuper = self.parametricResample()
            # Run tomography
            resampledSuper.tomography(rho_start)
            # modify tomographer to save space
            resampledSuper.tom.rho = []
            resampledSuper.tom.P_ij = []
         #   self.bootstrap_fidelities.append(
         #                               resampledSuper.tomography(rho_start))
            self.bootstrap_log.append(resampledSuper.tom)
        #self.bootstrap_fidelities = np.array(self.bootstrap_fidelities)
        #return self.bootstrap_fidelities

    def bootstrap_analysis(self, measure=None):
        """Calculate statistics from bootstrap"""
        if measure is not None:
            self.measure = measure
        self.inferred_loglikelihood = self.tom.loglikelihood_list[-1]
        # Check if Bootstrapped final loglikelihoods are consistent with
        # inferred final loglikelihood
        self.final_loglikelihood = [self.bootstrap_log[i].loglikelihood_list[-1]
                                    for i in range(self.boot_iters)]
        self.bootstrap_fidelities = [[self.measure(self.targets[k],
                                      self.bootstrap_log[i].est_rho_final[k])
                                      for k in range(self.num_rhos)]
                                      for i in range(self.boot_iters)]
        self.bootstrap_fidelities = np.array(self.bootstrap_fidelities)
        self.likelihood_percentile = percentileofscore(self.final_loglikelihood,
                                                       self.inferred_loglikelihood)
        print('Bootstrap mean ', np.mean(self.bootstrap_fidelities), 
              'pm ', np.std(self.bootstrap_fidelities))
        print('16-84 percentile', np.percentile(self.bootstrap_fidelities, 16),
              np.percentile(self.bootstrap_fidelities, 84))     
        #np.median(self.bootstrap_fidelities)
        self.auto_save()         # autosave

    def summary(self):
        print('Inferred Fidelities ', self.fidelity)
        print('Bootstrap standard deviations ',
              np.std(self.bootstrap_fidelities, axis=0))

    def tomography_stability(self, iterations = 200):
        """Use random starting points for tomography solvers to estimate
        stability of solution."""
        self.stability_log_list = []
        self.stability_fids = []
        self.stability_tom_list = []
        initial_q = self.estimate_q_quick()
        if initial_q is None:
            initial_q = np.ones((self.num_subs, self.bins)) / self.bins
        for i in range(iterations):
            # Generate random starting point for q
            #q_start = np.random.rand(self.num_subs, self.bins)-0.5
            q_start = initial_q + \
                      (np.random.rand(self.num_subs, self.bins)-0.5)/10
            q_start[q_start < 0] = 0  # zero out non probabilities
            q_start += 1e-6*np.ones(self.bins)  # regularize
            #q_start[q_start > 1] = 1  # zero out non probabilities
            q_start = q_start/np.sum(q_start, 1)[:, np.newaxis]
            self.tomography(q_start = q_start)
            self.stability_log_list.append(self.tom.loglikelihood_list[-1])
            self.stability_tom_list.append(self.tom.copy())
            #self.tomographyML(q_start = q_start)
            #log_log.append(self.info['likelihoodlist'][0][-1])
            #self.save(str(i))
            for i in range(self.num_rhos):
                self.stability_fids.append(self.measure(self.targets[i],
                                                        self.estRho[i]))
        self.stability_fids = np.array(self.stability_fids)
        self.stability_log_list = np.array(self.stability_log_list)
        self.stability_tom_list = np.array(self.stability_tom_list)
        return (self.stability_fids, self.stability_log_list,
                self.stability_tom_list)

    def tomography(self, rho_start=None, q_start=None):
        """Simultaneously estimate measurable state distributions and the
        density matrices that produced data histograms using an iterative ML
        method. Uses tomography.py

        Input:
            rho_start -- list of density matrices for each data set
                         to start tomography at
            q_start   -- list of ideal subspace distributions to start
                         tomography
        """
        initial_rho = [self.input_state[self.ref_ind][i]
                       for i in range(self.num_refs)]
        if rho_start is None:
            # First step, use maximally mixed density matrix and estimate
            # measurable state distributions
            initial_rho.extend([np.identity(self.dim)/self.dim
                                for k in range(self.num_data)])
        else:
            # Convert rho_start list into full list for tomography
                
            initial_rho.extend([self.state_label[self.data_ind][k]-1
                                for k in range(self.num_data)])
        initial_rho = np.array(initial_rho)
        if q_start is None:
            # Initial guess for measurable state distributions
            q_start = self.estimate_q_quick()
            #initial_q = None
            # If linear inversion failed, make simple initial point
            if q_start is None:
                q_start = np.ones((self.num_subs, self.bins)) / self.bins
        initial_q = q_start
        hists = self.convertListOfHistsToMatrix()
        self.tom = Tomographer(initial_q, initial_rho, hists, self.state_label,
                               self.P_ij)
        self.tom.iterative_tomography()
        self.estRho = self.tom.est_rho_final
        self.est_q = self.tom.q
        #log = self.tom.full_tomography()
        # Calculate fidelities
        self.remeasure()
        self.auto_save()
        return self.fidelity

    def remeasure(self, measure=None):
        """Recalculate property from estimated density matrix using function
        measure, 2nd argument is estimated density matrices.

        Recalculates inferred density matrices as well as all resamples
        from bootstrap"""
        if measure is not None:
            self.measure = measure
        # Calculate fidelities
        if self.targets is not None:
            self.fidelity = []
            for i in range(self.num_rhos):
                self.fidelity.append(self.measure(self.targets[i],
                                                  self.estRho[i]))
            self.fidelity = np.array(self.fidelity)

        if self.bootstrap_log:
            self.bootstrap_fidelities = [[self.measure(self.targets[k],
                                          self.bootstrap_log[i].est_rho_final[k])
                                          for k in range(self.num_rhos)]
                                          for i in range(self.boot_iters)]
            self.bootstrap_fidelities = np.array(self.bootstrap_fidelities)

    def infer_expectation(self, operators):
        """Estimate minimum and maximum epectation values for a
        list of operators for each inferred density matrix.

        For each operator, a tuple of expectations are returned corresponding
        to the minimum expectation over possible density matrix
        consistent with the data, the expectation from the inferred density
        matrix, and the maximum. A semidefinite program calculates the minimum
        and maximum.

        If the bootstrap has been run then a tuple is returned for each
        resample

        operators is list of hermitian operators with each entry corresponding
        to each inferred density matrix
        """
        self.startMatlabEng()
        # make sure operators is a list
        operators = np.array(operators)
        if operators.ndim < 3:
            operators = np.expand_dims(operators, axis=0)
        #print(operators.shape)
        bound_list = []
        log = self.make_full_log()
        for tom in log:
            inferred_rho = tom.rho[[self.tom.data_ind[k][0]
                                   for k in range(self.num_rhos)], ...]
            all_povms = tom.F[self.data_ind]
            #all_povms = tom.P_ij[self.data_ind]

            for p in range(self.num_rhos):
                # Pick out corresponding povm elements and "flatten"
                #povms = self.svd_POVM(all_povms[self.data_ind_data[p]])
                povms = all_povms[self.data_ind_data[p]]
                povms = povms.reshape((-1, self.dim, self.dim))
                bounds = measureOperator(operators[p], inferred_rho[p],
                                         povms, self.eng)
                bound_list.append(np.array(bounds))
        return bound_list

    def make_full_log(self):
        """Create list of tomographer objects, first element is tomographer
        for data, all other elements are from bootstrap"""
        #full_log = copy.deepcopy(self.bootstrap_log)
        full_log = [self.tom] + self.bootstrap_log
        #full_log.insert(0, self.tom)
        return full_log

    def auto_save(self):
        """Auto save AnalysisSet object if auto save is flag is True.
        This is to avoid bootstrap resampled objects from saving themselves
        and slowing the bootstrap down.

        See AnalysisSet.save(....)"""
        if self.autosave is True:
            self.save('autosave_' + self.name)

    def save(self, filename):
        """Save all relevant information (mostly)

        filename.hist saves this entire object

        filename.tomo saves tomography information for inference and
        for bootstrap. The first element is the info for the inference
        and the remaining entries are the bootstrap."""
        #pickle.dump(self.make_full_log(), open(filename + '.tomo', 'wb'))
        temp = self.eng
        self.eng = None
        pickle.dump(self, open(filename + '.hist', 'wb'))
        self.eng = temp

    def load(self, filename):
        """Load AnalysisSet Object from .hist file"""
        pass
        #self = pickle.load(open(filename+'.hist', 'rb'))
        # this doesn't work right?

    def simulate(self, state_sim=None, hist_sim=None, trials=5000,
                 trainFrac=0.1, **simkwargs):
        """ Simulate 1D histograms given function that
        simulates density matrices and unitaries

        Input:
            state_sim       -- function that returns:
                rho         -- list of density matrice to
                               simulate (logical basis)
                Us          -- list of unitaries
                input_state -- list of input states (density matrices or
                               labels for unknown state, 0 for reference,
                               anything else for non-reference). This is a
                               bit redundant for references given rho, but
                               reflects the analyses knowledge of state
                               preperations
               P_j          -- list of subspace projectors
            hist_sim        -- function that takes the above and below
                               arguments and returns:
                hists       -- list of histogram objects

            trials    -- number of trials for each hist (can be a list)
            trainFrac -- fraction of trials used for training for each hist
            simkwargs -- keyword arguments to pass to state_sim and hist_sim
                         assumes no inconsistent overlap in keyword arguments
                         among those functions. Assumes these functions accept
                         variable keyword arguments (**kwargs)!
        """
        if state_sim is None:
            state_sim = two_qubit_sym_bell_state
            #state_sim = two_qubit_asym_bell_state

        if hist_sim is None:
            hist_sim = one_dimensional_poisson_hists
            #hist_sim = n_dimensional_poisson_hists

        rho, Us, input_state, P_j = state_sim(**simkwargs)
        #print(input_state)
        hists = hist_sim(rho, Us, P_j, trials, **simkwargs)

        # Setup this object properly
        self.__init__(hists, input_state, Us, P_j, trainFrac=trainFrac,
                      name='simulated_'+str(datetime.date.today()),
                      measure=self.measure, targets=self.targets)

    def svd_POVM(self, povms=None):
        """Caclulate SVD of POVM elements as vectors

        Input:
            povms -- list of a set of povm elements
                            shape = (?, bins, dim, dim)
        """
        tol = 0.5  # tolerance on singular values to remove
        #povm_matrix = povms.reshape((self.num_data_each[p]*self.bins,-1))
        povm_matrix = povms.reshape((-1, self.dim**2))
        u, s, vt = np.linalg.svd(povm_matrix)
        row_basis = vt[s>tol]
        return row_basis.reshape((-1, self.dim, self.dim))

    def infer_expectation_python(self, operators):
        """Measure list of operators for each inferred density matrix.

        For each operator, a tuple of expectations are returned corresponding
        to the minimum expectation over possible density matrix
        consistent with the data, the expectation from the inferred density
        matrix, and the maximum. A semidefinite program calculates the minimum
        and maximum.

        If the bootstrap has been run then a tuple is returned for each
        resample

        operator is hermitian
        """
        if operators.ndim < 3:
            operators = np.expand_dims(operators, axis=0)
        bound_list = []
        log = self.make_full_log()
        for tom in log:
            inferred_rho = tom.rho[[self.tom.data_ind[k][0]
                                   for k in range(self.num_rhos)], ...]
            #all_povms = tom.F[self.data_ind, ...]
            all_povms = tom.P_ij[self.data_ind, ...]

            for p in range(self.num_rhos):  # for each estimated rho
                EV = pic.Problem()
                C = pic.new_param('C', operators[p])
                rho_est = inferred_rho[p]
                RHO = EV.add_variable('RHO',(self.dim, self.dim),'hermitian')
                          
                # Add Constraints          
                EV.add_constraint(RHO >> 0) # density matrix is positive              
                EV.add_constraint('I'|RHO == 1) # density matrix has trace 1
                #povms = self.svd_POVM(all_povms[self.data_ind_data[p]])
                povms = all_povms[self.data_ind_data[p]]
                povms = povms.reshape((-1, self.dim, self.dim))
                
#                EV.add_constraint(('I'|(RHO-rho_est)*povms[0]) == 0)
#                EV.add_constraint(('I'|(RHO-rho_est)*povms[1]) == 0)
#                EV.add_constraint(('I'|(RHO-rho_est)*povms[2]) == 0)
#                EV.add_constraint(('I'|(RHO-rho_est)*povms[3]) == 0)
#                EV.add_constraint(('I'|(RHO-rho_est)*povms[4]) == 0)
#                EV.add_constraint(('I'|(RHO-rho_est)*povms[5]) == 0)
#                EV.add_constraint(('I'|(RHO-rho_est)*povms[6]) == 0)
#                EV.add_constraint(('I'|(RHO-rho_est)*povms[7]) == 0)

                for povm in povms:
                    #EV.add_constraint(('I'|(RHO-rho_est)*povm) == 0)
                    EV.add_constraint(pic.trace(RHO*povm) == pic.trace(rho_est*povm))
            
#         
#            BB = []
#            for k in range(sum(self.data_ind_data == p)):
#            #for k in range(self.purestates):
#                BB.append([])
#                for c in range(self.bins):
##                    BB[k].append([])
#                    #BB[k][c] = pic.new_param('B_' + str(k) + '_' + str(c), B[k][c])
#                    #test = pic.new_param('B_' + str(k) + '_' + str(c), B[k][c])
#                    #EV.add_constraint('I'|((RHO-rho_est)*BB[k][c]) == 0)
#                    #EV.add_constraint(('I'|(RHO*BB[k][c])) == ('I'|(rho_est*BB[k][c])))
#                    #EV.add_constraint(('I'|(RHO*BB[k][c])) == np.real(np.trace(rho_est.dot(B[k][c]))))
#                    #EV.add_constraint(('I'|(RHO*test)) == np.real(np.trace(rho_est.dot(B[k][c]))))
#                    EV.add_constraint(((RHO|B[k][c])) == np.real(np.trace(rho_est.dot(B[k][c]))))
#                    #EV.add_constraint(((RHO|PI[k])) == np.real(np.trace(rho_est.dot(PI[k]))))
           
                # Solve for the maximum value
                EV.set_objective('max', 'I'|RHO*C)
                EV.solve(verbose = 1, solver='cvxopt')
                #EV.solve(verbose = 1, solver='smcp')
                max_EV = EV.obj_value()
                print(RHO, max_EV)
               
                # Solve for the minimum value
                EV.set_objective('min', 'I'|RHO*C)                    
                EV.solve(verbose = 1, solver='cvxopt')
                #EV.solve(verbose = 1, solver='smcp')
                min_EV = EV.obj_value()
                bound_list.append(np.array([min_EV, max_EV]))

            
# ----------------------------------------- #
### NECESSARY FUNCTIONS FOR ABOVE CLASSES ###
# ----------------------------------------- #

def measureOperator(O, inferred_rho, povms, eng):
    """ Measure operator for an inferred density matrix with constraints

    This function is a wrapper for measureOperator.m (see documentation)

    O, rho is a 2D numpy array with the same dimensions
    povms is a numpy array of 2D numpy arrays with the above dimension
        There can be any number of povms with any number of elements.
        For this purpose, the order and total number are irrelevant.
    eng is a matlab engine object"""

    # Need to make povm list a matlab list (roll the list axis)
    swapped = np.rollaxis(np.array(povms), 0, 3)
    povms = matlab.double(swapped.tolist(), is_complex=True)

    # Convert to matlab variables
    O = matlab.double(O.tolist(), is_complex=True)
    inferred_rho = matlab.double(inferred_rho.tolist(), is_complex=True)

    bounds = eng.measureOperator(O, inferred_rho, povms, nargout=1)
    return np.array(bounds)


def binDists(unbn, bnds):
    """Bin 1D probability distributions (or histograms)

    Inputs:
        unbn -- probability distributions without binning
        bnds -- new bin boundaries

    Assumes bins are contiguous, obviously
    """
    bins = np.size(bnds)-1
    bnds = bnds.astype(int)  ## enforcing integer boundaries for py3.5
    if unbn.ndim > 1:  # multiple distributions to bin
        binned = np.zeros([np.shape(unbn)[0], bins])
        for i in range(bins):
            binned[:, i] = np.sum(unbn[:, bnds[i]:bnds[i+1]], 1)
    elif unbn.ndim == 1:
        binned = np.zeros(bins)
        for i in range(bins):
            binned[i] = np.sum(unbn[bnds[i]:bnds[i+1]])
    return binned


def mutualInfo(pc, pr, p_cgr):
    """Compute mutual information between two random variables

    Input:
        pc    -- Unconditional probablity of random variable C
        pr    -- Unconditional probablity of random variable R
        p_cgr -- Conditional probabilities of C given R
                    each row is probability distribution of C given R_i
    """
    mutualInformation = 0
    for i in range(np.size(pr)):     # over states
        for j in range(np.size(pc)):    # over counts
            if p_cgr[i, j] > 0 and pc[j] > 0:
                mutualInformation += p_cgr[i, j]*pr[i] * \
                                     np.log2(p_cgr[i, j]/pc[j])
    return mutualInformation


def fidelity(rho, sigma):
    """Find fidelity between density matrices rho and sigma"""
    if np.linalg.matrix_rank(rho) == 1:      # rho is a pure state
        return np.trace(rho.dot(sigma))
    elif np.linalg.matrix_rank(sigma) == 1:  # sigma is a pure state
        return np.trace(sigma.dot(rho))
    else:
        rho_half = linalg.sqrtm(rho)
        F = (np.trace(linalg.sqrtm(rho_half.dot(sigma).dot(rho_half))))**2
        return F


def main():
    SE = AnalysisSet(seed=0)
    SE.simulate()
    #SE.simulate(two_qubit_asym_bell_state, n_dimensional_poisson_hists)
    SE.estimate_q_quick()
    print(SE.autobin())
    #SE.tomographyML()
    #SE.bootstrap(iters=2)
    #SE.killMatlabEng()
    return SE


if __name__ == "__main__":
    main()

