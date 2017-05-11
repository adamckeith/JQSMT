"""Hist class that defines a histogram

Author = Adam Keith

Assumptions
    Raw histogram bins MUST be integer spaced (problems arise when adding
         histograms that did not at least come from integer spaced bins)
    Assumes integer bin boundaries
        Maybe assumes input histogram bin boundaries are integer spaced
        (i.e. don't prebin except for continuous measurements)
        Not sure if violating this assumption actually causes any problems
    All histograms are multidimensional for binning purposes
        histograms are flattened using (np.flatten() (row major)) for computing
        things like loglikelihoods, etc.
        Thus, bin boundaries are a list of bin boundaries for each dimension,
        even in 1D, yes sorry.

Problems
    Adding histograms results in no binning. How do handle this properly?
    Rawhists have many zeros if loaded from some types of data. Not a problem
        but not efficient
    Not sure if code can handle general non-integer binbounds

Future Changes
    Change Hist cutoffs to make sense for tuples of counts (marginals)
    Merge 1D and multidimensional histogram simulation methods
"""

import numpy as np
import copy
from scipy.stats.contingency import margins


class Hist(object):
    """Base Class for Histograms.

    Assumes all counts are integers
    Aggregates counts from trials and stores them in a permanent unbinned
    histogram with integer boundaries
    """

    def __init__(self, counts=None):
        """Generate Histogram from Experiment. This will be changing soon - it
        will not accept experiment class objects in future versions"""
        # If no experiment make empty histogram
        self.rawhist = []       # Unbinned input raw histogram
        self.rawhistbins = []   # Unbinned input bin boundaries
        self.nspecies = 1       # Number of distinguishable objects
                                # or, number of values that one trial produces
        self.trials = 0         # Sum over entire histogram
        self.maxcounts = []     # Largest observed count along each dimension
        self.mincounts = []     # Smallest observed count along each dimension
        self.hist = []          # Numpy array of the mutable histogram
        self.binbounds = []     # boundaries of the mutable histogram
                                # (same structure as numpy.histogram)
        self.bins = []          # number of bins of mutable histogram
        if counts is not None:
            self.makeFromCounts(counts)

    def hist1D(self, raw=False):
        """Return row major flattened histogram"""
        if raw is True:
            return self.rawhist.flatten()
        return self.hist.flatten()

    def histND(self, flat):
        """Return roaw major reshaped multidimensional histogram
        or distribution"""
        return flat.reshape(self.bins)

    def makeFromCounts(self, counts):
        """Generate Histogram from a list of counts.

        Make permanent unbinned histogram and setup mutable
        histogram to be binned from count array

        Input:
            counts -- a list of observed values, may be list of tuples
        """
        counts = np.array(counts)
        counts = np.squeeze(counts)  # in case individual counts are lists

        if counts.ndim > 1:
            self.nspecies = np.shape(counts)[1]  # length of each count tuple
        else:  # counts are 1D
            self.nspecies = 1
            counts = np.expand_dims(counts, 1)  # need to add in dimension

        self.trials = np.shape(counts)[0]
        self.maxcounts = np.amax(counts, 0)
        self.mincounts = np.amin(counts, 0)
        # initial bin boundaries are integer spaced (assuming counts are)
        ranges = np.transpose(np.array([self.mincounts,
                                        self.maxcounts+1]))
        bins = self.maxcounts+1 - self.mincounts
        self.rawhist, self.rawhistbins = np.histogramdd(counts,
                                                      bins=bins,
                                                      range=ranges)                                                      
        #self.rawhistbins = [k.astype(int) for k in self.rawhistbins]                                              
        self.hist = copy.deepcopy(self.rawhist)
        self.binbounds = copy.deepcopy(self.rawhistbins)
        self.bins = np.squeeze(bins)

    def binRawHist(self, nwbnds, selfflag=True):
        """Bin raw histograms and return hist (user should use self.rebin())

        Inputs:
            nwbnds -- new bin boundaries
                1D array -> bin 1D histogram according to binbounds
                list of 1D array -> Each sublist of newbinbounds is
                                    bin boundaries in multidimensional
                                    histogram
            selfflag -- if True set self.hist to new binned histogram
                        and self.binbounds to nwbnds
        """
        bins = []
        # get number of bins in each dimension
        for n in range(self.nspecies):
            bins.append(np.size(nwbnds[n])-1)

        # make new n-dimensional histogram
        binned = copy.deepcopy(self.rawhist)
        histshape = np.asarray(self.rawhist.shape)

        # bin each dimension
        for n in range(self.nspecies):
            histshape[n] = bins[n]  # adjust current dimension to new bin #
            binnedtemp = np.zeros(histshape)
            # unbinned bounds in this dimension
            unbnbnds = self.rawhistbins[n][:-1]
            fullind1 = [slice(None)]*self.nspecies  # slicing indices
            fullind2 = [slice(None)]*self.nspecies
            for i in range(bins[n]):
                # find counts in unbinned in this new bin
                ind = np.where(np.logical_and(nwbnds[n][i] <= unbnbnds,
                                              unbnbnds < nwbnds[n][i+1]))
                # get rid of weird extra dimension
                ind = np.asarray(ind).flatten()
                if ind.size == 0:  # ind is empty (no counts observed)
                    continue       # for this new bin

                fullind1[n] = ind  # full index on partial binned histogram
                fullind2[n] = i    # full index on more binned histogram

                if binned[fullind1].ndim < self.nspecies:
                    # only one count for this bin in this dimension
                    binnedtemp[fullind2] = binned[fullind1]
                else:
                    binnedtemp[fullind2] = np.sum(binned[fullind1], axis=n)

            binned = binnedtemp  # this dimension is done

        if selfflag:
            self.hist = binned
            self.binbounds = nwbnds
            self.bins = self.hist.shape
        return binned

    def rebin(self, newbinbounds=None, show_warnings=True):
        """Rebin Histogram. Binning depends on interpretation of input.

        Input:
            newbinbounds -- interpretation of this value determines how
                            how histogram is binned.
                None     -> No compression (integer binning from mincount to
                            maxcount along each dimension)
                list of 1D array -> Each sublist of newbinbounds is
                                    bin boundaries in multidimensional
                                    histogram
            show_warnings -- flag to print warnings

        Be careful, a particular binning may not capture all counts
        Warnings are printed if this happens.

        All binning happens on raw, multidimensional histograms
        """
        # return original binning
        if newbinbounds is None:
            self.binbounds = copy.deepcopy(self.rawhistbins)
            self.hist = copy.deepcopy(self.rawhist)
            self.bins = self.hist.shape
            return
        self.binRawHist(newbinbounds)  # do the real work
        if np.sum(self.hist) != self.trials and show_warnings:
            print('This rebinning doesn\'t capture all counts.')
        return self.binbounds

    def countsAlongDim(self, dim):
        """Find counts for marginal of unbinned histogram for axis dim"""
        counts = makeCounts(marginal(self.rawhist, dim), 
                            [self.rawhistbins[dim]])
        np.random.shuffle(counts)
        return counts

    def makeFromHist(self, hist, binbounds):
        """Takes existing numpy histogram with bins and converts to Hist()
        only if binbounds are integer spaced

        Assumes input histogram is unbinned!
        """
        self.rawhist = copy.deepcopy(hist).astype(int)
        self.rawhistbins = copy.deepcopy(binbounds)
        self.hist = copy.deepcopy(hist).astype(int)
        self.binbounds = copy.deepcopy(binbounds)
        self.trials = np.sum(hist)
        self.nspecies = hist.ndim
        self.findMaxCount()

    def simPoisson(self, trials=50000, pops=[0.9995, 0.0005], mu=[2, 20],
                   make=True):
        """Simulate 1D counts from multiple Poisson distributions

        Input:
            trials -- the number of counts
            pops   -- list of desired populations, sum(pops)=1
            mu     -- list of means for multiple Poissons, same size as pops
            make   -- use counts to make histogram for this object
        """
        counts = []
        trialsPerPoisson = np.random.multinomial(trials, pops)
        for lam, N in zip(mu, trialsPerPoisson):
            counts.extend(np.random.poisson(lam, N).tolist())
        np.random.shuffle(counts)
        if make:
            self.makeFromCounts(counts)
        return counts

    def simulate_multi(self, nspecies=2, trials=50000,
                       statepops=[0.4, 0.1, 0.1, 0.4], mu=[[2, 20], [3, 25]]):
        """Simulate counts for multi-species experiment using simPoisson().

        Input:
            trials     -- the number of count tuples
            statepops  -- list of desired populations of multi-ion states
                          ((dark, dark), (bright, bright)), etc
            nspecies   -- number of dimensions (number of distinguishable ions)
            mu         -- for each ion,
                          list of means for single ion states (Poissons)
        Assumes ions only have two states (qubits) and binary ordering of
        states.
        """
        self.nspecies = nspecies
        self.trials = trials
        mu = np.array(mu)
        rawcounts = []
        trialsPerState = np.random.multinomial(trials, statepops)
        for i in range(2**self.nspecies):
            # get counts for each ion in that state
            statecounts = []
            for s in range(self.nspecies):
                # convert state to mean count for that ion in this state
                # binary string to see if ion is dark or bright
                m = mu[s, ord(bin(i)[2:].zfill(self.nspecies)[s])-48]
                # stack counts for each ion
                statecounts.append(self.simPoisson(trialsPerState[i], [1], [m],
                                                   make=False))
            # break into tuples
            rawcounts.extend(np.transpose(statecounts).tolist())
        self.makeFromCounts(rawcounts)

    def resample(self, N=None):
        """Return a resampled (with replacement) rawhist. N is the number of
        resampled trials"""
        if N is None:
            N = self.trials
        hist = self.rawhist.flatten()
        newhist = np.random.multinomial(N, hist/N)
        return self.histND(newhist)

    def findMaxCount(self):
        """Find max and min values along each dimension"""
        self.maxcounts = np.zeros(self.nspecies)
        self.mincounts = np.zeros(self.nspecies)
        for n in range(self.nspecies):
            # I could make this tighter, by check last nonzero item in marginal
            self.maxcounts[n] = self.rawhistbins[n][-2]  # second to last item
            self.mincounts[n] = self.rawhistbins[n][0]

    def trialsAboveCutoff(self, cutoff):
        """Return number of trials with counts above cutoff (inclusive) """
        # needs to be fixed for multiions
        unbnbnds = self.rawhistbins
        ind = np.where(unbnbnds == cutoff)
        return np.sum(self.rawhist[ind:])

    def trialsBelowCutoff(self, cutoff):
        """Return number of trials with counts below cutoff (inclusive)"""
        return (self.trials - self.trialsAboveCutoff(cutoff+1))

    def __add__(self, other):
        """Add Histograms.

        To subtract numbers or arrays, add the negative.
        """
        # Make new hist class instance
        addedhist = copy.deepcopy(self)
        addedhist += other
        return addedhist

    def __iadd__(self, other):
        """Add assign Histograms.

        Concatenate count lists and recreate histogram

        Note: This method removes previous binning on self
        """
        # Check if same type
        if isinstance(self, Hist) and isinstance(other, Hist):
            # Check if both hists have same nspecies
            if self.nspecies == other.nspecies:
                # Generate counts for each histogram, concatenate
                # and recreate histogram
                c1 = makeCounts(self.rawhist, self.rawhistbins)
                c2 = makeCounts(other.rawhist, other.rawhistbins)                
                counts = np.concatenate((c1, c2))
                #counts = np.concatenate((c1, c2), axis=0)
                self.makeFromCounts(counts)
            else:
                raise TypeError('Attempted to add histograms with '
                                'different histogram dimension')
        else:   # if other is array or single number, add it to each bin
            self.hist += other
        return self

    def __radd__(self, other):
        """Used for sum() function"""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def save(self, fname='histogram_counts.txt'):
        """Convert rawhist to counts and save as a text document"""
        c = makeCounts(self.rawhist, self.rawhistbins)
        np.random.shuffle(c)
        np.savetxt(fname, c, '%i')


def marginal(dist, dim):
    """Compute marginal of distribution dist along axis dim"""
    # Note, this computes all marginals and returns the one asked, so this
    # might be slow in some cases.
    ms = margins(dist)   # compute all marginals
    return np.squeeze(ms[dim])   # get the right one


def makeCounts(hist, binbounds):
    """Make a count list from histogram. The list is ordered and unshuffled"""
    hist = np.array(hist)  # make sure numpy array
    counts = []
    # I just realized how useless this is... could do this all on flattened
    # arrays
    ind = np.array(np.unravel_index(np.arange(hist.size), hist.shape))
    hist = hist.flatten()
    for c in range(hist.size):
        counts.extend(np.tile(ind[..., c], (hist[c], 1)).tolist())
    #np.random.shuffle(counts)
    counts = np.squeeze(np.array(counts))
    # Map to actual values using binbounds
    nspecies = len(binbounds)  # or len(hist.shape)
    if nspecies == 1:
        counts = np.expand_dims(counts, 1)  # need to add in dimension
    for n in range(nspecies):
        counts[:, n] = binbounds[n][counts[:, n]]
    return np.squeeze(counts)  # check this for multihist
