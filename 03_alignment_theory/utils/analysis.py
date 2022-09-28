from itertools import groupby
import numpy as np


def predecessor2path(P, best=None):
    if best is None:
        best = (P.shape[0]-1, P.shape[1]-1)
    elif np.issubdtype(type(best), np.integer):
        best = np.unravel_index(best, P.shape)
    i, j = best
    step = P[i, j]
    direction = ((0 ,  0), # stop
                 (-1, -1), # diagonal = match
                 (-1,  0), # vertical = insertion
                 (0 , -1)) # horizontal = deletion
    # build path from end to start
    I, J = [i], [j]
    while step != 0:
        di, dj = direction[step]
        i += di
        I.append(i)
        j += dj
        J.append(j)
        step = P[i, j]
    I, J = I[::-1], J[::-1]
    return I, J


def path2alignment(path, X, Y):
    I, J = path
    N = len(I)  # common for both
    alX, alY = '', ''
    for n in range(1, N): 
        # skip 0 as it's the start node and we care about edges
        i, j = I[n] - 1, J[n] - 1  # sequence index, -1 because we start with empty string (gap)
        alX += X[i] if I[n] > I[n-1] else '-'
        alY += Y[j] if J[n] > J[n-1] else '-'
    alM = ''.join('|' if x == y else ' ' for x, y in zip(alX, alY))
    return '\n'.join((alY, alM, alX))


def print_alignment(X, Y, P, best=None):
    path = predecessor2path(P, best)
    print(path2alignment(path, X, Y))

    
def get_local_extrema(X):
    """
    Given a 1D random walk it returns the the type and position of the local extrema.
    
    To recover the extrema we need the type of the 1st (minimum/maximum) and the
    positions. Their type switches back and forth.
    
    Argument:
        X: random walk series indexed by position
        
    Returns:
        start: type of first extremum {-1: minimum, 1: maximum}
        monotonic: position of extrema
    """
    dX = np.diff(X)
    # get length of runs of the same sign
    runs = [(sign, len(list(run))) for sign, run in groupby(dX, key=np.sign)]
    signs, runs = zip(*runs)
    idx = np.cumsum(runs)  # turns runs into indices
    signs, idx = zip(*[(s, i) for s, i in zip(signs, idx) if s != 0])
    start = signs[0]
    position = []
    for i in range(len(signs)-1):
        if signs[i] * signs[i+1] < 0:
            # we have a sign switch (we dropped zeros)
            position.append(idx[i])

    return start, position


def get_ladders_and_peaks(X):
    """
    Given a random walk X it returns the position of the ladders and peaks.
    """
    start, idx = get_local_extrema(X)
    if start > 0:  # first extremum is a maximum
        # every other position if a maximum/minimum
        local_max, local_min = idx[::2], idx[1::2]
        local_min = [0] + local_min  # a ladder must always preceide a peak
    else:  # first extremum is a minimum
        local_min, local_max = idx[::2], idx[1::2]

    ladders, peaks = [], []  # final lists
    cp = 0  # current peak index
    L, P = X[0]+1, X[cp]  # current ladder & peak
    for l, p in zip(local_min, local_max):
        if X[l] < L:  # new ladder
            ladders.append(l)
            peaks.append(cp)
            L, P = X[l], X[l]  # new ladder and peak reset
            cp = p
        if X[p] >= P:
            P, cp = X[p], p

    peaks = peaks[1:]
    return ladders, peaks
