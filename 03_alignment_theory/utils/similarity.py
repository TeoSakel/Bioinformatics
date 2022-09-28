from itertools import chain, groupby, product, combinations_with_replacement
import numpy as np
from scipy.optimize import bisect


# Sources:
#    - https://chagall.med.cornell.edu/BioinfoCourse/PDFs/Lecture2/Dayhoff1978.pdf table 22
#    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7127678/table/tbl1/?report=objectonly

AMINO_FREQ = {
    "Dayhoff": {'A': 0.087,  
                'C': 0.033,
                'D': 0.047,
                'E': 0.050,
                'F': 0.040,
                'G': 0.089,
                'H': 0.034,
                'I': 0.037,
                'K': 0.081,
                'L': 0.085,
                'M': 0.015,
                'N': 0.04,
                'P': 0.051,
                'Q': 0.038,
                'R': 0.041,
                'S': 0.07,
                'T': 0.058,
                'V': 0.065,
                'W': 0.010,
                'Y': 0.030},
    "PMC7127678": {'A': 0.0777, 
                   'C': 0.0157,
                   'D': 0.0530,
                   'E': 0.0656,
                   'F': 0.0405,
                   'G': 0.0691,
                   'H': 0.0227,
                   'I': 0.0591,
                   'K': 0.0595,
                   'L': 0.0960,
                   'M': 0.0238,
                   'N': 0.0427,
                   'P': 0.0469,
                   'Q': 0.0393,
                   'R': 0.0526,
                   'S': 0.0694,
                   'T': 0.0550,
                   'V': 0.0667,
                   'W': 0.0118,
                   'Y': 0.0311}
}

def get_amino_freq(source="PMC7127678"):
    amino_freq = AMINO_FREQ[source]
    aminos, amino_p = zip(*amino_freq.items())
    amino_p = np.array(amino_p) / sum(amino_p)
    amino_freq = dict(zip(aminos, amino_p))
    return amino_freq


class ScoreMatrix:
    def __init__(self, scores, gap_score=-1, gap_open=0, gap_char='-'):
        self.matrix = scores
        self.gap = gap_score
        self.gap_open = gap_open
        self.gap_char = gap_char
        self.alphabet = sorted(set(chain.from_iterable(scores.keys())))
        
    def __getitem__(self, key):
        assert len(key) == 2
        if self.gap_char in key:
            return self.gap
        x, y = key
        try:
            return self.matrix[x, y]
        except KeyError:
            return self.matrix[y, x] 
    
    def __call__(self, X, Y):
        '''
        Given 2 aligned strings compute the score of the alignment
        '''
        assert len(X) == len(Y)
        score = 0
        for x, y in zip(X, Y):
            score += self[x, y]
        Ngaps  = sum(c == self.gap_char for c, _ in groupby(X))
        Ngaps += sum(c == self.gap_char for c, _ in groupby(Y))
        return(score + Ngaps * self.gap_open)
    
    def __str__(self):
        # quick and dirty print
        mat = []
        for i, a in enumerate(self.alphabet):
            line = [a]
            for b in self.alphabet[:i+1]:
                s = self[a, b]
                line.append(f'{s:>3}')
            mat.append(' '.join(line))
        
        mat += ['  ' + ' '.join(f'{a:>3}' for a in self.alphabet)]
        return '\n'.join(mat)
                  
    def trace(self, X, Y):
        '''
        Given 2 aligned strings compute the cummulative score of the alignment.
        '''
        assert len(X) == len(Y)
        score = [0] * (len(X) + 1)
        gapOpen = False
        for i, (x, y) in enumerate(zip(X, Y)):
            score[i+1] = self[x, y]
            if x == self.gap_char or y == self.gap_char:
                if not gapOpen:
                    score[i] += self.gap_open
                    gapOpen = True
            else:
                gapOpen = False
        # cumsum
        for i in range(1, len(score)):
            score[i] += score[i-1]
        return score
    
    def to_array(self):
        L = len(self.alphabet)
        mat = np.zeros((L, L), dtype=int)
        for i, a in enumerate(self.alphabet):
            for j, b in enumerate(self.alphabet[i:]):
                mat[i, i+j] = mat[j+i, i] = self[a, b]
        return mat
    
    def calc_K_lambda(self, ref):
        '''
        lambda is the unique solution to: f(x) = sum(p_i * p_j * exp(S_{ij} *x)) == 1
        '''
        ref = ''.join(c for c in ref if c != self.gap_char)
        freq = self._get_char_freq(ref)
        def f(x):
            all_pairs = product(self.alphabet, repeat=2) 
            # we could do some caching to avoid needless calcs
            return sum(freq[a] * freq[b] * np.exp(self[a, b] * x) for a, b in all_pairs) - 1.

        # find interval where solution lies
        assert np.isclose(f(0.), 0.)
        l0, f0 = 0.0625, f(0.0625)  # lower bound
        l1, f1 = 0.1250, f(0.1250)  # upper bound
        assert f0 < 0 # sanity check
        while f0 * f1 > 0: # while both have the same sign (-)
            l0, l1 = l1, l1 * 2.
            f0, f1 = f1, f(l1)
        # solve for lambda using bisect method
        lam = bisect(f, l0, l1)
        
        # K is more involved obviously
        K = 1.  # TODO: compute K properly
        return K, lam
    
    def _get_char_freq(self, text):
        counts = {c: 0 for c in self.alphabet}
        for c in text:
            counts[c] += 1
        N = len(text)
        freq = {c: n/N for c, n in counts.items()}
        return freq

def UniformScore(alphabet, match, mismatch, **kwargs):
    scores = {(a, b): match if a == b else mismatch 
              for a, b in combinations_with_replacement(alphabet, 2)}
    return ScoreMatrix(scores, **kwargs)

BLOSUM62 = ScoreMatrix(scores = {
    ('A', 'A'):  4, ('A', 'C'):  0, ('A', 'D'): -2, ('A', 'E'): -1,
    ('A', 'F'): -2, ('A', 'G'):  0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'K'): -1, ('A', 'L'): -1, ('A', 'M'): -1, ('A', 'N'): -2, 
    ('A', 'P'): -1, ('A', 'Q'): -1, ('A', 'R'): -1, ('A', 'S'):  1, 
    ('A', 'T'):  0, ('A', 'V'):  0, ('A', 'W'): -3, ('A', 'Y'): -2, 
    ('C', 'C'):  9, ('C', 'D'): -3, ('C', 'E'): -4, ('C', 'F'): -2, 
    ('C', 'G'): -3, ('C', 'H'): -3, ('C', 'I'): -1, ('C', 'K'): -3, 
    ('C', 'L'): -1, ('C', 'M'): -1, ('C', 'N'): -3, ('C', 'P'): -3, 
    ('C', 'Q'): -3, ('C', 'R'): -3, ('C', 'S'): -1, ('C', 'T'): -1, 
    ('C', 'V'): -1, ('C', 'W'): -2, ('C', 'Y'): -2, ('D', 'D'):  6, 
    ('D', 'E'):  2, ('D', 'F'): -3, ('D', 'G'): -1, ('D', 'H'): -1, 
    ('D', 'I'): -3, ('D', 'K'): -1, ('D', 'L'): -4, ('D', 'M'): -3, 
    ('D', 'N'):  1, ('D', 'P'): -1, ('D', 'Q'):  0, ('D', 'R'): -2, 
    ('D', 'S'):  0, ('D', 'T'): -1, ('D', 'V'): -3, ('D', 'W'): -4,
    ('D', 'Y'): -3, ('E', 'E'):  5, ('E', 'F'): -3, ('E', 'G'): -2, 
    ('E', 'H'):  0, ('E', 'I'): -3, ('E', 'K'):  1, ('E', 'L'): -3, 
    ('E', 'M'): -2, ('E', 'N'):  0, ('E', 'P'): -1, ('E', 'Q'):  2, 
    ('E', 'R'):  0, ('E', 'S'):  0, ('E', 'T'): -1, ('E', 'V'): -2, 
    ('E', 'W'): -3, ('E', 'Y'): -2, ('F', 'F'):  6, ('F', 'G'): -3, 
    ('F', 'H'): -1, ('F', 'I'):  0, ('F', 'K'): -3, ('F', 'L'):  0, 
    ('F', 'M'):  0, ('F', 'N'): -3, ('F', 'P'): -4, ('F', 'Q'): -3, 
    ('F', 'R'): -3, ('F', 'S'): -2, ('F', 'T'): -2, ('F', 'V'): -1, 
    ('F', 'W'):  1, ('F', 'Y'):  3, ('G', 'G'):  6, ('G', 'H'): -2, 
    ('G', 'I'): -4, ('G', 'K'): -2, ('G', 'L'): -4, ('G', 'M'): -3, 
    ('G', 'N'):  0, ('G', 'P'): -2, ('G', 'Q'): -2, ('G', 'R'): -2, 
    ('G', 'S'):  0, ('G', 'T'): -2, ('G', 'V'): -3, ('G', 'W'): -2, 
    ('G', 'Y'): -3, ('H', 'H'):  8, ('H', 'I'): -3, ('H', 'K'): -1, 
    ('H', 'L'): -3, ('H', 'M'): -2, ('H', 'N'):  1, ('H', 'P'): -2, 
    ('H', 'Q'):  0, ('H', 'R'):  0, ('H', 'S'): -1, ('H', 'T'): -2,
    ('H', 'V'): -3, ('H', 'W'): -2, ('H', 'Y'):  2, ('I', 'I'):  4, 
    ('I', 'K'): -3, ('I', 'L'):  2, ('I', 'M'):  1, ('I', 'N'): -3, 
    ('I', 'P'): -3, ('I', 'Q'): -3, ('I', 'R'): -3, ('I', 'S'): -2, 
    ('I', 'T'): -1, ('I', 'V'):  3, ('I', 'W'): -3, ('I', 'Y'): -1, 
    ('K', 'K'):  5, ('K', 'L'): -2, ('K', 'M'): -1, ('K', 'N'):  0, 
    ('K', 'P'): -1, ('K', 'Q'):  1, ('K', 'R'):  2, ('K', 'S'):  0, 
    ('K', 'T'): -1, ('K', 'V'): -2, ('K', 'W'): -3, ('K', 'Y'): -2, 
    ('L', 'L'):  4, ('L', 'M'):  2, ('L', 'N'): -3, ('L', 'P'): -3, 
    ('L', 'Q'): -2, ('L', 'R'): -2, ('L', 'S'): -2, ('L', 'T'): -1, 
    ('L', 'V'):  1, ('L', 'W'): -2, ('L', 'Y'): -1, ('M', 'M'):  5, 
    ('M', 'N'): -2, ('M', 'P'): -2, ('M', 'Q'):  0, ('M', 'R'): -1, 
    ('M', 'S'): -1, ('M', 'T'): -1, ('M', 'V'):  1, ('M', 'W'): -1, 
    ('M', 'Y'): -1, ('N', 'N'):  6, ('N', 'P'): -2, ('N', 'Q'):  0, 
    ('N', 'R'):  0, ('N', 'S'):  1, ('N', 'T'):  0, ('N', 'V'): -3, 
    ('N', 'W'): -4, ('N', 'Y'): -2, ('P', 'P'):  7, ('P', 'Q'): -1, 
    ('P', 'R'): -2, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'V'): -2, 
    ('P', 'W'): -4, ('P', 'Y'): -3, ('Q', 'Q'):  5, ('Q', 'R'):  1, 
    ('Q', 'S'):  0, ('Q', 'T'): -1, ('Q', 'V'): -2, ('Q', 'W'): -2, 
    ('Q', 'Y'): -1, ('R', 'R'):  5, ('R', 'S'): -1, ('R', 'T'): -1, 
    ('R', 'V'): -3, ('R', 'W'): -3, ('R', 'Y'): -2, ('S', 'S'):  4, 
    ('S', 'T'):  1, ('S', 'V'): -2, ('S', 'W'): -3, ('S', 'Y'): -2, 
    ('T', 'T'):  5, ('T', 'V'):  0, ('T', 'W'): -2, ('T', 'Y'): -2, 
    ('V', 'V'):  4, ('V', 'W'): -3, ('V', 'Y'): -1, ('W', 'W'): 11, 
    ('W', 'Y'):  2, ('Y', 'Y'):  7,
}, gap_score=-4)


TransCost = ScoreMatrix(gap_score=-6, scores = {
    ('A', 'G'): -2, ('C', 'T'): -2,
    ('A', 'C'): -4, ('A', 'T'): -4, ('C', 'G'): -4, ('G', 'T'): -4,
    ('A', 'A'):  2, ('C', 'C'):  2, ('G', 'G'):  2, ('T', 'T'):  2})

