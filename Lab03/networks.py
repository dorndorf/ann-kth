import numpy as np


class HopfieldNetwork():

    def __init__(self, patterns):
        ### patterns need to have shape (num of patterns, number of units)
        self.num_pat = patterns.shape[0]
        self.num_units = patterns.shape[1]
        self.W = np.zeros((self.num_units, self.num_units))

    def calc_weight_matrix(self, patterns):
        #for i in range(self.W.shape[0]):
        #    for j in range(self.W.shape[1]):
        #        self.W[i, j] = np.sum(patterns[:, i] * patterns[:, j]) / self.num_pat
        self.W = np.matmul(patterns.transpose(), patterns) / self.num_pat


    def synchr_update(self, in_pat): #synchronous also called simultaneous
        out = np.zeros(in_pat.shape)
        for p in range(in_pat.shape[0]):
            for i in range(in_pat.shape[1]):
                out[p, i] = np.sign(np.sum(self.W[:, i] * in_pat[p]))
        return out

    def asynchr_update(self, in_pat): #asynchronous also called sequential
        out = np.copy(in_pat)
        for p in range(in_pat.shape[0]):
            for i in range(in_pat.shape[1]):
                    out[p, i] = np.sign(np.sum(self.W[:, i] * out[p]))
        return out
