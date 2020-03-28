import numpy as np

def sign(x):
    if x>=0:
        return 1
    else:
        return -1


class HopfieldNetwork():


    def __init__(self, patterns):
        ### patterns need to have shape (num of patterns, number of units)
        self.num_pat = patterns.shape[0]
        self.num_units = patterns.shape[1]
        self.W = np.zeros((self.num_units, self.num_units))
        self.energy = 0.0
        self.sparse_factor = np.sum(patterns) / (self.num_pat*self.num_units)
        self.sparse_W = np.zeros((self.num_units, self.num_units))

    def set_weight_matrix(self, weight_matrix):
        assert self.W.shape == weight_matrix.shape, "The weight matrix you provided has the wrong shape."
        self.W = weight_matrix

    def remove_weight_matrix_diag(self):
        np.fill_diagonal(self.W, 0.0)
        
    def calc_weight_matrix(self, patterns):
        #for i in range(self.W.shape[0]):
        #    for j in range(self.W.shape[1]):
        #        self.W[i, j] = np.sum(patterns[:, i] * patterns[:, j]) / self.num_pat
        self.W = np.matmul(patterns.transpose(), patterns) / self.num_pat # Denominator should be the number of units. I am not sure.
        #self.W = np.matmul(patterns.transpose(), patterns)/self.num_units


    def random_one_update(self,in_pat):
        out = np.copy(in_pat)
        for i in range(in_pat.shape[0]):
            j = np.random.randint(0,in_pat.shape[0])
            out[j] = sign(np.sum(self.W[j, :] * out))
        return out

    def synchr_update(self, in_pat): #synchronous also called simultaneous
        out = np.zeros(in_pat.shape)
        for p in range(in_pat.shape[0]):
            for i in range(in_pat.shape[1]):
                out[p, i] = sign(np.sum(self.W[i, :] * in_pat[p]))
        return out

    def synchr_one_update(self, in_pat): #synchronous also called simultaneous
        out = np.zeros(in_pat.shape)
        for i in range(in_pat.shape[0]):
            out[i] = sign(np.sum(self.W[i, :] * in_pat))
        return out
    

    def asynchr_one_update(self, in_pat): #asynchronous also called sequential
        out = np.copy(in_pat)
        for i in range(in_pat.shape[0]):
            out[i] = sign(np.sum(self.W[i, :] * out))
        return out

    def asynchr_update(self, in_pat): #asynchronous also called sequential
        out = np.copy(in_pat)
        for p in range(in_pat.shape[0]):
            for i in range(in_pat.shape[1]):
                    out[p, i] = sign(np.sum(self.W[i, :] * out[p]))
        return out

    def calc_energy(self, pattern):
        energy = 0.0
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                energy += self.W[i, j] * pattern[i] * pattern[j]
        energy= -energy
        self.energy = energy
        return energy

    def sparse_calc_weight_matrix(self, patterns, sp_fac=0.0):
        if sp_fac > 0.0:
            self.sparse_factor = sp_fac
        normed_pattern = patterns - self.sparse_factor
        self.sparse_W = np.matmul(normed_pattern.transpose(), normed_pattern) / self.num_units# Denominator should be the number of units. I am not sure.

    def sparse_synchr_update(self, in_pat, bias=0.1):  # synchronous also called simultaneous
        out = np.zeros(in_pat.shape)
        for p in range(in_pat.shape[0]):
            for i in range(in_pat.shape[1]):
                out[p, i] = 0.5 + 0.5 * sign(np.sum(self.sparse_W[i, :] * in_pat[p]) - bias)
        return out

    def sparse_asynchr_update(self, in_pat, bias=0.1):  # asynchronous also called sequential
        out = np.copy(in_pat)
        for p in range(in_pat.shape[0]):
            for i in range(in_pat.shape[1]):
                out[p, i] = 0.5 + 0.5 * sign(np.sum(self.sparse_W[i, :] * out[p]) - bias)
        return out


