from collections import Counter
import numpy as np
from numba import njit
from torch.utils.data import Dataset

from abc import ABC, abstractmethod


class DataSampler(ABC):
    @abstractmethod
    def fit(self, seqs):
        pass

    @abstractmethod
    def sampling(self):
        pass

    @abstractmethod
    def conditional_sampling(self, conditioned_on):
        pass


class nGramSampler(DataSampler):
    def __init__(
        self, window_length=10, context_window_type="double", buffer_size=1000
    ):
        self.window_length = window_length
        self.buffer_size = buffer_size
        self.context_window_type = context_window_type
        self.n_seqs = 0
        self.n_samples = 0
        self.seq_order = None
        self.seq_iter = 0
        self.buffer_idx = 0
        self.centers = np.array([])
        self.contexts = np.array([])

    def fit(self, seqs):
        self.seqs = seqs

        self.n_samples = 0
        for seq in seqs:
            n_pairs = count_center_context_pairs(
                self.window_length, len(seq), self.context_window_type
            )
            self.n_samples += n_pairs
        self.n_seqs = len(seqs)
        self.seq_order = np.random.choice(self.n_seqs, self.n_seqs, replace=False)
        self.seq_iter = 0
        self.buffer_idx = 0

    def __len__(self):
        return self.n_samples

    def sampling(self):

        if self.buffer_idx >=len(self.centers):
            self._generate_samples()
        
        cent = self.centers[self.buffer_idx]
        cont = self.contexts[self.buffer_idx]

        return cent, cont

    def conditional_sampling(self, conditioned_on=None):
        return -1

    def _generate_samples(self):
        self.centers = []
        self.contexts = []
        for _ in range(self.buffer_size):
            self.seq_iter += 1
            if self.seq_iter >= self.n_seqs:
                self.seq_iter = self.seq_iter % self.n_seqs
            seq_id = self.seq_order[self.seq_iter]
            cent, cont = _get_center_context_pairs(
                np.array(self.seqs[seq_id]),
                self.window_length,
                self.context_window_type,
            )
            self.centers.append(cent)
            self.contexts.append(cont)

        self.centers, self.contexts = (
            np.concatenate(self.centers).astype(int),
            np.concatenate(self.contexts).astype(int),
        )
        order = np.random.choice(
            len(self.centers), size=len(self.centers), replace=False
        )
        self.centers, self.contexts = self.centers[order], self.contexts[order]
        self.buffer_idx = 0


class FrequencyBasedSampler(DataSampler):
    def __init__(self, gamma=1):
        self.gamma = gamma
        self.n_elements = 0
        self.ele_null_prob = None

    def fit(self, seqs):
        counter = Counter()
        for seq in seqs:
            counter.update(seq)
        self.n_elements = int(max(counter.keys()) + 1)
        self.ele_null_prob = np.zeros(self.n_elements)
        for k, v in counter.items():
            self.ele_null_prob[k] = v
        self.ele_null_prob /= np.sum(self.ele_null_prob)

    def conditional_sampling(self, conditioned_on=None):
        return self.sampling()[0]

    def sampling(self):
        cent = np.random.choice(
            self.n_elements, size=1, p=self.ele_null_prob, replace=True
        )
        cont = np.random.choice(
            self.n_elements, size=1, p=self.ele_null_prob, replace=True
        )
        return cent[0], cont[0]


class Word2VecSampler(DataSampler):
    def __init__(self, in_vec, out_vec, alpha=1):
        self.alpha = alpha
        self.in_vec = in_vec
        self.out_vec = out_vec
        self.center_sampler = FrequencyBasedSampler()
        self.n_elements = out_vec.shape[0]

    def fit(self, seqs):
        self.center_sampler.fit(seqs)

    def conditional_sampling(self, conditioned_on=None):
        uv = self.in_vec[conditioned_on, :] @ self.out_vec.T
        uv = np.array(uv - np.mean(uv)).ravel()
        p = np.exp(self.alpha * uv)
        p /= np.sum(p)
        return np.random.choice(self.n_elements, size=1, p=p, replace=True)[0]

    def sampling(self):
        cent = self.center_sampler.sampling()[0]
        cont = self.conditional_sampling(cent)
        return cent, cont




#
# Triplet Dataset
#
class TripletDataset(Dataset):
    def __init__(
        self,
        pos_sampler,
        neg_sampler,
        epochs=1,
    ):
        self.epochs = epochs
        self.pos_sampler = pos_sampler
        self.neg_sampler = neg_sampler
        self.n_samples = len(self.pos_sampler)

    def __len__(self):
        return self.n_samples * self.epochs

    def __getitem__(self, idx):
        center, cont = self.pos_sampler.sampling()
        rand_cont = self.neg_sampler.conditional_sampling(conditioned_on=center)
        return center, cont, rand_cont


@njit(nogil=True)
def count_center_context_pairs(window_length, seq_len, context_window_type):
    # Count the number of center-context pairs.
    # Suppose that we sample, for each center word, a context word that proceeds the center k-words.
    # There are T-k words in the sequence, so that the number of pairs is given by summing this over k upto min(T-1, L), i.e.,
    # 2 \sum_{k=1}^{min(T-1, L)} (T-k)
    # where we cap the upper range to T-1 in case that the window covers the entire sentence, and
    # we double the sum because the word window extends over both proceeding and succeeding words.
    min_window_length = np.minimum(window_length, seq_len - 1)
    n = 2 * min_window_length * seq_len - min_window_length * (min_window_length + 1)
    if context_window_type == "double":
        return int(n)
    else:
        return int(n / 2)


@njit(nogil=True)
def _get_center_context_pairs(seq, window_length, context_window_type):
    """Get center-context pairs from sequence.

    :param seq: Sequence
    :type seq: numpy array
    :param window_length: Length of the context window
    :type window_length: int
    :return: Center, context pairs
    :rtype: tuple
    """
    n_seq = len(seq)
    n_pairs = count_center_context_pairs(window_length, n_seq, context_window_type)
    centers = -np.ones(n_pairs, dtype=np.int64)
    contexts = -np.ones(n_pairs, dtype=np.int64)
    idx = 0
    wstart, wend = 0, 2 * window_length + 1
    if context_window_type == "suc":
        wstart = window_length + 1
    if context_window_type == "prec":
        wend = window_length

    for i in range(n_seq):
        for j in range(wstart, wend):
            if (
                (j != window_length)
                and (i - window_length + j >= 0)
                and (i - window_length + j < n_seq)
            ):
                centers[idx] = seq[i]
                contexts[idx] = seq[i - window_length + j]
                idx += 1
    return centers, contexts
