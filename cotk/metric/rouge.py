r"""
Containing some classes and functions about bleu evaluating results of models.
"""
import random
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import tqdm
from .metric import MetricBase
from .._utils import hooks


def _replace_unk(_input, _unk_id, _target=-1):
    r'''Auxiliary function for replacing the unknown words:

    Arguments:
            _input (list): the references or hypothesis.
            _unk_id (int): id for unknown words.
            _target: the target word index used to replace the unknown words.

    Returns:

            * list: processed result.
    '''
    output = []
    for _list in _input:
        _output = []
        for ele in _list:
            _output.append(_target if ele == _unk_id else ele)
        output.append(_output)
    return output

def _get_ngrams(n, text):
    """ Calcualtes n-grams. """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count):
    # Handle edge case. This isn't mathematically correct, but it's good enough
    precision = 0.0 if not evaluated_count else overlapping_count / evaluated_count
    recall = 0.0 if not reference_count else overlapping_count / reference_count
    return 2.0 * ((precision * recall) / (precision + recall + 1e-8)), precision, recall

def rouge_n(evaluated_sentences, reference_sentences, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf
    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: The sentences from the reference set
      n: Size of ngram.  Defaults to 2.
    Returns:
      A tuple (f1, precision, recall) for ROUGE-N
    """
    reference_ngrams = [ _get_ngrams(n, _) for _ in reference_sentences ]
    evaluated_ngrams = [ _get_ngrams(n, _) for _ in evaluated_sentences ]
    rouge_score = [ f_r_p_rouge_n(len(e), len(r), len(e.intersection(r)))
                    for e, r in zip(evaluated_ngrams, reference_ngrams) ]
    evaluated_count = len(evaluated_sentences)
    return { "f": sum([ e for e, _, _ in rouge_score ]) / evaluated_count,
             "p": sum([ e for _, e, _ in rouge_score ]) / evaluated_count,
             "r": sum([ e for _, _, e in rouge_score ]) / evaluated_count
    }

def _recon_lcs(x, y):
    """
    Compute LCS_u(r_i, c_i) which is the LCS score of the union longest common
    subsequence between reference sentence r and candidate summary c.
    For example:
    if r = w1 w2 w3 w4 w5, and c contains two sentences: c = w1 w2 w6 w7 w8 w1 w3 w8 w9 w5
    The union longest common subsequence of r and c is "w1 w2 w3 w5"
    and LCS_u(r, c) = 4/5.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns:
      sequence: LCS of x and y
    """

    def _lcs(x, y):
        """
        Computes the length of the longest common subsequence (lcs) between two
        strings. The implementation below uses a DP programming algorithm and runs
        in O(nm) time where n = len(x) and m = len(y).
        Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
        Args:
        x: collection of words
        y: collection of words
        Returns:
        Table of dictionary of coord and len lcs
        """
        n, m = len(x), len(y)
        table = dict()
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 or j == 0:
                    table[i, j] = 0
                elif x[i - 1] == y[j - 1]:
                    table[i, j] = table[i - 1, j - 1] + 1
                else:
                    table[i, j] = max(table[i - 1, j], table[i, j - 1])
        return table

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    table = _lcs(x, y)
    return tuple(map(lambda x: x[0], _recon(len(x), len(y))))

def _union_lcs(evaluated_sentence, reference_sentence):
   """
    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary
    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentence: One of the sentences in the reference summaries
    Returns:
      Dict float: f, r, and p
    """
   m = len(set(reference_sentence))
   n = len(set(evaluated_sentence))
   union = _recon_lcs(reference_sentence, evaluated_sentence)
   llcs = len(union)
   r_lcs = llcs / m
   p_lcs = llcs / n
   beta = p_lcs / (r_lcs + 1e-12)
   num = (1 + (beta**2)) * r_lcs * p_lcs
   denom = r_lcs + ((beta**2) * p_lcs)
   f_lcs = num / (denom + 1e-12)
   return f_lcs, p_lcs, r_lcs

def rouge_l(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf
    """
    rouge_l_score = [ _union_lcs(e, r) for e, r in zip(evaluated_sentences, reference_sentences) ]
    evaluated_count = len(evaluated_sentences)
    return { "f": sum([ e for e, _, _ in rouge_l_score ]) / evaluated_count,
             "p": sum([ e for _, e, _ in rouge_l_score ]) / evaluated_count,
             "r": sum([ e for _, _, e in rouge_l_score ]) / evaluated_count
    }

class RougeCorpusMetric(MetricBase):
    '''Metric for calculating ROUGE.

    Arguments:
            {MetricBase.DATALOADER_ARGUMENTS}
            {MetricBase.REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
            {MetricBase.GEN_KEY_ARGUMENTS}

    Here is an exmaple:

            >>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
            >>> reference_allvocabs_key = "ref_allvocabs"
            >>> gen_key = "gen"
            >>> metric = cotk.metric.BleuCorpusMetric(dl,
            ...	    reference_allvocabs_key=reference_allvocabs_key,
            ...	    gen_key=gen_key)
            >>> data = {
            ...     reference_allvocabs_key: [[2, 10, 64, 851, 3], [2, 10, 48, 851, 3]],
            ...     # reference_allvocabs_key: [["<go>", "I", "like", "python", "<eos>"], ["<go>", "I", "use", "python", "<eos>"]],
            ...
            ...	    gen_key: [[10, 1028, 479, 285, 220, 3], [851, 17, 2451, 3]]
            ...	    # gen_key: [["I", "love", "java", "very", "much", "<eos>"], ["python", "is", "excellent", "<eos>"]],
            ... }
            >>> metric.forward(data)
            >>> metric.close()
            {'rouge': 0.08582363099612991,
            'rouge hashvalue': '70e019630fef24d9477034a3d941a5349fcbff5a3dc6978a13ea3d85290114fb'}

    '''

    _name = 'BleuCorpusMetric'
    _version = 1

    @hooks.hook_metric
    def __init__(self, dataloader, ignore_smoothing_error=False,
                 reference_allvocabs_key="ref_allvocabs", gen_key="gen"):
        super().__init__(self._name, self._version)
        self.dataloader = dataloader
        self.ignore_smoothing_error = ignore_smoothing_error
        self.reference_allvocabs_key = reference_allvocabs_key
        self.gen_key = gen_key
        self.refs = []
        self.hyps = []

    def forward(self, data):
        '''Processing a batch of data.

        Arguments:
                data (dict): A dict at least contains the following keys:

                        {MetricBase.FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS}
                        {MetricBase.FORWARD_GEN_ARGUMENTS}

                        Here is an example for data:

                                >>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
                                >>> #   "been", "to", "China"]
                                >>> data = {
                                ...     reference_allvocabs_key: [[2,4,3], [2,5,6,3]],
                                ...	    gen_key: [[4,5,3], [6,7,8,3]]
                                ... }
        '''
        super().forward(data)
        gen = data[self.gen_key]
        resp = data[self.reference_allvocabs_key]

        if not isinstance(gen, (np.ndarray, list)):
            raise TypeError("Unknown type for gen.")
        if not isinstance(resp, (np.ndarray, list)):
            raise TypeError("Unknown type for resp")

        if len(resp) != len(gen):
            raise ValueError("Batch num is not matched.")

        relevant_data = []
        for gen_sen, resp_sen in zip(gen, resp):
            self.hyps.append(self.dataloader.trim(gen_sen))
            reference = list(self.dataloader.trim(resp_sen[1:]))
            relevant_data.append(reference)
            self.refs.append(reference)
        self._hash_relevant_data(relevant_data)

    @hooks.hook_metric_close
    def close(self):
        '''
        Returns:
                (dict): Return a dict which contains

                * **rouge**: bleu value.
                * **rouge hashvalue**: hash value for rouge metric, same hash value stands
                  for same evaluation settings.
        '''
        result = super().close()
        if (not self.hyps) or (not self.refs):
            raise RuntimeError(
                "The metric has not been forwarded data correctly.")

        self.hyps = _replace_unk(self.hyps, self.dataloader.unk_id)

        rouge_1 = rouge_n(self.hyps, self.refs, n=1)
        rouge_2 = rouge_n(self.hyps, self.refs, n=2)
        rougel = rouge_l(self.hyps, self.refs)

        result.update( { "rouge-1-f": rouge_1["f"], #"rouge-1-p": rouge_1["p"], "rouge-1-r": rouge_1["r"],
                         "rouge-2-f": rouge_2["f"], #"rouge-2-p": rouge_2["p"], "rouge-2-r": rouge_2["r"],
                         "rouge-l-f": rougel["f"], #"rouge-l-p": rougel["p"], "rouge-l-r": rougel["r"]
                         "rouge hashvalue": self._hashvalue()
        } )

        return result
