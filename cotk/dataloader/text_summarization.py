"""Dataloader for language generation"""
from collections import Counter
from itertools import chain
import os
import json

import numpy as np

# from .._utils.unordered_hash import UnorderedSha256
from .._utils.file_utils import get_resource_file_path
from .._utils import hooks
from .dataloader import LanguageProcessingBase
from ..metric import MetricChain, PerplexityMetric, RougeCorpusMetric, SingleTurnDialogRecorder
#from ..metric import MetricChain, ,

# pylint: disable=W0223
class TextSummarization(LanguageProcessingBase):
    r"""Base class for sentence classification datasets. This is an abstract class.

    Arguments:{ARGUMENTS}

    Attributes:{ATTRIBUTES}
    """

    ARGUMENTS = LanguageProcessingBase.ARGUMENTS
    ATTRIBUTES = LanguageProcessingBase.ATTRIBUTES

    def get_batch(self, key, indexes):
        '''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}

                Returns:
                        (dict): A dict at least contains:

                        * **post_length** (:class:`numpy.ndarray`): A 1-d array, the length of post in each batch.
                          Size: ``[batch_size]``
                        * **post** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form in posts.
                          Only provide valid words. ``unk_id`` will be used if a word is not valid.
                          Size: ``[batch_size, max(sent_length)]``
                        * **post_allvocabs** (:class:`numpy.ndarray`): A 2-d padded array containing words of id
                          form in posts. Provide both valid and invalid vocabs.
                          Size: ``[batch_size, max(sent_length)]``
                        * **resp_length** (:class:`numpy.ndarray`): A 1-d array, the length of response in each batch.
                          Size: ``[batch_size]``
                        * **resp** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form
                          in responses. Only provide valid vocabs. ``unk_id`` will be used if a word is not valid.
                          Size: ``[batch_size, max(sent_length)]``
                        * **resp_allvocabs** (:class:`numpy.ndarray`):
                          A 2-d padded array containing words of id form in responses.
                          Provide both valid and invalid vocabs.
                          Size: ``[batch_size, max(sent_length)]``
                                        Examples:
                        # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
                        >>>
                        >>> #   "hello", "i", "am", "fine"]
                        >>> # vocab_size = 9
                        # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
                        >>>
                        >>> dataloader.get_batch('train', [0, 1])
                        {
                                "post_allvocabs": numpy.array([
                                        # first post:  <go> are you fine <eos>
                                        [2, 5, 6, 10, 3],
                                        # second post: <go> hello <eos> <pad> <pad>
                                        [2, 7, 3, 0, 0],
                                ]),
                                "post": numpy.array([
                                        # first post:  <go> are you <unk> <eos>
                                        [2, 5, 6, 1, 3],
                                        # second post: <go> hello <eos> <pad> <pad>
                                        [2, 7, 3, 0, 0],
                                ]),
                                "resp_allvocabs": numpy.array([
                                        # first response:  <go> i am fine <eos>
                                        [2, 8, 9, 10, 3],
                                        # second response: <go> hello <eos> <pad> <pad>
                                        [2, 7, 3, 0, 0],
                                ]),
                                "resp": numpy.array([
                                        # first response:  <go> i <unk> <unk> <eos>
                                        [2, 8, 1, 1, 3],
                                        # second response: <go> hello <eos> <pad> <pad>
                                        [2, 7, 3, 0, 0],
                                ]),
xs
                                # length of posts
                                "post_length": numpy.array([5, 3]),
                                # length of responses
                                "resp_length": numpy.array([5, 3]),
                        }
                '''

        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {}
        batch_size = len(indexes)
        res["post_length"] = np.array(
            list(map(lambda i: len(self.data[key]['post'][i]), indexes)))
        res["resp_length"] = np.array(
            list(map(lambda i: len(self.data[key]['resp'][i]), indexes)))
        res_post = res["post"] = np.zeros(
            (batch_size, np.max(res["post_length"])), dtype=int)
        res_resp = res["resp"] = np.zeros(
            (batch_size, np.max(res["resp_length"])), dtype=int)
        for i, j in enumerate(indexes):
            post = self.data[key]['post'][j]
            resp = self.data[key]['resp'][j]
            res_post[i, :len(post)] = post
            res_resp[i, :len(resp)] = resp

        res["post_allvocabs"] = res_post.copy()
        res["resp_allvocabs"] = res_resp.copy()
        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id
        return res

    def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob",
                                       invalid_vocab=False):
        '''
                    It contains:
        * :class:`.metric.PerplexityMetric`
        
            Arguments:
                gen_log_prob_key (str):  The key of predicted log probability over words.
                            Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.
                    invalid_vocab (bool): Whether ``gen_log_prob`` contains invalid vocab.
                            Refer to :class:`.metric.PerplexityMetric`. Default: ``False``.
        
            Returns:
                    A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(PerplexityMetric(self,
                                           reference_allvocabs_key="resp_allvocabs",
                                           reference_len_key="resp_length",
                                           gen_log_prob_key=gen_log_prob_key,
                                           invalid_vocab=invalid_vocab))
        return metric

    def get_inference_metric(self, gen_key="gen"):
        '''Get metrics for inference.
            It contains:
        * :class:`.metric.BleuCorpusMetric`
        * :class:`.metric.SingleTurnDialogRecorder`
        Arguments:
                gen_key (str): The key of generated sentences in index form.
                            Refer to :class:`.metric.BleuCorpusMetric` or
                            :class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.
            Returns:
                    A :class:`.metric.MetricChain` object.
            '''
        metric = MetricChain()
        metric.add_metric(RougeCorpusMetric(self, gen_key=gen_key, \
                                            reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
        return metric

    def get_metric(self, prediction_key="prediction"):
        pass
        '''Get metrics for accuracy. In other words, this function
        provides metrics for sentence classification task.

        It contains:

                * :class:`.metric.AccuracyMetric`

        Arguments:
                prediction_key (str): The key of prediction over sentences.
                        Refer to :class:`.metric.AccuracyMetric`. Default: ``prediction``.

        Returns:
                A :class:`.metric.MetricChain` object.
        '''
        """
        metric = MetricChain()
        metric.add_metric(AccuracyMetric(self,
                                         label_key='label',
                                         prediction_key=prediction_key))
        return metric
    """


class CNN(TextSummarization):
    '''A dataloader for preprocessed Cnn dataset.
    '''

    @hooks.hook_dataloader
    def __init__(self, file_id, min_vocab_times=10, max_doc_length=1000, invalid_vocab_times=0):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_doc_length = max_doc_length
        self._invalid_vocab_times = invalid_vocab_times
        super(CNN, self).__init__()

    def _load_data(self):
        r'''Loading dataset, invoked by `LanguageProcessingBase.__init__`
        '''
        return super()._general_load_data(self._file_path,
                                          [['post', 'Sentence'], [
                                              "resp", "Sentence"]],
                                          self._min_vocab_times,
                                          self._max_doc_length,
                                          None,
                                          self._invalid_vocab_times)

    def tokenize(self, sentence):
        return super().tokenize(sentence, True, 'space')
