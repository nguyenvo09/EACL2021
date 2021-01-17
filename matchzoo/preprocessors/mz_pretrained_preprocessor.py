
from tqdm import tqdm

from . import units
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .build_vocab_unit import build_vocab_unit
from .build_unit_from_data_pack import build_unit_from_data_pack
from .chain_transform import chain_transform
from handlers.output_handler import FileHandler
from pytorch_transformers import PreTrainedTokenizer
from pytorch_transformers.utils_glue import _truncate_seq_pair
from typing import List, Tuple
import pandas as pd
tqdm.pandas()


class PreTrainedModelsProcessor(PreTrainedTokenizer):
    """
    a preprocessor for transform DataPack.

    :param fixed_length_left: Integer, maximize length of :attr:`left` in the
        data_pack.
    :param fixed_length_right: Integer, maximize length of :attr:`right` in the
        data_pack.
    :param filter_mode: String, mode used by :class:`FrequenceFilterUnit`, Can
        be 'df', 'cf', and 'idf'.
    :param filter_low_freq: Float, lower bound value used by
        :class:`FrequenceFilterUnit`.
    :param filter_high_freq: Float, upper bound value used by
        :class:`FrequenceFilterUnit`.
    :param remove_stop_words: Bool, use :class:`StopRemovalUnit` unit or not.


    """

    def __init__(self, max_seq_length: int, fixed_length_left: int = -1,
                 fixed_length_right: int = -1,
                 filter_mode: str = 'df',
                 filter_low_freq: float = 2,
                 filter_high_freq: float = float('inf'),
                 remove_stop_words: bool = False,

                 tokenizer: PreTrainedTokenizer = None):
        """Initialization. We may need to store vocab path file, number of tokens, blah blah.
        """
        FileHandler.myprint("Query truncated to " + str(fixed_length_left) +
                            " Doc truncated to " + str(fixed_length_right))
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        assert fixed_length_left > 0 and fixed_length_right > 0
        self.fixed_length_left = fixed_length_left
        self.fixed_length_right = fixed_length_right
        assert self.fixed_length_left + self.fixed_length_right < self.max_seq_length, \
            "Left + right should be smaller than max length"


    def fit(self, data_pack: pd.DataFrame, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`BasicPreprocessor` instance.
        """
        raise NotImplementedError("Not coded yet")

    def transform(self, data_pack: pd.DataFrame, verbose: int = 1) -> Tuple[pd.DataFrame, List]:
        """
        Apply transformation on data, create fixed length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """


        # data_pack.append_text_length(inplace = True, verbose = verbose)
        # we need to split each text_left to an array of tokens, then we can convert them to ids
        converted_features = self._convert_examples_to_features(data_pack, label_list = [0, 1], max_seq_length = self.max_seq_length,
                                     tokenizer = self.tokenizer, output_mode = "classification")

        # data_pack.apply_on_text(str.split, mode = 'left', inplace = True, verbose = verbose)
        # data_pack.apply_on_text(self.tokenizer.convert_tokens_to_ids,
        #                         mode = 'left', inplace = True, verbose = verbose)

        # data_pack.apply_on_text(str.split, mode = 'right', inplace = True, verbose = verbose)
        # data_pack.apply_on_text(self.tokenizer.convert_tokens_to_ids,
        #                         mode = 'right', inplace = True, verbose = verbose)

        # max_len_left = self._fixed_length_left
        # max_len_right = self._fixed_length_right
        #
        # data_pack.left['length_left'] = \
        #     data_pack.left['length_left'].apply(
        #         lambda val: min(val, max_len_left))
        #
        # data_pack.right['length_right'] = \
        #     data_pack.right['length_right'].apply(
        #         lambda val: min(val, max_len_right))
        return data_pack, converted_features



    def _convert_examples_to_features(self, examples: pd.DataFrame, label_list, max_seq_length,
                                     tokenizer, output_mode,
                                     cls_token_at_end = False,
                                     cls_token = '[CLS]',
                                     cls_token_segment_id = 1,
                                     sep_token = '[SEP]',
                                     sep_token_extra = False,
                                     pad_on_left = False,
                                     pad_token = 0,
                                     pad_token_segment_id = 0,
                                     sequence_a_segment_id = 0,
                                     sequence_b_segment_id = 1,
                                     mask_padding_with_zero = True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        label_map = {label: i for i, label in enumerate(label_list)}
        # from tqdm import tqdm
        features = []
        ex_index = -1
        FileHandler.myprint("Processing text_left and text_right to make it a full sequence for BERT........")
        assert type(examples) == pd.DataFrame
        for q_id, text_a, doc_id, text_b, label in zip(examples["id_left"], examples["text_left"],
                                                       examples["id_right"], examples["text_right"], examples["label"]):
            ex_index += 1
            if ex_index % 10000 == 0: FileHandler.myprint("Processed xample %d of %d" % (ex_index, len(examples)))
            tokens_a = text_a.split()
            tokens_a = tokens_a[:self.fixed_length_left]
            tokens_b = None
            assert len(text_b) != 0, "Length of documents must be not zero!"
            if text_b:
                tokens_b = text_b.split()
                tokens_b = tokens_b[: self.fixed_length_right]
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
            else:
                # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
                special_tokens_count = 3 if sep_token_extra else 2
                if len(tokens_a) > max_seq_length - special_tokens_count:
                    tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if output_mode == "classification":
                label_id = label_map[label]
            elif output_mode == "regression":
                label_id = float(label)
            else:
                raise KeyError(output_mode)

            if ex_index < 5:
                FileHandler.myprint("*** Example ***")
                # FileHandler.myprint("guid: %s" % (example.guid))
                FileHandler.myprint("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                FileHandler.myprint("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                FileHandler.myprint("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                FileHandler.myprint("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                FileHandler.myprint("label: %s (id = %d)" % (label, label_id))

            features.append(
                InputFeatures(left_id = q_id, right_id = doc_id,
                              text_left = text_a, text_right = text_b,
                              input_ids = input_ids,
                              input_mask = input_mask,
                              segment_ids = segment_ids,
                              label_id = label_id))
        return features

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, left_id: int, right_id: int,
                 text_left: str, text_right: str,
                 input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.left_id = left_id
        self.right_id = right_id
        self.text_left = text_left
        self.text_right = text_right