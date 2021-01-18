import numpy as np
import pandas as pd
import matchzoo
import collections
from setting_keywords import KeyWordSettings
from handlers.output_handler import FileHandler


class MatchInteraction(object):
    """
        Interactions object. Contains (at a minimum) pair of user-item
        interactions, but can also be enriched with ratings, timestamps,
        and interaction weights.

        For *implicit feedback* scenarios, user ids and item ids should
        only be provided for user-item pairs where an interaction was
        observed. All pairs that are not provided are treated as missing
        observations, and often interpreted as (implicit) negative
        signals.

        For *explicit feedback* scenarios, user ids, item ids, and
        ratings should be provided for all user-item-rating triplets
        that were observed in the dataset.

         This class is designed specificlaly for matching models only. Since I don't want
        to use MatchZoo datapack at all.
        Parameters
        ----------

        data_pack:
        Attributes
        ----------

        unique_query_ids: `np.ndarray`array of np.int32
            array of user ids of the user-item pairs
        query_contents: array of np.int32
            array of item ids of the user-item pairs
        query_lengths: array of np.float32, optional
            array of ratings

        unique_doc_ids: array of np.int32, optional
            array of timestamps
        doc_contents: array of np.float32, optional
            array of weights
        doc_lengths: int, optional
            Number of distinct users in the dataset.

        pos_queries: list[int]
            Number of distinct items in the dataset.
        pos_docs: list[int]
            Number of distinct items in the dataset.
        negatives: dict

        """

    def __init__(self, data_pack: matchzoo.DataPack, **kargs):
        # Note that, these indices are not from 0.
        FileHandler.myprint("Converting DataFrame to Normal Dictionary of Data")
        self.unique_query_ids, \
        self.dict_query_contents, \
        self.dict_query_lengths, \
        self.dict_query_raw_contents, \
        self.dict_query_positions = self.convert_leftright(data_pack.left, text_key = "text_left",
                                                              length_text_key = "length_left",
                                                              raw_text_key = "raw_text_left")
        self.data_pack = data_pack
        assert len(self.unique_query_ids) == len(set(self.unique_query_ids)), "Must be unique ids"
        """ Why do I need to sort it? I have no idea why did I do it? """

        self.unique_doc_ids, \
        self.dict_doc_contents, \
        self.dict_doc_lengths, \
        self.dict_doc_raw_contents, \
        self.dict_doc_positions = self.convert_leftright(data_pack.right, text_key = "text_right",
                                                            length_text_key = "length_right",
                                                            raw_text_key = "raw_text_right")

        assert len(self.unique_doc_ids) == len(set(self.unique_doc_ids)), "Must be unique ids for doc ids"
        assert len(self.unique_query_ids) != len(self.unique_doc_ids), "Impossible to have equal number of docs and number of original tweets"

        self.pos_queries, \
        self.pos_docs, \
        self.negatives, \
        self.unique_queries_test = self.convert_relations(data_pack.relation)

        # for queries, padded
        self.np_query_contents = np.array([self.dict_query_contents[q] for q in self.pos_queries])
        self.np_query_lengths = np.array([self.dict_query_lengths[q] for q in self.pos_queries])
        self.query_positions = np.array([self.dict_query_positions[q] for q in self.pos_queries])

        # for docs, padded
        self.np_doc_contents = np.array([self.dict_doc_contents[d] for d in self.pos_docs])
        self.np_doc_lengths = np.array([self.dict_doc_lengths[d] for d in self.pos_docs])
        self.doc_positions = np.array([self.dict_doc_positions[d] for d in self.pos_docs])

        assert self.np_query_lengths.shape == self.np_doc_lengths.shape
        self.padded_doc_length = len(self.np_doc_contents[0])
        self.padded_query_length = len(self.np_query_contents[0])

    def convert_leftright(self, part: pd.DataFrame, text_key: str, length_text_key: str, raw_text_key: str, **kargs):
        """ Converting the dataframe of interactions """
        ids, contents_dict, lengths_dict, position_dict = [], {}, {}, {}
        raw_content_dict = {}
        # Why don't we use the queryID as the key for dictionary????
        FileHandler.myprint("[NOTICE] MatchZoo use queryID and docID as index in dataframe left and right, "
                            "therefore, iterrows will return index which is left_id or right_id")
        for index, row in part.iterrows(): # very dangerous, be careful because it may change order!!!
            ids.append(index)
            text_ = row[text_key]  # text_ here is converted to numbers and padded
            raw_content_dict[index] = row[raw_text_key]

            if length_text_key not in row: length_ = len(text_)
            else: length_ = row[length_text_key]
            assert length_ != 0
            assert index not in contents_dict
            contents_dict[index] = text_
            lengths_dict[index] = length_
            position_dict[index] = np.pad(np.arange(length_) + 1, (0, len(text_) - length_), 'constant')

        return np.array(ids), contents_dict, lengths_dict, raw_content_dict, position_dict

    def convert_relations(self, relation: pd.DataFrame):
        """ Convert relations.
        We want to retrieve positive interactions and negative interactions. Particularly,
        for every pair (query, doc) = 1, we get a list of negatives of the query q

        It is possible that a query may have multiple positive docs. Therefore, negatives[q]
        may vary the lengths but not too much.
        """
        queries, docs, negatives = [], [], collections.defaultdict(list)
        unique_queries = collections.defaultdict(list)

        for index, row in relation.iterrows():
            query = row["id_left"]
            doc = row["id_right"]
            label = row["label"]
            assert label == 0 or label == 1
            unique_queries[query] = unique_queries.get(query, [[], [], [], []]) #  doc, label, content, length
            a, b, c, d = unique_queries[query]
            a.append(doc)
            b.append(label)
            c.append(self.dict_doc_contents[doc])
            d.append(self.dict_doc_lengths[doc])

            if label == 1:
                queries.append(query)
                docs.append(doc)
            elif label == 0:
                negatives[query].append(doc)
        assert len(queries) == len(docs)
        return np.array(queries), np.array(docs), negatives, unique_queries

    def __repr__(self):

        return ('<Interactions dataset ({num_users} users x {num_items} items '
                'x {num_interactions} interactions)>'
            .format(
            num_users = self.num_users,
            num_items = self.num_items,
            num_interactions = len(self)
        ))

    def _check(self):
        pass


class BaseClassificationInteractions(object):
    """ Base classification interactions for fact-checking with evidences """

    def __init__(self, data_pack: matchzoo.DataPack, **kargs):
        # FileHandler.myprint("Converting DataFrame to Normal Dictionary of Data")
        self.output_handler = kargs[KeyWordSettings.OutputHandlerFactChecking]
        self.output_handler.myprint("Converting DataFrame to Normal Dictionary of Data")
        additional_field = {KeyWordSettings.FCClass.CharSourceKey: "char_claim_source"}
        self.unique_query_ids, \
        self.dict_claim_contents, \
        self.dict_claim_lengths, \
        self.dict_query_raw_contents, \
        self.dict_query_positions, \
        self.dict_claim_source, \
        self.dict_raw_claim_source, \
        self.dict_char_left_src = self.convert_leftright(data_pack.left, text_key="text_left",
                                     length_text_key="length_left", raw_text_key="raw_text_left",
                                     source_key="claim_source", raw_source_key="raw_claim_source", **additional_field)
        self.data_pack = data_pack
        assert len(self.unique_query_ids) == len(set(self.unique_query_ids)), "Must be unique ids"
        """ Why do I need to sort it? I have no idea why did I do it? """
        additional_field = {KeyWordSettings.FCClass.CharSourceKey: "char_evidence_source"}
        self.unique_doc_ids, \
        self.dict_doc_contents, \
        self.dict_doc_lengths, \
        self.dict_doc_raw_contents, \
        self.dict_doc_positions, \
        self.dict_evd_source,\
        self.dict_raw_evd_source, \
        self.dict_char_right_src = self.convert_leftright(data_pack.right, text_key="text_right",
                                                      length_text_key="length_right",
                                                      raw_text_key="raw_text_right", source_key="evidence_source",
                                                      raw_source_key="raw_evidence_source", **additional_field)

        assert len(self.unique_doc_ids) == len(set(self.unique_doc_ids)), "Must be unique ids for doc ids"
        assert len(self.unique_query_ids) != len(
            self.unique_doc_ids), "Impossible to have equal number of docs and number of original tweets"

    def convert_leftright(self, part: pd.DataFrame, text_key: str, length_text_key: str, raw_text_key: str,
                          source_key: str, raw_source_key: str, **kargs):
        """ Converting the dataframe of interactions """
        ids, contents_dict, lengths_dict, position_dict = [], {}, {}, {}
        raw_content_dict, sources, raw_sources, char_sources = {}, {}, {}, {}
        char_source_key = kargs[KeyWordSettings.FCClass.CharSourceKey]
        # Why don't we use the queryID as the key for dictionary????
        self.output_handler.myprint("[NOTICE] MatchZoo use queryID and docID as index in dataframe left and right, "
                            "therefore, iterrows will return index which is left_id or right_id")
        for index, row in part.iterrows(): # very dangerous, be careful because it may change order!!!
            ids.append(index)
            text_ = row[text_key]  # text_ here is converted to numbers and padded
            raw_content_dict[index] = row[raw_text_key]

            if length_text_key not in row: length_ = len(text_)
            else: length_ = row[length_text_key]
            assert length_ != 0
            assert index not in contents_dict
            contents_dict[index] = text_
            lengths_dict[index] = length_
            position_dict[index] = np.pad(np.arange(length_) + 1, (0, len(text_) - length_), 'constant')
            sources[index] = row[source_key]
            raw_sources[index] = row[raw_source_key]
            char_sources[index] = row[char_source_key]

        return np.array(ids), contents_dict, lengths_dict, raw_content_dict, \
               position_dict, sources, raw_sources, char_sources

    def convert_relations(self, relation: pd.DataFrame): pass


class ClassificationInteractions(BaseClassificationInteractions):
    """
    This class is for classification based on evidences.
    Query - [list of evidences] -> labels
    """

    def __init__(self, data_pack: matchzoo.DataPack, **kargs):
        super(ClassificationInteractions, self).__init__(data_pack, **kargs)

        # (1) unique claims, (2) labels for each claim and (3) info of each claim
        self.claims, self.claims_labels, self.dict_claims_and_evidences_test = \
            self.convert_relations(data_pack.relation)

        # for queries, padded
        self.np_query_contents = np.array([self.dict_claim_contents[q] for q in self.claims])
        self.np_query_lengths = np.array([self.dict_claim_lengths[q] for q in self.claims])
        self.np_query_char_source = np.array([self.dict_char_left_src[q] for q in self.claims])
        self.query_positions = np.array([self.dict_query_positions[q] for q in self.claims])

        # assert self.np_query_lengths.shape == self.np_doc_lengths.shape
        self.padded_doc_length = len(self.dict_doc_contents[self.unique_doc_ids[0]])
        self.padded_doc_char_source_length = len(self.dict_char_right_src[self.unique_doc_ids[0]])
        # self.padded_query_length = len(self.np_query_contents[0])

    def convert_relations(self, relation: pd.DataFrame):
        """ Convert relations.
        We want to retrieve positive interactions and negative interactions. Particularly,
        for every pair (query, doc) = 1, we get a list of negatives of the query q

        It is possible that a query may have multiple positive docs. Therefore, negatives[q]
        may vary the lengths but not too much.
        """
        queries = []  # , collections.defaultdict(list)
        queries_labels = []
        unique_queries = collections.defaultdict(list)
        set_queries = set()

        for index, row in relation.iterrows():
            query = row["id_left"]
            doc = row["id_right"]
            label = row["label"]
            # assert label == 0 or label == 1
            unique_queries[query] = unique_queries.get(query, [[], [], [], []])  # doc, label, content, length
            a, b, c, d = unique_queries[query]
            a.append(doc)
            b.append(label)
            c.append(self.dict_doc_contents[doc])
            d.append(self.dict_doc_lengths[doc])

            if query not in set_queries:
                queries.append(query)  # same as unique_queries
                queries_labels.append(label)
                set_queries.add(query)

        assert len(queries) == len(unique_queries)
        return np.array(queries), np.array(queries_labels), unique_queries
