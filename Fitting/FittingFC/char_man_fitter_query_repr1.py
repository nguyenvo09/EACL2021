import torch
import numpy as np
import torch_utils
from Models import base_model
import torch_utils as my_utils
import time
import interactions
from handlers.tensorboard_writer import TensorboardWrapper
from setting_keywords import KeyWordSettings
from Fitting.FittingFC.multi_level_attention_composite_fitter import MultiLevelAttentionCompositeFitter
from typing import List


class CharManFitterQueryRepr1(MultiLevelAttentionCompositeFitter):
    """
    I implement this class for testing if the padding all zeros sequences are the root cause of performance reduction.
    """

    def fit(self, train_iteractions: interactions.ClassificationInteractions,
            verbose = True,  # for printing out evaluation during training
            topN = 10,
            val_interactions: interactions.ClassificationInteractions = None,
            test_interactions: interactions.ClassificationInteractions = None):
        """
        Fit the model.
        Parameters
        ----------
        train_iteractions: :class:`interactions.ClassificationInteractions` The input sequence dataset.
        val_interactions: :class:`interactions.ClassificationInteractions`
        test_interactions: :class:`interactions.ClassificationInteractions`
        """
        self._initialize(train_iteractions)
        best_val_auc, best_val_f1_macro, best_epoch, test_auc = 0, 0, 0, 0
        test_results_dict = None
        iteration_counter = 0
        count_patience_epochs = 0

        for epoch_num in range(self._n_iter):

            # ------ Move to here ----------------------------------- #
            self._net.train(True)
            query_ids, left_contents, left_lengths, query_sources, query_char_sources, \
            evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, \
            pair_labels = self._sampler.get_train_instances_char_man(train_iteractions, self.fixed_num_evidences)

            queries, query_content, query_lengths, query_sources, query_char_sources, \
            evd_docs, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, \
            pair_labels = my_utils.shuffle(query_ids, left_contents, left_lengths, query_sources, query_char_sources,
                                           evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources,
                                           evd_cnt_each_query, evd_char_sources, pair_labels)
            epoch_loss, total_pairs = 0.0, 0
            t1 = time.time()
            for (minibatch_num,
                (batch_query, batch_query_content, batch_query_len, batch_query_sources, batch_query_chr_src,
                 batch_evd_docs, batch_evd_contents, batch_evd_lens, batch_evd_sources, # i.e. claim source
                 batch_evd_cnt_each_query, batch_evd_chr_src, batch_labels)) \
                    in enumerate(my_utils.minibatch(queries, query_content, query_lengths, query_sources, query_char_sources,
                                                    evd_docs, evd_docs_contents, evd_docs_lens, evd_sources,
                                                    evd_cnt_each_query, evd_char_sources, pair_labels, batch_size = self._batch_size)):

                batch_query = my_utils.gpu(torch.from_numpy(batch_query), self._use_cuda)
                batch_query_content = my_utils.gpu(torch.from_numpy(batch_query_content), self._use_cuda)
                # batch_query_len = my_utils.gpu(torch.from_numpy(batch_query_len), self._use_cuda)
                batch_query_sources = my_utils.gpu(torch.from_numpy(batch_query_sources), self._use_cuda)
                batch_query_chr_src = my_utils.gpu(torch.from_numpy(batch_query_chr_src), self._use_cuda)

                batch_evd_docs = my_utils.gpu(torch.from_numpy(batch_evd_docs), self._use_cuda)
                batch_evd_contents = my_utils.gpu(torch.from_numpy(batch_evd_contents), self._use_cuda)
                # batch_evd_lens = my_utils.gpu(torch.from_numpy(batch_evd_lens), self._use_cuda)
                batch_evd_sources = my_utils.gpu(torch.from_numpy(batch_evd_sources), self._use_cuda)
                batch_evd_cnt_each_query = my_utils.gpu(torch.from_numpy(batch_evd_cnt_each_query), self._use_cuda)
                batch_evd_chr_src = my_utils.gpu(torch.from_numpy(batch_evd_chr_src), self._use_cuda)

                batch_labels = my_utils.gpu(torch.from_numpy(batch_labels), self._use_cuda)
                # total_pairs += self._batch_size * self.
                additional_data = {KeyWordSettings.EvidenceCountPerQuery: batch_evd_cnt_each_query,
                                   KeyWordSettings.FCClass.QueryCharSource: batch_query_chr_src,
                                   KeyWordSettings.FCClass.DocCharSource: batch_evd_chr_src}
                self._optimizer.zero_grad()
                if self._loss in ["bpr", "hinge", "pce", "bce", "cross_entropy",
                                  "vanilla_cross_entropy", "regression_loss", "masked_cross_entropy"]:
                    loss = self._get_multiple_evidences_predictions_normal(
                        batch_query, batch_query_content, batch_query_len, batch_query_sources,
                        batch_evd_docs, batch_evd_contents, batch_evd_lens, batch_evd_sources,
                        batch_labels, self.fixed_num_evidences, **additional_data)
                # print("Loss: ", loss)
                epoch_loss += loss.item()
                iteration_counter += 1
                # if iteration_counter % 2 == 0: break
                TensorboardWrapper.mywriter().add_scalar("loss/minibatch_loss", loss.item(), iteration_counter)
                loss.backward()
                self._optimizer.step()
                # for name, param in self._net.named_parameters():
                #     self.tensorboard_writer.add_histogram(name + "/grad", param.grad, iteration_counter)
                #     self.tensorboard_writer.add_histogram(name + "/value", param, iteration_counter)
            # epoch_loss /= float(total_pairs)
            TensorboardWrapper.mywriter().add_scalar("loss/epoch_loss_avg", epoch_loss, epoch_num)
            # print("Number of Minibatches: ", minibatch_num, "Avg. loss of epoch: ", epoch_loss)
            t2 = time.time()
            epoch_train_time = t2 - t1
            if verbose:  # validation after each epoch
                f1_macro_val, auc_val = self._output_results_every_epoch(topN, val_interactions, test_interactions,
                                                                         epoch_num, epoch_train_time, epoch_loss)
                if (f1_macro_val > best_val_f1_macro) or \
                        (f1_macro_val == best_val_f1_macro and auc_val > best_val_auc):  # prioritize f1_macro
                    # if (hits + ndcg) > (best_hit + best_ndcg):
                    count_patience_epochs = 0
                    with open(self.saved_model, "wb") as f:
                        torch.save(self._net.state_dict(), f)
                    # test_results_dict = result_test
                    best_val_auc, best_val_f1_macro, best_epoch = auc_val, f1_macro_val, epoch_num
                    # test_hit, test_ndcg = hits_test, ndcg_test
                else: count_patience_epochs += 1
                if self._early_stopping_patience and count_patience_epochs > self._early_stopping_patience:
                    self.output_handler.myprint("Early Stopped due to no better performance in %s epochs" % count_patience_epochs)
                    break

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'.format(epoch_loss))
        self._flush_training_results(best_val_auc, best_epoch)

    def _get_multiple_evidences_predictions_normal(self, query_ids: torch.Tensor,
                                                   query_contents: torch.Tensor,
                                                   query_lens: np.ndarray,
                                                   query_sources: torch.Tensor,
                                                   evd_doc_ids: torch.Tensor,
                                                   evd_doc_contents: torch.Tensor,
                                                   evd_docs_lens: np.ndarray,
                                                   evd_sources: torch.Tensor,
                                                   labels: np.ndarray,
                                                   n: int, **kargs) -> torch.Tensor:
        """
        compute cross entropy loss
        Parameters
        ----------
        query_ids: (B, )
        query_contents: (B, L)
        query_lens: (B, )
        evd_doc_ids: (B, n)
        evd_doc_contents: (B, n, R)
        evd_docs_lens: (B, n)
        evd_sources: (B, n)
        labels: (B, ) labels of pair
        n: `int` is the number of evidences for each claim/query
        kargs: `dict`
        Returns
        -------
            loss value based on a loss function
        """
        evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]  # (B, )
        query_char_source = kargs[KeyWordSettings.FCClass.QueryCharSource]
        doc_char_source = kargs[KeyWordSettings.FCClass.DocCharSource]
        assert evd_doc_ids.size() == evd_docs_lens.shape
        assert query_ids.size(0) == evd_doc_ids.size(0)
        assert query_lens.shape == labels.size()
        assert query_contents.size(0) == evd_doc_contents.size(0)  # = batch_size
        _, L = query_contents.size()
        batch_size = query_ids.size(0)
        # prunning at this step to remove padding\
        e_lens, e_conts, q_conts, q_lens = [], [], [], []
        e_chr_src_conts = []
        expaned_labels = []
        for evd_cnt, q_cont, q_len, evd_lens, evd_doc_cont, evd_chr_src, label in \
                zip(evd_count_per_query, query_contents, query_lens,
                    evd_docs_lens, evd_doc_contents, doc_char_source, labels):
            evd_cnt = int(torch_utils.cpu(evd_cnt).detach().numpy())
            e_lens.extend(list(evd_lens[:evd_cnt]))
            e_conts.append(evd_doc_cont[:evd_cnt, :])  # stacking later
            e_chr_src_conts.append(evd_chr_src[:evd_cnt, :])
            q_lens.extend([q_len] * evd_cnt)
            q_conts.append(q_cont.unsqueeze(0).expand(evd_cnt, L))
            expaned_labels.extend([int(torch_utils.cpu(label).detach().numpy())] * evd_cnt)
        # concat
        e_conts = torch.cat(e_conts, dim = 0)  # (n1 + n2 + ..., R)
        e_chr_src_conts = torch.cat(e_chr_src_conts, dim = 0)  # (n1 + n2 + ... , R)
        e_lens = np.array(e_lens)  # (n1 + n2 + ..., )
        q_conts = torch.cat(q_conts, dim = 0)  # (n1 + n2 + ..., R)
        q_lens = np.array(q_lens)
        assert q_conts.size(0) == q_lens.shape[0] == e_conts.size(0) == e_lens.shape[0]

        d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(e_lens)
        e_lens = my_utils.gpu(torch.from_numpy(e_lens), self._use_cuda)
        x = query_lens
        q_new_indices, q_restoring_indices = torch_utils.get_sorted_index_and_reverse_index(x)
        x = my_utils.gpu(torch.from_numpy(x), self._use_cuda)
        # query_lens = my_utils.gpu(torch.from_numpy(query_lens), self._use_cuda)
        additional_paramters = {
            KeyWordSettings.Query_lens: x,
            KeyWordSettings.Doc_lens: evd_docs_lens,
            KeyWordSettings.DocLensIndices: (d_new_indices, d_old_indices, e_lens),
            KeyWordSettings.QueryLensIndices: (q_new_indices, q_restoring_indices, x),
            KeyWordSettings.QuerySources: query_sources,
            KeyWordSettings.DocSources: evd_sources,
            KeyWordSettings.TempLabel: labels,
            KeyWordSettings.DocContentNoPaddingEvidence: e_conts,
            KeyWordSettings.QueryContentNoPaddingEvidence: q_conts,
            KeyWordSettings.EvidenceCountPerQuery: evd_count_per_query,
            KeyWordSettings.FCClass.QueryCharSource: query_char_source,  # (B, 1, L)
            KeyWordSettings.FCClass.DocCharSource: e_chr_src_conts,
            KeyWordSettings.FIXED_NUM_EVIDENCES: n
        }
        predictions = self._net(query_contents, evd_doc_contents, **additional_paramters)  # (B, )
        # labels.unsqueeze(-1).expand(batch_size, n).reshape(batch_size * n)
        # labels = torch_utils.gpu(torch.from_numpy(np.array(expaned_labels)), self._use_cuda)
        # print("Labels: ", labels)
        # mask = (evd_doc_ids >= 0).view(batch_size * n).float()
        return self._loss_func(predictions, labels.float())

    def evaluate(self, testRatings: interactions.ClassificationInteractions, K: int, output_ranking = False, **kargs):
        """
        Compute evaluation metrics. No swearing in code please!!!
        Parameters
        ----------
        testRatings
        K
        output_ranking: whether we should output predictions
        kargs

        Returns
        -------

        """
        all_labels = []
        all_final_probs = []
        list_error_analysis = []
        for query, evidences_info in testRatings.dict_claims_and_evidences_test.items():
            evd_ids, labels, evd_contents, evd_lengths = evidences_info
            assert len(set(labels)) == 1, "Must have only one label due to same claim"
            all_labels.append(labels[0])
            claim_content = testRatings.dict_claim_contents[query]
            claim_source = np.array([testRatings.dict_claim_source[query]])  # (1, )
            claim_char_src = np.array([testRatings.dict_char_left_src[query]])
            evd_sources = np.array([testRatings.dict_evd_source[e] for e in evd_ids])  # (len(labels), 1)
            evd_sources = self._pad_article_sources(evd_sources)  # (1, 30)
            evd_char_src = np.array([testRatings.dict_char_right_src[e] for e in evd_ids])  # (len(labels), 1)
            query_len = np.array([testRatings.dict_claim_lengths[query]])  # shape = (1, ) where B =1
            # doc_lens = [testRatings.dict_doc_lengths[d] for d in docs]

            claim_content = np.tile(claim_content, (1, 1))  # (1, L)
            L = claim_content.shape[1]
            evd_contents = np.array(evd_contents).reshape(1, len(labels), -1)  # shape = (1, n, R)
            padded_evd_contents = self._pad_evidences(evd_contents)
            # claim_content = my_utils.gpu(claim_content)
            # evd_contents = my_utils.gpu(evd_contents)

            claim_content = my_utils.gpu(my_utils.numpy2tensor(claim_content, dtype=torch.int), self._use_cuda)
            evd_contents = my_utils.gpu(my_utils.numpy2tensor(evd_contents, dtype=torch.int), self._use_cuda)  # (1, x, R)
            padded_evd_contents = my_utils.gpu(my_utils.numpy2tensor(padded_evd_contents, dtype=torch.int), self._use_cuda)  # (1, x, R)

            # for evidences
            evd_lengths = np.array(evd_lengths)
            d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(evd_lengths)
            evd_lengths = my_utils.gpu(my_utils.numpy2tensor(evd_lengths, dtype=torch.int), self._use_cuda)
            x = query_len # np.repeat(query_len, len(labels))
            q_new_indices, q_restoring_indices = torch_utils.get_sorted_index_and_reverse_index(x)
            x = my_utils.gpu(my_utils.numpy2tensor(x, dtype=torch.int), self._use_cuda)

            # for sources
            claim_source = my_utils.gpu(my_utils.numpy2tensor(claim_source, dtype=torch.int), self._use_cuda)
            evd_sources = my_utils.gpu(my_utils.numpy2tensor(evd_sources, dtype=torch.int), self._use_cuda)
            claim_char_src = my_utils.gpu(my_utils.numpy2tensor(claim_char_src, dtype=torch.int), self._use_cuda)
            evd_char_src = my_utils.gpu(my_utils.numpy2tensor(evd_char_src, dtype=torch.int), self._use_cuda)
            additional_information = {
                KeyWordSettings.Query_lens: x,
                KeyWordSettings.QueryLensIndices: (q_new_indices, q_restoring_indices, x),
                KeyWordSettings.Doc_lens: evd_lengths,
                KeyWordSettings.DocLensIndices: (d_new_indices, d_old_indices, evd_lengths),
                KeyWordSettings.QuerySources: claim_source,
                KeyWordSettings.DocSources: evd_sources,  # (B = 1, n = 30)
                KeyWordSettings.DocContentNoPaddingEvidence: evd_contents.view(1 * len(labels), -1),  # (B1, R)
                KeyWordSettings.FIXED_NUM_EVIDENCES: self.fixed_num_evidences,
                KeyWordSettings.EvidenceCountPerQuery: torch_utils.gpu(torch.from_numpy(np.array([len(labels)])), self._use_cuda),
                KeyWordSettings.QueryContentNoPaddingEvidence: claim_content.expand(len(labels), L),
                KeyWordSettings.OutputRankingKey: output_ranking,
                KeyWordSettings.FCClass.QueryCharSource: claim_char_src.long(),
                KeyWordSettings.FCClass.DocCharSource: evd_char_src.long()
            }
            # padded_evd_contents = self._pad_evidences(evd_contents) # 1, 30, R
            probs = self._net.predict(claim_content, padded_evd_contents, **additional_information)  # shape = (len(labels), )
            if output_ranking:
                probs, att_scores = probs
                more_info = {KeyWordSettings.FCClass.AttentionWeightsInfo: att_scores}
                list_error_analysis.append(self._prepare_error_analysis(testRatings, query, evd_ids, probs, labels, **more_info))

            all_final_probs.append(float(my_utils.cpu(probs).detach().numpy().flatten()))

        results = self._computing_metrics(true_labels = all_labels, predicted_probs = all_final_probs)
        if output_ranking: return results, list_error_analysis  # sorted(list_error_analysis, key=lambda x: x["qid"])
        return results
