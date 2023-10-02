#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo, Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import MerlionPretrainDataModule
from conformer import ConformerMTL

from lhotse.cut import Cut
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint
from icefall.decode import (
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_attention_decoder,
    rescore_with_n_best_list,
    rescore_with_rnn_lm,
    rescore_with_whole_lattice,
)
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.rnn_lm.model import RnnLmModel
from icefall.utils import (
    AttributeDict,
    get_texts,
    load_averaged_model,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=77,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="It specifies the checkpoint to use for decoding.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=55,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="attention-decoder",
        help="""Decoding method.
        Supported values are:
            - (0) ctc-decoding. Use CTC decoding. It uses a sentence piece
              model, i.e., lang_dir/bpe.model, to convert word pieces to words.
              It needs neither a lexicon nor an n-gram LM.
            - (1) 1best. Extract the best path from the decoding lattice as the
              decoding result.
            - (2) nbest. Extract n paths from the decoding lattice; the path
              with the highest score is the decoding result.
            - (3) nbest-rescoring. Extract n paths from the decoding lattice,
              rescore them with an n-gram LM (e.g., a 4-gram LM), the path with
              the highest score is the decoding result.
            - (4) whole-lattice-rescoring. Rescore the decoding lattice with an
              n-gram LM (e.g., a 4-gram LM), the best path of rescored lattice
              is the decoding result.
            - (5) attention-decoder. Extract n paths from the LM rescored
              lattice, the path with the highest score is the decoding result.
            - (6) rnn-lm. Rescoring with attention-decoder and RNN LM. We assume
              you have trained an RNN LM using ./rnn_lm/train.py
            - (7) nbest-oracle. Its WER is the lower bound of any n-best
              rescoring method can achieve. Useful for debugging n-best
              rescoring method.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""Number of paths for n-best based decoding method.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, attention-decoder, rnn-lm, and nbest-oracle
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, attention-decoder, rnn-lm, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_500",
        help="The lang dir",
    )

    parser.add_argument(
        "--lm-dir",
        type=str,
        default="data/lm",
        help="""The n-gram LM dir.
        It should contain either G_4_gram.pt or G_4_gram.fst.txt
        """,
    )

    parser.add_argument(
        "--rnn-lm-exp-dir",
        type=str,
        default="rnn_lm/exp",
        help="""Used only when --method is rnn-lm.
        It specifies the path to RNN LM exp dir.
        """,
    )

    parser.add_argument(
        "--rnn-lm-epoch",
        type=int,
        default=7,
        help="""Used only when --method is rnn-lm.
        It specifies the checkpoint to use.
        """,
    )

    parser.add_argument(
        "--rnn-lm-avg",
        type=int,
        default=2,
        help="""Used only when --method is rnn-lm.
        It specifies the number of checkpoints to average.
        """,
    )

    parser.add_argument(
        "--rnn-lm-embedding-dim",
        type=int,
        default=2048,
        help="Embedding dim of the model",
    )

    parser.add_argument(
        "--rnn-lm-hidden-dim",
        type=int,
        default=2048,
        help="Hidden dim of the model",
    )

    parser.add_argument(
        "--rnn-lm-num-layers",
        type=int,
        default=4,
        help="Number of RNN layers the model",
    )
    parser.add_argument(
        "--rnn-lm-tie-weights",
        type=str2bool,
        default=False,
        help="""True to share the weights between the input embedding layer and the
        last output linear layer
        """,
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
            "subsampling_factor": 4,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            "feature_dim": 80,
            "nhead": 8,
            "attention_dim": 512,
            "num_decoder_layers": 6,
            # parameters for decoding
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
            "env_info": get_env_info(),
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    device,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if no rescoring is used, the key is the string `no_rescore`.
               If LM rescoring is used, the key is the string `lm_scale_xxx`,
               where `xxx` is the value of `lm_scale`. An example key is
               `lm_scale_0.7`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding without LM rescoring.
        - params.method is "nbest", it uses nbest decoding without LM rescoring.
        - params.method is "nbest-rescoring", it uses nbest LM rescoring.
        - params.method is "whole-lattice-rescoring", it uses whole lattice LM
          rescoring.

      model:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]

    nnet_output, lid_logits, memory, memory_key_padding_mask = model(feature, supervisions)
    # nnet_output is (N, T, C)
    # lid_logits is (1, N, LID_C)

    # Simply choose the argmax as the prediction
    lid_logits = lid_logits.squeeze(0)
    hyps = torch.argmax(lid_logits, dim=-1)
    # lid_pred is (N,)
    key = "lid-cls"
    return {key: hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    device,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    label2id = dict(Chinese=0, English=1, NON_SPEECH=2)
    for batch_idx, batch in enumerate(dl):
        if batch_idx == 5:
            break
        supervisions = batch["supervisions"]
        langids = torch.LongTensor([label2id[cut.supervisions[0].language] if cut.supervisions[0].language in label2id else label2id['NON_SPEECH'] for cut in supervisions['cut']]).to(device)
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            batch=batch,
            device=device,
        )

        if hyps_dict is not None:
            for lm_scale, hyps in hyps_dict.items():
                this_batch = []
                assert len(hyps) == len(langids)
                for cut_id, hyp_lids, ref_lids in zip(cut_ids, hyps, langids):
                    this_batch.append((cut_id, ref_lids, hyp_lids))

                results[lm_scale].extend(this_batch)
        else:
            raise RuntimeError("Empty hyps_dict.")

        num_cuts += len(langids)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(
                f"batch {batch_str}, cuts processed until now is {num_cuts}"
            )
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[int], List[int]]]],
):
    # res_path = params.exp_dir / f"results.txt"
    # with open(res_path, "w") as f:
    breakpoint()
    correct = 0
    total = 0
    for (uttid, ref, hyp) in results_dict['lid-cls']:
        ref = ref.item()
        hyp = hyp.item()
        if ref == hyp:
            correct += 1
        total += 1
    print(f"Accuracy: {correct/total}")

    # if params.method in ("attention-decoder", "rnn-lm"):
    #     # Set it to False since there are too many logs.
    #     enable_log = False
    # else:
    #     enable_log = True
    # test_set_wers = dict()
    # for key, results in results_dict.items():
    #     recog_path = params.exp_dir / f"recogs-{test_set_name}-{key}.txt"
    #     results = sorted(results)
    #     store_transcripts(filename=recog_path, texts=results)
    #     if enable_log:
    #         logging.info(f"The transcripts are stored in {recog_path}")

    #     # # The following prints out WERs, per-word error statistics and aligned
    #     # # ref/hyp pairs.
    #     # errs_filename = params.exp_dir / f"errs-{test_set_name}-{key}.txt"
    #     # with open(errs_filename, "w") as f:
    #     #     wer = write_error_stats(
    #     #         f, f"{test_set_name}-{key}", results, enable_log=enable_log
    #     #     )
    #     #     test_set_wers[key] = wer

    #     # if enable_log:
    #     #     logging.info(
    #     #         "Wrote detailed error stats to {}".format(errs_filename)
    #     #     )

    # test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    # errs_info = params.exp_dir / f"wer-summary-{test_set_name}.txt"
    # with open(errs_info, "w") as f:
    #     print("settings\tWER", file=f)
    #     for key, val in test_set_wers:
    #         print("{}\t{}".format(key, val), file=f)

    # s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    # note = "\tbest for {}".format(test_set_name)
    # for key, val in test_set_wers:
    #     s += "{}\t{}{}\n".format(key, val, note)
    #     note = ""
    # logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    MerlionPretrainDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.lm_dir = Path(args.lm_dir)

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-{params.method}/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    params.num_classes = num_classes

    model = ConformerMTL(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=params.vgg_frontend,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )

    if params.avg == 1:
        if params.batch > 0:
            load_checkpoint(f"{params.exp_dir}/finetune-epoch-{params.epoch}-batch-{params.batch}.pt", model)
        elif params.batch == 0:
            load_checkpoint(f"{params.exp_dir}/finetune-epoch-{params.epoch}.pt", model)
        else:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        model = load_averaged_model(
            params.exp_dir, model, params.epoch, params.avg, device
        )

    # breakpoint()
    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    merlion = MerlionPretrainDataModule(args)

    test_cuts = merlion.test_cuts()
    # test_cuts = merlion.dev_cuts()

    test_dl = merlion.test_dataloaders(test_cuts)

    test_sets = ["test"]
    test_dl = [test_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            device=device,
        )

        save_results(
            params=params, test_set_name=test_set, results_dict=results_dict
        )

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
