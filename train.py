import argparse
import os
import torch
from asr import CTCAttentionASRParser, CLASCTCAttentionASRParser, CopyNEASRParser
from supar.utils.logging import init_logger, logger
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    if "LOCAL_RANK" in os.environ:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main():
    ddp_setup()
    parser = argparse.ArgumentParser(description="Train CopyNE / ASR model", allow_abbrev=False)
    # Common arguments
    parser.add_argument('--path', help='path to model file')
    parser.add_argument('--pre_model', type=str, default="None")
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--e2ener', action='store_true', help='whether it is an e2ener model')
    parser.add_argument('--char_dict', default='data/sp_ner/chinese_char.txt', help='path to the char dict file')
    parser.add_argument('--cmvn', default='data_to_upload/aishell1_global_cmvn_mel80', help='global cmvn file')
    parser.add_argument('--config', default='conf/ctc_mel80.yaml', help='config file')
    parser.add_argument('--add_bert', action='store_true', help='whether to add bert')
    parser.add_argument('--bert', default='bert-base-chinese', help='which bert model to use')
    parser.add_argument('--frame_length', default=25, type=int)
    parser.add_argument('--frame_shift', default=10, type=int)
    parser.add_argument('--max_frame_num', default=10000, type=int)
    parser.add_argument('--add_context', action='store_true', help='whether to add context')
    parser.add_argument('--pad_context', default=3, type=float)
    parser.add_argument('--train_ne_dict', default='data/end2end/aishell_train_ner_most-all.vocab')
    parser.add_argument('--dev_ne_dict', default='data/end2end/aishell_dev_ner_random-500.vocab')
    parser.add_argument('--att_type', default='simpleatt', type=str, choices=['contextual', 'crossatt', 'simpleatt'])
    parser.add_argument('--add_copy_loss', action='store_true')
    parser.add_argument('--no_concat', action='store_true')
    parser.add_argument('--use_avg', action='store_true')

    # Train-specific arguments
    parser.add_argument('--train', default='data/sp_ner/new_train.json', help='path to train file')
    parser.add_argument('--dev', default='data/end2end/dev_single_bracket.json', help='path to dev file')
    parser.add_argument('--test', default='data/sp_ner/new_test.json', help='path to test file')

    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args.mode = 'train'

    torch.manual_seed(args.seed)
    if int((torch.__version__)[0]) > 1:
        torch.set_float32_matmul_precision('high')
    init_logger(logger, os.path.join(args.path, f"{args.mode}.log"))
    logger.info('\n' + str(args))

    if not args.add_context:
        parser_instance = CTCAttentionASRParser(args)
    else:
        if not args.add_copy_loss:
            parser_instance = CLASCTCAttentionASRParser(args)
        else:
            parser_instance = CopyNEASRParser(args)
    logger.info(f'{parser_instance.model}\n')
    parser_instance.train()

    if torch.distributed.is_initialized():
        destroy_process_group()

if __name__ == '__main__':
    main()
