import logging
import os
import random

import numpy as np
import torch
from mlflow import log_metric
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
# from colbert import base_config
from colbert.modeling.tokenization.utils import CostomTokenizer
from colbert.parameters import DEVICE
from colbert.training.losses import listnet_loss, listMLEWeighted, BiEncoderNllLoss
from colbert.training.training_utils import SequentialDistributedSampler, moving_average, sample_T_scheduler, split_parameters, MAverage, coef_scheduler, distributed_concat, collection_qd_masks, \
    mix_qd, pre_batch_enable, get_t5_optimizer
from colbert.training.training_utils import scheduler_neg
from colbert.utils.amp import MixedPrecisionManager
from conf import reader_config, colbert_config, model_config, p_num, padded_p_num, index_config, pos_num, neg_num, SCORE_TEMPERATURE, opt_num, pretrain, lr, eval_p_num, load_trained, \
    calc_re_loss, save_every_eval, teacher_aug
from ir_score_silver import eval_metric_for_data, eval_metric_for_data_noopt
from file_utils import dump_json
from tests.pyserini_search import evaluate_retrieval
from colbert.modeling.colbert_list_qa_gen import ColBERT_List_qa, load_model_helper
from colbert.training.CBQADataset_gen import CBQADataset, collate_fun
from collections import deque

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO if int(os.environ.get('LOCAL_RANK', 0)) in [-1, 0] else logging.WARN)
logger = logging.getLogger(__name__)
from colbert.utils import distributed
# import bitsandbytes as bnb
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()


def setseed(seed, rank):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.cuda.manual_seed_all(seed)


def cleanup():
    torch.distributed.destroy_process_group()


def train(args):
    setseed(12345, args.rank)

    colbert_config['init'] = True
    colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config, load_old=False)
    if load_trained:
        colbert_qa.load(load_trained)

    colbert_qa.to(DEVICE)
    if args.distributed:
        colbert_qa = DDP(colbert_qa, device_ids=[args.rank], find_unused_parameters=True)

    colbert_qa.train()

    tokenizer = CostomTokenizer.from_pretrained(pretrain)
    train_dataset = CBQADataset('webq-train-0', tokenizer=tokenizer, doc_maxlen=colbert_config['doc_maxlen'],
                                query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'])
    train_sampler = DistributedSampler(train_dataset, rank=args.rank, num_replicas=args.nranks) if args.distributed else RandomSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler, pin_memory=True, drop_last=True,
                                  batch_size=args.batch_size, collate_fn=collate_fun())
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epoch
    # for param in colbert_qa.colbert.bert.parameters():
    # for param in colbert_qa.colbert.parameters():
    # param.requires_grad = False

    # params_decay, params_no_decay = [], []
    # lrs = [args.retriever_lr, args.retriever_lr * 2]
    # lrs = [args.retriever_lr, 1e-3, 1e-3]
    lrs = [lr, 1e-3, 1e-3]
    logger.info("learning rate is " + str(lrs))
    params = []
    weight_decay = 0
    # optimized_modules = [colbert_qa.module.colbert, colbert_qa.module.decoder]
    # optimized_modules = [colbert_qa.module.colbert]
    if args.distributed:
        # optimized_modules = [[colbert_qa.module.model, colbert_qa.module.linear]]
        optimized_modules = [[colbert_qa]]
    else:
        optimized_modules = [[colbert_qa.model, colbert_qa.linear]]

    for idx, modules in enumerate(optimized_modules):
        # for idx, module in enumerate([colbert_qa.module.colbert]):
        # for module in [colbert_qa.module.colbert]:
        for module in modules:
            pd, pnd = split_parameters(module)
            params.append({
                'params': pd, 'weight_decay': weight_decay, 'lr': lrs[idx]
            })
            params.append({
                'params': pnd, 'weight_decay': 0.0, 'lr': lrs[idx]
            })

    # retriever_optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.module.colbert.parameters()), lr=args.retriever_lr, eps=1e-8)
    # retriever_optimizer = torch.optim.Adam([
    #     {'params': params_decay},
    #     {'params': params_no_decay, 'weight_decay': 0.0}], lr=args.retriever_lr, weight_decay=1e-2)
    if pretrain.find("t5") != -1:
        retriever_optimizer = get_t5_optimizer(colbert_qa, lr=lr)
    else:
        retriever_optimizer = torch.optim.Adam(params)
    # retriever_optimizer = bnb.optim.Adam8bit(filter(lambda p: p.requires_grad, colbert_qa.module.colbert.parameters()), lr=args.retriever_lr)

    # optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.reader.parameters()), lr=args.lr, eps=1e-8)
    # reader_optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.module.reader.parameters()), lr=args.lr, eps=1e-8)

    retriever_optimizer.zero_grad()
    # reader_optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)

    args.max_grad_norm = 1.0

    # t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epoch
    # train_sampler = RandomSampler(train_dataset)

    warm_up = 0.1
    retriever_scheduler = get_linear_schedule_with_warmup(retriever_optimizer, num_warmup_steps=int(warm_up * t_total), num_training_steps=t_total)

    # reader_scheduler = get_linear_schedule_with_warmup(reader_optimizer, num_warmup_steps=int(warm_up * t_total), num_training_steps=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    model = colbert_qa

    global_step = 0
    # global_log_step = math.floor(len(train_dataloader) / args.gradient_accumulation_steps) // args.logging_steps
    global_log_step = (len(train_dataloader) // args.gradient_accumulation_steps) // args.logging_steps
    logger.info(f"global log step = {global_log_step * args.gradient_accumulation_steps}")

    model_helper = load_model_helper(args.rank)
    # load_model_helper()
    # retriever_criterion = listnet_loss

    # loss_fun = AutomaticWeightedLoss(num=2)
    best_metrics = float('inf')
    retriever_labels = torch.zeros((args.batch_size * opt_num, args.batch_size * p_num * opt_num + padded_p_num), dtype=torch.float, device=DEVICE)

    neg_weight_mask = torch.zeros((args.batch_size * opt_num, args.batch_size * p_num * opt_num + padded_p_num), dtype=torch.float, device=DEVICE)
    qd_queue = deque(maxlen=1)
    batch_queue = deque(maxlen=2)
    for epoch in tqdm(range(args.epoch)):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        if args.distributed:
            train_sampler.set_epoch(epoch)
        epoch_iterator = tqdm(train_dataloader) if args.rank == 0 else train_dataloader
        tr_loss = MAverage()
        re_loss = MAverage()
        ar_loss = MAverage()
        qr_loss = MAverage()
        dr_loss = MAverage()
        teacher_aug_re_loss = MAverage()
        ans_sim_loss = MAverage()
        re_ans_loss = MAverage()
        q_answer_re_loss = MAverage()
        q_answer_kl_loss = MAverage()

        reader_loss_total = 0
        pos_weight, hard_neg_weight, neg_weight = scheduler_neg(epoch, args.epoch)
        logger.info(f"now neg weight={scheduler_neg(epoch, args.epoch)}")
        for i in range(args.batch_size * opt_num):
            # labels[i, i * pn_num:i * pn_num + pos_num] = score[i]
            base_offset = (i // opt_num) * p_num * opt_num
            # next_base_offset = (i // 4 + 1) * p_num * 4
            neg_weight_mask[i, base_offset:base_offset + pos_num] = pos_weight
            neg_weight_mask[i, base_offset + pos_num:base_offset + pos_num + neg_num] = hard_neg_weight
            neg_weight_mask[i, 0:base_offset] = neg_weight
            neg_weight_mask[i, base_offset + pos_num + neg_num:] = neg_weight

        train_dataset.sample_T = sample_T_scheduler(epoch, args.epoch)
        logger.info(f"now sample_T = {sample_T_scheduler(epoch, args.epoch)}")
        coef = coef_scheduler(epoch, args.epoch)
        logger.info(f"now coef = {coef}")

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            # Q0, q_word_mask0, D0, d_word_mask0 = [], [], [], []
            # if len(batch_queue) > 0 and True:
            #     for b in batch_queue:
            #         with torch.no_grad():
            #             Q1, q_word_mask1, D1, d_word_mask1 = \
            #                 model(b, train_dataset, merge=False, doc_enc_training=True, pad_p_num=padded_p_num)
            #         # Q1 = Q1.clone().detach().requires_grad_(True)
            #         # q_word_mask1 = q_word_mask1.clone().detach().requires_grad_(False)
            #         # D1 = D1.clone().detach().requires_grad_(True)
            #         # d_word_mask1 = d_word_mask1.clone().detach().requires_grad_(False)
            #         # model.requires_grad_(True)
            #         Q0.append(Q1)
            #         q_word_mask0.append(q_word_mask1)
            #         D0.append(D1)
            #         d_word_mask0.append(d_word_mask1)
            model.train()
            cur_retriever_loss, cur_q_answer_retriever_loss, cur_q_answer_kl_loss, cur_ar_loss, cur_teacher_aug_retriever_loss \
                = [torch.tensor(0.0).cuda() for _ in range(5)]
            with amp.context():
                # scores = \
                # scores, query_reconstruction_loss, doc_reconstruction_loss = \
                # scores = \
                # torch.autograd.set_detect_anomaly(True)
                # Q, q_word_mask, D, d_word_mask, cur_ar_loss = \
                # cur_retriever_loss, cur_ar_loss, cur_teacher_aug_retriever_loss = \
                cur_retriever_loss, cur_q_answer_retriever_loss, cur_q_answer_kl_loss = \
                    model(batch, train_dataset, merge=False, doc_enc_training=True, pad_p_num=padded_p_num)
                # scores, D_scores = model(batch, train_dataset, merge=False, doc_enc_training=True, pad_p_num=padded_p_num)

                # D_word_mask = D_word_mask.view(Q.size(0), p_num, D.size(1))
                # D = D.view(Q.size(0), p_num, D.size(1), D.size(2))
                # scores = model.query_wise_score(Q, D, q_mask=word_mask, d_mask=D_word_mask)

                # for i in range(len(batch) * opt_num):
                #     # labels[i, i * pn_num:i * pn_num + pos_num] = score[i]
                #     retriever_labels[i, (i // opt_num) * p_num * opt_num:(i // opt_num + 1) * p_num * opt_num] = D_scores[i, :p_num * opt_num]
                #     if padded_p_num:
                #         retriever_labels[i, -padded_p_num:] = D_scores[i, -padded_p_num:]
                # Q0.append(Q)
                # q_word_mask0.append(q_word_mask)
                # D0.append(D)
                # d_word_mask0.append(d_word_mask)
                #
                # Q, q_word_mask, D, d_word_mask = [torch.cat(_, dim=0) for _ in [Q0, q_word_mask0, D0, d_word_mask0]]

                # tQ, tq_word_mask, tD, td_word_mask = collection_qd_masks(Q, q_word_mask, D, d_word_mask, args.rank)
                # Q, q_word_mask, D, d_word_mask, aug_mask, positive_idxes = mix_qd(Q, q_word_mask, D, d_word_mask, aug_num=Q.size(0) // 4, p_num=2, alpha=None)
                # Q, q_word_mask, D, d_word_mask = tQ.clone(), tq_word_mask.clone(), tD.clone(), td_word_mask.clone()
                # if len(qd_queue) != 0 and pre_batch_enable(epoch, args.epoch):
                #     for q1, q_word_mask1, d1, d_word_mask1 in qd_queue:
                #         Q = torch.cat([Q, q1], dim=0)
                #         q_word_mask = torch.cat([q_word_mask, q_word_mask1], dim=0)
                #         D = torch.cat([D, d1], dim=0)
                #         d_word_mask = torch.cat([d_word_mask, d_word_mask1], dim=0)
                # cur_retriever_loss = torch.tensor(0.0).cuda()
                # if calc_re_loss:
                #     if args.distributed:
                #         Q, q_word_mask, D, d_word_mask, teacher_aug_Q, teacher_aug_Q_mask = \
                #             collection_qd_masks([Q, q_word_mask, D, d_word_mask], args.rank)
                #         scores = model.module.score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)
                #         if teacher_aug:
                #             teacher_aug_scores = model.module.score(teacher_aug_Q, D, q_mask=teacher_aug_Q_mask, d_mask=d_word_mask)
                #     else:
                #         scores = model.score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)
                #         if teacher_aug:
                #             teacher_aug_scores = model.module.score(teacher_aug_Q, D, q_mask=teacher_aug_Q_mask, d_mask=d_word_mask)
                # scores[aug_mask.bool()] = -1e4

                ###########scores = (scores / q_word_mask.bool().sum(1)[:, None])################
                # positive_idxes = torch.tensor([_ * p_num for _ in range(Q.size(0))])
                # retriever_loss = retriever_criterion(y_pred=scores, y_true=retriever_labels[:scores.size(0), :scores.size(1)])
                # cur_retriever_loss = retriever_criterion(y_pred=scores / SCORE_TEMPERATURE, y_true=retriever_labels[:scores.size(0), :scores.size(1)], neg_weight_mask=neg_weight_mask)
                # cur_retriever_loss = retriever_criterion(y_pred=scores / SCORE_TEMPERATURE, y_true=retriever_labels[:scores.size(0), :], neg_weight_mask=neg_weight_mask)
                # cur_retriever_loss = retriever_criterion(scores=scores / SCORE_TEMPERATURE,
                #                                          positive_idx_per_question=positive_idxes,
                #                                          hard_negative_idx_per_question=None)
                #
                # if teacher_aug:
                #     cur_teacher_aug_retriever_loss = retriever_criterion(scores=teacher_aug_scores / SCORE_TEMPERATURE,
                #                                                          positive_idx_per_question=positive_idxes,
                #                                                          hard_negative_idx_per_question=None)

                # cur_retriever_loss = retriever_criterion(y_pred=scores / Temperature, y_true=retriever_labels[:scores.size(0), :scores.size(1)] / Temperature)
                # word_num = word_mask.sum(-1).unsqueeze(-1)
                # scores = scores / word_num.cuda()
                # print(retriever_labels)
                # input()
                if (step) % 16 == 111:
                    idx = 0
                    # print(scores[idx])
                    print(retriever_labels[idx])
                    idx = idx // 4
                    print(batch[idx]['background'], '\n', batch[idx]['question'], '\n', batch[idx]['A'], '\n', '\n'.join([_['paragraph'] for _ in batch[idx]['paragraph_a'][:2]]))
                    # input()

                #
                pass
                # scores = scores[:, :-padded_p_num]  # remove padded negs
                # all_input_ids, all_input_mask, all_segment_ids, all_segment_lens, all_labels = [_.cuda() for _ in train_dataset.tokenize_for_reader(batch)]
                # reader_loss, *_ = model.reader(input_ids=all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask,
                #                                segment_lens=all_segment_lens, retriever_bias=scores, labels=all_labels)
                # total_loss = retriever_loss
                # reader_loss = torch.tensor(0.0, requires_grad=True).cuda()
                # total_loss = retriever_loss + reader_loss
                # total_loss = cur_retriever_loss + 1 * answer_reconstruction_loss + 1 * query_reconstruction_loss + 1 * doc_reconstruction_loss
                # total_loss = cur_retriever_loss + cur_ans_sim_loss + cur_retriever_loss_ans
                # total_loss = coef * cur_retriever_loss + (1 - coef) * query_reconstruction_loss + (1 - coef) * 0.5 * doc_reconstruction_loss
                # total_loss = coef * cur_retriever_loss + (1 - coef) * answer_reconstruction_loss
                # total_loss = 1 * cur_retriever_loss + (1 - 1) * answer_reconstruction_loss
                # if calc_re_loss:
                #     if teacher_aug:
                #         total_loss = cur_ar_loss + cur_retriever_loss + 0 * cur_teacher_aug_retriever_loss  # * int(epoch > 15)
                #     else:
                #         # total_loss = cur_ar_loss + cur_retriever_loss
                #         total_loss = cur_retriever_loss * 2 + cur_q_answer_retriever_loss * 2 + cur_q_answer_kl_loss
                # else:
                #     total_loss = cur_ar_loss
                total_loss = cur_retriever_loss
                # total_loss = cur_retriever_loss
                # total_loss = cur_retriever_loss
                # total_loss = cur_retriever_loss
                # total_loss = cur_retriever_loss + answer_reconstruction_loss
                # total_loss = cur_retriever_loss
                if args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / args.gradient_accumulation_steps

            # reader_loss_total += 0
            # reader_loss_total += reader_loss.item()
            # input(total_loss)
            amp.backward(total_loss)

            # batch_queue.append(batch)
            # if pre_batch_enable(epoch, args.epoch):
            #     qd_queue.append((_.detach().requires_grad_(False) for _ in [tQ, tq_word_mask, tD, td_word_mask]))
            if args.distributed:
                # if not torch.isfinite(cur_retriever_loss):
                #     logger.warning('nan loss')
                cur_retriever_loss, cur_q_answer_retriever_loss, cur_q_answer_kl_loss, total_loss = [
                    distributed_concat(tensor=_.unsqueeze(0), num_total_examples=None).mean()
                    for _ in
                    [cur_retriever_loss, cur_q_answer_retriever_loss, cur_q_answer_kl_loss, total_loss]
                ]

            re_loss.add(cur_retriever_loss.item() if torch.isfinite(cur_retriever_loss) else 0)
            tr_loss.add(total_loss.item())
            q_answer_kl_loss.add(cur_q_answer_kl_loss.item())
            q_answer_re_loss.add(cur_q_answer_retriever_loss.item())

            # ar_loss.add(cur_ar_loss.item())
            # teacher_aug_re_loss.add(cur_teacher_aug_retriever_loss.item())
            # ar_loss.add(distributed_concat(tensor=answer_reconstruction_loss.unsqueeze(0), num_total_examples=1).mean().item())
            # avg_qr_loss = qr_loss.send(distributed_concat(tensor=query_reconstruction_loss.unsqueeze(0), num_total_examples=1).mean().item())
            # avg_dr_loss = dr_loss.send(distributed_concat(tensor=doc_reconstruction_loss.unsqueeze(0), num_total_examples=1).mean().item())
            # avg_ans_sim_loss = ans_sim_loss.send(cur_ans_sim_loss.item())
            # avg_re_ans_loss = re_ans_loss.send(cur_retriever_loss_ans.item())

            # avg_ans_sim_loss = ans_sim_loss.send(0)

            # print(total_loss.item(), ir_loss.item(), reader_loss.item())
            # tr_loss += total_loss.item() * args.gradient_accumulation_steps
            # ir_loss_total += ir_loss.item()
            # reader_loss_total += reader_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # print(torch.allclose(model.module.old_colbert.linear.weight, model.module.colbert.linear.weight))
                # amp.step([model.colbert, model.reader], [retriever_optimizer, reader_optimizer])
                optimizers = [retriever_optimizer]
                # amp.step([[model.module.colbert]], optimizers)
                amp.step(optimized_modules, optimizers)

                for optimizer in optimizers:
                    optimizer.zero_grad()
                # reader_scheduler.step()
                retriever_scheduler.step()
                global_step += 1
                # torch.distributed.barrier()

            # continue
            if (step + 1) % args.gradient_accumulation_steps == 0 and (step + 1) % global_log_step == 0:
                # distributed.barrier(args.rank)
                # results, _, _ = evaluate(args, model)
                eval_loss = eval_retrieval(args, model)
                # eval_loss = 0
                if args.rank == 0:
                    # model.save(save_dir=args.output_dir)
                    if save_every_eval or eval_loss < best_metrics:
                        # best_metrics = results['accuracy']
                        best_metrics = min(eval_loss, best_metrics)
                        if args.distributed:
                            model.module.save(args.output_dir)
                        else:
                            model.save(args.output_dir)
                        # save_model(args, model, tokenizer)
                        # best_results = {'epoch': epoch, 'global_step': global_step}
                        # best_results.update(results)
                    # save_model(args, model, tokenizer, step=global_step)

                    logger.info("Current best loss = %s", best_metrics)
                    logger.info("global_log_step = %s", global_log_step)
                    logger.info("re_loss = %s", re_loss.get_average())
                    # logger.info("re_ans_loss = %s", avg_re_ans_loss)
                    logger.info("tr_loss = %s", tr_loss.get_average())
                    logger.info("ar_loss = %s", ar_loss.get_average())
                    logger.info("teacher_aug_re_loss = %s", teacher_aug_re_loss.get_average())
                    logger.info("q_answer_re_loss = %s", q_answer_re_loss.get_average())
                    logger.info("q_answer_kl_loss = %s", q_answer_kl_loss.get_average())
                    logger.info("qr_loss = %s", qr_loss.get_average())
                    logger.info("dr_loss = %s", dr_loss.get_average())
                    # logger.info("ans_sim_loss = %s", avg_ans_sim_loss)
                    logger.info("eval_loss = %s", eval_loss)

                    # log_metric("eval_loss", eval_loss)
                    # log_metric('ar_loss', avg_ar_loss)
                    # log_metric('qr_loss', avg_qr_loss)
                    # log_metric('train_loss', avg_tr_loss)
                if args.distributed:
                    distributed.barrier(args.rank)

            if args.rank <= 0:
                epoch_iterator.set_postfix(avg_ir='%.4f' % re_loss.get_average(), avg_reader='%.4f' % (reader_loss_total / (step + 1)),  # elapsed='%.4d' % elapsed,
                                           avg_total='%.4f' % tr_loss.get_average(), lr='%.1Ef' % retriever_scheduler.get_last_lr()[0])  # reader_lr='%.4e' % (scheduler.get_last_lr()[0]))
            # ir_lr='%.4f' % (retriever_scheduler.get_last_lr()[0]))
        # model.module.old_colbert = load_model(colbert_config).cuda()
        # model.module.old_colbert.load_state_dict(model.module.colbert.state_dict())
        # print(torch.allclose(model.module.old_colbert.linear.weight, model.module.colbert.linear.weight))

        # logger.info("refreshed colbert")
        # if args.rank == 0:
        #     eval_retrieval(args, model.module)
        # torch.distributed.barrier()

    # colbert_qa.module.save(os.path.join(args.output_dir, "rouge12_model"))


def eval_retrieval(args, colbert_qa=None):
    # setseed(12345 + args.rank)
    if colbert_qa is None:
        # colbert_config['checkpoint'] = f"output/webq/{save_model_name}/pytorch.bin"
        exit()
        colbert_config['init'] = True
        colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config, load_old=False)
        colbert_qa.load(colbert_config['checkpoint'])
    colbert_qa.to(DEVICE)
    # if args.distributed:
    #     colbert_qa = DDP(colbert_qa, device_ids=[args.rank]).module

    colbert_qa.eval()
    # for param in colbert_qa.colbert.bert.parameters():
    # for param in colbert_qa.colbert.parameters():
    # param.requires_grad = False

    amp = MixedPrecisionManager(args.amp)

    tokenizer = CostomTokenizer.from_pretrained(pretrain)
    train_dataset = CBQADataset('webq-dev-0', tokenizer=tokenizer, doc_maxlen=colbert_config['doc_maxlen'],
                                query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'])
    # t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epoch
    # train_sampler = SequentialSampler(train_dataset)
    train_sampler = DistributedSampler(train_dataset) if args.distributed else SequentialSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler, pin_memory=True, drop_last=False,
                                  batch_size=args.batch_size, collate_fn=collate_fun())
                                  # batch_size=args.batch_size * 2, collate_fn=collate_fun())
    model = colbert_qa

    model_helper = load_model_helper(args.rank)
    # load_model_helper()
    # retriever_criterion = listMLE

    # loss_fun = AutomaticWeightedLoss(num=2)
    best_metrics = 0
    # retriever_labels = torch.zeros((args.batch_size * 4, args.batch_size * p_num * 4 + padded_p_num), dtype=torch.float, device=DEVICE)
    epoch_iterator = tqdm(train_dataloader) if args.rank == 0 else train_dataloader
    tr_loss = 0
    retriever_loss_total = 0
    reader_loss_total = 0
    output_data = []
    # global padded_p_num, p_num
    # t_pnum, tpad_pnum = p_num, padded_p_num
    # p_num, padded_p_num = 100, 0
    # eval_p_num = 5
    # eval_pad_p_num = 5
    # retriever_labels = torch.zeros((args.batch_size * 4, args.batch_size * eval_p_num * 4), dtype=torch.float, device=DEVICE)
    # retriever_labels = torch.zeros((args.batch_size * 4, args.batch_size * eval_p_num * 4 + eval_pad_p_num), dtype=torch.float, device=DEVICE)

    eval_loss = 0
    eval_steps = 0
    logger.info("doing evaluating")
    # ar_loss = MAverage()
    # re_loss = MAverage()
    re_loss = torch.tensor(0.0).cuda()
    ar_loss = torch.tensor(0.0).cuda()
    teacher_aug_re_loss = torch.tensor(0.0).cuda()
    q_answer_re_loss = torch.tensor(0.0).cuda()
    q_answer_kl_loss = torch.tensor(0.0).cuda()
    # qr_loss = MAverage()
    # dr_loss = MAverage()
    for step, batch in enumerate(epoch_iterator):
        # Skip past any already trained steps if resuming training
        model.eval()
        with torch.no_grad():
            # ids, mask, word_mask = train_dataset.tokenize_for_retriever(batch)
            # Q = model.colbert.query(ids, mask)
            # retrieval_scores, d_paras = model.retriever_forward(Q, q_word_mask=word_mask, labels=None)
            # model_helper.merge_to_reader_input(batch, d_paras)
            cur_retriever_loss, cur_q_answer_retriever_loss, cur_q_answer_kl_loss = [torch.tensor(0.0).cuda() for _ in range(3)]
            with amp.context():
                # scores, D_scores, answer_reconstruction_loss, query_reconstruction_loss = model(batch, train_dataset, is_evaluating=True, merge=False, doc_enc_training=True, eval_p_num=eval_p_num, pad_p_num=eval_pad_p_num)
                # scores, answer_reconstruction_loss = model(batch, train_dataset, is_evaluating=True, merge=False, doc_enc_training=True, eval_p_num=eval_p_num, pad_p_num=eval_pad_p_num)
                # scores, query_reconstruction_loss, doc_reconstruction_loss\
                # scores \
                cur_retriever_loss, cur_q_answer_retriever_loss, cur_q_answer_kl_loss = \
                    model(batch, train_dataset, is_evaluating=True, merge=False, doc_enc_training=True, eval_p_num=eval_p_num, pad_p_num=0)
                # scores = model(batch, train_dataset, is_evaluating=True, merge=False, doc_enc_training=True, eval_p_num=eval_p_num, pad_p_num=eval_pad_p_num)
                # pass
                # output_data += batch
                # padded_negs = [model_helper.all_paras[_] for _ in np.random.randint(1, len(model_helper.all_paras), padded_p_num)]

                # D_ids, D_mask, D_word_mask, D_scores = train_dataset.tokenize_for_train_retriever(batch, [])
                # D = model.colbert.doc(D_ids, D_mask)
                #
                # scores = model.colbert.score(Q, D, q_mask=word_mask, d_mask=D_word_mask)
                # D_word_mask = D_word_mask.view(Q.size(0), p_num, D.size(1))
                # D = D.view(Q.size(0), p_num, D.size(1), D.size(2))
                # scores = model.query_wise_score(Q, D, q_mask=word_mask, d_mask=D_word_mask)
                # for i in range(len(batch) * opt_num):
                #     # labels[i, i * pn_num:i * pn_num + pos_num] = score[i]
                #     retriever_labels[i, (i // opt_num) * eval_p_num * opt_num:(i // opt_num + 1) * eval_p_num * opt_num] = D_scores[i, :eval_p_num * opt_num] / Temperature
                #     if padded_p_num:
                #         retriever_labels[i, -padded_p_num:] = D_scores[i, -padded_p_num:] / Temperature
                # if args.rank > 0:
                #     torch.distributed.barrier()
                # assert args.distributed
                # scores = model.module.colbert.score(all_Q, all_D, q_mask=all_q_word_mask, d_mask=all_d_word_mask)
                # if calc_re_loss:
                #     if args.distributed:
                #         Q, q_word_mask, D, d_word_mask = collection_qd_masks([Q, q_word_mask, D, d_word_mask, args.rank])
                #         scores = model.module.score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)
                #     else:
                #         scores = model.score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)
                #
                #     positive_idxes = torch.tensor([_ * eval_p_num for _ in range(Q.size(0))])
                #     cur_retriever_loss = retriever_criterion(scores=scores / SCORE_TEMPERATURE,
                #                                              positive_idx_per_question=positive_idxes,
                #                                              hard_negative_idx_per_question=None)
                # print(retriever_labels)
                # input()
                if (step) % 25 == 111:
                    idx = 0
                    # print(scores)
                    # print(retriever_labels)
                    idx = idx // 4
                    print(batch[idx]['background'], '\n', batch[idx]['question'], '\n', batch[idx]['A'], '\n', '\n'.join([_['paragraph'] for _ in batch[idx]['paragraph_a'][:2]]))
                    # input()
                # if args.rank == 0 and args.distributed:
                #     torch.distributed.barrier()

                # retriever_loss = retriever_criterion(y_pred=scores, y_true=retriever_labels[:scores.size(0), :scores.size(1)])
                # retriever_loss = retriever_criterion(y_pred=scores, y_true=retriever_labels[:scores.size(0), :scores.size(1)])
            # eval_loss += retriever_loss
            re_loss += cur_retriever_loss
            q_answer_re_loss += cur_q_answer_retriever_loss
            q_answer_kl_loss += cur_q_answer_kl_loss

            # ar_loss += cur_ar_loss
            # teacher_aug_re_loss += cur_teacher_aug_retriever_loss
            # ar_loss.add(cur_ar_loss.item())
            # qr_loss.add(distributed_concat(tensor=query_reconstruction_loss.unsqueeze(0), num_total_examples=1).mean().item())
            # dr_loss.add(distributed_concat(tensor=doc_reconstruction_loss.unsqueeze(0), num_total_examples=1).mean().item())
            # ar_loss.add(0)
            eval_steps += 1
            # word_num = word_mask.sum(-1).unsqueeze(-1)
            # scores = scores / word_num.cuda()
            #

            # pass
            # scores = scores[:, :-padded_p_num]  # remove padded negs
            # all_input_ids, all_input_mask, all_segment_ids, all_segment_lens, all_labels = [_.cuda() for _ in train_dataset.tokenize_for_reader(batch)]
            # reader_loss, *_ = model.reader(input_ids=all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask,
            #                                segment_lens=all_segment_lens, retriever_bias=scores, labels=all_labels)
            # total_loss = retriever_loss
            # reader_loss = torch.tensor(0.0, requires_grad=True).cuda()
            # total_loss = retriever_loss + reader_loss
            # total_loss = retriever_loss

        # retriever_loss_total += retriever_loss.item()
        # reader_loss_total += 0
        # reader_loss_total += reader_loss.item()
        # tr_loss += total_loss.item()

        if args.rank <= 0:
            epoch_iterator.set_postfix(avg_ir='%.4f' % (retriever_loss_total / (step + 1)), avg_reader='%.4f' % (reader_loss_total / (step + 1)),  # elapsed='%.4d' % elapsed,
                                       avg_total='%.4f' % (tr_loss / (step + 1)), )
    # torch.distributed.barrier()
    if args.distributed:
        re_loss, q_answer_re_loss, q_answer_kl_loss = [
            distributed_concat(tensor=_.unsqueeze(0), num_total_examples=None).mean()
            for _ in
            [re_loss / len(train_dataloader), q_answer_re_loss / len(train_dataloader), q_answer_kl_loss / len(train_dataloader)]
        ]

    eval_losses = {"eval_re_loss": re_loss.item(),
                   "eval_ar_loss": ar_loss.item(),
                   "eval_teacher_aug_re_loss": teacher_aug_re_loss.item(),
                   "q_answer_kl_loss": q_answer_kl_loss.item(),

                   # "eval_qr_loss": qr_loss.get_average(),
                   # "eval_dr_loss": dr_loss.get_average(),
                   # "eval_tr_loss": re_loss.get_average() + ar_loss.get_average()}
                   "eval_tr_loss": re_loss.item()}
    for k, v in eval_losses.items():
        logger.info(f"{k} = %s", v)

    # eval_metric_for_data(output_data)
    # p_num, padded_p_num = t_pnum, tpad_pnum
    return eval_losses['eval_re_loss']


def eval_retrieval_for_model(args):
    # setseed(12345 + args.rank)
    colbert_config['checkpoint'] = args.checkpoint
    colbert_config['init'] = True
    colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config, load_old=False)
    colbert_qa.load(colbert_config['checkpoint'])
    colbert_qa.to(DEVICE)
    # if args.distributed:
    #     colbert_qa = DDP(colbert_qa, device_ids=[args.rank]).module

    colbert_qa.eval()
    # for param in colbert_qa.colbert.bert.parameters():
    # for param in colbert_qa.colbert.parameters():
    # param.requires_grad = False

    amp = MixedPrecisionManager(args.amp)

    tokenizer = CostomTokenizer.from_pretrained(pretrain)
    train_dataset = CBQADataset('webq-test-0', tokenizer=tokenizer, doc_maxlen=colbert_config['doc_maxlen'],
                                query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'])
    # t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epoch
    train_sampler = SequentialSampler(train_dataset)
    # train_sampler = DistributedSampler(train_dataset) if args.distributed else SequentialSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler, pin_memory=True, drop_last=False,
                                  batch_size=args.batch_size, collate_fn=collate_fun())
    model = colbert_qa

    index_config['index_path'] = args.index_path
    index_config['faiss_index_path'] = args.index_path + "ivfpq.2000.faiss"
    # index_config["faiss_depth"] = 64
    # index_config["n_probe"] = 320
    # model_helper = load_model_helper(args.rank)
    # model_helper = load_model_helper(0)
    model_helper = load_model_helper()
    # retriever_criterion = listnet_loss

    # loss_fun = AutomaticWeightedLoss(num=2)
    best_metrics = 0
    # retriever_labels = torch.zeros((args.batch_size * 4, args.batch_size * p_num * 4 + padded_p_num), dtype=torch.float, device=DEVICE)
    epoch_iterator = tqdm(train_dataloader) if args.rank == 0 else train_dataloader
    tr_loss = 0
    retriever_loss_total = 0
    reader_loss_total = 0
    output_data = []
    # global padded_p_num, p_num
    # t_pnum, tpad_pnum = p_num, padded_p_num
    # p_num, padded_p_num = 100, 0
    eval_p_num = 10
    retriever_labels = torch.zeros((args.batch_size * 4, args.batch_size * eval_p_num * 4), dtype=torch.float, device=DEVICE)
    eval_loss = 0
    eval_steps = 0
    for step, batch in enumerate(epoch_iterator):
        # Skip past any already trained steps if resuming training
        model.eval()
        with amp.context(), torch.no_grad():
            # ids, mask, word_mask = train_dataset.tokenize_for_retriever(batch)
            # Q = model.colbert.query(ids, mask)
            # retrieval_scores, d_paras = model.retriever_forward(Q, q_word_mask=word_mask, labels=None)
            # model_helper.merge_to_reader_input(batch, d_paras)
            model(batch, train_dataset, is_testing_retrieval=True)
            # pass
            output_data += batch
            continue
            # padded_negs = [model_helper.all_paras[_] for _ in np.random.randint(1, len(model_helper.all_paras), padded_p_num)]

            # D_ids, D_mask, D_word_mask, D_scores = train_dataset.tokenize_for_train_retriever(batch, [])
            # D = model.colbert.doc(D_ids, D_mask)
            #
            # scores = model.colbert.score(Q, D, q_mask=word_mask, d_mask=D_word_mask)
            # D_word_mask = D_word_mask.view(Q.size(0), p_num, D.size(1))
            # D = D.view(Q.size(0), p_num, D.size(1), D.size(2))
            # scores = model.query_wise_score(Q, D, q_mask=word_mask, d_mask=D_word_mask)
            for i in range(len(batch) * 4):
                # labels[i, i * pn_num:i * pn_num + pos_num] = score[i]
                retriever_labels[i, (i // 4) * eval_p_num * 4:(i // 4 + 1) * eval_p_num * 4] = D_scores[i, :eval_p_num * 4]
            # if args.rank > 0:
            #     torch.distributed.barrier()

            # print(retriever_labels)
            # input()
            # if (step) % 25 == 26:
            #     idx = 0
            #     print(scores[idx])
            #     print(retriever_labels[idx] * Temperature)
            #     idx = idx // 4
            #     print(batch[idx]['background'], '\n', batch[idx]['question'], '\n', batch[idx]['A'], '\n', '\n'.join([_['paragraph'] for _ in batch[idx]['paragraph_a'][:2]]))
            #     # input()
            # if args.rank == 0 and args.distributed:
            #     torch.distributed.barrier()

            # retriever_loss = retriever_criterion(y_pred=scores, y_true=retriever_labels[:scores.size(0), :scores.size(1)])
            retriever_loss = retriever_criterion(y_pred=scores, y_true=retriever_labels[:scores.size(0), :scores.size(1)])
            # eval_loss += retriever_loss
            eval_loss += distributed_concat(tensor=retriever_loss, num_total_examples=1).item()

            eval_steps += 1
            # word_num = word_mask.sum(-1).unsqueeze(-1)
            # scores = scores / word_num.cuda()
            #

            # pass
            # scores = scores[:, :-padded_p_num]  # remove padded negs
            # all_input_ids, all_input_mask, all_segment_ids, all_segment_lens, all_labels = [_.cuda() for _ in train_dataset.tokenize_for_reader(batch)]
            # reader_loss, *_ = model.reader(input_ids=all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask,
            #                                segment_lens=all_segment_lens, retriever_bias=scores, labels=all_labels)
            # total_loss = retriever_loss
            # reader_loss = torch.tensor(0.0, requires_grad=True).cuda()
            # total_loss = retriever_loss + reader_loss
            # total_loss = retriever_loss

        # retriever_loss_total += retriever_loss.item()
        # reader_loss_total += 0
        # reader_loss_total += reader_loss.item()
        # tr_loss += total_loss.item()

        if args.rank <= 0:
            epoch_iterator.set_postfix(avg_ir='%.4f' % (retriever_loss_total / (step + 1)), avg_reader='%.4f' % (reader_loss_total / (step + 1)),  # elapsed='%.4d' % elapsed,
                                       avg_total='%.4f' % (tr_loss / (step + 1)), )
    # torch.distributed.barrier()
    # logger.info(f"eval loss = {eval_loss / eval_steps}")
    print(f'''data size {len(output_data)}''')
    # eval_metric_for_data_noopt(output_data)

    evaluate_retrieval(output_data)
    # dump_json(output_data, file="data/bm25/result.json")
    # p_num, padded_p_num = t_pnum, tpad_pnum
    # return eval_loss / eval_steps
    model_helper.close()


def evaluate(args, model, mode='dev'):
    results = {}
    exit()
    tokenizer = CostomTokenizer.from_pretrained(pretrain)
    dev_dataset = CBQADataset(f'rougelr-{mode}-0', tokenizer=tokenizer, doc_maxlen=colbert_config['doc_maxlen'],
                              query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'])
    # colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config)

    # model = colbert_qa.load(args.output_dir)

    preds, gts, loss = get_model_predict(args, model=model, dataset=dev_dataset)

    preds_label = np.argmax(preds, axis=1)
    acc = np.mean(preds_label == gts)
    # metrics = np.array(preds)
    results.update({
        "accuracy": acc
    })
    results.update({'loss': loss})

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds.tolist(), gts


def get_model_predict(args, model, dataset):
    eval_sampler = SequentialDistributedSampler(dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(dataset=dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.batch_size, collate_fn=collate_fun())
    # test!
    # logger.info("***** Running prediction *****")
    # logger.info("  Num examples = %d", len(dataset))
    # logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    all_preds = None
    out_labels = None
    model_helper = load_model_helper(args.rank)
    amp = MixedPrecisionManager(args.amp)
    retriever_criterion = listnet_loss
    retriever_labels = torch.zeros((args.batch_size * 4, args.batch_size * p_num * 4), dtype=torch.float, device=DEVICE)
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.rank != 0):
        model.eval()
        with torch.no_grad():
            with amp.context():
                ids, mask, word_mask = dataset.tokenize_for_retriever(batch)
                Q = model.colbert.query(ids, mask)
                retrieval_scores, d_paras = model.retriever_forward(Q, q_word_mask=word_mask, labels=None)
                model_helper.merge_to_reader_input(batch, d_paras)
                D_ids, D_mask, D_word_mask, D_scores = dataset.tokenize_for_train_retriever(batch)
                D = model.colbert.doc(D_ids, D_mask)
                # scores = model.colbert.score(Q, D, q_mask=word_mask, d_mask=D_word_mask)
                D_word_mask = D_word_mask.view(Q.size(0), p_num, D.size(1))
                D = D.view(Q.size(0), p_num, D.size(1), D.size(2))
                scores = model.query_wise_score(Q, D, q_mask=word_mask, d_mask=D_word_mask)
                for i in range(len(batch)):
                    # labels[i, i * pn_num:i * pn_num + pos_num] = score[i]
                    retriever_labels[i, i * p_num * 4:(i + 1) * p_num * 4] = D_scores[i, ...]
                retriever_loss = retriever_criterion(y_pred=scores, y_true=retriever_labels[:scores.size(0), :scores.size(1)])

                all_input_ids, all_input_mask, all_segment_ids, all_segment_lens, all_labels = [_.cuda() for _ in dataset.tokenize_for_reader(batch)]
                reader_loss, preds, labels = model.reader(input_ids=all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask,
                                                          segment_lens=all_segment_lens, retriever_bias=scores, labels=all_labels)
            logits = distributed_concat(tensor=preds, num_total_examples=len(batch))
            labels = distributed_concat(tensor=labels, num_total_examples=len(batch))
            eval_loss += reader_loss

        label_ids = labels
        nb_eval_steps += 1
        if all_preds is None:
            all_preds = logits.detach().cpu().numpy()
            out_labels = label_ids.detach().cpu().numpy()
        else:
            all_preds = np.append(all_preds, logits.detach().cpu().numpy(), axis=0)
            out_labels = np.append(out_labels, label_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    np.set_printoptions(threshold=np.inf)
    # preds = np.argmax(preds, axis=1)
    return all_preds, out_labels, eval_loss.cpu().tolist()


def do_eval(args, mode='dev'):
    args.n_gpu = 1
    colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config)
    colbert_qa.to(DEVICE)
    import os
    basedir = os.path.dirname(args.output_dir)
    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)
    colbert_qa.load(os.path.join(args.output_dir, 'pytorch.bin'))
    model = colbert_qa
    model.eval()
    evaluate(args, model, mode=mode)
