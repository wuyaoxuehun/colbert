import logging
import math
import random

import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from colbert.modeling.tokenization.utils import CostomTokenizer
from colbert.modeling.tokenization.utils import CostomTokenizer
from colbert.training.CBQADataset import CBQADataset, collate_fun
from colbert.modeling.colbert_list_qa import ColBERT_List_qa
from colbert.modeling.colbert_list_qa import load_model_helper, load_model
from colbert.parameters import DEVICE
from colbert.utils.amp import MixedPrecisionManager
from conf import reader_config, colbert_config, model_config, p_num, Temperature, padded_p_num
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from colbert import base_config
from colbert.training.losses import listnet_loss
from colbert.utils import distributed
from torch.distributed.optim import ZeroRedundancyOptimizer
from colbert.training.training_utils import SequentialDistributedSampler

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def setseed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup():
    torch.distributed.destroy_process_group()


def train(args):
    setseed(12345 + args.rank)

    colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config)
    colbert_qa.to(DEVICE)
    if args.distributed:
        colbert_qa = DDP(colbert_qa, device_ids=[args.rank]).module

    colbert_qa.train()
    # for param in colbert_qa.colbert.bert.parameters():
    # for param in colbert_qa.colbert.parameters():
    # param.requires_grad = False

    if args.distributed and False:
        retriever_optimizer = ZeroRedundancyOptimizer(filter(lambda p: p.requires_grad, colbert_qa.colbert.parameters()),
                                                      optimizer_class=torch.optim.Adam,
                                                      lr=args.retriever_lr)
        reader_optimizer = ZeroRedundancyOptimizer(filter(lambda p: p.requires_grad, colbert_qa.colbert.parameters()),
                                                   optimizer_class=torch.optim.Adam,
                                                   lr=args.lr)
    else:
        retriever_optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.colbert.parameters()), lr=args.retriever_lr, eps=1e-8)

        # optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.reader.parameters()), lr=args.lr, eps=1e-8)
        reader_optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.reader.parameters()), lr=args.lr, eps=1e-8)

    retriever_optimizer.zero_grad()
    reader_optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)

    args.max_grad_norm = 1.0

    tokenizer = CostomTokenizer.from_pretrained(base_config.pretrain)
    train_dataset = CBQADataset('rougelr-train-0', tokenizer=tokenizer, doc_maxlen=colbert_config['doc_maxlen'],
                                query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'])
    # t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epoch
    # train_sampler = RandomSampler(train_dataset)
    train_sampler = DistributedSampler(train_dataset) if args.distributed else RandomSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler, pin_memory=True, drop_last=False,
                                  batch_size=args.batch_size, collate_fn=collate_fun())
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epoch
    warm_up = 0
    retriever_scheduler = get_linear_schedule_with_warmup(retriever_optimizer, num_warmup_steps=int(warm_up * t_total), num_training_steps=t_total)

    reader_scheduler = get_linear_schedule_with_warmup(reader_optimizer, num_warmup_steps=int(warm_up * t_total), num_training_steps=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    model = colbert_qa

    # args.n_gpu = 1
    # if args.n_gpu > 1:
    #     print("data parrallel")
    #     model = torch.nn.DataParallel(model)

    global_step = 0
    global_log_step = math.floor(len(train_dataloader) / args.gradient_accumulation_steps) // args.logging_steps
    logger.info(f"global log step = {global_log_step * args.gradient_accumulation_steps}")

    model_helper = load_model_helper(args.rank)
    # load_model_helper()
    retriever_criterion = listnet_loss

    # loss_fun = AutomaticWeightedLoss(num=2)
    best_metrics = 0
    retriever_labels = torch.zeros((args.batch_size * 4, args.batch_size * p_num * 4 + padded_p_num), dtype=torch.float, device=DEVICE)
    for epoch in range(args.epoch):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        if args.distributed:
            train_sampler.set_epoch(epoch)
        epoch_iterator = tqdm(train_dataloader) if args.rank == 0 else train_dataloader
        tr_loss = 0
        retriever_loss_total = 0
        reader_loss_total = 0

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            model.train()
            with amp.context():
                # batch, labels = batch
                # q_ids, q_mask, q_word_mask = model_helper.query_tokenize(batch)
                # scores, ir_loss, d_paras = model.retriever_forward((q_ids, q_mask), q_word_mask, labels=labels)
                # model_helper.merge_to_reader_input(batch, d_paras)
                # graphs, labels = batch_transform_bert_fever(batch, reader_config['max_seq_length'], reader_config['p_num'], device=DEVICE)
                # reader_loss, preds = model.reader_forward(graphs, labels)
                # ir_loss, reader_loss, ir_preds, reader_preds = model(batch, labels)
                ids, mask, word_mask = train_dataset.tokenize_for_retriever(batch)
                # Q = model.colbert.query(ids, mask)
                with torch.no_grad():
                    Q = model.old_colbert.query(ids, mask)
                # Q = model.colbert.query(ids, mask)
                # retrieval_scores, d_paras = model.retriever_forward(Q, q_word_mask=word_mask, labels=None)
                # print(batch[0]['question'], '\n', batch[0]['A'], '\n', d_paras[0][:2])
                # input()
                Q = model.colbert.query(ids, mask)
                pass
                # model_helper.merge_to_reader_input(batch, d_paras)

                padded_negs = [model_helper.all_paras[_] for _ in np.random.randint(1, len(model_helper.all_paras), padded_p_num)]

                D_ids, D_mask, D_word_mask, D_scores = train_dataset.tokenize_for_train_retriever(batch, padded_negs)

                # D = model.colbert.doc(D_ids, D_mask)
                # with torch.no_grad():
                #     D = model.old_colbert.doc(D_ids, D_mask)
                D = model.colbert.doc(D_ids, D_mask)
                # if model.old_colbert.training:
                # D = D.requires_grad_(requires_grad=True)

                scores = model.colbert.score(Q, D, q_mask=word_mask, d_mask=D_word_mask)
                # D_word_mask = D_word_mask.view(Q.size(0), p_num, D.size(1))
                # D = D.view(Q.size(0), p_num, D.size(1), D.size(2))
                # scores = model.query_wise_score(Q, D, q_mask=word_mask, d_mask=D_word_mask)

                for i in range(len(batch) * 4):
                    # labels[i, i * pn_num:i * pn_num + pos_num] = score[i]
                    retriever_labels[i, (i // 4) * p_num * 4:(i // 4 + 1) * p_num * 4 + padded_p_num] = D_scores[i, ...] / Temperature
                if args.rank > 0:
                    torch.distributed.barrier()
                if (step) % 25 == 0:
                    idx = 0
                    print(scores[idx])
                    print(retriever_labels[idx] * Temperature)
                    idx = idx // 4
                    print(batch[idx]['background'], '\n', batch[idx]['question'], '\n', batch[idx]['A'], '\n', '\n'.join([_['paragraph'] for _ in batch[idx]['paragraph_a'][:2]]))
                    # input()
                if args.rank == 0 and args.distributed:
                    torch.distributed.barrier()

                retriever_loss = retriever_criterion(y_pred=scores, y_true=retriever_labels[:scores.size(0), :scores.size(1)])
                # retriever_loss = retriever_criterion(y_pred=scores, y_true=retriever_labels[:scores.size(0), :scores.size(1)])
                word_num = word_mask.sum(-1).unsqueeze(-1)
                scores = scores / word_num.cuda()
                #

                pass
                # scores = scores[:, :-padded_p_num]  # remove padded negs
                # all_input_ids, all_input_mask, all_segment_ids, all_segment_lens, all_labels = [_.cuda() for _ in train_dataset.tokenize_for_reader(batch)]
                # reader_loss, *_ = model.reader(input_ids=all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask,
                #                                segment_lens=all_segment_lens, retriever_bias=scores, labels=all_labels)
                # total_loss = retriever_loss
                reader_loss = torch.tensor(0.0, requires_grad=True).cuda()
                total_loss = retriever_loss + reader_loss
                if args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / args.gradient_accumulation_steps

            retriever_loss_total += retriever_loss.item()
            # reader_loss_total += 0
            reader_loss_total += reader_loss.item()
            tr_loss += total_loss.item()
            amp.backward(total_loss)
            # print(total_loss.item(), ir_loss.item(), reader_loss.item())
            # tr_loss += total_loss.item() * args.gradient_accumulation_steps
            # ir_loss_total += ir_loss.item()
            # reader_loss_total += reader_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # amp.step([model.colbert, model.reader], [retriever_optimizer, reader_optimizer])
                amp.step([colbert_qa.colbert], [retriever_optimizer])
                reader_scheduler.step()
                retriever_scheduler.step()
                global_step += 1

            # continue
            if (step + 1) % args.gradient_accumulation_steps == 0 and global_step % global_log_step == 0:
                # distributed.barrier(args.rank)
                continue
                # results, _, _ = evaluate(args, model)

                if args.rank == 0:
                    # model.save(save_dir=args.output_dir)

                    if results['accuracy'] > best_metrics:
                        best_metrics = results['accuracy']
                        model.save(save_dir=args.output_dir)
                        # save_model(args, model, tokenizer)
                        # best_results = {'epoch': epoch, 'global_step': global_step}
                        # best_results.update(results)
                    # save_model(args, model, tokenizer, step=global_step)

                    logger.info("Current best Accuracy = %s", best_metrics)
                    logger.info("global_log_step = %s", global_log_step)
                    logger.info("tr_loss = %s", tr_loss / (step + 1))
                # distributed.barrier(args.rank)

            if args.rank <= 0:
                epoch_iterator.set_postfix(avg_ir='%.4f' % (retriever_loss_total / (step + 1)), avg_reader='%.4f' % (reader_loss_total / (step + 1)),  # elapsed='%.4d' % elapsed,
                                           avg_total='%.4f' % (tr_loss / (step + 1)), )  # reader_lr='%.4e' % (scheduler.get_last_lr()[0]))
            # ir_lr='%.4f' % (retriever_scheduler.get_last_lr()[0]))
        # model.old_colbert = model.colbert
        # model.colbert = load_model(colbert_config).cuda()
        # model.colbert.load_state_dict(model.colbert.state_dict())
        # logger.info("refreshed colbert")


def evaluate(args, model, mode='dev'):
    results = {}
    tokenizer = CostomTokenizer.from_pretrained(base_config.pretrain)
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


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


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
                    retriever_labels[i, i * p_num * 4:(i + 1) * p_num * 4] = D_scores[i, ...] / Temperature
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
