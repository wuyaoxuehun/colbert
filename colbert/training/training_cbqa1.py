import logging
import math
import random

import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# from colbert.modeling.colbert import ColBERT
from CBQADataset import CBQADataset, batcher_cbqa
from colbert.modeling.colbert_list_qa import ColBERT_List_qa
from colbert.modeling.colbert_list_qa import load_model_helper
from colbert.modeling.utils_xh import batch_transform_bert_fever
from colbert.parameters import DEVICE
from colbert.utils.amp import MixedPrecisionManager
from conf import reader_config, colbert_config, model_config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    # if args.distributed:
    #     torch.cuda.manual_seed_all(12345)

    colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config)
    colbert_qa.to(DEVICE)

    colbert_qa.train()

    for param in colbert_qa.colbert.bert.parameters():
    # for param in colbert_qa.colbert.parameters():
        param.requires_grad = False

    retriever_optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.colbert.parameters()), lr=args.retriever_lr, eps=1e-8)

    # optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.reader.parameters()), lr=args.lr, eps=1e-8)
    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert_qa.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)

    args.max_grad_norm = 1.0

    args.n_gpu = torch.cuda.device_count()

    train_dataset = CBQADataset(files=args.train_files)
    t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epoch
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size * args.n_gpu, collate_fn=batcher_cbqa())
    retriever_scheduler = get_linear_schedule_with_warmup(retriever_optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    model = colbert_qa

    # args.n_gpu = 1
    if args.n_gpu > 1:
        print("data parrallel")
        model = torch.nn.DataParallel(model)

    global_step = 0
    global_log_step = math.floor(len(train_dataloader) / args.gradient_accumulation_steps) // args.logging_steps
    logger.info(f"global log step = {global_log_step * args.gradient_accumulation_steps}")

    model_helper = load_model_helper()
    # load_model_helper()

    # loss_fun = AutomaticWeightedLoss(num=2)
    best_metrics = 0

    for epoch in range(args.epoch):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_iterator = tqdm(train_dataloader)
        tr_loss = 0
        ir_loss_total = 0
        reader_loss_total = 0
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            model.train()

            with amp.context():
                batch, labels = batch
                # q_ids, q_mask, q_word_mask = model_helper.query_tokenize(batch)
                # scores, ir_loss, d_paras = model.retriever_forward((q_ids, q_mask), q_word_mask, labels=labels)
                # model_helper.merge_to_reader_input(batch, d_paras)
                # graphs, labels = batch_transform_bert_fever(batch, reader_config['max_seq_length'], reader_config['p_num'], device=DEVICE)
                # reader_loss, preds = model.reader_forward(graphs, labels)
                # ir_loss, reader_loss, ir_preds, reader_preds = model(batch, labels)
                ir_loss, ir_preds = model(batch, labels)

                # total_loss = loss_fun(ir_loss, reader_loss)
                # total_loss = ir_loss + reader_loss
                total_loss = ir_loss
                if args.n_gpu > 1:
                    total_loss = total_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / args.gradient_accumulation_steps

            amp.backward(total_loss)
            # print(total_loss.item(), ir_loss.item(), reader_loss.item())
            tr_loss += total_loss.item() * args.gradient_accumulation_steps
            ir_loss_total += ir_loss.item()
            # reader_loss_total += reader_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # amp.step([model.colbert, model.reader], [retriever_optimizer, optimizer])
                # retriever_scheduler.step()
                amp.step([colbert_qa.colbert], [retriever_optimizer])
                # scheduler.step()
                retriever_scheduler.step()
                global_step += 1

            # continue
            if (step + 1) % args.gradient_accumulation_steps == 0 and global_step % global_log_step == 0:
                results, _, _ = evaluate(args, model)
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

            epoch_iterator.set_postfix(avg_ir='%.4f' % (ir_loss_total / (step + 1)), avg_reader='%.4f' % (reader_loss_total / (step + 1)),  # elapsed='%.4d' % elapsed,
                                       avg_total='%.4f' % (tr_loss / (step + 1)), )#reader_lr='%.4e' % (scheduler.get_last_lr()[0]))
                                       # ir_lr='%.4f' % (retriever_scheduler.get_last_lr()[0]))


def evaluate(args, model, mode='dev'):
    results = {}
    dev_dataset = CBQADataset(files=args.dev_files if mode == 'dev' else args.test_files)
    preds, gts, loss = get_model_predict(args, model=model,
                                         dataset=dev_dataset)

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
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset=dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.batch_size * args.n_gpu, collate_fn=batcher_cbqa())
    # test!
    # logger.info("***** Running prediction *****")
    # logger.info("  Num examples = %d", len(dataset))
    # logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    all_preds = None
    out_labels = None
    model_helper = load_model_helper()
    amp = MixedPrecisionManager(args.amp)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch, labels = batch

        with torch.no_grad():
            with amp.context():
                q_ids, q_mask, q_word_mask = model_helper.query_tokenize(batch)
                scores, ir_loss, d_paras = model.retriever_forward((q_ids, q_mask), q_word_mask, labels=labels)
                model_helper.merge_to_reader_input(batch, d_paras)
                graphs, labels = batch_transform_bert_fever(batch, reader_config['max_seq_length'], reader_config['p_num'], device=DEVICE)
                reader_loss, preds = model.reader_forward(graphs, labels)
            logits = preds
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
