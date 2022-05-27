from collections import deque

import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from colbert.modeling.colbert_list_qa_gen_medqa import *
from colbert.training.CBQADataset_gen_medqa import CBQADataset, collate_fun
# from colbert import base_config
# from colbert.parameters import DEVICE
from colbert.utils.amp import MixedPrecisionManager
from awutils.file_utils import dump_json, load_json
from colbert.training.training_utils import *
from colbert.modeling.tokenization import CostomTokenizer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO if int(os.environ.get('LOCAL_RANK', 0)) in [-1, 0] else logging.WARN)
logger = logging.getLogger(__name__)
from colbert.utils import distributed
# import bitsandbytes as bnb
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()


def cleanup():
    torch.distributed.destroy_process_group()


def train(args):
    if hasattr(args, "scoretemperature"):
        logger.info(f"temperature set to {args.scoretemperature}")
    colbert_config['init'] = True
    colbert_qa = ColBERT_List_qa(load_old=False)
    if load_trained:
        colbert_qa.load(load_trained)

    colbert_qa.to(DEVICE)
    if args.distributed:
        colbert_qa = DDP(colbert_qa, device_ids=[args.rank], find_unused_parameters=True)

    colbert_qa.train()

    tokenizer = CostomTokenizer.from_pretrained(pretrain)
    train_dataset = CBQADataset(train_task, tokenizer=tokenizer, doc_maxlen=colbert_config['doc_maxlen'],
                                query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'], mode='train')
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
    # lrs = [lr, lr * 2, lr * 2, lr * 2]

    # optimized_modules = [colbert_qa.module.colbert, colbert_qa.module.decoder]
    # optimized_modules = [colbert_qa.module.colbert]
    t = colbert_qa
    if args.distributed:
        t = colbert_qa.module
        # optimized_modules = [[colbert_qa.module.model, colbert_qa.module.linear]]
    # optimized_modules = [[t.model, t.linear, t.q_word_weight_linear, t.d_word_weight_linear]]
    # optimized_modules = [[t.model, t.linear, t.q_word_weight_linear, t.esim_linear_42, t.esim_linear_80]]
    # optimized_modules = [[t.model, t.linear, t.q_word_weight_linear, t.comatch]]
    optimized_modules = [[t.model]]
    lrs = [[lr]]
    logger.info("learning rate is " + str(lrs))
    params = []
    weight_decay = 0.0
    for modules, cur_lrs in zip(optimized_modules, lrs):
        assert len(modules) == len(cur_lrs), (len(modules), len(cur_lrs))
        for module, cur_lr in zip(modules, cur_lrs):
            pd, pnd = split_parameters(module)
            params.append({
                'params': pd, 'weight_decay': weight_decay, 'lr': cur_lr
            })
            params.append({
                'params': pnd, 'weight_decay': 0.0, 'lr': cur_lr
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

    warm_up = 0.0
    retriever_scheduler = get_linear_schedule_with_warmup(retriever_optimizer, num_warmup_steps=int(warm_up * t_total), num_training_steps=t_total)
    # retriever_scheduler = get_constant_schedule_with_warmup(retriever_optimizer, num_warmup_steps=int(warm_up * t_total))
    # reader_scheduler = get_linear_schedule_with_warmup(reader_optimizer, num_warmup_steps=int(warm_up * t_total), num_training_steps=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    model = colbert_qa

    global_step = 0
    # global_log_step = math.floor(len(train_dataloader) / args.gradient_accumulation_steps) // args.logging_steps
    global_log_step = math.floor(len(train_dataloader) / args.logging_steps)
    # global_log_step = (len(train_dataloader) // args.gradient_accumulation_steps) // args.logging_steps
    logger.info(f"global log step = {global_log_step}")
    # print(args.batch_size, args.gradient_accumulation_steps, args.logging_steps, len(train_dataloader))
    # input()
    model_helper = load_model_helper(args.rank)
    # load_model_helper()
    # retriever_criterion = listnet_loss

    # loss_fun = AutomaticWeightedLoss(num=2)
    best_metrics = float('inf')
    # retriever_labels = torch.zeros((args.batch_size * opt_num, args.batch_size * p_num * opt_num + padded_p_num), dtype=torch.float, device=DEVICE)

    # neg_weight_mask = torch.zeros((args.batch_size * opt_num, args.batch_size * p_num * opt_num + padded_p_num), dtype=torch.float, device=DEVICE)
    qd_queue = deque(maxlen=1)
    batch_queue = deque(maxlen=2)
    config = args
    self = args
    # loss_names = ["tr_loss", "re_loss"]
    loss_names = ["tr_loss", ] + [f"loss_{i}" for i in range(3)]
    tr_losses = [MAverage() for _ in range(len(loss_names))]

    for epoch in tqdm(range(config.epoch)):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        if epoch > stop_epoch: break
        if self.distributed:
            train_sampler.set_epoch(epoch)
        args.cur_epoch = epoch
        neg_weight_mask = None
        if schedule:
            neg_weight_mask, sample_T = get_epoch_scheduler(epoch, args, eval_p_num=p_num)
            train_dataset.sample_T = sample_T
            # neg_weight_mask = None

        epoch_iterator = tqdm(train_dataloader) if self.rank == 0 else train_dataloader
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # cur_retriever_loss, cur_q_answer_retriever_loss, cur_q_answer_kl_loss, cur_ar_loss, cur_teacher_aug_retriever_loss \
            #     = [torch.tensor(0.0).cuda() for _ in range(5)]
            with amp.context():
                # batch = [_.cuda() for _ in batch]
                losses = model(batch, train_dataset, neg_weight_mask=neg_weight_mask, args=args)
                # if (step) % 16 == 111:
                #     idx = 0
                #     # print(scores[idx])
                #     print(retriever_labels[idx])
                #     idx = idx // 4
                #     print(batch[idx]['background'], '\n', batch[idx]['question'], '\n', batch[idx]['A'], '\n', '\n'.join([_['paragraph'] for _ in batch[idx]['paragraph_a'][:2]]))
                pass
                # total_loss = losses

                if config.gradient_accumulation_steps > 1:
                    losses = [loss / config.gradient_accumulation_steps for loss in losses]
                total_loss = sum(losses)

            amp.backward(total_loss)

            losses = [total_loss] + list(losses)
            if self.distributed:
                losses = [distributed_concat(tensor=_.unsqueeze(0), num_total_examples=None).mean()
                          for _ in losses]

            for avg_loss, loss in zip(tr_losses, losses):
                avg_loss.add(loss.item())
            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                # print(torch.allclose(model.module.old_colbert.linear.weight, model.module.colbert.linear.weight))
                # amp.step([model.colbert, model.reader], [retriever_optimizer, reader_optimizer])
                optimizers = [retriever_optimizer]
                # amp.step([[model.module.colbert]], optimizers)
                amp.step(optimized_modules, optimizers)

                for optimizer in optimizers:
                    optimizer.zero_grad()

                retriever_scheduler.step()
                global_step += 1

            # continue
            if ((step + 1) % config.gradient_accumulation_steps == 0) and ((step + 1) % global_log_step == 0):
                eval_loss = eval_retrieval(args, model)
                # eval_loss = 0
                if self.rank == 0:
                    # model.save(save_dir=args.output_dir)
                    if save_every_eval or eval_loss < best_metrics:
                        # if eval_loss < best_metrics:
                        # best_metrics = results['accuracy']
                        best_metrics = min(eval_loss, best_metrics)
                        # save_dir = config.output_dir + f"_{epoch}"
                        save_dir = config.output_dir
                        if (epoch + 1) % 5 == 0 or True:
                            if self.distributed:
                                model.module.save(save_dir)
                            else:
                                model.save(save_dir)

                    for loss_name, avg_loss in zip(loss_names, tr_losses):
                        logger.info(f"{loss_name} = %s", avg_loss.get_average())
                    # logger.info("ans_sim_loss = %s", avg_ans_sim_loss)
                    logger.info("eval_loss = %s", eval_loss)
                if self.distributed:
                    distributed.barrier(self.rank)

            if self.rank <= 0:
                # epoch_iterator.set_postfix(avg_ir='%.4f' % re_loss.get_average(), avg_reader='%.4f' % (reader_loss_total / (step + 1)),  # elapsed='%.4d' % elapsed,
                #                            avg_total='%.4f' % tr_loss.get_average(), lr='%.1Ef' % retriever_scheduler.get_last_lr()[0])  # reader_lr='%.4e' % (scheduler.get_last_lr()[0]))
                postfix = {loss_name: '%.4f' % loss.get_average() for loss_name, loss in zip(loss_names, tr_losses)}
                postfix['lr'] = '%.1Ef' % retriever_scheduler.get_last_lr()[0]
                epoch_iterator.set_postfix(postfix)
            # ir_lr='%.4f' % (retriever_scheduler.get_last_lr()[0]))
        # model.module.old_colbert = load_model(colbert_config).cuda()
        # model.module.old_colbert.load_state_dict(model.module.colbert.state_dict())
        # print(torch.allclose(model.module.old_colbert.linear.weight, model.module.colbert.linear.weight))

        # logger.info("refreshed colbert")
        # if args.rank == 0:
        #     eval_retrieval(args, model.module)
        # torch.distributed.barrier()
        # input()
    # colbert_qa.module.save(os.path.join(args.output_dir, "rouge12_model"))


def eval_retrieval(args, colbert_qa=None):
    # setseed(12345 + args.rank)
    if colbert_qa is None:
        # colbert_config['checkpoint'] = f"output/webq/{save_model_name}/pytorch.bin"
        exit()
        colbert_config['init'] = True
        colbert_qa = ColBERT_List_qa(load_old=False)
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
    train_dataset = CBQADataset(dev_task, tokenizer=tokenizer, doc_maxlen=colbert_config['doc_maxlen'],
                                query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'], mode='dev')
    # t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epoch
    # train_sampler = SequentialSampler(train_dataset)
    train_sampler = DistributedSampler(train_dataset) if args.distributed else SequentialSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler, pin_memory=True, drop_last=False,
                                  # batch_size=args.batch_size, collate_fn=collate_fun())
                                  batch_size=int(args.batch_size * 1), collate_fn=collate_fun())
    model = colbert_qa
    epoch_iterator = tqdm(train_dataloader) if args.rank == 0 else train_dataloader
    eval_steps = 0
    logger.info("doing evaluating")
    loss_names = ["eval_tr_loss", ] + [f"eval_loss_{i}" for i in range(3)]
    tr_losses = [torch.tensor(0.0).cuda() for _ in range(len(loss_names))]
    neg_weight_mask = None
    # if schedule:
    neg_weight_mask, _ = get_epoch_scheduler(args.epoch, args, eval_p_num=eval_p_num, is_evaluating=True)
    for step, batch in enumerate(epoch_iterator):
        # Skip past any already trained steps if resuming training
        model.eval()
        with torch.no_grad():
            with amp.context():
                losses = \
                    model(batch, train_dataset, is_evaluating=True, merge=False,
                          doc_enc_training=True, eval_p_num=eval_p_num, pad_p_num=0,
                          neg_weight_mask=neg_weight_mask, args=args)
                total_loss = sum(losses)
                losses = [total_loss] + list(losses)
                if (step) % 25 == 111:
                    idx = 0
                    # print(scores)
                    # print(retriever_labels)
                    idx = idx // 4
                    print(batch[idx]['background'], '\n', batch[idx]['question'], '\n', batch[idx]['A'], '\n', '\n'.join([_['paragraph'] for _ in batch[idx]['paragraph_a'][:2]]))
            for i in range(min(len(loss_names), len(losses))):
                tr_losses[i] += losses[i]

            eval_steps += 1

        if args.rank <= 0:
            postfix = {loss_name: '%.4f' % (float(loss) / eval_steps) for loss_name, loss in zip(loss_names, tr_losses)}
            epoch_iterator.set_postfix(postfix)
    # torch.distributed.barrier()
    if args.distributed:
        losses = [distributed_concat(tensor=_.unsqueeze(0), num_total_examples=None).mean() / eval_steps
                  for _ in tr_losses]

    eval_losses = {loss_name: (float(loss)) for loss_name, loss in zip(loss_names, losses)}
    for k, v in eval_losses.items():
        logger.info(f"{k} = %s", v)

    # eval_metric_for_data(output_data)
    # p_num, padded_p_num = t_pnum, tpad_pnum
    return eval_losses[loss_names[1]]


def eval_retrieval_for_model(args):
    # setseed(12345 + args.rank)
    colbert_config['checkpoint'] = args.checkpoint
    colbert_config['init'] = True
    colbert_qa = ColBERT_List_qa(load_old=False)
    colbert_qa.load(colbert_config['checkpoint'])
    # print(colbert_qa.span_len_embedding)
    colbert_qa.to(DEVICE)
    # if args.distributed:
    #     colbert_qa = DDP(colbert_qa, device_ids=[args.rank]).module

    colbert_qa.eval()
    # for param in colbert_qa.colbert.bert.parameters():
    # for param in colbert_qa.colbert.parameters():
    # param.requires_grad = False

    amp = MixedPrecisionManager(args.amp)

    tokenizer = CostomTokenizer.from_pretrained(pretrain)
    train_dataset = CBQADataset(test_task, tokenizer=tokenizer, doc_maxlen=colbert_config['doc_maxlen'],
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
    # retriever_labels = torch.zeros((args.batch_size * 4, args.batch_size * eval_p_num * 4), dtype=torch.float, device=DEVICE)
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
            model(batch, train_dataset, is_testing_retrieval=True, eval_p_num=30)
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
    # eval_res = eval_metric_for_data(output_data)
    # eval_res = eval_dataset(output_data)

    # evaluate_retrieval(output_data)
    print(save_result_file)
    dump_json(output_data, file=save_result_file)
    eval_dureader(output_data)
    # eval_res["model"] = args.checkpoint
    # with open("/home2/awu/testcb/res/result.txt", 'a') as f:
    #     f.write(json.dumps(eval_res, indent=2))
    # p_num, padded_p_num = t_pnum, tpad_pnum
    # return eval_loss / eval_steps
    model_helper.close()


def eval_dataset(data):
    topk = [10, 20, 30, 100, 200, 500]
    accuracy = {k: [] for k in topk}
    max_k = max(topk)

    # for qid in tqdm(list(retrieval.keys())):
    #     answers = retrieval[qid]['answers']
    #     contexts = retrieval[qid]['contexts']
    for t in tqdm(data):
        answer = t['answers'][0]
        contexts = t['res']
        has_ans_idx = max_k  # first index in contexts that has answers
        for idx, ctx in enumerate(contexts):
            # print(ctx)
            # input()
            if idx >= max_k:
                break
            # text = ctx['text'].split('\n')[1]  # [0] is title, [1] is text
            # text = ctx['text']
            if answer in ctx['paragraph']:
                has_ans_idx = idx
                break

        for k in topk:
            accuracy[k].append(0 if has_ans_idx >= k else 1)
            t['hit@' + str(k)] = accuracy[k][-1]

    for k in topk:
        print(f'Top{k}\taccuracy: {np.mean(accuracy[k])}')
    return accuracy


def eval_dureader(output_data):
    # dureader_corpus_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
    # passage_id_map = load_json(dureader_corpus_dir + "passage2id.map.json")
    topk = 10
    recall_topk = 60
    res = 0
    recall_res = 0
    for t in output_data:
        # input(len(t['res']))
        # print('*' * 100)
        # print(t['question'])
        # print(t['res'][0]['paragraph_cut'])
        # for i in t['positive_ctxs']:
        #     print(i)
        # input(t['res'][0]['paragraph_cut'] in t['positive_ctxs'])
        # continue

        for i in range(topk):
            if t['res'][i]['paragraph_cut'] in t['positive_ctxs']:
                res += 1 / (i + 1)
                # print(t['res'][i]['paragraph_cut'])
                # print(t['positive_ctxs'])
                # input()
                break
        for i in range(recall_topk):
            if t['res'][i]['paragraph_cut'] in t['positive_ctxs']:
                recall_res += 1
                break

    print(f"mrr@10 = {res / len(output_data)}")
    print(f"recall@50 = {recall_res / len(output_data)}")


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
