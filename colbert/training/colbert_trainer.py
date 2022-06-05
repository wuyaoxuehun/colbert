from typing import Union, Any, Optional, Tuple

from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, ProgressCallback

from awutils.awtrainer import AWTrainer
from awutils.file_utils import dump_json
from colbert.modeling.colbert import *
# from colbert.training.CBQADataset_gen_medqa import CBQADataset, collate_fun
from colbert.training.colbert_dataset import ColbertDataset, collate_fun
from colbert.training.training_utils import *

logger = logging.getLogger(__name__)
from colbert.utils import distributed


# from transformers.utils.logging import set_verbosity_error
# set_verbosity_error()

class MyProgressCallback(ProgressCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    """

    def __init__(self):
        super().__init__()

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step
            self.training_bar.set_postfix({"train_loss": "%.4f" % float(state.train_avg_loss)})


class ColbertTrainer(AWTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool = True,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        inputs['is_evaluating'] = True
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        return loss, None, None


def train(args, trainer_args):
    model = ColbertModel(args.dense_training_args)
    train_dataset = ColbertDataset(args.dense_training_args, task=args.dense_training_args.train_task)
    eval_dataset = ColbertDataset(args.dense_training_args, task=args.dense_training_args.dev_task)
    trainer = ColbertTrainer(model=model, args=trainer_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fun,
                             callbacks=[])
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(MyProgressCallback)
    trainer.train()



def eval_retrieval(args, colbert_qa=None):
    if colbert_qa is None:
        exit()
        colbert_config['init'] = True
        colbert_qa = ColbertModel(load_old=False)
        colbert_qa.load(colbert_config['checkpoint'])
    colbert_qa.to(DEVICE)
    colbert_qa.eval()

    amp = MixedPrecisionManager(args.amp)

    train_dataset = CBQADataset(dev_task, doc_maxlen=colbert_config['doc_maxlen'],
                                query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'], mode='dev')
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
    for step, batch in enumerate(epoch_iterator):
        # Skip past any already trained steps if resuming training
        model.eval()
        with torch.no_grad():
            with amp.context():
                losses = \
                    model(batch, train_dataset, is_evaluating=True, merge=False,
                          doc_enc_training=True, eval_p_num=eval_p_num, pad_p_num=0,
                          args=args)
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
    if args.distributed:
        losses = [distributed_concat(tensor=_.unsqueeze(0), num_total_examples=None).mean() / eval_steps
                  for _ in tr_losses]

    eval_losses = {loss_name: (float(loss)) for loss_name, loss in zip(loss_names, losses)}
    for k, v in eval_losses.items():
        logger.info(f"{k} = %s", v)
    return eval_losses[loss_names[1]]


def eval_retrieval_for_model(args):
    colbert_config['checkpoint'] = args.checkpoint
    colbert_config['init'] = True
    colbert_qa = ColbertModel(load_old=False)
    colbert_qa.load(colbert_config['checkpoint'])
    colbert_qa.to(DEVICE)

    colbert_qa.eval()
    amp = MixedPrecisionManager(args.amp)
    train_dataset = CBQADataset(test_task, doc_maxlen=colbert_config['doc_maxlen'],
                                query_maxlen=colbert_config['query_maxlen'], reader_max_seq_length=reader_config['max_seq_length'])
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler, pin_memory=True, drop_last=False,
                                  batch_size=args.batch_size, collate_fn=collate_fun())
    model = colbert_qa

    index_config['index_path'] = args.index_path
    index_config['faiss_index_path'] = args.index_path + "ivfpq.2000.faiss"
    model_helper = load_model_helper()
    epoch_iterator = tqdm(train_dataloader) if args.rank == 0 else train_dataloader
    output_data = []
    for step, batch in enumerate(epoch_iterator):
        model.eval()
        with amp.context(), torch.no_grad():
            model(batch, train_dataset, is_testing_retrieval=True, eval_p_num=30)
            output_data += batch
    print(f'''data size {len(output_data)}''')
    # eval_res = eval_dataset(output_data)
    print(save_result_file)
    dump_json(output_data, file=save_result_file)
    eval_dureader(output_data)
    # model_helper.close()


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
    topk = 10
    recall_topk = 60
    res = 0
    recall_res = 0
    for t in output_data:
        for i in range(topk):
            if t['res'][i]['paragraph_cut'] in t['positive_ctxs']:
                res += 1 / (i + 1)
                break
        for i in range(recall_topk):
            if t['res'][i]['paragraph_cut'] in t['positive_ctxs']:
                recall_res += 1
                break

    print(f"mrr@10 = {res / len(output_data)}")
    print(f"recall@50 = {recall_res / len(output_data)}")
