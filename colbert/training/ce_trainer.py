import copy
import logging
from typing import Union, Any, Optional, Tuple, List, Dict

import os

from torch import nn
from torch.cuda.amp import autocast
from transformers import ProgressCallback, TrainerCallback, TrainingArguments, TrainerState, IntervalStrategy, TrainerControl, DefaultFlowCallback
from awutils.awtrainer import AWTrainer
from awutils.file_utils import dump_json
from awutils.mytrainer_callbacks import MyProgressCallback, MyDefaultFlowCallback
from colbert.modeling.ce_model import CEModel
from colbert.training.colbert_dataset import ColbertDataset, collate_fun
from colbert.training.training_utils import *

# logger = logging.getLogger(__name__)
logger = logging.getLogger("transformers")


class CETrainer(AWTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool = True,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        inputs['mode'] = 'dev'
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        return loss, None, None


class CETrainerTest(AWTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool = True,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        inputs['mode'] = 'test'
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        # input(outputs)
        return loss, outputs['logits'], None


def train(args, trainer_args):
    model = CEModel(args)
    train_dataset = ColbertDataset(task=args.ce_training_args.train_task)
    eval_dataset = ColbertDataset(task=args.ce_training_args.dev_task)
    trainer = CETrainer(model=model, args=trainer_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fun,
                        callbacks=[])
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(DefaultFlowCallback)
    trainer.add_callback(MyProgressCallback)
    trainer.add_callback(MyDefaultFlowCallback)
    trainer.train()


def evaluate(args, trainer_args):
    model = CEModel(args)
    eval_dataset = ColbertDataset(task=args.ce_training_args.dev_task)
    trainer_args.per_device_eval_batch_size = 32
    trainer = CETrainer(model=model, args=trainer_args, eval_dataset=eval_dataset, data_collator=collate_fun,
                        callbacks=[])
    checkpoint_dir = "./temp/checkpoint_ce/"
    checkpoints = os.listdir(checkpoint_dir)
    for checkpoint in checkpoints:
        path = checkpoint_dir + checkpoint
        # input(path)
        # if os.path.isdir(checkpoint):
        args.dense_training_args.checkpoint = path
        if trainer_args.local_rank == 0:
            print(args.dense_training_args.checkpoint)
        model.load(checkpoint=args.dense_training_args.checkpoint + "/pytorch_model.bin")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if trainer_args.local_rank == 0:
            print(metrics)


def test(args, trainer_args):
    model = CEModel(args)
    # eval_dataset = ColbertDataset(task=args.ce_training_args.test_task)
    eval_dataset = ColbertDataset(task=args.ce_training_args.test_task)
    trainer_args.per_device_eval_batch_size = 1
    trainer = CETrainerTest(model=model, args=trainer_args, eval_dataset=eval_dataset, data_collator=collate_fun,
                            callbacks=[])
    model.load(checkpoint=args.ce_test_args.checkpoint + "/pytorch_model.bin")
    output = trainer.predict(test_dataset=eval_dataset)
    from proj_utils.dureader_utils import eval_dureader

    if trainer_args.local_rank <= 0:
        output_res = []
        # output_ori = []
        for t, t_pred in zip(eval_dataset.data, output.predictions):
            t_res = [(None, float(score), para) for para, score in zip(t['retrieval_res'], t_pred)]
            # t['res'] = t_res
            # output_ori.append(copy.deepcopy(t))
            t_res.sort(key=lambda x: x[1], reverse=True)
            t['res'] = t_res
            output_res.append(t)
            # print(len(t['res']))
            # input([_[1] for _ in t['res']])

        eval_dureader(output_res)
        dump_json(output_res, f"data/{args.ce_training_args.test_task}.json")
        # eval_dureader(output_ori)
