import logging
from typing import Union, Any, Optional, Tuple

import os
from torch.cuda.amp import autocast
from transformers import ProgressCallback, DefaultFlowCallback
from awutils.awtrainer import AWTrainer
from awutils.mytrainer_callbacks import MyProgressCallback, MyDefaultFlowCallback
from colbert.modeling.colbert_model import *
from colbert.training.colbert_dataset import ColbertDataset, collate_fun
from colbert.training.training_utils import *

# logger = logging.getLogger(__name__)
logger = logging.getLogger("transformers")


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
    model = ColbertModel(args)
    # model.load("./temp/checkpoint/checkpoint-6360/pytorch_model.bin")
    train_dataset = ColbertDataset(task=args.dense_training_args.train_task)
    eval_dataset = ColbertDataset(task=args.dense_training_args.dev_task)
    trainer = ColbertTrainer(model=model, args=trainer_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fun,
                             callbacks=[])
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(DefaultFlowCallback)
    trainer.add_callback(MyProgressCallback)
    trainer.add_callback(MyDefaultFlowCallback)
    trainer.train()


def evaluate(args, trainer_args):
    model = ColbertModel(args)
    eval_dataset = ColbertDataset(task=args.dense_training_args.dev_task)
    trainer_args.per_device_eval_batch_size = 80
    trainer = ColbertTrainer(model=model, args=trainer_args, eval_dataset=eval_dataset, data_collator=collate_fun,
                             callbacks=[])
    # checkpoints = os.listdir(trainer_args.output_dir)
    checkpoint_dir = "./temp/checkpoint_colbert1/"
    checkpoints = os.listdir(checkpoint_dir)
    for checkpoint in checkpoints:
        path = checkpoint_dir + checkpoint
        # input(path)
        # if os.path.isdir(checkpoint):
        args.dense_training_args.checkpoint = path
        if trainer_args.local_rank == 0:
            print(args.dense_training_args.checkpoint)
        model.load(checkpoint=path + "/pytorch_model.bin")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if trainer_args.local_rank == 0:
            print(metrics)
