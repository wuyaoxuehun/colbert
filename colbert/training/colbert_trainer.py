import logging
from typing import Union, Any, Optional, Tuple
from torch.cuda.amp import autocast
from transformers import ProgressCallback
from awutils.awtrainer import AWTrainer
from colbert.modeling.colbert_model import *
from colbert.training.colbert_dataset import ColbertDataset, collate_fun
from colbert.training.training_utils import *

# logger = logging.getLogger(__name__)
logger = logging.getLogger("transformers")

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
    model = ColbertModel(args)
    train_dataset = ColbertDataset(args.dense_training_args, task=args.dense_training_args.train_task)
    eval_dataset = ColbertDataset(args.dense_training_args, task=args.dense_training_args.dev_task)
    trainer = ColbertTrainer(model=model, args=trainer_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fun,
                             callbacks=[])
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(MyProgressCallback)
    trainer.train()
