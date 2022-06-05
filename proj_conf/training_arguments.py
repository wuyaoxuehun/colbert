from dataclasses import field, dataclass
from typing import Optional, List

from transformers import TrainingArguments, IntervalStrategy


@dataclass
class MyTraniningArgs(TrainingArguments):
    xpu_backend: Optional[str] = field(
        default=None,
        metadata={"help": "The backend to be used for distributed training on Intel XPU.", "choices": ["mpi", "ccl"]},
    )
    eval_steps: Optional[int] = field(default=None, metadata={"help": "Run an evaluation every X steps."})

    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    push_to_hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to which push the `Trainer`."}
    )
    push_to_hub_organization: Optional[str] = field(
        default=None, metadata={"help": "The name of the organization in with to which push the `Trainer`."}
    )
    push_to_hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    disable_tqdm: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    evaluation_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use."},
    )
    save_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    logging_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The logging strategy to use."},
    )
    fp16_backend: str = field(
        default="amp",
        metadata={"help": "The backend to be used for mixed precision.", "choices": ["auto", "amp", "apex"]},
    )
