from dataclasses import field, dataclass
from typing import Optional

from transformers import TrainingArguments

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
