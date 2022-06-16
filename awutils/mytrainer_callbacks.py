from transformers import TrainingArguments, TrainerState, TrainerControl, IntervalStrategy, ProgressCallback, TrainerCallback


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


class MyDefaultFlowCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles the default flow of the training loop for logs, evaluation
    and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True

        # Evaluate
        if (args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step % args.eval_steps == 0) or \
                (args.evaluation_strategy == IntervalStrategy.EPOCH and state.global_step % int(state.steps_in_epoch / 2) == 0):
            control.should_log = True
            control.should_evaluate = True
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        # if args.logging_strategy == IntervalStrategy.EPOCH:
        #     control.should_log = True
        #
        # # Evaluate
        # if args.evaluation_strategy == IntervalStrategy.EPOCH:
        #     control.should_evaluate = True
        #
        # # Save
        # if args.save_strategy == IntervalStrategy.EPOCH:
        #     control.should_save = True

        control.should_log = False
        control.should_evaluate = False
        control.should_save = False

        return control