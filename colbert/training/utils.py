import os

from colbert.parameters import SAVED_CHECKPOINTS
from colbert.utils.runs import Run
from colbert.utils.utils import save_checkpoint


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, colbert, optimizer, epoch):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    # if batch_idx % 2000 == 0:
    #     name = os.path.join(path, "colbert.dnn")
    #     save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

    # if batch_idx in SAVED_CHECKPOINTS:
    name = os.path.join(path, "colbert-{}.dnn".format(epoch))
    save_checkpoint(name, epoch, 0, colbert, optimizer, arguments)
