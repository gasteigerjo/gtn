import logging
import time

import torch.nn.functional as F
from tqdm.autonotebook import tqdm


def evaluate(
    model, device, dataloader, metrics_tracker, disable_tqdm=False
):

    model.eval()  # Set model to evaluate mode
    metrics_tracker.reset()

    eval_start = time.time()
    for graphs1, graphs2, labels in tqdm(dataloader, disable=disable_tqdm):

        # send stuff to device
        graphs1 = {k: v.to(device) for k, v in graphs1.items()}
        graphs2 = {k: v.to(device) for k, v in graphs2.items()}
        labels = labels.to(device)

        # forward
        outputs = model(graphs1, graphs2)

        loss = F.mse_loss(outputs, labels)

        # statistics
        metrics_tracker.update(outputs, labels, loss)

    eval_str = f"{metrics_tracker.get_string()} ({time.time() - eval_start:.2f}s)"
    logging.info(eval_str)

    stat = metrics_tracker.get_log_dict()

    return stat
