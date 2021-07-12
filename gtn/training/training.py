import copy
import logging
import os
import random
import socket
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def get_logdir():
    host = socket.gethostname()
    current_time = datetime.now().strftime("%b%d_%H-%M-%S_") + str(
        np.random.randint(10, 99)
    )

    location = "logs"

    return os.path.join(location, current_time + "_" + host)


def train(
    model,
    device,
    dataloaders,
    optimizer,
    lr_scheduler,
    metrics_trackers,
    ex=None,
    print_step=None,
    num_epochs=25,
    config_str="",
    save_weights=False,
):

    if ex is not None and ex.current_run:
        writer = SummaryWriter(log_dir=get_logdir(), flush_secs=5)
        writer.add_text("config", config_str)
        stats = ex.current_run.info
    else:
        writer = None
        stats = {}

    phases = ["train", "val"]

    for phase in phases:
        stats[phase] = {}
    stats["best_epoch"] = -1

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):

        epoch_title = f"Epoch {epoch:2d}/{num_epochs - 1:2d}"
        if print_step is not None:
            logging.info("")
            logging.info(epoch_title)
            logging.info("-" * len(epoch_title))

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase not in dataloaders:
                logging.warn(f"Skipping {phase} due to missing dataloader")
                continue
            epoch_start = time.time()

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            if print_step is not None:
                iter_metrics = metrics_trackers["iter"][phase]
            epoch_metrics = metrics_trackers["epoch"][phase]

            if print_step is not None:
                iter_metrics.reset()
            epoch_metrics.reset()

            # Iterate over data.
            if print_step is not None:
                iter_time_start = time.time()
            curr_iter = 0
            next_print = print_step

            for graphs1, graphs2, labels in dataloaders[phase]:

                # send stuff to device
                graphs1 = {k: v.to(device) for k, v in graphs1.items()}
                graphs2 = {k: v.to(device) for k, v in graphs2.items()}
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(graphs1, graphs2)
                    batch_size = outputs.size(0)

                    loss = F.mse_loss(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                # statistics
                with torch.set_grad_enabled(False):
                    if print_step is not None:
                        iter_metrics.update(outputs, labels, loss)
                    epoch_metrics.update(outputs, labels, loss)

                curr_iter += batch_size
                if print_step is not None and curr_iter >= next_print:
                    next_print += print_step
                    iter_str = f"{phase:5} {curr_iter:8d} "
                    iter_str += iter_metrics.get_string()
                    iter_str += f" ({time.time() - iter_time_start:.2f}s)"
                    logging.info(iter_str)
                    iter_metrics.reset()
                    iter_time_start = time.time()

            stat = epoch_metrics.get_log_dict()
            stats[phase][str(epoch)] = stat

            #############
            # Tensorboard
            if writer:
                for name, value in stat.items():
                    writer.add_scalar(
                        str(phase) + "/" + name.replace("@", "_"), value, epoch
                    )

                if phase == "val":
                    writer.add_scalar(
                        "zisc/learning_rate", optimizer.param_groups[0]["lr"], epoch
                    )

                    if epoch % 5 == 0:
                        for name, param in model.named_parameters():
                            writer.add_histogram(name, param, epoch)
            # END Tensorboard
            #################

            epoch_str = "Epoch "
            if print_step is None:
                temp = f"{epoch}/{num_epochs - 1},"
                epoch_str += f"{temp:10}"
            epoch_str += f"{phase:8} {epoch_metrics.get_string()} ({time.time() - epoch_start:.2f}s)"
            logging.info(epoch_str)

            if phase == "train":
                lr_scheduler.step()

            if phase == "val" and epoch_metrics.is_best_val():
                stats["best_epoch"] = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch_metrics.is_patience_over():
            break

    time_elapsed = time.time() - start
    logging.info("")
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logging.info(
        f"Best validation result: {metrics_trackers['epoch']['val'].get_best_values_string()} "
        f"(epoch {stats['best_epoch']})"
    )

    if ex is not None and ex.current_run and save_weights:
        temp_path = f"/tmp/graph_distance_model_weights_{ex.current_run.config['seed']}_{random.randint(0, sys.maxint)}"
        if "val" not in dataloaders:
            best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, temp_path)
        ex.current_run.add_artifact(temp_path, "model_wts")
    model.load_state_dict(best_model_wts)

    result_dict = {
        "time": time_elapsed,
        "best_epoch": stats["best_epoch"],
        "last_epoch": epoch,
    }

    stats_dict = {
        stat: {phase: stats[phase][str(stats["best_epoch"])][stat] for phase in phases}
        for stat in metrics_trackers["epoch"]["train"].metrics_list
    }
    result_dict.update(stats_dict)

    return result_dict
