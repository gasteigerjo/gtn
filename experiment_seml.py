import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import seml
import torch
from sacred import Experiment
from torch.utils.data import DataLoader

from gtn import GTN
from gtn.dataloader.graphbatch_collator import GraphBatchCollator
from gtn.dataloader.graphdist_dataset import GraphDistDataset
from gtn.dataloader.io import load_from_npz
from gtn.dataloader.pyg_ged import get_pyg_ged_gcolls
from gtn.model import aggregation, geometric_gnn
from gtn.training.metrics import Metrics
from gtn.training.optimizer import add_weight_decay
from gtn.training.training import train
from gtn.training.validation import evaluate

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(
    dataname: str = "pref_att",
    graph_distance: str = "ged",
    pyg_data_path: Optional[str] = None,
    extensive: bool = True,
    similarity: bool = False,
    emb_size: int = 32,
    nlayers: int = 3,
    act_fn: str = "leaky_relu",
    weight_decay: float = 0.0,
    deg_norm_hidden: bool = False,
    sinkhorn_reg: float = 0.1,
    sinkhorn_niter: int = 50,
    unbalanced_mode: dict = {"name": "balanced"},
    num_heads: int = 1,
    multihead_scale_basis: float = 1.0,
    nystrom: Optional[dict] = None,
    sparse: Optional[dict] = None,
    num_epochs: int = 20,
    batch_size: int = 1000,
    learning_rate: float = 0.01,
    lr_stepsize: int = 100,
    lr_gamma: float = 0.1,
    print_step: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
    save_weights: bool = False,
    test: bool = False,
):
    """
    Main function for training GTN.

    Arguments
    ---------
    dataname:               Dataset name
    graph_distance:         Which distance to fit (GED or PM)
    pyg_data_path:          Path to PyTorch Geometric data directory (only for Linux dataset)
    extensive:              The target label scales with the number of nodes
    similarity:             The target label is a similarity (predicted via exp(-distance))
    emb_size:               GNN embedding size
    nlayers:                Number of GNN layers
    act_fn:                 Activation function. Options: linear, relu, sigmoid, leaky_relu
    weight_decay:           Weight decay for weight regularization
    deg_norm_hidden:        Use symmetric degree normalization in all GNN layers except the first
    sinkhorn_reg:           Entropy regularization used for the Sinkhorn distance
    sinkhorn_niter:         Number of Sinkhorn iterations
    unbalanced_mode:        Mode for handling asymmetric numbers of nodes. Options: bp_matrix, balanced, entropy_reg
    num_heads:              Number of OT heads
    multihead_scale_basis:  Basis for varying the Sinkhorn regularization across heads
    nystrom:                Whether to use Nyström approximation, and its settings
    sparse:                 Whether to use a sparse approximation, and its settings
    num_epochs:             Number of epochs
    batch_size:             Batch size
    learning_rate:          Learning rate
    lr_stepsize:            Number of steps after which to decrease the learning rate
    lr_gamma:               Factor with which to reduce the learning rate every lr_stepsize steps
    print_step:             Print loss and metrics every X steps. Default: Once per epoch.
    seed:                   Random seed for NumPy, PyTorch, and Python
    device:                 Which device to use, e.g. cpu, cuda
    save_weights:           Save the best model weights after training
    test:                   Also evaluate the test set. Only set this once everything is done!
    """

    my_dict = locals().copy()
    run_config = json.dumps(my_dict, indent=4, sort_keys=True)

    logging.info("Run config:" + run_config)

    logging.info("Seed: " + str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device)
    graph_distance = graph_distance.lower()
    dataname = dataname.lower()

    # Load data
    data_path = Path.cwd() / "data"
    if dataname in ["linux"]:
        dataname = dataname.upper()
        gcolls, pair_idxs = get_pyg_ged_gcolls(
            pyg_data_path, dataname, use_norm_ged=True, similarity=similarity
        )
    else:
        gcolls = {
            dataset: load_from_npz(
                data_path / f"{dataname}_{graph_distance}_{dataset}.npz"
            )
            for dataset in ["train", "val", "test"]
        }
        pair_idxs = {dataset: None for dataset in ["train", "val", "test"]}
    node_onehot = True
    if node_onehot:
        node_feat_size = int(
            max(
                (
                    max((np.max(graph.attr_matrix) for graph in gcoll))
                    for gcoll in gcolls.values()
                )
            )
            + 1
        )
    else:
        node_feat_size = gcolls["train"][0].attr_matrix.shape[1]
    if gcolls["train"][0].edge_attr_matrix is None:
        edge_feat_size = 0
    else:
        edge_feat_size = int(
            max(
                (
                    max((np.max(graph.edge_attr_matrix) for graph in gcoll))
                    for gcoll in gcolls.values()
                )
            )
            + 1
        )

    # Get datasets
    datasets = {}
    for key, gcoll in gcolls.items():
        datasets[key] = GraphDistDataset(
            gcoll,
            node_feat_size,
            edge_feat_size,
            node_onehot=node_onehot,
            edge_onehot=True,
            pair_idx=pair_idxs[key],
        )

    # Get dataloader
    collator = GraphBatchCollator()
    dataloaders = {}
    for key, dataset in datasets.items():
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=1,
        )

    # Get Metrics
    metrics_list = ["rmse", "cvrmse", "label_std"]
    metrics_trackers = {"iter": {}, "epoch": {}}
    metric_to_stop_on = "rmse"
    minimize_stop_on = True
    patience = np.inf
    if print_step is not None:
        metrics_trackers["iter"]["train"] = Metrics(metrics_list)
    metrics_trackers["epoch"]["train"] = Metrics(metrics_list)
    if print_step is not None:
        metrics_trackers["iter"]["val"] = Metrics(metrics_list)
    metrics_trackers["epoch"]["val"] = Metrics(
        metrics_list, metric_to_stop_on, minimize_stop_on, patience
    )
    if test:
        metrics_trackers["epoch"]["test"] = Metrics(metrics_list)

    # Select activation function
    if act_fn == "linear":
        act_fn = lambda x: x
    elif act_fn == "relu":
        act_fn = torch.nn.functional.relu
    elif act_fn == "sigmoid":
        act_fn = torch.nn.functional.sigmoid
    elif act_fn == "leaky_relu":
        act_fn = torch.nn.functional.leaky_relu
    else:
        raise ValueError(f"Invalid act_fn '{act_fn}'.")

    # Select layer aggregation function
    assert num_heads >= 1
    if num_heads == 1:
        layer_aggregation = aggregation.MLP(
            emb_size=emb_size, nlayers=nlayers, output_size=emb_size
        )
    else:
        layer_aggregation = aggregation.All()

    # Average degree used to prevent embedding magnitude changes in non-normalized aggregation
    avg_degree = np.mean([graph.adj_matrix.sum(1).mean() for graph in gcolls["train"]])

    # Get GNN
    gnn = geometric_gnn.Net(
        node_feat_size=node_feat_size,
        edge_feat_size=edge_feat_size,
        emb_size=emb_size,
        nlayers=nlayers,
        layer_aggregation=layer_aggregation,
        device=device,
        act_fn=act_fn,
        avg_degree=avg_degree,
        deg_norm_hidden=deg_norm_hidden,
    )

    # Statistics for normalizing embeddings used for Sinkhorn
    emb_dist_scale = np.mean(gcolls["train"].dists.A[datasets["train"].pair_idx])
    if extensive:
        emb_dist_scale /= np.mean(
            [
                gcolls["train"][idx].num_nodes()
                for idx in datasets["train"].pair_idx.flatten()
            ]
        )

    # Overall GTN model
    model = GTN(
        gnn=gnn,
        emb_dist_scale=emb_dist_scale,
        device=device,
        sinkhorn_reg=sinkhorn_reg,
        sinkhorn_niter=sinkhorn_niter,
        unbalanced_mode=unbalanced_mode,
        nystrom=nystrom,
        sparse=sparse,
        extensive=extensive,
        num_heads=num_heads,
        multihead_scale_basis=multihead_scale_basis,
        similarity=similarity,
    )

    # Training
    parameters = add_weight_decay(model, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_stepsize, gamma=lr_gamma
    )

    logging.info("Start training")
    result = train(
        model,
        device,
        dataloaders,
        optimizer,
        lr_scheduler,
        metrics_trackers,
        ex=ex,
        num_epochs=num_epochs,
        print_step=print_step,
        config_str=run_config,
        save_weights=save_weights,
    )

    if test:
        logging.info("Evaluating on test")
        result_test = evaluate(
            model,
            device,
            dataloaders["test"],
            metrics_trackers["epoch"]["test"],
            disable_tqdm=True,
        )
        for key in metrics_list:
            result[key]["test"] = result_test[key]

    return result
