from collections import defaultdict
from typing import Sequence, Tuple

import numpy as np
import torch
import torchmetrics as tm
import seaborn as sns
import matplotlib.pyplot as plt

from . import utils

FONT_SCALE = 1.9


@torch.no_grad()
def get_n_logvars(model):
    tensors = [param.data.cpu() for param in model.filter["n_logvars"]]
    logvars = torch.stack(tensors, dim=0)
    return logvars.numpy()


def distance_matrix(params, num_tasks=None, dist="l1"):
    if num_tasks is None:
        num_tasks = params.shape[0]
    mat = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(num_tasks):
            if dist == "l1":
                mat[i, j] = np.sum(np.abs(params[i, :] - params[j, :]))
            elif dist == "cos":
                mat[i, j] = 1 - np.abs(
                    params[i, :].dot(params[j, :]) /
                    (np.linalg.norm(params[i, :]) * torch.norm(params[j, :]))
                )
    return mat


def dataloader2numpy(dataloader):
    batches = {key: list() for key in ["x", "y", "s"]}
    for batch in dataloader:
        batches["x"].append(batch[0])
        batches["y"].append(batch[1])
        batches["s"].append(batch[2])

    x = torch.cat(batches["x"]).numpy()
    y = torch.cat(batches["y"]).numpy()
    s = torch.cat(batches["s"]).numpy()
    return x, y, s


def distances_heatmap(distances, labels):
    """ Correlation Matrix Heatmap """
    sns.set(font_scale=FONT_SCALE)

    fig, ax = plt.subplots(figsize=(8, 8))

    # labels for x-axis
    axis_labels = [repr(lbl) for lbl in labels]
    sns.heatmap(distances,
                xticklabels=axis_labels,
                yticklabels=axis_labels,
                annot=True,
                ax=ax,
                cmap="coolwarm",
                fmt='.1f',
                linewidths=.05,
                # annot_kws={"fontsize": 16},
                # cbar_kws={"labelsize": 16}
                )
    fig.subplots_adjust(top=0.93)
    ax.set_xticklabels(labels=axis_labels, rotation=-45, rotation_mode="anchor", ha="left")
    ax.set_yticklabels(labels=axis_labels, rotation=0, rotation_mode="anchor", ha="right")
    # ax.tick_params(labelsize=16)
    plt.tight_layout()

    # fig.suptitle("Distance Heatmap", fontsize=16)
    return fig


def vectors_heatmap(vectors, labels):
    """ Correlation Matrix Heatmap """
    sns.set(font_scale=FONT_SCALE)

    fig, ax = plt.subplots(figsize=(8, 8))

    # labels for x-axis
    y_axis_labels = [repr(lbl) for lbl in labels]

    sns.heatmap(vectors,
                annot=True,
                cmap="coolwarm",
                fmt='.1f',
                linewidths=.05,
                # annot_kws={"fontsize": 16},
                # cbar_kws={"labelsize": 16}
                ax=ax
                )
    fig.subplots_adjust(top=0.93)
    ax.set_yticklabels(labels=y_axis_labels, rotation=0, rotation_mode="anchor", ha="right")

    # ax.tick_params(labelsize=16)
    plt.tight_layout()

    # fig.suptitle("log-variance Heatmap", fontsize=14)
    return fig


def eval_distance_matrix(distances, idxs: Sequence[Tuple[int, int]]):
    """
    4 subgroups:
        idxs: [(0,1),(2,3)]
    6 subgroups:
        idxs: [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]
    """

    # logvars = n_logvars(model)
    # distances = distance_matrix(logvars)
    num_tasks = distances.shape[0]
    in_distances, out_distances = list(), list()
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if (i, j) in idxs:
                in_distances.append(distances[i, j])
            else:
                out_distances.append(distances[i, j])

    results = dict(
        max_in=max(in_distances),
        min_out=min(out_distances),
        avg_in=np.mean(in_distances),
        avg_out=np.mean(out_distances),

    )
    results["avg_diff"] = results["avg_out"] - results["avg_in"]
    results["minmax_diff"] = results["min_out"] - results["max_in"]
    results["success"] = results["max_in"] < results["min_out"]
    return results


def reshape_results(results, metadata=None):
    if metadata is None:
        metadata = dict()

    output = list()
    for idx, entry in enumerate(results):  # Iterate tasks
        for k, v in entry.items():  # Iterate evaluators
            new_entry = dict()
            new_entry["metric"] = k
            new_entry["value"] = v
            new_entry["task"] = idx
            new_entry.update(metadata)
            output.append(new_entry)
    return output
