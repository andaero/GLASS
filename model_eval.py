import time
import json
import os
import copy
import numpy as np
import glob
import pandas as pd
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from models import Model,geo_CGNN
from data_utils import AtomGraphDatasetPrediction, Atomgraph_collate_prediction
from datetime import datetime
from process_geo_CGNN import create_model
from data_utils import load_graph_data, AtomGraph

def setup():
    """Adds all argument parser info and creates a model"""
    import argparse

    parser = argparse.ArgumentParser(description="Crystal Graph Neural Networks")

    parser.add_argument("--n_hidden_feat", type=int, default=128,
        help='the dimension of node features')
    parser.add_argument("--conv_bias", type=bool, default=False,
        help='use bias item or not in the linear layer')
    parser.add_argument("--n_GCN_feat", type=int, default=128)
    parser.add_argument("--N_block", type=int, default=6)
    parser.add_argument("--N_shbf", type=int, default=6)
    parser.add_argument("--N_srbf", type=int, default=6)
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument("--max_nei", type=int, default=12)
    parser.add_argument("--n_MLP_LR", type=int, default=3)
    parser.add_argument("--n_grid_K", type=int, default=4)
    parser.add_argument("--n_Gaussian", type=int, default=64)
    parser.add_argument("--node_activation", type=str, default="Sigmoid")
    parser.add_argument("--MLP_activation", type=str, default="Elu")
    parser.add_argument("--use_node_batch_norm", type=bool, default=True)
    parser.add_argument("--use_edge_batch_norm", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--milestones", nargs='+', type=int, default=[20])
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--cosine_annealing", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--dataset_path", type=str, default='database')
    parser.add_argument("--datafile_name", type=str, default="my_graph_data_OQMD_8_12_100")
    parser.add_argument("--database", type=str, default="OQMD")
    parser.add_argument("--target_name", type=str, default='formation_energy_per_atom')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--pred", action='store_true')
    parser.add_argument("--pre_trained_model_path", type=str, default='./pre_trained/model_Ef_OQMD.pth')
    options = vars(parser.parse_args())

    # set cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    model_param_names = ['n_hidden_feat','conv_bias','n_GCN_feat','N_block','N_shbf','N_srbf','cutoff','max_nei','n_MLP_LR','node_activation','MLP_activation','use_node_batch_norm','use_edge_batch_norm','n_grid_K','n_Gaussian']
    model_param = {k : options[k] for k in model_param_names if options[k] is not None}
    if model_param["node_activation"].lower() == 'none':
        model_param["node_activation"] = None
    if model_param["MLP_activation"].lower() == 'none':
        model_param["MLP_activation"] = None
    print("Model_param:", model_param)
    print()

    # Optimizer parameters
    optimizer_param_names = ["optim", "lr", "weight_decay", "clip_value"]
    optimizer_param = {k : options[k] for k in optimizer_param_names if options[k] is not None}
    if optimizer_param["clip_value"] == 0.0:
        optimizer_param["clip_value"] = None
    print("Optimizer:", optimizer_param)
    print()

    # Scheduler parameters
    scheduler_param_names = ["milestones", "gamma", "cosine_annealing"]
    #scheduler_param_names = ["milestones", "gamma"]
    scheduler_param = {k : options[k] for k in scheduler_param_names if options[k] is not None}
    print("Scheduler:", scheduler_param)
    print()

    # Dataset parameters
    dataset_param_names = ["dataset_path",'datafile_name','database', "target_name","test_ratio"]
    dataset_param = {k : options[k] for k in dataset_param_names if options[k] is not None}
    print("Dataset:", dataset_param)
    print()

    # Dataloader parameters
    dataloader_param_names = ["num_workers", "batch_size"]
    dataloader_param = {k : options[k] for k in dataloader_param_names if options[k] is not None}
    print("Dataloader:", dataloader_param)
    print()

    # Dataset creation
    current_time = datetime.now().strftime("%m_%d_%H-%M")
    dataset = AtomGraphDatasetPrediction(dataset_param["dataset_path"],dataset_param['datafile_name'],dataset_param["database"], model_param['cutoff'],model_param['N_shbf'],model_param['N_srbf'],model_param['n_grid_K'],model_param['n_Gaussian'])

    dataloader_param["collate_fn"] = Atomgraph_collate_prediction

    model_param['n_node_feat'] = dataset.graph_data[0].nodes.shape[1]

    model = create_model(current_time, device, model_param, optimizer_param, scheduler_param, options["load_model"])

    return model, options["pre_trained_model_path"], dataset, dataloader_param, dataset_param['datafile_name']

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

#MAIN CODE
model, pre_trained_model_path, dataset, dataloader_param, datafile_name = setup()
print(f"Loading weights from {pre_trained_model_path}.pth")
model.load(model_path=pre_trained_model_path)
print("Model loaded at: {}".format(pre_trained_model_path))

pred_dl = DataLoader(dataset, pin_memory=True, **dataloader_param)

outputs,all_graph_vec = model.predict(pred_dl)

names = dataset.graph_names

df_predictions = pd.DataFrame({"name": names, "prediction": outputs})

model_date = splitall(pre_trained_model_path)[1]
print(model_date)
if not os.path.exists(f"predictions/{model_date}"):
    os.makedirs(f"predictions/{model_date}")
df_predictions.to_csv(f"predictions/{model_date}/test_predictions_{datafile_name}.csv", index=False)
print("\nEND")

