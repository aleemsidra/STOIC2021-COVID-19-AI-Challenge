# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from torch.nn import BCEWithLogitsLoss
# from fastai.data.core import DataLoader, DataLoaders
# from fastai.vision.learner import Learner
# from fastai.callback.all import *
# from fastai.metrics import AccumMetric

# from algorithm.i3d.i3dpt import I3D
# from ctdataset import CTDataset
# from config import get_config
# from metrics import acc_probseverecovid
# from metrics import acc_probcovid
# from metrics import roc_probseverecovid
# from metrics import roc_probcovid

# import torch
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'

# CONFIGFILE = "/opt/train/config/baseline.json"

# def get_datasets(config, data_dir):
#     image_dir = os.path.join(data_dir, "data/mha/")
#     reference_path = os.path.join(data_dir, "metadata/reference.csv")
#     df = pd.read_csv(reference_path)

#     df["x"] = df.apply(lambda row: os.path.join(image_dir, str(row["PatientID"]) + ".mha"), axis=1)
#     df["y"] = df.apply(lambda row: [row["probCOVID"], row["probSevere"]], axis=1)
#     df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)

#     data_train = df_train[["x", "y"]].to_dict("records")
#     data_valid = df_valid[["x", "y"]].to_dict("records")

#     train_ds = CTDataset(data_train, config["preprocess_dir"])
#     valid_ds = CTDataset(data_valid, config["preprocess_dir"])
#     return train_ds, valid_ds


# def get_learn(config, data_dir, artifact_dir):
#     train_ds, valid_ds = get_datasets(config, data_dir)
#     train_dl = DataLoader(train_ds, bs=config["batch_size"], num_workers=config["num_workers"],
#                           shuffle=True, drop_last=True)
#     valid_dl = DataLoader(valid_ds, bs=config["batch_size"], num_workers=config["num_workers"])

#     model = I3D(nr_outputs=2).to(device)

#     metrics = [
#         AccumMetric(acc_probcovid, flatten=False),
#         AccumMetric(roc_probcovid, flatten=False),
#         AccumMetric(acc_probseverecovid, flatten=False),
#         AccumMetric(roc_probseverecovid, flatten=False)
#     ]

#     return Learner(DataLoaders(train_dl, valid_dl),
#                    model=model,
#                    metrics=metrics,
#                    loss_func=BCEWithLogitsLoss(),
#                    model_dir=artifact_dir
#                    ).to_fp16()


# def train(learn, config):
#     cbs = [SaveModelCallback(monitor="roc_probseverecovid",
#                              fname=config["experiment_name"]
#                              )
#            ]
#     learn.fit(config["epochs"],
#               lr=config["lr"],
#               cbs=cbs)


# def do_learning(data_dir, artifact_dir):
#     """
#     You can implement your own solution to the STOIC2021 challenge by editing this function.
#     :param data_dir: Input directory that the training Docker container has read access to. This directory has the same
#         structure as the stoic2021-training S3 bucket (see https://registry.opendata.aws/stoic2021-training/)
#     :param artifact_dir: Output directory that, after training has completed, should contain all artifacts (e.g. model
#         weights) that the inference Docker container needs. It is recommended to continuously update the contents of
#         this directory during training.
#     :returns: A list of filenames that are needed for the inference Docker container. These are copied into artifact_dir
#         in main.py. If your model already produces all necessary artifacts into artifact_dir, an empty list can be
#         returned. Note: To limit the size of your inference Docker container, please make sure to only place files that 
#         are necessary for inference into artifact_dir.
#     """
#     config = get_config(CONFIGFILE)
#     learn = get_learn(config, data_dir, artifact_dir)
#     train(learn, config)

#     artifacts = [] # empty list because train() already writes all artifacts to artifact_dir

#     # If your code does not produce all necessary artifacts for the inference Docker container into artifact_dir, return 
#     # their filenames:
#     # artifacts = ["/tmp/model_checkpoint.pth", "/tmp/some_other_artifact.json"]
    
#     return artifacts


import sys, os

print('path in train ', os.path.dirname(__file__))


from image_classification_pytorch.train import main
import time


def do_learning(config, suffix):
    """
    You can implement your own solution to the STOIC2021 challenge by editing this function.
    :param data_dir: Input directory that the training Docker container has read access to. This directory has the same
        structure as the stoic2021-training S3 bucket (see https://registry.opendata.aws/stoic2021-training/)
    :param artifact_dir: Output directory that, after training has completed, should contain all artifacts (e.g. model
        weights) that the inference Docker container needs. It is recommended to continuously update the contents of
        this directory during training.
    :returns: A list of filenames that are needed for the inference Docker container. These are copied into artifact_dir
        in main.py. If your model already produces all necessary artifacts into artifact_dir, an empty list can be
        returned. Note: To limit the size of your inference Docker container, please make sure to only place files that 
        are necessary for inference into artifact_dir.
    """

    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    d = {'config':config, 'suffix':suffix, 'wandb_mode':'disabled', 'test':None}

    args = Bunch(d)


    now = time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time()))

    # print('before main ', args.config)
    # asd

    artifact1 = main(args, now)

    artifacts = [] # empty list because train() already writes all artifacts to artifact_dir

    artifacts.append(os.path.join(artifact1, 'auc_model.pth'))
    # If your code does not produce all necessary artifacts for the inference Docker container into artifact_dir, return 
    # their filenames:
    # artifacts = ["/tmp/model_checkpoint.pth", "/tmp/some_other_artifact.json"]

    return artifacts