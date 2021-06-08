#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg
import os
# import sys
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

def main():
    config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()
    gpus = cfg.MODEL.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus.split('_')[1:])
    print("Using GPUs: {:} for training.\n".format(gpus.split('_')[1:]))
    # exit()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)


if __name__ == "__main__":
    main()
