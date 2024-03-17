# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .efficient_sam import build_efficient_sam

import os
def build_efficient_sam_vitt():
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="weights/efficient_sam_vitt.pt",
    ).eval()
import zipfile
# Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
# with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    # zip_ref.extractall("weights")
def build_efficient_sam_vits(weights_dir_path="weights"):
    # /efficient_sam_vits.pt
    checkpoint_path = os.path.join(weights_dir_path, "efficient_sam_vits.pt")
    # check if file exists, if not then extract from zip
    if not os.path.exists(checkpoint_path):
        with zipfile.ZipFile(os.path.join(weights_dir_path, "efficient_sam_vits.pt.zip"), 'r') as zip_ref:
            zip_ref.extractall(weights_dir_path)
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=checkpoint_path,
    ).eval()
