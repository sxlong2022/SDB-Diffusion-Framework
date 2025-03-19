# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .unet_video import UNetVideoModel
from .ema import ExponentialMovingAverage


def get_model(config):
    kwargs = dict(in_channels=config.num_channels, 
                  model_channels=config.nf, 
                  out_channels=config.num_channels, 
                  num_res_blocks=config.num_res_blocks, 
                  attention_resolutions=config.attn_resolutions, 
                  image_size=config.image_size, 
                  dropout=config.dropout, 
                  channel_mult=config.ch_mult,
                  conv_resample=True,
                  dims=config.dims,
                  num_heads=config.num_heads,
                  use_rpe_net=True)
    return UNetVideoModel(**kwargs)


def get_ema(parameters, decay, use_num_updates=True):
    return ExponentialMovingAverage(parameters, decay, use_num_updates=use_num_updates)
