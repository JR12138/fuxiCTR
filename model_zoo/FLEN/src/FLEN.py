# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# =========================================================================

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, InnerProductInteraction, LogisticRegression


class FLEN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FLEN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(FLEN, self).__init__(feature_map, 
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map)
        self.mf_interaction = InnerProductInteraction(num_fields=3, output="elementwise_product")
        self.fm_interaction = InnerProductInteraction(feature_map.num_fields, output="bi_interaction")
        self.dnn = MLP_Block(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=None, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.r_ij = nn.Linear(3, 1, bias=False)
        self.r_mm = nn.Linear(3, 1, bias=False)
        self.w_FwBI = nn.Sequential(nn.Linear(embedding_dim + 1, embedding_dim + 1, bias=False),
                                    nn.ReLU())
        self.w_F = nn.Linear(dnn_hidden_units[-1] + embedding_dim + 1, 1, bias=False)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        emb_user = self.embedding_layer.dict2tensor(feature_emb_dict, feature_source="user")
        emb_item = self.embedding_layer.dict2tensor(feature_emb_dict, feature_source="item")
        emb_context = self.embedding_layer.dict2tensor(feature_emb_dict, feature_source="context")
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        lr_out = self.lr_layer(X)
        field_emb = torch.stack([emb_user.sum(dim=1), emb_item.sum(dim=1), emb_context.sum(dim=1)], dim=1)
        h_MF = self.r_ij(self.mf_interaction(field_emb).transpose(1, 2))
        h_FM = self.r_mm(torch.stack([self.fm_interaction(emb_user), self.fm_interaction(emb_item), 
                                      self.fm_interaction(emb_context)], dim=1).transpose(1, 2))
        h_FwBI = self.w_FwBI(torch.cat([lr_out, (h_MF + h_FM).squeeze(-1)], dim=-1))
        h_L = self.dnn(feature_emb.flatten(start_dim=1))
        y_pred = self.w_F(torch.cat([h_FwBI, h_L], dim=-1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
