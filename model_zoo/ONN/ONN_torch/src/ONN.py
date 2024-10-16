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


from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class ONN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="ONN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=2, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU",
                 embedding_dropout=0,
                 net_dropout = 0,
                 batch_norm = False,
                 **kwargs):
        super(ONN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.num_fields = feature_map.num_fields
        input_dim = embedding_dim * self.num_fields + int(self.num_fields * (self.num_fields - 1) / 2)
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.embedding_layers = nn.ModuleList([FeatureEmbedding(feature_map, embedding_dim) 
                                               for _ in range(self.num_fields)]) # f x f x bs x dim
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        field_aware_emb_list = [each_layer(X) for each_layer in self.embedding_layers] # list of emb tensors
        diag_embedding = field_aware_emb_list[0].flatten(start_dim=1)
        ffm_out = self.field_aware_interaction(field_aware_emb_list[1:])
        dnn_input = torch.cat([diag_embedding, ffm_out], dim=1)
        y_pred = self.dnn(dnn_input)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def field_aware_interaction(self, field_aware_emb_list):
        interaction = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                v_ij = field_aware_emb_list[j - 1][:, i, :]
                v_ji = field_aware_emb_list[i][:, j, :]
                dot = torch.sum(v_ij * v_ji, dim=1, keepdim=True)
                interaction.append(dot)
        return torch.cat(interaction, dim=1)



