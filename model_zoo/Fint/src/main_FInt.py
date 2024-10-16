
import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CompressedInteractionNet, LogisticRegression

class FINT(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="FINT",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FINT, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        # 常规操作，先构建embedding层，然后由输入fields和embedding维度确定输入的维度
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim

        # 构建FI层和FNN层
        self.field_inter_layers = FInet(feature_map.num_fields, num_layers)
        self.fnn = MLP_Layer(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             use_bias=True)

        # 常规操作，最后的全连接层、输出的激活函数、优化器、损失、学习率、初始化参数、移动模型到device上
        self.fc = nn.Linear(dnn_hidden_units[-1], 1)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        # 常规操作，分离x和y并构建特征embedding
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        FInet_out = self.field_inter_layers(feature_emb)
        FNN_out = self.fnn(torch.flatten(FInet_out, start_dim=1, end_dim=2))
        # 常规操作，对隐藏层输出进行线性使出，并使用激活函数，最后返回数据的标签和预测值的字典
        y_pred = self.fc(FNN_out)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class FIInteractionLayer(nn.Module):
    def __init__(self, field_num):
        super(FIInteractionLayer, self).__init__()
        # 残差连接的参数层
        self.res_weight = nn.Parameter(torch.Tensor(field_num, 1))
        self.interaction_weight = nn.Parameter(torch.Tensor(field_num, field_num))
        # 注意在用tensor设定参数层时要使用初始化
        nn.init.xavier_normal_(self.res_weight)
        nn.init.xavier_normal_(self.interaction_weight)

    def forward(self, V_0, V_i):
        # 这里遵循文章公式，先吧V0和权重矩阵相乘，然后再和VI哈达玛积，然后加上有权重的VI (这里的vi为 embed_dim * field_num)
        interaction_out = V_i * torch.matmul(self.interaction_weight, V_0) + self.res_weight * V_i
        return interaction_out


class FInet(nn.Module):
    def __init__(self, field_num, num_layers):
        super(FInet, self).__init__()
        self.num_layers = num_layers
        # 交叉网络的结果
        self.FI_net = nn.ModuleList(FIInteractionLayer(field_num) for _ in range(num_layers))

    def forward(self, V_0):
        V_i = V_0
        for i in range(self.num_layers):
            V_i = self.FI_net[i](V_0, V_i)
        return V_i