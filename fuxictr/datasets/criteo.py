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


import numpy as np

#引入 fuxictr.preprocess.FeatureProcessor，作为 CustomizedFeatureProcessor 的基类。
from fuxictr.preprocess import FeatureProcessor

#引入 polars 库，用于处理列式数据（类似于 pandas），pl 是 polars 的别名。
import polars as pl

class CustomizedFeatureProcessor(FeatureProcessor):
    #定义了 convert_to_bucket 方法，用于将列中的值分桶（bucketization）。col_name 是列的名称，表示要对哪一列进行分桶处理。
    def convert_to_bucket(self, col_name):

        #在方法内部定义了一个局部函数 _convert_to_bucket，用于将单个值进行分桶转换。局部函数是为了在 polars 的 apply 方法中调用。
        def _convert_to_bucket(value):

            #是否大于2。如果是，则对自然对数的值进行平方运算 向下取整 转换为整数
            if value > 2:
                value = int(np.floor(np.log(value) ** 2))
            else:
                value = int(value)
            return value
        #pl.col(col_name)：获取 polars 中指定列 col_name 的数据。
        # .apply(_convert_to_bucket)：对该列中的每个值应用 _convert_to_bucket 函数，进行分桶处理。
        # .cast(pl.Int32)：将处理后的结果强制转换为 Int32 类型，确保数据类型一致性。
        return pl.col(col_name).apply(_convert_to_bucket).cast(pl.Int32)
