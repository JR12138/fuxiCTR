<<<<<<< HEAD

import os
#设置当前工作目录为脚本所在的目录   确保对文件的操作使用相对路径时不会出错
# os.path.realpath(__file__) 获取该路径的绝对路径
#  __file__ 是一个特殊的变量，表示当前正在执行的 Python 脚本的路径
#os.chdir() 用于更改当前工作目录
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import sys
import logging
# import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset

import gc
import argparse   #argparse 来解析命令行参数。
from pathlib import Path

from model_zoo.xDeepFM import src
# import src    #import src 由于不在xdeepFM文件下 无法直接导入


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    #通过 argparse 获取命令行参数，包括配置文件目录 (--config)、实验 ID (--expid) 和 GPU 设备 ID (--gpu)。
    parser = argparse.ArgumentParser()

    #注意 config路径！！
    parser.add_argument('--config', type=str, default='./config/xdeepfm_criteo_x4', help='The config directory.')
    #注意 实验id  这个可以根据你不同的实验进行更改
    parser.add_argument('--expid', type=str, default='xDeepFM_criteo_x4:', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    print("Config directory: ", args['config'])
    print("Experiment ID: ", args['expid'])
    print("GPU: ", args['gpu'])

    #load_config 加载指定配置目录和实验 ID 的配置文件。
    # set_logger 根据配置文件设置日志。
    # seed_everything 设置随机种子，确保实验结果的可重复性
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # data_root 和 dataset_id 生成数据目录
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    #如果数据格式是 CSV，使用 FeatureProcessor 处理特征并通过 build_dataset 构建数据集。
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    #加载 feature_map.json 中的特征映射文件，方便模型使用特征信息。
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    

    #通过 getattr(src, params['model']) 动态加载指定模型类。
    #实例化模型并调用 count_parameters() 方法，打印模型的参数数量。
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    #使用 RankDataLoader 为训练和验证数据创建迭代器。
    # 调用模型的 fit 方法进行训练，并传入训练集和验证集。
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    #如果有测试数据，使用同样的方法对模型进行测试评估
    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)
    
    #将实验结果、验证结果和测试结果记录到一个 CSV 文件中。
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))
=======

import os
#设置当前工作目录为脚本所在的目录   确保对文件的操作使用相对路径时不会出错
# os.path.realpath(__file__) 获取该路径的绝对路径
#  __file__ 是一个特殊的变量，表示当前正在执行的 Python 脚本的路径
#os.chdir() 用于更改当前工作目录
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import sys
import logging
# import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset

import gc
import argparse   #argparse 来解析命令行参数。
from pathlib import Path

from model_zoo.xDeepFM import src
# import src    #import src 由于不在xdeepFM文件下 无法直接导入


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    #通过 argparse 获取命令行参数，包括配置文件目录 (--config)、实验 ID (--expid) 和 GPU 设备 ID (--gpu)。
    parser = argparse.ArgumentParser()

    #注意 config路径！！
    parser.add_argument('--config', type=str, default='./config/xdeepfm_criteo_x4', help='The config directory.')
    #注意 实验id  这个可以根据你不同的实验进行更改
    parser.add_argument('--expid', type=str, default='xDeepFM_criteo_x4:', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    print("Config directory: ", args['config'])
    print("Experiment ID: ", args['expid'])
    print("GPU: ", args['gpu'])

    #load_config 加载指定配置目录和实验 ID 的配置文件。
    # set_logger 根据配置文件设置日志。
    # seed_everything 设置随机种子，确保实验结果的可重复性
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # data_root 和 dataset_id 生成数据目录
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    #如果数据格式是 CSV，使用 FeatureProcessor 处理特征并通过 build_dataset 构建数据集。
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    #加载 feature_map.json 中的特征映射文件，方便模型使用特征信息。
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    

    #通过 getattr(src, params['model']) 动态加载指定模型类。
    #实例化模型并调用 count_parameters() 方法，打印模型的参数数量。
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    #使用 RankDataLoader 为训练和验证数据创建迭代器。
    # 调用模型的 fit 方法进行训练，并传入训练集和验证集。
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    #如果有测试数据，使用同样的方法对模型进行测试评估
    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)
    
    #将实验结果、验证结果和测试结果记录到一个 CSV 文件中。
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))
>>>>>>> 6be6bae (first commit)
