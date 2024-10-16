from datetime import datetime
import gc
import argparse
import fuxictr_version
from fuxictr import autotuner 

if __name__ == '__main__':
    #创建一个参数解析器。
    parser = argparse.ArgumentParser()
    #--config: 用于指定配置文件路径。默认值
    parser.add_argument('--config', type=str, default='../config/tuner_config.yaml', 
                        help='The config file for para tuning.')
    #--tag: 可选参数，用于指定实验 ID 标记
    parser.add_argument('--tag', type=str, default=None, help='Use the tag to determine which expid to run (e.g. 001 for the first expid).')
    parser.add_argument('--gpu', nargs='+', default=[-1], help='The list of gpu indexes, -1 for cpu.')

    #parser.parse_args()：解析命令行输入的参数。
    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    expid_tag = args['tag']

    # generate parameter space combinations autotuner.
    # enumerate_params(config_path): 根据配置文件生成参数空间组合，返回一个配置目录。
    #在参数空间内执行网格搜索，利用指定的 GPU 列表和实验 ID 进行实验。
    config_dir = autotuner.enumerate_params(args['config'])
    autotuner.grid_search(config_dir, gpu_list, expid_tag)

