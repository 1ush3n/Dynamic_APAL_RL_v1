import os
import time
import logging
import pandas as pd
from datetime import datetime

def init_logger(args, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_basename = os.path.splitext(os.path.basename(getattr(args, 'data_path', 'default')))[0]
    exp_dir = os.path.join(getattr(args, 'result_dir', 'results'), f"{experiment_name}_{data_basename}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    log_file = os.path.join(exp_dir, f"{experiment_name}.log")
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"========== {experiment_name} 实验开始 ==========")
    logger.info(f"实验参数: {vars(args) if hasattr(args, '__dict__') else args}")
    logger.info(f"结果将归档至: {exp_dir}")
    
    return logger, exp_dir

def record_experiment_time(logger, start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    logger.info(f"实验结束。总耗时: {hours}小时{minutes}分钟{seconds}秒")
    return elapsed
