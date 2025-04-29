'''
Author       : JiaYu.Wu
PersonalEmail: 602023230055@smail.nju.edu.cn
OfficeEmail  : jiayu.wu@magicdepth.com
Company      : Magic Depth
Date         : 2025-04-29 18:00:17
LastEditTime : 2025-04-29 18:03:53
LastEditors  : JiaYu.Wu
Description  : #*  *#
FilePath     : /SoundSourceLocalization/logger.py
'''
import logging
import time
import datetime
logger = logging.getLogger()  # 不加名称设置root logger
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
cnt_time = datetime.datetime.now()
fh = logging.FileHandler(f'./{cnt_time}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
