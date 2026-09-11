# 转发模块: 使 legacy/ 下的历史脚本仍能导入仓库根目录的 paths
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import *  # noqa
