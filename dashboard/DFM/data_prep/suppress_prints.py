#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子进程print抑制模块
"""

import os
import builtins

# 保存原始print函数
_original_print = builtins.print

def _conditional_print(*args, **kwargs):
    """条件化的print函数"""
    # 检查是否在多进程环境中且需要静默
    if os.getenv('DFM_SILENT_WARNINGS', 'true').lower() == 'true':
        # 只抑制包含特定关键词的打印
        if args and len(args) > 0:
            first_arg = str(args[0])
            # 抑制config相关的警告
            if '警告: config 模块中没有' in first_arg:
                return
            # 抑制其他重复的警告
            if '[警告抑制]' in first_arg:
                return
    
    # 其他情况正常打印
    _original_print(*args, **kwargs)

# 替换内置的print函数
builtins.print = _conditional_print
