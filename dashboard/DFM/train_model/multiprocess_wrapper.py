#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程包装器 - 解决DFM训练中的重复打印问题
保持并行处理性能的同时消除子进程的重复输出
"""

import os
import sys
import io
import contextlib
from functools import wraps

def setup_subprocess_silence():
    """设置子进程静默模式"""
    # 设置环境变量标识这是子进程
    os.environ['DFM_SUBPROCESS_MODE'] = 'true'
    os.environ['DFM_SILENT_WARNINGS'] = 'true'
    
    # 重定向子进程的stdout和stderr到null
    if os.getenv('DFM_SUBPROCESS_MODE', 'false').lower() == 'true':
        # 创建null设备
        devnull = open(os.devnull, 'w')
        
        # 保存原始的stdout和stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # 重定向到null设备
        sys.stdout = devnull
        sys.stderr = devnull
        
        return original_stdout, original_stderr, devnull
    
    return None, None, None

def restore_subprocess_output(original_stdout, original_stderr, devnull):
    """恢复子进程的输出"""
    if original_stdout and original_stderr and devnull:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        devnull.close()

def silent_subprocess_wrapper(func):
    """装饰器：让函数在子进程中静默运行"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 检查是否在子进程中
        is_subprocess = os.getenv('DFM_SUBPROCESS_MODE', 'false').lower() == 'true'
        
        if is_subprocess:
            # 在子进程中，使用静默模式
            original_stdout, original_stderr, devnull = setup_subprocess_silence()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                restore_subprocess_output(original_stdout, original_stderr, devnull)
        else:
            # 在主进程中，正常运行
            return func(*args, **kwargs)
    
    return wrapper

@contextlib.contextmanager
def silent_subprocess_context():
    """上下文管理器：在子进程中静默执行代码块"""
    is_subprocess = os.getenv('DFM_SUBPROCESS_MODE', 'false').lower() == 'true'
    
    if is_subprocess:
        original_stdout, original_stderr, devnull = setup_subprocess_silence()
        try:
            yield
        finally:
            restore_subprocess_output(original_stdout, original_stderr, devnull)
    else:
        yield

# 全局初始化函数，必须在模块级别定义以便序列化
def _init_worker():
    """子进程初始化函数"""
    import os
    import builtins

    # 设置环境变量标识这是子进程
    os.environ['DFM_SUBPROCESS_MODE'] = 'true'
    os.environ['DFM_SILENT_WARNINGS'] = 'true'

    # 重写print函数为静默版本
    original_print = builtins.print

    def silent_print(*args, **kwargs):
        """静默的print函数，只在非静默模式下打印"""
        if os.getenv('DFM_SILENT_WARNINGS', 'true').lower() != 'true':
            original_print(*args, **kwargs)

    builtins.print = silent_print

def create_silent_executor(max_workers=None):
    """创建一个静默的ProcessPoolExecutor"""
    import concurrent.futures
    import multiprocessing

    # 设置多进程启动方式
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误

    # 返回配置好的executor
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker
    )

def patch_print_for_subprocess():
    """为子进程打补丁，重写print函数"""
    import builtins
    
    original_print = builtins.print
    
    def silent_print(*args, **kwargs):
        """静默的print函数"""
        if os.getenv('DFM_SUBPROCESS_MODE', 'false').lower() != 'true':
            original_print(*args, **kwargs)
    
    # 只在子进程中替换print函数
    if os.getenv('DFM_SUBPROCESS_MODE', 'false').lower() == 'true':
        builtins.print = silent_print

# 在模块导入时自动设置
if __name__ != "__main__":
    # 如果这是作为模块被导入，检查是否需要设置静默模式
    if os.getenv('DFM_SUBPROCESS_MODE', 'false').lower() == 'true':
        patch_print_for_subprocess()

def test_multiprocess_silence():
    """测试多进程静默功能"""
    import concurrent.futures
    import time
    
    def test_worker(worker_id):
        """测试工作函数"""
        print(f"Worker {worker_id}: 这条消息应该被抑制")
        print(f"Worker {worker_id}: 警告: config 模块中没有 EXCEL_DATA_FILE 属性")
        time.sleep(1)
        return f"Worker {worker_id} completed"
    
    print("=== 测试多进程静默功能 ===")
    print("主进程: 这条消息应该显示")
    
    # 使用静默的executor
    with create_silent_executor(max_workers=2) as executor:
        futures = [executor.submit(test_worker, i) for i in range(4)]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"主进程收到结果: {result}")
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    test_multiprocess_silence()
