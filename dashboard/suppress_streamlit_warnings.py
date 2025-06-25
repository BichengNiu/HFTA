#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抑制Streamlit重复警告消息的脚本
用于减少控制台输出中的噪音
"""

import warnings
import logging
import sys
import os

# 全局标志，确保只打印一次消息
_warnings_suppressed = False

def suppress_streamlit_warnings():
    """抑制Streamlit常见的重复警告"""
    
    global _warnings_suppressed
    
    # 如果已经执行过，直接返回，不再打印消息
    if _warnings_suppressed:
        return
    
    # 🔥 重定向stderr以完全阻止警告输出
    import sys
    from io import StringIO
    
    class WarningFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
            self.buffer = StringIO()
            
        def write(self, text):
            # 过滤掉所有ScriptRunContext相关的警告
            if ("ScriptRunContext" not in text and 
                "Session state does not function" not in text and
                "to view a Streamlit app" not in text and
                "No runtime found" not in text):
                self.original_stderr.write(text)
                
        def flush(self):
            self.original_stderr.flush()
    
    # 重定向stderr到我们的过滤器
    sys.stderr = WarningFilter(sys.stderr)
    
    # 🔥 强力抑制所有ScriptRunContext相关警告
    warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
    warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
    warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.runtime")
    warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.runtime.scriptrunner_utils")
    
    # 设置所有相关的streamlit日志级别为CRITICAL，完全消除输出
    loggers_to_silence = [
        "streamlit",
        "streamlit.runtime", 
        "streamlit.runtime.scriptrunner_utils",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.caching",
        "streamlit.runtime.caching.cache_data_api",
        "streamlit.runtime.state",
        "streamlit.runtime.state.session_state_proxy"
    ]
    
    for logger_name in loggers_to_silence:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # 只显示最严重的错误
        logger.propagate = False  # 阻止传播到父级logger
        logger.disabled = True  # 完全禁用这些logger
    
    # 抑制其他常见警告
    warnings.filterwarnings("ignore", message=".*VegaFusion.*")
    warnings.filterwarnings("ignore", message=".*Altair.*")
    warnings.filterwarnings("ignore", message=".*to view a Streamlit app.*")
    warnings.filterwarnings("ignore", message=".*Session state does not function.*")
    
    # 设置标志位，表示已经执行过
    _warnings_suppressed = True

    # 优化：移除重复的警告抑制消息
    # print("[警告抑制] Streamlit警告消息已被抑制，控制台输出将更加清洁。")

# 🔥 自动执行：确保模块被导入时立即抑制警告
suppress_streamlit_warnings()

if __name__ == "__main__":
    print("Streamlit警告抑制脚本运行完成。") 