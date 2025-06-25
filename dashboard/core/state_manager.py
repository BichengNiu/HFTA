# -*- coding: utf-8 -*-
"""
统一状态管理器

提供集中化的状态管理、模块协调和数据共享功能
"""

import streamlit as st
from typing import Any, Dict, List, Optional, Set, Callable
import pandas as pd
from datetime import datetime
import logging
from .state_keys import StateKeys, ModuleStateKeys


class StateManager:
    """统一状态管理器"""
    
    def __init__(self, session_state=None):
        """
        初始化状态管理器
        
        Args:
            session_state: Streamlit session state对象，默认使用st.session_state
        """
        self.session_state = session_state if session_state is not None else st.session_state
        self.logger = logging.getLogger(__name__)
        self._cleanup_callbacks: Dict[str, List[Callable]] = {}
        self._initialized_modules: Set[str] = set()
        
        # 初始化核心状态
        self._init_core_state()
    
    def _init_core_state(self):
        """初始化核心状态管理相关的状态键"""
        if 'state_manager_initialized' not in self.session_state:
            self.session_state.state_manager_initialized = True
            self.session_state.state_manager_version = "1.0.0"
            self.session_state.state_manager_modules = set()
            
            # 初始化暂存数据
            if 'staged_data' not in self.session_state:
                self.session_state.staged_data = {}
            
            self.logger.info("StateManager initialized successfully")
    
    def register_module(self, module_name: str, auto_init: bool = True) -> bool:
        """
        注册模块到状态管理器
        
        Args:
            module_name: 模块名称
            auto_init: 是否自动初始化模块状态键
            
        Returns:
            bool: 注册是否成功
        """
        try:
            if module_name in self._initialized_modules:
                self.logger.warning(f"Module {module_name} already registered")
                return True
            
            # 获取模块状态键定义
            module_keys = StateKeys.get_module_keys(module_name)
            if not module_keys:
                self.logger.error(f"No state keys defined for module: {module_name}")
                return False
            
            # 自动初始化模块状态键
            if auto_init:
                self._init_module_state(module_name, module_keys)
            
            # 记录已注册模块
            self._initialized_modules.add(module_name)
            self.session_state.state_manager_modules.add(module_name)
            
            self.logger.info(f"Module {module_name} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register module {module_name}: {e}")
            return False
    
    def _init_module_state(self, module_name: str, state_keys: List[str]):
        """初始化模块状态键的默认值"""
        for key in state_keys:
            if key not in self.session_state:
                # 根据键名推断默认值类型
                if 'df' in key.lower():
                    self.session_state[key] = pd.DataFrame()
                elif 'map' in key.lower() or 'dict' in key.lower():
                    self.session_state[key] = {}
                elif 'list' in key.lower() or 'industries' in key:
                    self.session_state[key] = []
                elif 'file' in key.lower():
                    self.session_state[key] = None
                elif 'status' in key.lower():
                    self.session_state[key] = "未初始化"
                elif 'counter' in key.lower():
                    self.session_state[key] = 0
                elif 'flag' in key.lower() or key.endswith('_loaded'):
                    self.session_state[key] = False
                else:
                    self.session_state[key] = None
    
    def get_module_state(self, module_name: str) -> Dict[str, Any]:
        """
        获取指定模块的所有状态
        
        Args:
            module_name: 模块名称
            
        Returns:
            Dict[str, Any]: 模块状态字典
        """
        module_keys = StateKeys.get_module_keys(module_name)
        return {key: self.session_state.get(key) for key in module_keys}
    
    def set_module_state(self, module_name: str, key: str, value: Any) -> bool:
        """
        设置模块状态值
        
        Args:
            module_name: 模块名称
            key: 状态键
            value: 状态值
            
        Returns:
            bool: 设置是否成功
        """
        try:
            # 验证键是否属于该模块
            module_keys = StateKeys.get_module_keys(module_name)
            if key not in module_keys:
                self.logger.warning(f"Key {key} not defined for module {module_name}")
                return False
            
            self.session_state[key] = value
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set state {key} for module {module_name}: {e}")
            return False

    def set_state(self, module_name: str, key: str, value: Any) -> bool:
        """
        设置状态值（简化接口）

        Args:
            module_name: 模块名称
            key: 状态键
            value: 状态值

        Returns:
            bool: 设置是否成功
        """
        return self.set_module_state(module_name, key, value)

    def get_state(self, module_name: str, key: str, default: Any = None) -> Any:
        """
        获取状态值（简化接口）

        Args:
            module_name: 模块名称
            key: 状态键
            default: 默认值

        Returns:
            Any: 状态值
        """
        return self.session_state.get(key, default)

    def clear_module_state(self, module_name: str, exclude_shared: bool = True):
        """
        清理指定模块的状态
        
        Args:
            module_name: 模块名称
            exclude_shared: 是否排除共享状态键
        """
        try:
            module_keys = StateKeys.get_module_keys(module_name)
            shared_keys = StateKeys.get_shared_keys(module_name) if exclude_shared else []
            
            for key in module_keys:
                if key not in shared_keys:
                    if key in self.session_state:
                        del self.session_state[key]
            
            # 执行清理回调
            if module_name in self._cleanup_callbacks:
                for callback in self._cleanup_callbacks[module_name]:
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error(f"Cleanup callback failed for {module_name}: {e}")
            
            self.logger.info(f"Cleared state for module: {module_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to clear state for module {module_name}: {e}")
    
    def register_cleanup_callback(self, module_name: str, callback: Callable):
        """
        注册模块清理回调函数
        
        Args:
            module_name: 模块名称
            callback: 清理回调函数
        """
        if module_name not in self._cleanup_callbacks:
            self._cleanup_callbacks[module_name] = []
        self._cleanup_callbacks[module_name].append(callback)
    
    def share_data(self, source_key: str, target_key: str, transform_func: Optional[Callable] = None):
        """
        在模块间共享数据
        
        Args:
            source_key: 源状态键
            target_key: 目标状态键
            transform_func: 数据转换函数（可选）
        """
        try:
            if source_key in self.session_state:
                source_data = self.session_state[source_key]
                
                if transform_func:
                    target_data = transform_func(source_data)
                else:
                    target_data = source_data
                
                self.session_state[target_key] = target_data
                self.logger.debug(f"Shared data from {source_key} to {target_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to share data from {source_key} to {target_key}: {e}")
    
    def cleanup_on_module_switch(self, old_module: str, new_module: str):
        """
        模块切换时的状态清理
        
        Args:
            old_module: 旧模块名称
            new_module: 新模块名称
        """
        if old_module and old_module != new_module:
            self.clear_module_state(old_module, exclude_shared=True)
            self.logger.info(f"Cleaned up state when switching from {old_module} to {new_module}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态管理器摘要信息"""
        staged_data = self.session_state.get('staged_data', {})
        staged_data_count = len(staged_data) if staged_data is not None else 0

        return {
            'initialized_modules': list(self._initialized_modules),
            'total_state_keys': len(self.session_state.keys()),
            'state_manager_version': self.session_state.get('state_manager_version'),
            'staged_data_count': staged_data_count
        }
    
    def validate_state_integrity(self) -> Dict[str, List[str]]:
        """验证状态完整性"""
        issues = {
            'missing_keys': [],
            'invalid_key_names': [],
            'orphaned_keys': []
        }
        
        # 检查已注册模块的必需键
        for module_name in self._initialized_modules:
            module_keys = StateKeys.get_module_keys(module_name)
            for key in module_keys:
                if key not in self.session_state:
                    issues['missing_keys'].append(f"{module_name}.{key}")
        
        # 检查键名规范
        for key in self.session_state.keys():
            if not StateKeys.validate_key_name(key):
                issues['invalid_key_names'].append(key)
        
        return issues
