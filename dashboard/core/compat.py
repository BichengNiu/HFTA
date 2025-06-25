# -*- coding: utf-8 -*-
"""
向后兼容适配器

确保新的状态管理系统与现有代码兼容，提供平滑的迁移路径
"""

import streamlit as st
from typing import Any, Dict, Optional
import logging
from .state_keys import StateKeys


class CompatibilityAdapter:
    """向后兼容适配器"""
    
    # 旧键名到新键名的映射
    KEY_MAPPING = {
        # 导航相关
        'selected_main_module': 'nav_selected_main_module',
        'selected_sub_module': 'nav_selected_sub_module',
        'active_tab': 'nav_active_tab',
        'data_import_reset_counter': 'nav_data_import_reset_counter',
        
        # 数据预览相关（保持不变，已经使用preview_前缀）
        'data_loaded': 'preview_data_loaded_files',  # 特殊映射
        'weekly_df': 'preview_weekly_df',
        'monthly_df': 'preview_monthly_df',
        'source_map': 'preview_source_map',
        'indicator_industry_map': 'preview_indicator_industry_map',
        'weekly_industries': 'preview_weekly_industries',
        'monthly_industries': 'preview_monthly_industries',
        'clean_industry_map': 'preview_clean_industry_map',
        'weekly_summary_cache': 'preview_weekly_summary_cache',
        'monthly_summary_cache': 'preview_monthly_summary_cache',
        
        # 应用工具相关
        'ts_tool_uploaded_file': 'tools_ts_uploaded_file',
        'ts_tool_data_raw': 'tools_ts_data_raw',
        'ts_tool_data_processed': 'tools_ts_data_processed',
        'ts_tool_data_final': 'tools_ts_data_final',
        'stationarity_uploaded_file_tool': 'tools_stationarity_uploaded_file',
        'correlation_selected_df': 'tools_correlation_selected_df',
        'correlation_selected_df_name': 'tools_correlation_selected_df_name',
    }
    
    # 反向映射（新键名到旧键名）
    REVERSE_KEY_MAPPING = {v: k for k, v in KEY_MAPPING.items()}
    
    def __init__(self, session_state=None):
        """
        初始化兼容适配器
        
        Args:
            session_state: Streamlit session state对象
        """
        self.session_state = session_state if session_state is not None else st.session_state
        self.logger = logging.getLogger(__name__)
        self._migration_log = []
    
    def migrate_existing_state(self) -> Dict[str, int]:
        """
        迁移现有状态到新的命名规范
        
        Returns:
            Dict[str, int]: 迁移统计信息
        """
        stats = {
            'migrated': 0,
            'skipped': 0,
            'errors': 0
        }
        
        # 创建现有键的快照，避免在迭代过程中修改字典
        existing_keys = list(self.session_state.keys())
        
        for old_key in existing_keys:
            if old_key in self.KEY_MAPPING:
                new_key = self.KEY_MAPPING[old_key]
                try:
                    # 如果新键不存在，则迁移
                    if new_key not in self.session_state:
                        self.session_state[new_key] = self.session_state[old_key]
                        self._migration_log.append(f"Migrated: {old_key} -> {new_key}")
                        stats['migrated'] += 1
                    else:
                        stats['skipped'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Failed to migrate {old_key} to {new_key}: {e}")
                    stats['errors'] += 1
        
        self.logger.info(f"State migration completed: {stats}")
        return stats
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        获取状态值，自动处理新旧键名映射
        
        Args:
            key: 状态键（可以是新键名或旧键名）
            default: 默认值
            
        Returns:
            Any: 状态值
        """
        # 首先尝试直接获取
        if key in self.session_state:
            return self.session_state[key]
        
        # 如果是旧键名，尝试映射到新键名
        if key in self.KEY_MAPPING:
            new_key = self.KEY_MAPPING[key]
            if new_key in self.session_state:
                return self.session_state[new_key]
        
        # 如果是新键名，尝试映射到旧键名
        if key in self.REVERSE_KEY_MAPPING:
            old_key = self.REVERSE_KEY_MAPPING[key]
            if old_key in self.session_state:
                return self.session_state[old_key]
        
        return default
    
    def set_value(self, key: str, value: Any, use_new_key: bool = True):
        """
        设置状态值，自动处理新旧键名映射
        
        Args:
            key: 状态键
            value: 状态值
            use_new_key: 是否优先使用新键名
        """
        if use_new_key:
            # 优先使用新键名
            if key in self.KEY_MAPPING:
                new_key = self.KEY_MAPPING[key]
                self.session_state[new_key] = value
            else:
                self.session_state[key] = value
        else:
            # 使用原键名
            self.session_state[key] = value
    
    def ensure_compatibility(self, module_name: str):
        """
        确保指定模块的兼容性
        
        Args:
            module_name: 模块名称
        """
        module_keys = StateKeys.get_module_keys(module_name)
        
        for new_key in module_keys:
            # 如果新键不存在，但对应的旧键存在，则创建映射
            if new_key not in self.session_state:
                old_key = self.REVERSE_KEY_MAPPING.get(new_key)
                if old_key and old_key in self.session_state:
                    self.session_state[new_key] = self.session_state[old_key]
                    self.logger.debug(f"Created compatibility mapping: {old_key} -> {new_key}")
    
    def create_legacy_accessors(self):
        """
        为旧代码创建访问器，使其能够透明地访问新状态键
        
        注意：这是一个高级功能，需要谨慎使用
        """
        class LegacyStateProxy:
            def __init__(self, adapter):
                self.adapter = adapter
            
            def __getattr__(self, name):
                return self.adapter.get_value(name)
            
            def __setattr__(self, name, value):
                if name == 'adapter':
                    super().__setattr__(name, value)
                else:
                    self.adapter.set_value(name, value)
            
            def get(self, key, default=None):
                return self.adapter.get_value(key, default)
            
            def __contains__(self, key):
                return (key in self.adapter.session_state or 
                       key in self.adapter.KEY_MAPPING or 
                       key in self.adapter.REVERSE_KEY_MAPPING)
        
        return LegacyStateProxy(self)
    
    def get_migration_log(self) -> list:
        """获取迁移日志"""
        return self._migration_log.copy()
    
    def validate_compatibility(self) -> Dict[str, list]:
        """
        验证兼容性状态
        
        Returns:
            Dict[str, list]: 兼容性检查结果
        """
        issues = {
            'missing_mappings': [],
            'conflicting_keys': [],
            'orphaned_old_keys': []
        }
        
        # 检查是否有旧键没有对应的新键
        for old_key in self.session_state.keys():
            if (old_key not in self.KEY_MAPPING and 
                not StateKeys.validate_key_name(old_key) and
                not old_key.startswith('state_manager_')):
                issues['orphaned_old_keys'].append(old_key)
        
        # 检查是否有冲突的键（新旧键同时存在但值不同）
        for old_key, new_key in self.KEY_MAPPING.items():
            if (old_key in self.session_state and 
                new_key in self.session_state):
                old_val = self.session_state[old_key]
                new_val = self.session_state[new_key]
                
                # 简单的值比较（对于复杂对象可能需要更精细的比较）
                try:
                    if str(old_val) != str(new_val):
                        issues['conflicting_keys'].append(f"{old_key} != {new_key}")
                except:
                    # 如果比较失败，记录为潜在冲突
                    issues['conflicting_keys'].append(f"{old_key} <-> {new_key} (comparison failed)")
        
        return issues
