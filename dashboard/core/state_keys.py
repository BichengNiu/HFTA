# -*- coding: utf-8 -*-
"""
统一状态键命名规范

定义所有模块的状态键命名标准，确保一致性和避免冲突
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class ModuleStateKeys:
    """模块状态键定义"""
    module_name: str
    keys: List[str]
    shared_keys: List[str] = None  # 与其他模块共享的键
    
    def __post_init__(self):
        if self.shared_keys is None:
            self.shared_keys = []


class StateKeys:
    """状态键命名规范管理器"""
    
    # 命名前缀规范
    PREFIXES = {
        'navigation': 'nav_',
        'preview': 'preview_',
        'dfm': 'dfm_',
        'tools': 'tools_',
        'staged': 'staged_',
        'temp': 'temp_',
        'ui': 'ui_',
        'cache': 'cache_'
    }
    
    # 导航相关状态键
    NAVIGATION = ModuleStateKeys(
        module_name='navigation',
        keys=[
            'nav_selected_main_module',
            'nav_selected_sub_module',
            'nav_active_tab',
            'nav_data_import_reset_counter'
        ]
    )
    
    # 数据预览模块状态键
    PREVIEW = ModuleStateKeys(
        module_name='preview',
        keys=[
            'preview_data_loaded_files',
            'preview_weekly_df',
            'preview_monthly_df',
            'preview_daily_df',
            'preview_source_map',
            'preview_indicator_industry_map',
            'preview_weekly_industries',
            'preview_monthly_industries',
            'preview_daily_industries',
            'preview_clean_industry_map',
            'preview_weekly_summary_cache',
            'preview_monthly_summary_cache',
            'preview_daily_summary_cache',
            'preview_monthly_growth_summary_df'
        ],
        shared_keys=[
            'preview_weekly_df',
            'preview_monthly_df',
            'preview_source_map',
            'preview_indicator_industry_map'
        ]
    )
    
    # DFM模块状态键
    DFM = ModuleStateKeys(
        module_name='dfm',
        keys=[
            'dfm_model_file_indep',
            'dfm_metadata_file_indep',
            'dfm_training_status',
            'dfm_model_results',
            'dfm_training_log',
            'dfm_model_results_paths'
        ],
        shared_keys=[
            'dfm_model_file_indep',
            'dfm_metadata_file_indep'
        ]
    )
    
    # 应用工具模块状态键
    TOOLS = ModuleStateKeys(
        module_name='tools',
        keys=[
            'tools_ts_uploaded_file',
            'tools_ts_data_raw',
            'tools_ts_data_processed',
            'tools_ts_data_final',
            'tools_stationarity_uploaded_file',
            'tools_correlation_selected_df',
            'tools_correlation_selected_df_name'
        ]
    )
    
    # 暂存数据状态键
    STAGED = ModuleStateKeys(
        module_name='staged',
        keys=[
            'staged_data',  # 全局暂存数据字典
            'staged_data_metadata'  # 暂存数据元信息
        ],
        shared_keys=[
            'staged_data'
        ]
    )
    
    # 所有模块状态键集合
    ALL_MODULES = [NAVIGATION, PREVIEW, DFM, TOOLS, STAGED]
    
    @classmethod
    def get_module_keys(cls, module_name: str) -> List[str]:
        """获取指定模块的所有状态键"""
        for module in cls.ALL_MODULES:
            if module.module_name == module_name:
                return module.keys
        return []
    
    @classmethod
    def get_shared_keys(cls, module_name: str) -> List[str]:
        """获取指定模块的共享状态键"""
        for module in cls.ALL_MODULES:
            if module.module_name == module_name:
                return module.shared_keys
        return []
    
    @classmethod
    def get_all_keys(cls) -> Set[str]:
        """获取所有状态键"""
        all_keys = set()
        for module in cls.ALL_MODULES:
            all_keys.update(module.keys)
        return all_keys
    
    @classmethod
    def validate_key_name(cls, key: str) -> bool:
        """验证状态键命名是否符合规范"""
        if not key:
            return False
        
        # 检查是否使用了标准前缀
        for prefix in cls.PREFIXES.values():
            if key.startswith(prefix):
                return True
        
        # 允许一些特殊的全局键
        special_keys = {'active_tab', 'data_loaded'}
        return key in special_keys
    
    @classmethod
    def suggest_key_name(cls, module_name: str, key_description: str) -> str:
        """根据模块名和描述建议状态键名称"""
        if module_name in cls.PREFIXES:
            prefix = cls.PREFIXES[module_name]
        else:
            prefix = f"{module_name}_"

        # 清理描述文本
        clean_desc = key_description.lower().replace(' ', '_').replace('-', '_')
        return f"{prefix}{clean_desc}"

    @classmethod
    def get_navigation_keys(cls):
        """获取导航模块的状态键"""
        return cls.NAVIGATION.keys

    @classmethod
    def get_preview_keys(cls):
        """获取预览模块的状态键"""
        return cls.PREVIEW.keys

    @classmethod
    def get_dfm_keys(cls):
        """获取DFM模块的状态键"""
        return cls.DFM.keys

    @classmethod
    def get_tools_keys(cls):
        """获取工具模块的状态键"""
        return cls.TOOLS.keys

    @classmethod
    def get_staged_keys(cls):
        """获取暂存模块的状态键"""
        return cls.STAGED.keys
