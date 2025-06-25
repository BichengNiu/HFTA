# -*- coding: utf-8 -*-
"""
Dashboard核心状态管理模块

提供统一的状态管理、数据共享和模块协调功能
"""

from .state_manager import StateManager
from .state_keys import StateKeys
from .compat import CompatibilityAdapter

__version__ = "1.0.0"
__all__ = ["StateManager", "StateKeys", "CompatibilityAdapter"]
