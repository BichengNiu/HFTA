# -*- coding: utf-8 -*-
"""
新闻分析前端模块
提供独立的新闻分析UI界面，包括参数设置、后端调用和结果显示
"""

import streamlit as st
import pandas as pd
import os
import traceback

# --- 新增：导入状态管理器 ---
try:
    from ...core.state_manager import StateManager
    from ...core.compat import CompatibilityAdapter
    from ...core.state_keys import StateKeys
    DFM_STATE_MANAGER_AVAILABLE = True
except ImportError:
    DFM_STATE_MANAGER_AVAILABLE = False
    print("[DFM UI] Warning: State manager not available, using legacy state management")


def get_dfm_state_manager_instance():
    """获取状态管理器实例（DFM模块专用）"""
    if DFM_STATE_MANAGER_AVAILABLE and hasattr(st.session_state, 'state_manager_initialized'):
        try:
            state_manager = StateManager(st.session_state)
            compat_adapter = CompatibilityAdapter(st.session_state)
            return state_manager, compat_adapter
        except Exception as e:
            print(f"[DFM UI] Error getting state manager: {e}")
            return None, None
    return None, None


def get_dfm_state(key, default=None, session_state=None):
    """获取DFM状态值（兼容新旧状态管理）"""
    state_manager, compat_adapter = get_dfm_state_manager_instance()
    
    if compat_adapter is not None:
        return compat_adapter.get_value(key, default)
    else:
        # 回退到传统方式
        if session_state is not None:
            return getattr(session_state, key, default) if hasattr(session_state, key) else session_state.get(key, default)
        else:
            return st.session_state.get(key, default)


def set_dfm_state(key, value, session_state=None):
    """设置DFM状态值（兼容新旧状态管理）"""
    state_manager, compat_adapter = get_dfm_state_manager_instance()
    
    if compat_adapter is not None:
        compat_adapter.set_value(key, value, use_new_key=True)
    else:
        # 回退到传统方式
        if session_state is not None:
            if hasattr(session_state, key):
                setattr(session_state, key, value)
            else:
                session_state[key] = value
        else:
            st.session_state[key] = value
# --- 结束新增 ---

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# 导入后端执行函数
try:
    from .news_analysis_backend import execute_news_analysis
    # 移除重复的打印信息，避免控制台污染
    # print("[News Frontend] Successfully imported news_analysis_backend.execute_news_analysis")
    backend_available = True
except ImportError as e_backend:
    # st.error(f"无法导入新闻分析后端模块: {e_backend}。新闻分析功能不可用。")
    # print(f"[News Frontend] ERROR: Failed to import news_analysis_backend: {e_backend}")
    execute_news_analysis = None
    backend_available = False

def render_news_analysis_tab(st_instance, session_state):
    """
    渲染新闻分析标签页
    
    Args:
        st_instance: Streamlit实例
        session_state: 会话状态对象
    """
    
    if not backend_available:
        st_instance.error("新闻分析后端不可用，请检查模块导入。")
        return
    
    # === 参数设置区域 ===
    st_instance.markdown("##### 📅 分析参数设置")
    
    # 目标月份选择
    target_month_date_selected = st_instance.date_input(
        "目标月份",
        value=datetime.now().replace(day=1),  # 默认当月第一天
        min_value=datetime(2000, 1, 1),      # 合理的最小可选日期
        max_value=datetime.now().replace(day=1) + timedelta(days=365*5),  # 限制可选的最大日期
        key="news_target_month_date_selector_frontend",
        help="选择您希望进行新闻归因分析的目标月份。"
    )
    st_instance.caption("选择目标月份后，程序将自动使用该年和月份进行分析。")
    
    # === 文件检查区域 ===
    st_instance.markdown("##### 📁 模型文件检查")
    
    model_file = get_dfm_state('dfm_model_file_indep', None, session_state)
    metadata_file = get_dfm_state('dfm_metadata_file_indep', None, session_state)
    
    # 显示文件状态
    col_file1, col_file2 = st_instance.columns(2)
    with col_file1:
        if model_file is not None:
            file_name = getattr(model_file, 'name', '未知文件')
            st_instance.success(f"✅ 模型文件: {file_name}")
        else:
            st_instance.error("❌ 未找到模型文件")

    with col_file2:
        if metadata_file is not None:
            file_name = getattr(metadata_file, 'name', '未知文件')
            st_instance.success(f"✅ 元数据文件: {file_name}")
        else:
            st_instance.error("❌ 未找到元数据文件")

    if model_file is None or metadata_file is None:
        st_instance.warning("⚠️ 请先在 **模型分析** 标签页上传必要的模型文件和元数据文件。")
        st_instance.info("💡 提示：模型文件通常为 .joblib 格式，元数据文件通常为 .pkl 格式。")
        return
    
    # === 执行按钮 ===
    st_instance.markdown("---")
    
    if st_instance.button(
        "🚀 运行新闻分析和生成图表", 
        key="run_news_analysis_frontend_button",
        type="primary",
        use_container_width=True
    ):
        _execute_news_analysis(st_instance, session_state, target_month_date_selected, model_file, metadata_file)

def _execute_news_analysis(st_instance, session_state, target_month_date_selected, model_file, metadata_file):
    """
    执行新闻分析的内部函数
    
    Args:
        st_instance: Streamlit实例
        session_state: 会话状态对象
        target_month_date_selected: 选择的目标月份
        model_file: 模型文件
        metadata_file: 元数据文件
    """
    
    # ❌ 移除本地工作目录创建 - 改为完全基于内存处理
    # dashboard_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # 回到dashboard目录
    # dashboard_workspace_dir = os.path.abspath(os.path.join(dashboard_base_dir, "dashboard_workspace"))
    # os.makedirs(dashboard_workspace_dir, exist_ok=True)
    
    with st_instance.spinner("🔄 正在执行新闻分析，请稍候..."):
        try:
            # 准备参数
            target_month_str = target_month_date_selected.strftime('%Y-%m') if target_month_date_selected else None
            plot_start_date_str = None
            plot_end_date_str = None
            
            # ✅ 修改：直接传递文件内容，不创建临时目录
            # 调用后端执行
            backend_results = execute_news_analysis(
                dfm_model_file_content=model_file.getbuffer(),
                dfm_metadata_file_content=metadata_file.getbuffer(),
                target_month=target_month_str,
                plot_start_date=plot_start_date_str,
                plot_end_date=plot_end_date_str,
                # ❌ 移除：base_workspace_dir=dashboard_workspace_dir  # 不再传递本地目录
            )
            
            # 处理结果
            _display_analysis_results(st_instance, backend_results, target_month_str)
            
        except Exception as e_call_backend:
            st_instance.error(f"❌ 调用后端脚本时发生错误: {e_call_backend}")
            st_instance.error("📋 详细错误信息:")
            st_instance.code(traceback.format_exc())

def _display_analysis_results(st_instance, backend_results: Dict[str, Any], target_month_str: Optional[str]):
    """
    显示分析结果
    
    Args:
        st_instance: Streamlit实例
        backend_results: 后端返回的结果字典
        target_month_str: 目标月份字符串
    """
    
    if backend_results["returncode"] == 0:
        st_instance.success("✅ 后端脚本执行成功！")
        
        # === 结果展示区域 ===
        st_instance.markdown("---")
        
        # 图表展示
        _display_charts(st_instance, backend_results)
        
        # 数据下载
        _display_download_section(st_instance, backend_results)
        
    else:
        st_instance.error(f"❌ 后端脚本执行失败 (返回码: {backend_results['returncode']})")
        
        if backend_results.get("error_message"):
            st_instance.error(f"🔍 错误详情: {backend_results['error_message']}")
        
        # 显示调试信息
        if backend_results.get("stderr"):
            with st_instance.expander("🐛 查看错误日志", expanded=False):
                st_instance.code(backend_results["stderr"], language="text")

def _display_charts(st_instance, backend_results: Dict[str, Any]):
    """
    显示图表
    
    Args:
        st_instance: Streamlit实例
        backend_results: 后端返回的结果字典
    """
    
    col_left_chart, col_right_chart = st_instance.columns(2)
    
    # 左列：演变图
    with col_left_chart:
        st_instance.markdown("##### 📈 Nowcast 演变图")
        evo_plot_path = backend_results.get("evolution_plot_path")
        
        if evo_plot_path and os.path.exists(evo_plot_path):
            if evo_plot_path.lower().endswith(".html"):
                try:
                    with open(evo_plot_path, 'r', encoding='utf-8') as f_html_evo:
                        html_content_evo = f_html_evo.read()
                    st_instance.components.v1.html(html_content_evo, height=500, scrolling=True)
                    print(f"[News Frontend] Displayed Evolution HTML plot: {evo_plot_path}")
                except Exception as e_html_evo:
                    st_instance.error(f"读取或显示演变图时出错: {e_html_evo}")
            else:
                st_instance.warning(f"演变图文件格式未知: {evo_plot_path}")
        else:
            st_instance.warning("⚠️ 未找到 Nowcast 演变图文件或路径无效。")
    
    # 右列：分解图
    with col_right_chart:
        st_instance.markdown("##### 🔍 新闻贡献分解图")
        decomp_plot_path = backend_results.get("decomposition_plot_path")
        
        if decomp_plot_path and os.path.exists(decomp_plot_path):
            if decomp_plot_path.lower().endswith(".html"):
                try:
                    with open(decomp_plot_path, 'r', encoding='utf-8') as f_html_decomp:
                        html_content_decomp = f_html_decomp.read()
                    st_instance.components.v1.html(html_content_decomp, height=500, scrolling=True) 
                    print(f"[News Frontend] Displayed Decomposition HTML plot: {decomp_plot_path}")
                except Exception as e_html_decomp:
                    st_instance.error(f"读取或显示分解图时出错: {e_html_decomp}")
            else:
                st_instance.warning(f"分解图文件格式未知: {decomp_plot_path}")
        else:
            st_instance.warning("⚠️ 未找到新闻贡献分解图文件或路径无效。")

def _display_download_section(st_instance, backend_results: Dict[str, Any]):
    """
    显示下载区域
    
    Args:
        st_instance: Streamlit实例
        backend_results: 后端返回的结果字典
    """
    
    evo_csv_path = backend_results.get("evo_csv_path")
    news_csv_path = backend_results.get("news_csv_path")
    
    col_dl1, col_dl2 = st_instance.columns(2)
    
    with col_dl1:
        if evo_csv_path and os.path.exists(evo_csv_path):
            with open(evo_csv_path, "rb") as fp_evo:
                st_instance.download_button(
                    label="📈 下载 Nowcast 演变数据 (CSV)",
                    data=fp_evo,
                    file_name=os.path.basename(evo_csv_path),
                    mime="text/csv",
                    key="download_evo_csv_frontend",
                    use_container_width=True
                )
        else:
            st_instance.caption("⚠️ Nowcast 演变 CSV 未生成。")
    
    with col_dl2:
        if news_csv_path and os.path.exists(news_csv_path):
            with open(news_csv_path, "rb") as fp_news:
                st_instance.download_button(
                    label="🔍 下载新闻分解数据 (CSV)",
                    data=fp_news,
                    file_name=os.path.basename(news_csv_path),
                    mime="text/csv",
                    key="download_news_csv_frontend",
                    use_container_width=True
                )
        else:
            st_instance.caption("⚠️ 新闻分解 CSV 未生成。")

# === 主函数 ===
def render_news_analysis_main_interface(st_instance, session_state):
    """
    渲染新闻分析主界面
    
    Args:
        st_instance: Streamlit实例
        session_state: 会话状态对象
    """
    
    # 标题和说明
    st_instance.markdown("# 📰 新闻分析模块")
    st_instance.markdown("""
    **新闻分析**功能可以帮助您：
    - 🔍 分析特定月份的Nowcast演变过程
    - 📊 识别影响预测变化的关键因素
    - 🎯 量化新闻事件对预测的贡献度
    - 📈 可视化新闻影响的时间序列
    """)
    
    st_instance.markdown("---")
    
    # 渲染主要功能
    render_news_analysis_tab(st_instance, session_state)

if __name__ == "__main__":
    # 测试模块
    print("这是新闻分析前端模块。请通过主dashboard调用。")
