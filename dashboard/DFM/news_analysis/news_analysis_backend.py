import streamlit as st
import subprocess
import os
import sys
import shutil
from typing import Dict, Any, Optional
import tempfile
from datetime import datetime
import traceback

# --- BEGIN UPDATED SYS.PATH SETUP ---
# Get the directory of the current file (news_analysis_backend.py, which is in news_analysis folder)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the DFM directory (parent of news_analysis)
dfm_directory = os.path.abspath(os.path.join(current_script_dir, '..'))
# Get the dashboard directory (parent of DFM) # Added for clarity
dashboard_actual_dir = os.path.abspath(os.path.join(dfm_directory, '..'))
# Get the project root directory (parent of dashboard)
project_root_dir = os.path.abspath(os.path.join(dashboard_actual_dir, '..'))

# Add project root directory to sys.path
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# Add dashboard directory to sys.path  
if dashboard_actual_dir not in sys.path:
    sys.path.insert(0, dashboard_actual_dir)

# Add DFM directory to sys.path for potential imports from DFM or other subdirectories like news_analysis
if dfm_directory not in sys.path:
    sys.path.insert(0, dfm_directory)
    # print(f"[Backend] Added dfm_directory ('{dfm_directory}') to sys.path.")
# --- END UPDATED SYS.PATH SETUP ---

# 不再使用统一配置的outputs目录，强制使用系统临时目录
_CONFIG_AVAILABLE = False
DFM_NEWS_OUTPUT_DIR = None  # 强制使用系统临时目录
print("[Backend] 强制使用系统临时目录，不再使用项目内outputs目录")

# 尝试导入本地配置
try:
    # 可以在此导入本地配置文件（如果存在）
    # import config as local_config
    # print("[Backend] Successfully imported local config")
    pass  # 暂时不导入任何配置
except ImportError as e_config_backend:
    print(f"[Backend Startup] 使用默认配置设置: {e_config_backend}")
    pass

def execute_news_analysis(
    dfm_model_file_content: bytes,
    dfm_metadata_file_content: bytes,
    target_month: Optional[str],
    plot_start_date: Optional[str],
    plot_end_date: Optional[str],
    base_workspace_dir: str = None  # ✅ 保留参数但默认为None，使用系统临时目录
) -> Dict[str, Any]:
    """
    执行新闻分析的后端逻辑
    
    Args:
        dfm_model_file_content: DFM模型文件内容（bytes）
        dfm_metadata_file_content: DFM元数据文件内容（bytes）
        target_month: 目标月份 (YYYY-MM格式，可选)
        plot_start_date: 绘图开始日期 (YYYY-MM-DD格式，可选)
        plot_end_date: 绘图结束日期 (YYYY-MM-DD格式，可选)
        base_workspace_dir: 基础工作目录（如果不提供，使用系统临时目录）
    
    Returns:
        Dict: 包含分析结果、文件路径和可能的错误信息
    """
    
    # 强制使用系统临时目录，不再使用项目内outputs目录
    if base_workspace_dir is None:
        # 始终创建临时目录用于此次分析
        temp_base_dir = tempfile.mkdtemp(prefix="news_analysis_")
        base_workspace_dir = temp_base_dir
        print(f"[Backend] 使用系统临时目录: {base_workspace_dir}")
    
    print(f"[Backend] Starting news analysis execution...")
    print(f"[Backend]   Base workspace directory: {base_workspace_dir}")
    
    # 1. 设置路径 - 仍使用子目录结构但在临时目录中
    news_analysis_output_dir = os.path.join(base_workspace_dir, "news_analysis_output_backend")
    temp_model_input_dir = os.path.join(base_workspace_dir, "temp_dfm_inputs_for_backend")
    base_plot_filename_stem = "news_analysis_plot_backend"
    
    # 实际的图表文件名将由 run_nowcasting_evolution.py 通过添加后缀生成
    expected_evo_plot_filename = f"{base_plot_filename_stem}_evo.html"
    expected_decomp_plot_filename = f"{base_plot_filename_stem}_decomp.html"
    
    plot_full_path_for_script_arg = os.path.join(news_analysis_output_dir, f"{base_plot_filename_stem}.html")

    evo_plot_full_path = os.path.join(news_analysis_output_dir, expected_evo_plot_filename)
    decomp_plot_full_path = os.path.join(news_analysis_output_dir, expected_decomp_plot_filename)

    evo_csv_filename = "nowcast_evolution_data_T.csv"
    news_csv_filename = "news_decomposition_grouped.csv"
    evo_csv_full_path = os.path.join(news_analysis_output_dir, evo_csv_filename)
    news_csv_full_path = os.path.join(news_analysis_output_dir, news_csv_filename)

    results = {
        "evolution_plot_path": None,      # <<< 新增
        "decomposition_plot_path": None,  # <<< 新增
        "evo_csv_path": None,
        "news_csv_path": None,
        "stdout": "",
        "stderr": "",
        "returncode": -1, # Default error code
        "error_message": None
    }

    try:
        # 2. 创建目录
        print(f"[Backend]   Creating directories...")
        os.makedirs(news_analysis_output_dir, exist_ok=True)
        os.makedirs(temp_model_input_dir, exist_ok=True)
        print(f"[Backend]     Output dir: {news_analysis_output_dir}")
        print(f"[Backend]     Temp input dir: {temp_model_input_dir}")

        # 3. 准备输入文件 (写入临时目录)
        # 使用标准化的文件名，从上传的文件重命名
        model_filename_from_config = "final_dfm_model.joblib"
        metadata_filename_from_config = "final_model_metadata.pkl"

        temp_model_path = os.path.join(temp_model_input_dir, model_filename_from_config)
        temp_metadata_path = os.path.join(temp_model_input_dir, metadata_filename_from_config)

        print(f"[Backend]   Writing temporary model file to: {temp_model_path}")
        with open(temp_model_path, "wb") as f:
            f.write(dfm_model_file_content)
        
        print(f"[Backend]   Writing temporary metadata file to: {temp_metadata_path}")
        with open(temp_metadata_path, "wb") as f:
            f.write(dfm_metadata_file_content)

        # 4. 定位 run_nowcasting_evolution.py 脚本
        # 此后端脚本 (news_analysis_backend.py) 位于 news_analysis/
        # run_nowcasting_evolution.py 也位于 news_analysis/
        script_dir = os.path.dirname(__file__)
        script_to_run_path = os.path.join(script_dir, "run_nowcasting_evolution.py")
        
        if not os.path.exists(script_to_run_path):
            results["error_message"] = f"Target script not found at: {script_to_run_path}"
            print(f"[Backend] ERROR: {results['error_message']}")
            return results
            
        print(f"[Backend]   Target script path: {script_to_run_path}")

        # 5. 构建命令
        cmd = [
            sys.executable, 
            script_to_run_path,
            "--model_files_dir", temp_model_input_dir, 
            "--evolution_output_dir", news_analysis_output_dir,
            "--plot_output_file", plot_full_path_for_script_arg, # <<< 传递基础HTML名给脚本
        ]
        if target_month:
            cmd.extend(["--target_month", target_month])
        if plot_start_date:
            cmd.extend(["--plot_start_date", plot_start_date])
        if plot_end_date:
            cmd.extend(["--plot_end_date", plot_end_date])

        print(f"[Backend]   Executing command: {' '.join(cmd)}")

        # 6. 执行脚本
        process = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

        # 7. 收集结果
        results["stdout"] = process.stdout
        results["stderr"] = process.stderr
        results["returncode"] = process.returncode

        print(f"[Backend]   Script finished with return code: {process.returncode}")
        # --- 新增：无论成功与否，都记录完整的子脚本标准输出和错误 --- 
        print(f"[Backend]   >> Subprocess STDOUT START <<")
        print(process.stdout)
        print(f"[Backend]   >> Subprocess STDOUT END <<")
        if process.stderr:
            print(f"[Backend]   >> Subprocess STDERR START <<")
            print(process.stderr)
            print(f"[Backend]   >> Subprocess STDERR END <<")
        # --- 结束新增 ---

        if process.returncode == 0:
            # 检查输出文件是否存在
            if os.path.exists(evo_plot_full_path):
                results["evolution_plot_path"] = evo_plot_full_path
                print(f"[Backend]   Evolution plot found: {evo_plot_full_path}")
            else:
                print(f"[Backend] WARNING: Evolution plot file not found after successful execution: {evo_plot_full_path}")
            
            if os.path.exists(decomp_plot_full_path):
                results["decomposition_plot_path"] = decomp_plot_full_path
                print(f"[Backend]   Decomposition plot found: {decomp_plot_full_path}")
            else:
                print(f"[Backend] WARNING: Decomposition plot file not found after successful execution: {decomp_plot_full_path}")

            # 旧的单一 plot_path 检查 (可以移除或注释)
            # if os.path.exists(plot_full_path): # plot_full_path 现在是基础名，不应该直接检查
            #     results["plot_path"] = plot_full_path 
            #     print(f"[Backend]   Plot found: {plot_full_path}")
            # else:
            #     print(f"[Backend] WARNING: Plot file not found after successful execution: {plot_full_path}")
                
            if os.path.exists(evo_csv_full_path):
                results["evo_csv_path"] = evo_csv_full_path
                print(f"[Backend]   Evolution CSV found: {evo_csv_full_path}")
            else:
                print(f"[Backend] WARNING: Evolution CSV file not found: {evo_csv_full_path}")
                
            if os.path.exists(news_csv_full_path):
                results["news_csv_path"] = news_csv_full_path
                print(f"[Backend]   News CSV found: {news_csv_full_path}")
            else:
                 print(f"[Backend] WARNING: News CSV file not found: {news_csv_full_path}")
        else:
            results["error_message"] = f"Script execution failed with return code {process.returncode}. Check stderr for details."
            print(f"[Backend] ERROR: {results['error_message']}")
            # Optional: Log more details from stderr here

    except Exception as e:
        error_msg = f"An unexpected error occurred in the backend: {e}"
        print(f"[Backend] UNEXPECTED ERROR: {error_msg}")
        traceback.print_exc() # Print full traceback to console/log
        results["error_message"] = error_msg
        results["stderr"] += f"\nBackend Exception: {traceback.format_exc()}"

    # 8. (可选) 清理临时输入目录 - 考虑是否需要保留用于调试
    # try:
    #     print(f"[Backend]   Cleaning up temporary input directory: {temp_model_input_dir}")
    #     shutil.rmtree(temp_model_input_dir)
    # except Exception as e_clean:
    #     print(f"[Backend] WARNING: Failed to clean up temporary directory {temp_model_input_dir}: {e_clean}")

    print(f"[Backend] News analysis execution finished.")
    return results

if __name__ == '__main__':
    # 此处可以添加一些用于直接测试后端脚本的逻辑
    print("This is the news_analysis_backend script. It should be called via its functions.")
    # Example test (requires placeholder files):
    # base_dir_test = "."
    # model_content_test = b"model data"
    # meta_content_test = b"meta data"
    # with open("placeholder_model.joblib", "wb") as f: f.write(model_content_test)
    # with open("placeholder_meta.pkl", "wb") as f: f.write(meta_content_test)
    # results_test = execute_news_analysis(model_content_test, meta_content_test, "2023-10", None, None, base_dir_test)
    # print("\nTest Results:")
    # import json
    # print(json.dumps(results_test, indent=2))
    print("Please run through the Streamlit dashboard UI.") 