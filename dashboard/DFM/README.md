# DFM模块使用指南

## 概述

DFM (Dynamic Factor Model) 模块是一个完整的动态因子模型系统，提供从数据预处理到模型训练、分析和可视化的全流程功能。本模块已经过重构，采用统一的输出目录管理，消除了文件重复和路径分散的问题。

## 功能特性

### 🔧 核心功能
- **数据预处理**: 自动化数据清洗、转换和标准化
- **模型训练**: 支持超参数调优和变量选择的DFM训练
- **新闻分析**: 基于DFM的新闻影响分析
- **预测演化**: Nowcasting预测的时间演化分析
- **可视化**: 丰富的图表和交互式可视化

### 🎯 高级功能
- **超参数网格搜索**: 自动寻找最优因子数量
- **变量选择**: 全局和分块后向选择算法
- **详细分析**: PCA分析、因子贡献、R²统计
- **专业报告**: 生成包含所有分析结果的Excel报告

## 目录结构

### 文件输出方式
所有DFM相关输出现在通过UI界面下载获得，不再使用固定的输出目录：

- **训练结果**: 通过训练模块UI下载模型文件、元数据、Excel报告等
- **新闻分析**: 通过新闻分析UI下载HTML可视化和分析数据
- **模型分析**: 通过模型分析UI下载因子载荷、时间序列数据等
- **数据预处理**: 通过数据准备UI下载处理后的数据和映射文件

### 模块结构
```
dashboard/DFM/
├── config.py                 # 统一配置文件
├── README.md                 # 本文档
├── data_prep/               # 数据预处理模块
│   ├── data_preparation.py
│   └── data_prep_ui.py
├── train_model/             # 模型训练模块
│   ├── tune_dfm.py          # 主训练管道
│   ├── DynamicFactorModel.py
│   ├── analysis_utils.py
│   ├── results_analysis.py
│   ├── variable_selection.py
│   └── train_model_ui.py
├── model_analysis/         # 模型分析模块
│   ├── dfm_backend.py
│   └── dfm_ui.py
└── news_analysis/          # 新闻分析模块
    ├── news_analysis_backend.py
    ├── run_nowcasting_evolution.py
    └── DFM_Nowcasting.py
```

## 快速开始

### 1. 环境准备
确保安装了必要的依赖包：
```bash
pip install pandas numpy matplotlib seaborn plotly streamlit openpyxl
```

### 2. 数据预处理
1. 准备Excel数据文件（默认：`data/经济数据库0508.xlsx`）
2. 在Streamlit界面的"数据准备"标签页进行数据预处理
3. 配置平稳性检验和数据转换参数

### 3. 模型训练
1. 在"模型训练"标签页选择目标变量和预测指标
2. 配置训练参数：
   - **时间设置**: 训练期间和验证集分割
   - **模型参数**: 因子数量、迭代次数等
   - **优化设置**: 超参数调优、变量选择
   - **分析设置**: 详细分析选项
3. 点击"启动完整DFM优化管道"开始训练

### 4. 结果分析
训练完成后，可以：
- 查看可视化结果（因子图、拟合图、预测图）
- 下载各类数据文件（模型、因子、载荷等）
- 获取综合Excel分析报告

## 文件说明

### 训练结果文件

| 文件 | 说明 |
|------|------|
| `final_dfm_model.pkl` | 最终DFM模型（pickle格式） |
| `final_dfm_model.joblib` | 最终DFM模型（joblib格式） |
| `final_dfm_metadata.pkl` | 完整元数据信息 |
| `smoothed_factors.csv` | 平滑因子时间序列 |
| `factor_loadings.csv` | 因子载荷矩阵 |
| `fitted_observables.csv` | 拟合观测值 |
| `hyperparameter_grid_search_results.csv` | 超参数搜索结果 |
| `pca_analysis.csv` | PCA分析结果 |
| `factor_contributions.csv` | 因子贡献度分析 |
| `comprehensive_dfm_report_{timestamp}.xlsx` | 综合Excel报告 |

### 可视化文件

| 文件 | 说明 |
|------|------|
| `smoothed_factors_plot.png` | 平滑因子时间序列图 |
| `fitted_vs_actual_plot.png` | 拟合vs实际值对比图 |
| `final_nowcast_plot_{timestamp}.png` | 最终预测结果图 |

## 配置说明

### 核心配置 (`config.py`)
- **文件命名**: 标准化的文件命名规范
- **数据源**: Excel数据文件和列名配置
- **模型参数**: 默认的训练和分析参数

### 主要参数

#### 模型参数
- `n_factors`: 因子数量
- `factor_order`: 因子阶数（当前固定为1）
- `idio_ar_order`: 特异性误差阶数（当前固定为1）
- `em_max_iter`: EM算法最大迭代次数

#### 优化参数
- `enable_hyperparameter_tuning`: 启用超参数调优
- `k_factors_range`: 因子数量搜索范围
- `enable_variable_selection`: 启用变量选择
- `validation_split_ratio`: 验证集分割比例

#### 分析参数
- `enable_detailed_analysis`: 启用详细分析
- `generate_excel_report`: 生成Excel报告
- `pca_n_components`: PCA主成分数量

## 文件管理

所有结果文件现在通过UI界面下载获得：

- **训练结果**: 在训练完成后通过下载按钮获取
- **分析报告**: 通过各个分析模块的下载功能获取
- **可视化图表**: 直接在UI中查看或下载
- **数据文件**: 通过相应模块的数据下载功能获取

## 最佳实践

### 数据准备
1. **数据质量**: 确保输入数据的时间序列完整性
2. **变量选择**: 选择有经济意义的指标变量
3. **平稳性**: 适当配置平稳性检验参数

### 模型训练
1. **超参数调优**: 建议启用以获得最佳性能
2. **变量选择**: 对于变量较多的情况建议启用
3. **验证期设置**: 合理设置验证期比例（建议0.7-0.8）

### 结果分析
1. **查看Excel报告**: 包含最全面的分析结果
2. **关注关键指标**: Hit Rate、RMSE、R²等
3. **因子解释**: 结合载荷矩阵理解因子经济含义

## 故障排除

### 常见问题

1. **训练卡在"正在训练"状态**
   - 检查数据质量和完整性
   - 减少变量数量或因子数量
   - 查看训练日志获取详细错误信息

2. **Excel报告生成失败**
   - 确保有足够的磁盘空间
   - 检查文件权限
   - 查看错误日志

3. **可视化显示异常**
   - 检查图表文件是否正确生成
   - 确认浏览器支持图片格式
   - 刷新界面重新加载

### 调试技巧
1. 查看训练日志获取详细信息
2. 使用较小的数据集进行测试
3. 逐步增加复杂度（关闭某些高级功能）

## 更新日志

### v2.0 (当前版本)
- ✅ 移除所有后台文件输出功能
- ✅ 所有结果通过UI下载获得
- ✅ 完整参数一致性检查
- ✅ 增强错误处理和日志
- ✅ 简化文件管理流程

### v1.0 (历史版本)
- 基础DFM训练功能
- 分散的输出路径
- 基本可视化功能

## 技术支持

如遇到问题，请：
1. 查看训练日志和错误信息
2. 检查输出目录结构和权限
3. 参考本文档的故障排除部分
4. 保留问题现场以便技术支持分析

---

📝 **文档版本**: v2.0  
📅 **更新时间**: 2024年12月  
�� **维护团队**: DFM开发组 