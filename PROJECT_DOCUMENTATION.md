# 项目文档: 动态因子模型 (DFM) 调优与临近预测

本文档概述了 `dym_estimate` 文件夹中 Python 脚本 (`tune_dfm.py`) 的主要流程和方法，用于动态因子模型 (DFM) 的超参数调优和基于模型的临近预测 (Nowcasting)。

## 1. 数据准备 (`data_preparation.py`)

数据准备的目标是将不同来源、不同频率、可能包含缺失值的原始经济指标转换为适用于 DFM 模型输入的、统一频率（周度）的数据集。

**主要方法:**

1.  **数据加载:**
    *   从指定的 Excel 文件 (`经济数据库.xlsx`) 加载数据。
    *   自动发现 Excel 文件中的所有 Sheet 页。
    *   **识别特定 Sheet:**
        *   目标变量 Sheet: 基于 `TARGET_SHEET_NAME` 参数指定。
        *   预测变量 Sheet: 仅加载名称以 `-日度` 或 `-周度` 结尾的 Sheet。
        *   其他名称格式的 Sheet 会被忽略。
    *   从目标 Sheet 中仅提取 `TARGET_VARIABLE_NAME` 指定的目标变量列。
2.  **数据清理 (加载时):**
    *   将 Sheet 索引转换为 DatetimeIndex，移除无法转换或为 NaT 的行。
    *   移除完全由 NaN 组成的行或列。
    *   将所有数据转换为数值类型，无法转换的值变为 `NaN`。
    *   **高缺失率变量移除:** 移除在**原始**日度或周度数据中缺失值比例超过 50% 的**预测变量**列。
3.  **频率统一与时间对齐:**
    *   **合并:** 将加载并清理后的目标变量（来自月度 Sheet）、日度预测变量和周度预测变量使用 `pd.concat(..., axis=1, join='outer')` 合并到一个 DataFrame 中。外连接 (`outer`) 会保留所有时间点，并在数据缺失处填充 `NaN`。
    *   **重采样至目标频率:** 调用 `.resample(target_freq).asfreq()` 将合并后的 DataFrame 的时间索引统一为目标频率 (`TARGET_FREQ`，例如 `'W-FRI'`)。
        *   对于**目标变量**（原始为月度），`.asfreq()` 会将月度值放置在该月**最后一个周五**的时间戳上，该月内其他周五为 `NaN`。
        *   对于预测变量，日度数据会聚合到周五，周度数据保持不变（除非频率结尾不同）。
    *   **结果:** 返回的 `all_data_aligned_weekly` DataFrame 包含了统一为周度频率的数据，其中包含了因合并和重采样产生的 `NaN` 值。
4.  **数据标准化 (在 `tune_dfm.py` 中):**
    *   在后续 DFM 模型的每次评估和最终训练前，会对输入数据进行标准化处理 (减去均值，除以标准差)。
    *   标准化所用的均值和标准差会被保存，用于后续将模型输出（预测值）反标准化回原始尺度。

## 2. 模型选择思路 (动态因子模型 - DFM)

本项目选用**动态因子模型 (DFM)** 进行临近预测。

**选择理由:**

1.  **处理大量序列:** DFM 能够有效地从大量经济指标（预测变量）中提取少数共同的潜在因子 (common factors)，捕捉宏观经济的主要动态。
2.  **利用混合频率信息:** 通过将所有数据对齐到周度，DFM 可以利用高频数据的信息来预测低频（原始月度）的目标变量。
3.  **处理缺失值:** DFM 框架结合卡尔曼滤波和 EM 算法，能够内在处理输入数据中的缺失值 (`NaN`)，**这对于处理 `data_preparation.py` 中合并与重采样后产生的 `NaN` 至关重要。**
4.  **临近预测能力:** DFM 能够利用最新发布的高频数据，对目标变量进行及时的预测（临近预测）。

**具体实现:**

*   使用了 `DynamicFactorModel.py` 中实现的 `DFM_EMalgo` 类。
*   该类使用基于 **EM (期望最大化) 算法** 迭代估计模型参数。
*   **E步 (期望步):** 应用**卡尔曼滤波 (Kalman Filter)** 和 **固定区间平滑 (Fixed Interval Smoother, FIS)** 来估计给定当前参数下的潜在因子状态。
    *   卡尔曼滤波实现 (`DiscreteKalmanFilter.py`) 中包含处理观测数据中 `NaN` 的逻辑。
*   **M步 (最大化步):** 基于 E 步得到的平滑状态，重新估计模型参数 (因子载荷 `Lambda`、状态转移矩阵 `A`、过程噪声协方差 `Q`、观测噪声协方差 `R` 等)。
*   **参数初始化:** 模型的初始参数（如 `Lambda`, `A`, `Q`, `R`）是基于对输入数据进行**主成分分析 (PCA)** 的结果来设定的。

## 3. 模型训练与超参数调优 (`tune_dfm.py`)

模型训练的核心是确定最优的变量组合和超参数（主要是因子数量 `k_factors`），以获得最佳的样本外 (Out-of-Sample, OOS) 预测性能。

**主要思路与流程:**

1.  **数据预处理 (变量级平稳性转换):**
    *   在每次 DFM 评估和最终模型训练前，对当前的**预测变量**子集应用 `apply_stationarity_transforms` 函数。
    *   该函数对**每个预测变量**单独进行 ADF (Augmented Dickey-Fuller) 单位根检验。
    *   如果变量是平稳的 (p-value < 阈值)，则使用其**原始水平值**。
    *   如果变量非平稳，则对其进行**一阶差分** (`.diff(1)`)。
    *   **目标变量始终保持原始水平值，不进行此转换。**
    *   *注: 代码中存在一个未使用的 `USE_LOG_YOY_TRANSFORM` 标志和相应的 `apply_log_yoy_transform` 函数，当前模型拟合过程不使用此逻辑。*
2.  **标准化:** 对经过平稳性转换的数据进行标准化。
3.  **忽略特定时期的目标值 (训练时):**
    *   在将数据**输入 DFM 模型进行参数估计时**（包括调优评估和最终模型训练），将每年 1 月和 2 月对应的**目标变量**观测值设置为 `NaN`。
    *   **目的:** 减轻春节等季节性因素对模型参数估计的潜在干扰。
    *   **注意:** 这**不影响**后续模型评估指标的计算，因为评估时使用的是未经修改的原始目标序列。
4.  **确定因子数范围:**
    *   根据初始变量集，按照 `var_type_map` (从 Excel 文件读取的指标类型) 进行分组。
    *   将变量数少于 3 个的组合并到 "其他" 组。
    *   最终的因子数范围 (`K_FACTORS_RANGE`) 基于分组后的块数动态确定 (从 1 到块数)。
5.  **变量选择 (分块后向剔除):**
    *   **初始化:** 使用所有变量和所有候选因子数进行评估，确定一个初始的最佳参数组合和性能基准（基于优化目标）。
    *   **分块处理:** 将变量按照类型分成不同的块。
    *   **迭代剔除:** 对每个块进行迭代：
        *   尝试移除块中的每一个变量。
        *   对于每次移除，使用剩余变量和所有候选因子数重新评估 DFM 模型性能。
        *   选择能最大程度**改进优化目标**（优先最大化平均胜率，其次最小化平均 **RMSE**）的移除操作。
        *   如果找到了改进，则**永久移除**该变量，更新当前最佳变量集、最佳参数和最佳性能得分，并继续在该块内尝试移除下一个变量。
        *   如果块内所有剩余变量的移除都**无法改进**当前最佳性能，则该块处理完毕，进入下一个块。
6.  **并行计算:** 使用 `concurrent.futures.ProcessPoolExecutor` 并行执行 DFM 评估，加速调优过程。
7.  **最终模型训练:**
    *   使用后向剔除后最终确定的最佳变量组合和最佳因子数 (`final_variables`, `final_params`)。
    *   再次进行变量级平稳性转换和标准化。
    *   同样在训练时忽略 1 月和 2 月的目标变量值。
    *   使用**全部可用数据**（经过转换和标准化处理）重新训练 DFM 模型，得到最终的模型参数（因子载荷 `Lambda`、状态转移矩阵 `A` 等）和平滑后的因子序列 (`x_sm`)。

## 4. 模型评估思路

模型评估的目的是衡量模型在样本内 (In-Sample, IS) 和样本外 (Out-of-Sample, OOS) 的预测性能。

**评估流程 (`evaluate_dfm_params` 函数):**

1.  **数据准备:** 获取当前评估所需的变量子集和对应的因子数参数。
2.  **数据处理:**
    *   应用变量级平稳性转换 (`apply_stationarity_transforms`)。
    *   标准化数据。
    *   训练时忽略 1 月/2 月目标值 (`NaN`)。
3.  **模型拟合:** 使用处理后的数据拟合 DFM 模型 (`DFM_EMalgo`)。
4.  **生成预测值 (Nowcast):**
    *   获取平滑后的因子序列 (`factors_sm = dfm_results.x_sm`) 和因子载荷矩阵 (`lambda_matrix = dfm_results.Lambda`)。
    *   提取目标变量对应的载荷向量 (`lambda_target`)。
    *   计算标准化的预测值: `nowcast_standardized = factors_sm @ lambda_target`。
    *   **反标准化:** 将预测值转换回原始尺度: `nowcast_orig_values = nowcast_standardized * target_std + target_mean`。
5.  **指标计算:**
    *   **对齐:** 将反标准化后的预测序列 (`nowcast_series_orig`) 与**未经任何转换的原始目标变量序列** (`original_target_series_full`) 在时间上对齐。
    *   **划分:** 将对齐后的数据划分为训练期 (`<= TRAIN_END_DATE`) 和验证期 (`VALIDATION_START_DATE` 到 `VALIDATION_END_DATE`)。
    *   **计算指标:**
        *   **均方根误差 (RMSE):** 分别计算训练期和验证期的 RMSE (`is_rmse`, `oos_rmse`)。 `RMSE = sqrt(mean((Actual - Predicted)^2))`
        *   **胜率 (Hit Rate):**
            *   计算预测值和实际值的**一阶差分**。
            *   比较差分符号是否一致 (`sign(Actual_diff) == sign(Predicted_diff)`)。
            *   胜率 = (符号一致且实际值变化非零的点数) / (实际值变化非零的总点数) * 100%。
            *   分别计算训练期和验证期的胜率 (`is_hit_rate`, `oos_hit_rate`)。
6.  **优化目标 (用于调优):**
    *   **主要目标:** 最大化**平均胜率** `(is_hit_rate + oos_hit_rate) / 2`。
    *   **次要目标:** 在平均胜率相同时，最小化**平均 RMSE** `(is_rmse + oos_rmse) / 2`。

**最终结果分析 (`analyze_and_save_final_results` 函数):**

*   使用最终模型生成的预测值和原始目标值，重新计算训练期和验证期的 RMSE 和 Hit Rate。
*   生成预测值与实际值的对比图 (`final_nowcast_comparison.png`)，图中不绘制 1 月和 2 月的实际观测点。
*   计算因子贡献度，分析每个变量主要由哪个因子解释。
*   生成因子解释文本（基于载荷绝对值 Top 5 的变量）。
*   为每个估计出的因子生成单独的时间序列图（例如 `factor_1_timeseries.png`）。
*   将所有配置、结果指标、最终变量列表及类型、因子载荷、因子序列、预测值与真实值对比、因子贡献度、因子解释、**对齐后的原始数据 (`Aligned_Original_Data`)**、**输入 DFM 模型的数据 (`Data_Input_to_DFM`)** 等保存到 Excel 文件 (`result.xlsx`) 的不同 Sheet 页中。 

## 5. 最终结果分析与输出 (`analyze_and_save_final_results` 函数)

此函数负责分析最终训练好的模型，计算最终指标，并生成图表和 Excel 文件。

**主要输出:**

1.  **最终指标计算:** (与之前一致)
    *   使用最终模型生成的**周度**预测值和**周度对齐的**原始目标值，重新计算训练期和验证期的 RMSE 和 Hit Rate。
2.  **预测对比图 (`final_nowcast_comparison_{timestamp}.png`):** (更新)
    *   生成预测值与实际值的对比图。
    *   图中绘制**完整**的**周度**预测序列 (`Nowcast (原始水平)`)，直至数据的最末端（包含未来预测）。
    *   **实际观测值 (`Target (原始水平)`)** 只在它们存在的点上绘制（通常是每个月的最后一个周五，并屏蔽 1/2 月的数据点）。
3.  **因子载荷热力图 (`factor_loadings_heatmap_{timestamp}.png`):** (与之前一致)
    *   生成热力图展示最终模型的因子载荷，帮助理解每个因子与哪些变量关联。
4.  **因子时间序列图 (`all_factors_timeseries_{timestamp}.png`):** (与之前一致)
    *   生成一张包含所有估计出的因子（标准化后）时间序列的图。
5.  **结果 Excel 文件 (`result_{timestamp}.xlsx`):** (完全重写)
    *   包含多个 Sheet 页，汇总了模型配置、结果和相关数据。缺失值 (`NaN`) 在 Excel 中通常显示为空白单元格 (`""`)。
    *   **`Summary_Overview`:**
        *   包含关键的运行参数（最终变量数、最佳因子数、是否使用对数同比转换标志位状态等）。
        *   最终模型的性能指标（训练期/验证期 Hit Rate、验证期 RMSE、调优过程中的最佳平均指标等）。
        *   总运行时间。
        *   追加了**分析文本 (`Analysis Text`)**，提供对最终结果的文字总结。
        *   追加了**PCA 解释方差 (`PCA Explained Variance`)** 表格，显示基于最终输入数据计算的 PCA 结果。
        *   追加了**因子对目标贡献度 (`Factor Contribution to Target Variance`)** 表格。
    *   **`Final_Selected_Variables`:**
        *   列出最终模型选择使用的变量名称 (`Variable Name`)。
        *   每个变量的类型 (`Variable Type`，来自输入 Excel)。
        *   每个变量在模型拟合前应用的平稳性转换 (`Transformation`，值为 'level', 'diff', 'skipped_empty', 'level_constant', 'level_error', 或 'target_level')。
        *   每个变量对各个估计出的因子（`Factor1`, `Factor2`, ...）的载荷值。
    *   **`Selected_Vars_Transformed`:**
        *   展示最终选择的变量集（包含目标变量和预测变量）在应用了**变量级平稳性转换**（差分或保持原值）**之后**的数据。这是接近输入给 DFM 模型（标准化之前）的数据形态。
    *   **`Nowcast_vs_Target`:**
        *   提供**周度**（例如 'W-FRI'）时间索引 (`index`)。
        *   `Nowcast (原始水平)`: 包含**完整**的、反标准化后的**周度**预测序列，覆盖整个时间范围（包括历史和未来预测期）。
        *   `Target (原始水平)`: 包含**原始月度目标值**，但已按周对齐，只在每个月**最后一个周五**显示该月的值，其他周五为空白。
        *   `Monthly_Forecast`: 只在每个月**最后一个周五**显示该周对应的**周度 Nowcast 值**（作为月度预测的代表值），其他周五为空白。
    *   **`Factor_Interpretation`:**
        *   为每个估计出的因子提供文字解释，列出对其载荷绝对值最高的变量（正向和负向），并尝试给出基于变量类型的初步经济含义解释。**文本会正确换行。**
    *   **`Data_Input_to_DFM`:**
        *   展示最终输入给 DFM 模型进行**最后一次训练**的**标准化后**的数据（`final_data_std_masked_for_fit` 的近似，但可能包含训练时掩码的目标值，且通常四舍五入）。
    *   **`Full_Aligned_Data_Orig`:**
        *   包含由 `data_preparation.py` 返回的、所有**初始加载**的变量（目标+预测）在经过**频率统一和时间对齐**（重采样到周度 'W-FRI'）后的**原始水平**数据。包含了因对齐和重采样产生的 `NaN` 值。
        
**已移除的旧 Sheets:**

*   `Final_Factor_Loadings` (合并到 `Final_Selected_Variables`)
*   `Selected_Vars_Level_or_LogYoY` (被 `Selected_Vars_Transformed` 替代)
*   `Final_Factors` (已移除)
*   `Factor_Contribution_Analysis` (合并到 `Summary_Overview`)
*   `Aligned_Original_Data` (与 `Full_Aligned_Data_Orig` 重复)
*   `Full_Nowcast_Output` (信息已整合进 `Nowcast_vs_Target`)
*   `Monthly_Forecast_Aggregated` / `Monthly_Forecast_2025` (逻辑整合进 `Nowcast_vs_Target` 的 `Monthly_Forecast` 列) 