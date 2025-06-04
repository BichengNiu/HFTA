import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tools.time_series_property import combined_lead_lag_backend # Import the new backend

def interpret_lead_lag(k_corr, corr_value, k_kl, kl_value, corr_threshold=0.3, kl_significant_change_threshold=0.1, lag_agreement_tolerance=1):
    """
    Provides a textual interpretation of the combined lead-lag results.
    Adjust thresholds as needed.
    """
    # Significance checks
    corr_significant = pd.notna(corr_value) and abs(corr_value) >= corr_threshold
    # For KL, a lower value is better. A "significant" KL result means a clear minimum was found and it's not excessively large.
    # This interpretation is tricky as KL scale is not standardized like correlation.
    # We might infer significance if kl_value is substantially lower than other KL values for that pair.
    # For simplicity here, we just check if it's a valid number.
    kl_significant = pd.notna(kl_value) and kl_value != np.inf

    # Lag agreement
    lags_agree = pd.notna(k_corr) and pd.notna(k_kl) and abs(k_corr - k_kl) <= lag_agreement_tolerance
    
    # Determine lead/lag string for an agreed lag k
    def get_lead_lag_str(k_val, method_name):
        if k_val > 0:
            return f"Candidate leads Target by {int(k_val)} periods (via {method_name})"
        elif k_val < 0:
            return f"Candidate lags Target by {int(abs(k_val))} periods (via {method_name})"
        else: # k_val == 0
            return f"Synchronous relationship (via {method_name})"

    if corr_significant and kl_significant and lags_agree:
        if k_corr == 0: # and k_kl is also 0 or very close by tolerance
            return f"Strong Synchronous Agreement: Corr={corr_value:.2f}, KL={kl_value:.2f}. {get_lead_lag_str(k_corr, 'Both')}."
        lead_lag_description = get_lead_lag_str(k_corr, 'Both') # Use k_corr as the agreed lag
        return f"Strong Agreement: {lead_lag_description}. Corr={corr_value:.2f}, KL={kl_value:.2f}."
    
    # Partial agreements or single method significance
    interpretations = []
    if corr_significant:
        interpretations.append(f"{get_lead_lag_str(k_corr, 'Correlation')} (Corr={corr_value:.2f})")
    if kl_significant:
        # Check if k_kl is similar to k_corr even if one method wasn't 'significant' by threshold but other was
        if corr_significant and not lags_agree and pd.notna(k_corr) and pd.notna(k_kl) and abs(k_corr - k_kl) <= lag_agreement_tolerance + 1: # slightly wider tolerance if one is borderline
             interpretations.append(f"{get_lead_lag_str(k_kl, 'KL Div.')} (KL={kl_value:.2f}, lag similar to Corr.)")
        else:
            interpretations.append(f"{get_lead_lag_str(k_kl, 'KL Div.')} (KL={kl_value:.2f})")

    if interpretations:
        if len(interpretations) > 1 and not lags_agree:
            return f"Mixed Signals: {'; '.join(interpretations)}."
        return f"Potential Relationship: {'; '.join(interpretations)}."
        
    if pd.notna(k_corr) or pd.notna(k_kl): # Results were computed but not deemed significant by thresholds
        return "Weak or Unclear Relationship: Metrics did not meet significance thresholds or provide a clear signal."

    return "No significant relationship found or unable to compute."


def plot_combined_lead_lag_charts(st_obj, full_correlogram_df, full_kl_divergence_df, target_var, candidate_var):
    """Plots Correlation vs. Lag and KL Divergence vs. Lag."""
    
    if full_correlogram_df.empty and full_kl_divergence_df.empty:
        st_obj.caption("No data available for plotting.")
        return

    # Correlation Plot
    if not full_correlogram_df.empty and 'Lag' in full_correlogram_df.columns and 'Correlation' in full_correlogram_df.columns:
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Bar(
            x=full_correlogram_df['Lag'], 
            y=full_correlogram_df['Correlation'],
            name='Correlation',
            marker_color='#1f77b4'
        ))
        if full_correlogram_df['Correlation'].notna().any():
            optimal_corr_idx = full_correlogram_df['Correlation'].abs().idxmax()
            k_corr_val = full_correlogram_df.loc[optimal_corr_idx, 'Lag']
            corr_at_k_corr_val = full_correlogram_df.loc[optimal_corr_idx, 'Correlation']
            fig_corr.add_vline(x=k_corr_val, line_width=2, line_dash="dash", line_color="red", 
                               annotation_text=f"Optimal Lag (Corr): {k_corr_val}", annotation_position="top left")
        
        fig_corr.update_layout(
            title_text=f'Time-Lagged Correlation: {target_var} vs {candidate_var}',
            xaxis_title_text='Lag of Candidate relative to Target (periods)',
            yaxis_title_text='Pearson Correlation',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0')
        )
        fig_corr.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#555555')
        fig_corr.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#555555')
        st_obj.plotly_chart(fig_corr, use_container_width=True)
    else:
        st_obj.caption(f"Correlation data for {candidate_var} is not available or is empty.")

    # KL Divergence Plot
    if not full_kl_divergence_df.empty and 'Lag' in full_kl_divergence_df.columns and 'KL_Divergence' in full_kl_divergence_df.columns:
        # Make a copy before replacing inf for plotting
        plot_kl_df = full_kl_divergence_df.copy()
        plot_kl_df['KL_Divergence_Plot'] = plot_kl_df['KL_Divergence'].replace([np.inf, -np.inf], np.nan)
        # For plotting, if NaNs are too disruptive, consider a placeholder or omitting them.
        # Here, plotly handles NaNs by breaking lines/bars.
        # If all are NaN/Inf after replace, then skip plotting.
        if plot_kl_df['KL_Divergence_Plot'].notna().any():
            fig_kl = go.Figure()
            fig_kl.add_trace(go.Scatter(
                x=plot_kl_df['Lag'], 
                y=plot_kl_df['KL_Divergence_Plot'],
                mode='lines+markers',
                name='K-L Divergence',
                marker_color='#ff7f0e'
            ))

            # Find optimal KL from the original (non-NaN replaced for plotting) data if needed for annotation
            valid_kl_for_min = full_kl_divergence_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['KL_Divergence'])
            if not valid_kl_for_min.empty:
                optimal_kl_idx = valid_kl_for_min['KL_Divergence'].idxmin()
                k_kl_val = full_kl_divergence_df.loc[optimal_kl_idx, 'Lag']
                # kl_at_k_kl_val = full_kl_divergence_df.loc[optimal_kl_idx, 'KL_Divergence']
                fig_kl.add_vline(x=k_kl_val, line_width=2, line_dash="dash", line_color="green",
                                 annotation_text=f"Optimal Lag (KL): {k_kl_val}", annotation_position="top right")

            fig_kl.update_layout(
                title_text=f'Time-Lagged K-L Divergence: D(P_{target_var} || P_{candidate_var}@lag)',
                xaxis_title_text='Lag of Candidate relative to Target (periods)',
                yaxis_title_text='K-L Divergence (nats)',
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0')
            )
            fig_kl.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#555555')
            fig_kl.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#555555', type="log") # Log scale often helpful for KL
            st_obj.plotly_chart(fig_kl, use_container_width=True)
        else:
            st_obj.caption(f"K-L Divergence data for {candidate_var} (after handling Inf/NaN) is not plottable.")
    else:
        st_obj.caption(f"K-L Divergence data for {candidate_var} is not available or is empty.")


def display_combined_lead_lag_analysis_tab(st_obj, session_state):
    st_obj.markdown("### 综合领先滞后分析 (时差相关性与K-L信息量)")
    st_obj.markdown("此工具结合使用时差相关性和时差K-L信息量来分析变量间的领先/滞后关系。" 
                    "正滞后值表示候选变量领先目标变量，负滞后值表示候选变量滞后目标变量。")

    selected_df_name = session_state.get('tlc_own_selected_df_name') # Reuse from original TLC tab for data source
    df = session_state.get('tlc_own_selected_df')

    if df is None or df.empty:
        st_obj.info("请先在主界面选择或上传一个有效的数据集，并在此标签页上方的数据选择器中确认。")
        return

    tab_prefix = f"combined_ll_{selected_df_name if selected_df_name else 'default'}"

    # Initialize session state keys specific to this tab
    if f'{tab_prefix}_target_var' not in session_state: session_state[f'{tab_prefix}_target_var'] = None
    if f'{tab_prefix}_candidate_vars' not in session_state: session_state[f'{tab_prefix}_candidate_vars'] = []
    if f'{tab_prefix}_max_lags' not in session_state: session_state[f'{tab_prefix}_max_lags'] = 12
    if f'{tab_prefix}_kl_bins' not in session_state: session_state[f'{tab_prefix}_kl_bins'] = 10
    if f'{tab_prefix}_results' not in session_state: session_state[f'{tab_prefix}_results'] = None
    if f'{tab_prefix}_errors' not in session_state: session_state[f'{tab_prefix}_errors'] = []
    if f'{tab_prefix}_warnings' not in session_state: session_state[f'{tab_prefix}_warnings'] = []
    if f'{tab_prefix}_selected_plot_candidate' not in session_state: session_state[f'{tab_prefix}_selected_plot_candidate'] = None
    
    # Reset results if dataframe changes
    dataset_id_key = f'{tab_prefix}_current_dataset_id'
    current_dataset_id = id(df)
    if session_state.get(dataset_id_key) != current_dataset_id:
        session_state[f'{tab_prefix}_results'] = None
        session_state[f'{tab_prefix}_errors'] = []
        session_state[f'{tab_prefix}_warnings'] = []
        session_state[f'{tab_prefix}_selected_plot_candidate'] = None
        session_state[dataset_id_key] = current_dataset_id
        # Also reset variable selections if they are no longer valid for the new df
        if session_state[f'{tab_prefix}_target_var'] not in df.columns: session_state[f'{tab_prefix}_target_var'] = None
        session_state[f'{tab_prefix}_candidate_vars'] = [cv for cv in session_state[f'{tab_prefix}_candidate_vars'] if cv in df.columns]

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        st_obj.warning("选定的数据集中没有可用的数值列进行分析。")
        return

    st_obj.markdown("##### 1. 参数设置")
    col_param1, col_param2 = st_obj.columns(2)
    with col_param1:
        # Target Variable
        current_target = session_state.get(f'{tab_prefix}_target_var', numeric_cols[0] if numeric_cols else None)
        if current_target not in numeric_cols and numeric_cols : current_target = numeric_cols[0] # Default if invalid
        target_var = st_obj.selectbox("选择目标变量 (A):", numeric_cols, 
                                      index=numeric_cols.index(current_target) if current_target in numeric_cols else 0,
                                      key=f"{tab_prefix}_sb_target")
        session_state[f'{tab_prefix}_target_var'] = target_var

        # Max Lags
        max_lags_val = st_obj.number_input("最大领先/滞后周期数:", min_value=1, max_value=max(24, len(df)//2 if df is not None else 24), 
                                           value=session_state.get(f'{tab_prefix}_max_lags'), 
                                           key=f"{tab_prefix}_ni_maxlags")
        session_state[f'{tab_prefix}_max_lags'] = max_lags_val
    
    with col_param2:
        # Candidate Variables
        candidate_options = [col for col in numeric_cols if col != target_var]
        # Ensure previous selections are valid for current options
        current_candidates_selection = [c for c in session_state.get(f'{tab_prefix}_candidate_vars', []) if c in candidate_options]
        candidate_vars = st_obj.multiselect("选择候选变量 (B, C, ...):", candidate_options, 
                                            default=current_candidates_selection,
                                            key=f"{tab_prefix}_ms_candidates")
        session_state[f'{tab_prefix}_candidate_vars'] = candidate_vars
        
        # KL Bins
        kl_bins_val = st_obj.number_input("K-L 信息量直方图区间(bins)数:", min_value=2, max_value=100, 
                                          value=session_state.get(f'{tab_prefix}_kl_bins'), 
                                          key=f"{tab_prefix}_ni_klbins")
        session_state[f'{tab_prefix}_kl_bins'] = kl_bins_val

    if st_obj.button("运行综合分析", key=f"{tab_prefix}_bt_run", disabled=(not target_var or not candidate_vars)):
        with st_obj.spinner("正在执行综合领先滞后分析..."):
            results, errors, warnings = combined_lead_lag_backend.perform_combined_lead_lag_analysis(
                df_input=df,
                target_variable_name=target_var,
                candidate_variable_names_list=candidate_vars,
                max_lags_config=max_lags_val,
                kl_bins_config=kl_bins_val
            )
            session_state[f'{tab_prefix}_results'] = results
            session_state[f'{tab_prefix}_errors'] = errors
            session_state[f'{tab_prefix}_warnings'] = warnings
            session_state[f'{tab_prefix}_selected_plot_candidate'] = None # Reset plot selection
            if errors: st_obj.error("\n".join(errors))
            if warnings: st_obj.warning("\n".join(warnings))
            if not results and not errors and not warnings:
                st_obj.info("分析完成，但没有生成结果。请检查输入数据和参数。")
            elif results : 
                st_obj.success("综合分析完成！")
    
    # --- Display Results ---
    analysis_results = session_state.get(f'{tab_prefix}_results')
    if analysis_results is not None:
        st_obj.markdown("##### 2. 分析结果汇总")
        
        summary_data = []
        for res in analysis_results:
            interpretation = interpret_lead_lag(
                res.get('k_corr'), res.get('corr_at_k_corr'), 
                res.get('k_kl'), res.get('kl_at_k_kl')
            )
            summary_data.append({
                "候选变量": res.get('candidate_variable'),
                "最优滞后 (相关性)": res.get('k_corr'),
                "相关系数 @最优滞后": res.get('corr_at_k_corr'),
                "最优滞后 (K-L)": res.get('k_kl'),
                "K-L值 @最优滞后": res.get('kl_at_k_kl'),
                "综合解读": interpretation,
                "备注": res.get('notes', '')
            })
        
        summary_df = pd.DataFrame(summary_data)
        st_obj.dataframe(summary_df, 
                         column_config={
                             "相关系数 @最优滞后": st.column_config.NumberColumn(format="%.3f"),
                             "K-L值 @最优滞后": st.column_config.NumberColumn(format="%.3f")
                         },
                         use_container_width=True, hide_index=True)
        
        # 添加综合解读说明附注
        st_obj.markdown("""
        **综合解读说明：**
        - **强一致性**：相关性和K-L散度分析得出相同或相近的最优滞后期
        - **混合信号**：两种方法得出的最优滞后期不一致
        - **潜在关系**：只有一种方法显示明显的关系
        """)
        
        st_obj.markdown("##### 3. 详细图表")
        plottable_candidates = [res['candidate_variable'] for res in analysis_results if res.get('full_correlogram_df') is not None or res.get('full_kl_divergence_df') is not None]
        
        if not plottable_candidates:
            st_obj.caption("没有可供详细绘图的候选变量结果。")
        else:
            current_plot_selection = session_state.get(f'{tab_prefix}_selected_plot_candidate', plottable_candidates[0])
            if current_plot_selection not in plottable_candidates: current_plot_selection = plottable_candidates[0]

            selected_candidate_for_plot = st_obj.selectbox("选择一个候选变量查看详细图表:", 
                                                             options=plottable_candidates,
                                                             index=plottable_candidates.index(current_plot_selection),
                                                             key=f"{tab_prefix}_sb_plot_candidate")
            session_state[f'{tab_prefix}_selected_plot_candidate'] = selected_candidate_for_plot

            chosen_result_details = next((r for r in analysis_results if r['candidate_variable'] == selected_candidate_for_plot), None)
            if chosen_result_details:
                plot_combined_lead_lag_charts(st_obj, 
                                              chosen_result_details.get('full_correlogram_df', pd.DataFrame()), 
                                              chosen_result_details.get('full_kl_divergence_df', pd.DataFrame()),
                                              chosen_result_details.get('target_variable'),
                                              chosen_result_details.get('candidate_variable'))
            else:
                st_obj.warning(f"无法加载 '{selected_candidate_for_plot}' 的详细数据。")
    else:
        st_obj.info('请设置参数并点击 "运行综合分析" 以查看结果。')

# For standalone testing if needed:
if __name__ == '__main__':
    # This would require mocking st.session_state and potentially the backend
    # or providing a dummy DataFrame directly.
    st.title("Combined Lead/Lag Analysis Tester")
    # Setup dummy session state
    if 'tlc_own_selected_df_name' not in st.session_state:
        st.session_state.tlc_own_selected_df_name = "DummyDF_Combined"
    if 'tlc_own_selected_df' not in st.session_state:
        idx_test = pd.date_range('2023-01-01', periods=50, freq='D')
        st.session_state.tlc_own_selected_df = pd.DataFrame({
            'TargetA': np.random.randn(50).cumsum(),
            'CandidateB': np.roll(np.random.randn(50).cumsum(), 3) + np.random.randn(50)*0.2,
            'CandidateC': np.random.randn(50).cumsum(),
            'NonNumericD': ['X'] * 50
        }, index=idx_test)
    
    display_combined_lead_lag_analysis_tab(st, st.session_state) 