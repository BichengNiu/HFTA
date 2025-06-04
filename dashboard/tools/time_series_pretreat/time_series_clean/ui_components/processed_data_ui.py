import streamlit as st
import pandas as pd
import numpy as np

# Assuming utils are in a sibling directory to ui_components
from ..utils.data_processing import apply_rename_rules, batch_rename_columns_by_text_replacement
# from .time_column_ui import display_time_column_settings # Removed due to circular import and being unused
from ..utils.time_analysis import identify_time_column

# --- <<< 新增：重置UI状态的回调函数 >>> ---
def reset_processed_ui_state():
    # Restore DataFrame from snapshot
    if 'df_processed_snapshot_at_entry' in st.session_state and \
       st.session_state.df_processed_snapshot_at_entry is not None:
        st.session_state.ts_tool_data_processed = st.session_state.df_processed_snapshot_at_entry.copy()
        print("[DEBUG processed_data_ui] ts_tool_data_processed has been reset from snapshot.")
    
    if 'df_processed_FULL_snapshot_at_entry' in st.session_state and \
       st.session_state.df_processed_FULL_snapshot_at_entry is not None and \
       'ts_tool_data_processed_FULL' in st.session_state: # Ensure _FULL exists before trying to reset it
        st.session_state.ts_tool_data_processed_FULL = st.session_state.df_processed_FULL_snapshot_at_entry.copy()
        print("[DEBUG processed_data_ui] ts_tool_data_processed_FULL has been reset from snapshot.")

    # Reset individual rename rule states
    st.session_state.ts_tool_rename_rules = []
    # Explicitly reset text inputs for rename rules if they exist in session_state
    if 'rename_select_orig' in st.session_state:
        # Finding a sensible default for selectbox is tricky without knowing options; None might clear it
        # Or we might need to re-evaluate if direct reset is best or rely on rerun with empty rules.
        # For now, clearing rules should be the primary reset.
        pass 
    if 'rename_input_new' in st.session_state:
        st.session_state.rename_input_new = ""


    # Reset batch text replacement states
    # Ensure keys for text inputs are correct
    if 'br_find_text_in_left_col' in st.session_state: # Key for "要查找的文本"
      st.session_state.br_find_text_in_left_col = "" 
    if 'br_replace_text_in_left_col' in st.session_state: # Key for "替换为"
      st.session_state.br_replace_text_in_left_col = ""

    st.session_state.br_preview_df = None
    st.session_state.br_modified_map = None
    st.session_state.br_status_message = None
    st.session_state.br_status_type = None
    st.session_state.confirm_batch_rename_apply = False
    
    print("[DEBUG processed_data_ui] UI state for processed_data_ui has been reset.")
    # st.success("数据和相关UI设置已重置到此部分初始状态。") # Commented out green success message
    # It's important to trigger a rerun for UI to update with reset values
    # st.rerun() # Commented out to prevent 'no-op' warning
# --- <<< 结束新增 >>> ---

# --- NEW FUNCTION: For Custom NaN Definition ---
def display_custom_nan_definition_ui(st, session_state, df_key='ts_tool_data_processed', cols_to_process_key='ts_tool_data_processed_cols_for_custom_nan'):
    """
    UI for defining and applying custom NaN values.
    Modifies session_state[df_key] in place.
    """
    st.markdown("<h5>自定义识别额外的缺失值</h5>", unsafe_allow_html=True)
    st.caption("指定额外文本/数字作为缺失值(NaN)。此操作会直接修改当前预览数据，建议在标准填充前或独立使用。")

    if df_key not in session_state or session_state[df_key] is None or session_state[df_key].empty:
        st.info("尚无数据可供处理。")
        # Return default/empty values if no data, so caller doesn't break
        return { 
            "nan_values_to_replace": [],
            "selected_cols_for_nan": [],
            "nan_values_str_input": ""
        }

    df_to_process = session_state[df_key]
    
    # Ensure a list of columns to process is available, default to all numeric/object
    if cols_to_process_key not in session_state or session_state[cols_to_process_key] is None:
        session_state[cols_to_process_key] = [
            col for col in df_to_process.columns 
            if pd.api.types.is_numeric_dtype(df_to_process[col]) or pd.api.types.is_object_dtype(df_to_process[col])
        ]

    # Initialize the session_state key for the text input if it doesn't exist
    # This ensures that the text_input widget can read its value from session_state directly
    # without needing the `value` param if the key is already set.
    if 'custom_nan_input_text_global' not in session_state:
        session_state['custom_nan_input_text_global'] = session_state.get('ts_tool_custom_nan_input_str', "0, N/A, --, NULL")

    nan_values_str = st.text_input(
        "要识别为缺失的文本/数字 (逗号分隔):", 
        # value=default_nan_str, # REMOVED to avoid warning, value will be taken from session_state[key]
        key="custom_nan_input_text_global" 
    )
    # Synchronize ts_tool_custom_nan_input_str with the widget's current value from session_state
    # This is important if other parts of the code read from ts_tool_custom_nan_input_str directly
    session_state['ts_tool_custom_nan_input_str'] = session_state.get("custom_nan_input_text_global", "0, N/A, --, NULL")

    # Prepare list of values to replace
    # Handles numbers and strings, strips whitespace from string values
    values_to_replace = []
    if nan_values_str:
        raw_values = [val.strip() for val in nan_values_str.split(',')] 
        for val_str in raw_values:
            if not val_str: continue 
            values_to_replace.append(val_str)
            try:
                num_val = float(val_str)
                values_to_replace.append(num_val) 
                if num_val == int(num_val):
                    values_to_replace.append(int(num_val))
            except ValueError:
                pass
        values_to_replace = list(dict.fromkeys(values_to_replace))

    all_available_cols = df_to_process.columns.tolist()
    
    # Ensure current_cols_for_multiselect is valid and initialized in session_state for the widget
    # The widget with key 'custom_nan_selected_cols_global' will read its value from session_state.
    # cols_to_process_key ('ts_tool_data_processed_cols_for_custom_nan') stores the logical selection for processing.
    # We need to initialize custom_nan_selected_cols_global based on cols_to_process_key or defaults.

    if 'custom_nan_selected_cols_global' not in session_state: # Initialize if widget state doesn't exist
        potential_default_cols = session_state.get(cols_to_process_key, [])
        valid_default_cols = [col for col in potential_default_cols if col in all_available_cols]
        
        # --- MODIFICATION: Exclude identified time column from default selection ---
        identified_time_col_name = session_state.get('ts_tool_time_col_info', {}).get('name')
        cols_for_widget_default = []

        if valid_default_cols:
            cols_for_widget_default = [col for col in valid_default_cols if col != identified_time_col_name]
        elif all_available_cols:
            cols_for_widget_default = [col for col in all_available_cols if col != identified_time_col_name]
        # --- END MODIFICATION ---
            
        session_state['custom_nan_selected_cols_global'] = cols_for_widget_default

    else:
        # If custom_nan_selected_cols_global already exists, ensure its contents are valid and exclude time col
        identified_time_col_name = session_state.get('ts_tool_time_col_info', {}).get('name') # Get time col again
        current_selection_in_state = session_state['custom_nan_selected_cols_global']
        
        # Filter out non-existent columns and the identified time column
        valid_and_non_time_cols = [col for col in current_selection_in_state if col in all_available_cols and col != identified_time_col_name]
        session_state['custom_nan_selected_cols_global'] = valid_and_non_time_cols
        
        # If this results in an empty list and there are available non-time columns, consider re-defaulting
        if not session_state['custom_nan_selected_cols_global'] and all_available_cols:
            non_time_available_cols = [col for col in all_available_cols if col != identified_time_col_name]
            if non_time_available_cols: # Check if there are any non-time columns left to default to
                session_state['custom_nan_selected_cols_global'] = non_time_available_cols

    selected_cols_for_nan_widget = st.multiselect(
        "选择要应用此规则的列 (留空则为所有数值/对象类型列):",
        options=all_available_cols,
        # default=current_cols_for_multiselect, # REMOVED to avoid warning
        key="custom_nan_selected_cols_global" 
    )
    # Synchronize the logical selection state (cols_to_process_key) with the widget's current state
    session_state[cols_to_process_key] = selected_cols_for_nan_widget

    return {
        "nan_values_to_replace": values_to_replace,
        "selected_cols_for_nan": selected_cols_for_nan_widget, 
        "nan_values_str_input": nan_values_str
    }

# --- <<< NEW FUNCTION: Logic for Applying Custom NaN Values >>> ---
def apply_custom_nan_values_processing(st, session_state, df_key, cols_to_process_key, 
                                     nan_values_to_replace_list, selected_cols_list, 
                                     nan_values_str_input_for_warning,
                                     rerun_after_processing: bool = True):
    """
    Applies custom NaN value processing based on provided inputs.
    Modifies session_state[df_key] and session_state[df_key + '_FULL'] in place.
    This function contains the logic previously under the button in display_custom_nan_definition_ui.
    """
    if df_key not in session_state or session_state[df_key] is None or session_state[df_key].empty:
        st.info("尚无数据可供处理 (from apply_custom_nan_values_processing)。")
        return

    df_to_process = session_state[df_key] # Get the DataFrame to modify

    if not nan_values_str_input_for_warning or not nan_values_to_replace_list: # Check based on inputs
        st.warning("请输入要识别为缺失的值。")
    else:
        # Determine columns to apply: use selected_cols_list if provided, else default
        cols_to_apply = selected_cols_list if selected_cols_list else [
            col for col in df_to_process.columns 
            if pd.api.types.is_numeric_dtype(df_to_process[col]) or pd.api.types.is_object_dtype(df_to_process[col])
        ]
        
        if not cols_to_apply:
            st.warning("没有选定列或没有合适的列可应用此规则。")
        else:
            try:
                modified_cols_count = 0
                for col_name in cols_to_apply:
                    if col_name in df_to_process.columns:
                        original_dtype = df_to_process[col_name].dtype
                        df_to_process[col_name] = df_to_process[col_name].replace(nan_values_to_replace_list, np.nan)
                        modified_cols_count +=1
                        if df_to_process[col_name].isnull().all():
                            if pd.api.types.is_numeric_dtype(original_dtype):
                                 df_to_process[col_name] = df_to_process[col_name].astype(float)
                        else:
                            try:
                                if pd.api.types.is_numeric_dtype(original_dtype) or pd.api.types.is_datetime64_any_dtype(original_dtype):
                                   df_to_process[col_name] = pd.to_numeric(df_to_process[col_name], errors='ignore')
                            except Exception: # nosemgrep
                                pass 
                
                session_state[df_key] = df_to_process # Update the main DataFrame in session_state
                
                # Also update _FULL if it exists
                if session_state.get(f'{df_key}_FULL') is not None: # Standardized to use f-string for _FULL key
                    df_full = session_state[f'{df_key}_FULL']
                    for col_name_full in cols_to_apply:
                        if col_name_full in df_full.columns:
                            # Store original dtype
                            original_dtype_full = df_full[col_name_full].dtype
                            df_full[col_name_full] = df_full[col_name_full].replace(nan_values_to_replace_list, np.nan)
                             # Similar dtype handling for _FULL
                            if df_full[col_name_full].isnull().all():
                                if pd.api.types.is_numeric_dtype(original_dtype_full):
                                     df_full[col_name_full] = df_full[col_name_full].astype(float)
                            else:
                                try:
                                    if pd.api.types.is_numeric_dtype(original_dtype_full) or pd.api.types.is_datetime64_any_dtype(original_dtype_full):
                                       df_full[col_name_full] = pd.to_numeric(df_full[col_name_full], errors='ignore')
                                except Exception: # nosemgrep
                                    pass
                    session_state[f'{df_key}_FULL'] = df_full

                st.success(f"已在 {modified_cols_count} 列中将指定值标记为缺失(NaN)。")
                print(f"[DEBUG CustomNaN Apply] Applied to {df_key}. Values: {nan_values_to_replace_list}, Cols: {cols_to_apply}")
                
                # Update the default column selection for next time display_custom_nan_definition_ui is called
                # if user left it blank (applied to default)
                if not selected_cols_list: # If applied to default columns
                    session_state[cols_to_process_key] = [
                        col for col in session_state[df_key].columns 
                        if pd.api.types.is_numeric_dtype(session_state[df_key][col]) or pd.api.types.is_object_dtype(session_state[df_key][col])
                    ]
                else: # If specific columns were selected, make them the default for next time
                    session_state[cols_to_process_key] = selected_cols_list

                # MODIFIED: Conditional rerun
                if rerun_after_processing:
                    st.rerun()
            except Exception as e:
                st.error(f"处理自定义缺失值时出错: {e}")
                print(f"[ERROR CustomNaN Apply] Error: {e}")
# --- <<< END NEW FUNCTION >>> ---

# --- NEW FUNCTION: For Column Selector (intended for staging prep) ---
def display_column_selector_for_staging_ui(st, session_state, df_key='ts_tool_data_processed', selected_cols_session_key='ts_tool_staging_selected_columns'):
    """
    UI for selecting columns, result stored in session_state[selected_cols_session_key].
    """
    st.markdown("##### **选择要保留的列 (用于暂存)**")
    st.caption("选择您希望在下一步（例如数据暂存或导出）中保留的列。")

    if df_key not in session_state or session_state[df_key] is None or session_state[df_key].empty:
        st.info("尚无数据可供选择列。")
        if selected_cols_session_key in session_state: # Clear previous selection if data disappears
            del session_state[selected_cols_session_key]
        return

    df_current = session_state[df_key]
    all_cols = df_current.columns.tolist()

    # Initialize or update selected columns: default to all, or keep previous selection if valid
    if selected_cols_session_key not in session_state or \
       not all(col in all_cols for col in session_state[selected_cols_session_key]):
        session_state[selected_cols_session_key] = all_cols

    selected_cols = st.multiselect(
        "选择列:",
        options=all_cols,
        default=session_state[selected_cols_session_key],
        key=f"{selected_cols_session_key}_multiselect" # Ensure unique key
    )
    session_state[selected_cols_session_key] = selected_cols
    
    if not selected_cols:
        st.warning("请至少选择一列。如果所有列都被取消选择，后续操作可能失败或产生空数据。")
    else:
        st.caption(f"已选择 {len(selected_cols)} 列。")


def display_processed_data_and_renaming(st, session_state, add_rename_rule_callback):
    """Displays processed data, column selection, and batch rename functionality."""

    # Custom CSS to adjust expander header font size
    st.markdown("""
    <style>
    div[data-testid="stExpander"] summary > div > p {
        font-size: 1.15em !important; /* Adjust size as needed */
        font-weight: bold !important; /* Optionally make it bold */
    }
    </style>
    """, unsafe_allow_html=True)

    if not session_state.ts_tool_processing_applied or session_state.ts_tool_data_processed is None:
        return # Don't display this section if data hasn't been processed

    # --- <<< 新增：数据快照逻辑 >>> ---
    # Snapshot for reset functionality
    # Check if the snapshot needs to be created or refreshed
    # The flag 'force_processed_ui_snapshot_refresh_flag' is expected to be set by the calling module (e.g., time_series_clean_tab.py)
    # when new data is loaded into ts_tool_data_processed, signaling that this UI section is being entered "freshly".
    if 'df_processed_snapshot_at_entry' not in session_state or \
       session_state.get('force_processed_ui_snapshot_refresh_flag', False):
        if session_state.get('ts_tool_data_processed') is not None: # Check if data actually exists
            session_state.df_processed_snapshot_at_entry = session_state.ts_tool_data_processed.copy()
            print("[DEBUG processed_data_ui] Created/Updated df_processed_snapshot_at_entry.")
            if session_state.get('ts_tool_data_processed_FULL') is not None:
                session_state.df_processed_FULL_snapshot_at_entry = session_state.ts_tool_data_processed_FULL.copy()
                print("[DEBUG processed_data_ui] Created/Updated df_processed_FULL_snapshot_at_entry.")
        else:
             # If data is None, clear snapshots too to avoid using stale snapshot if data disappears then reappears
            if 'df_processed_snapshot_at_entry' in session_state:
                del session_state.df_processed_snapshot_at_entry
            if 'df_processed_FULL_snapshot_at_entry' in session_state:
                del session_state.df_processed_FULL_snapshot_at_entry
            print("[DEBUG processed_data_ui] ts_tool_data_processed is None, cleared snapshots if they existed.")

        # Reset the flag after attempting to update the snapshot
        if 'force_processed_ui_snapshot_refresh_flag' in session_state: # Only reset if it exists
            session_state.force_processed_ui_snapshot_refresh_flag = False
            print("[DEBUG processed_data_ui] Reset force_processed_ui_snapshot_refresh_flag.")

    # --- <<< 新增：标题与重置按钮 >>> ---
    col_title_proc, col_reset_button_proc = st.columns([0.8, 0.2]) 
    with col_title_proc:
        st.subheader("变量处理与预览") 
    with col_reset_button_proc:
        # --- MODIFIED: Removed CSS, simplified button ---
        if st.button("重置",  # Changed button text
                     key="reset_variable_processing_button", 
                     on_click=reset_processed_ui_state,  
                     help="将数据和此部分的相关设置（如重命名规则）恢复到进入此部分时的状态。"):
            pass 
        # --- END OF MODIFICATION ---

    # --- <<< 新增：报告被移除的重复列 >>> ---
    if session_state.get('ts_tool_removed_duplicate_cols'):
        removed_cols_list = session_state.ts_tool_removed_duplicate_cols
        if removed_cols_list:
            st.warning(f"以下列因重复而被自动移除，系统保留了每组重复列中的第一个版本：")
            # Display as a list or formatted string
            removed_cols_str = ", ".join([f"'{col}'" for col in removed_cols_list])
            st.markdown(f"- {removed_cols_str}")
            # Clear it from session_state after displaying once to avoid repeated messages on reruns without new load
            # session_state.ts_tool_removed_duplicate_cols = [] # Or handle display more carefully
    # --- <<< 结束新增 >>> ---
   
    st.info("在此处预览经过初步处理的数据。您可以批量修改列名。自定义缺失值、列选择和时间筛选功能已移至后续步骤。") # MODIFIED: Updated info text

    # --- Section for Renaming and Batch Text Replacement ---
    with st.container(): # Replaces rename_expander
        st.markdown("##### **批量修改变量名**") # New H5 title for this section

        if session_state.get('ts_tool_data_processed') is not None and not session_state.ts_tool_data_processed.empty:
            current_cols_for_rename = session_state.ts_tool_data_processed.columns.tolist()
            
            col_add_rule, col_view_rules = st.columns(2) # Was rename_expander.columns(2)

            with col_add_rule:
                originals_in_rules = {rule['original'] for rule in session_state.ts_tool_rename_rules}
                available_cols_for_new_rule = [col for col in current_cols_for_rename if col not in originals_in_rules]

                if not available_cols_for_new_rule:
                    col_add_rule.info("所有列都已设置重命名规则，或没有可用的列进行重命名。")
                else:
                    col_add_rule.selectbox(
                        "选择原始列名:", 
                        options=available_cols_for_new_rule, 
                        key='rename_select_orig', 
                        index=0 if available_cols_for_new_rule else None
                    )
                    col_add_rule.text_input("输入新列名:", key='rename_input_new')

                    btn_col1, btn_col2, btn_col3 = col_add_rule.columns(3)

                    with btn_col1:
                        btn_col1.button("添加规则", on_click=add_rename_rule_callback, key="add_rule_btn_grouped")

                    with btn_col2:
                        if btn_col2.button("应用规则", key="apply_rules_btn_grouped"):
                            if not session_state.ts_tool_rename_rules:
                                st.warning("没有设置任何重命名规则。") # Was rename_expander.warning
                            else:
                                try:
                                    # Original df_processed for this operation
                                    df_to_rename_main = session_state.ts_tool_data_processed 
                                    
                                    df_renamed_main = apply_rename_rules(df_to_rename_main, session_state.ts_tool_rename_rules)
                                    session_state.ts_tool_data_processed = df_renamed_main
                                    
                                    # Now, also apply to _FULL if it exists
                                    if session_state.get('ts_tool_data_processed_FULL') is not None:
                                        df_to_rename_full = session_state.ts_tool_data_processed_FULL
                                        try:
                                            # We need to ensure that rules applied to _FULL are only those whose original columns exist in _FULL
                                            rules_for_full = []
                                            full_cols_str_set = set(str(col) for col in df_to_rename_full.columns) # Ensure string comparison
                                            for rule in session_state.ts_tool_rename_rules:
                                                if str(rule['original']) in full_cols_str_set: # Compare as strings
                                                    rules_for_full.append(rule)
                                            
                                            if rules_for_full: # Only apply if there are relevant rules
                                                df_renamed_full = apply_rename_rules(df_to_rename_full, rules_for_full)
                                                session_state.ts_tool_data_processed_FULL = df_renamed_full
                                                print(f"[DEBUG APPLY RULES] ts_tool_data_processed_FULL also renamed. New cols: {session_state.ts_tool_data_processed_FULL.columns.tolist()}")
                                            else:
                                                print("[DEBUG APPLY RULES] No applicable rules for ts_tool_data_processed_FULL or it's already consistent.")
                                        except ValueError as ve_full: # Catch specific error for more targeted message
                                            st.warning(f"注意：主数据已重命名，但在更新备份数据(FULL)时出现问题: {ve_full}。数据预览可能正确，但后续操作可能受影响。") # Was rename_expander.warning
                                        except Exception as e_full: # Catch any other exception during FULL update
                                             st.warning(f"注意：主数据已重命名，但在更新备份数据(FULL)时发生未知错误: {e_full}。") # Was rename_expander.warning

                                    # Update ts_tool_cols_to_keep and ts_tool_manual_time_col if they were affected by renaming
                                    new_col_map = {rule['original']: rule['new'] for rule in session_state.ts_tool_rename_rules}
                                    
                                    # Update ts_tool_cols_to_keep (used by time_column_ui for initial selection)
                                    if 'ts_tool_cols_to_keep' in session_state and session_state.ts_tool_cols_to_keep:
                                        session_state.ts_tool_cols_to_keep = [new_col_map.get(col, col) for col in session_state.ts_tool_cols_to_keep]
                                    
                                    # Update ts_tool_manual_time_col (manual selection for time column)
                                    if 'ts_tool_manual_time_col' in session_state and session_state.ts_tool_manual_time_col in new_col_map:
                                        session_state.ts_tool_manual_time_col = new_col_map[session_state.ts_tool_manual_time_col]
                                    
                                    # Update ts_tool_time_col_info if its 'name' was renamed
                                    if 'ts_tool_time_col_info' in session_state and session_state.ts_tool_time_col_info.get('name') in new_col_map:
                                        session_state.ts_tool_time_col_info['name'] = new_col_map[session_state.ts_tool_time_col_info['name']]

                                    session_state.ts_tool_rename_rules = [] # Clear rules after applying
                                    st.success("已成功应用重命名规则，规则列表已清空。") # Was rename_expander.success
                                    st.session_state.force_time_ui_snapshot_refresh_flag = True # <<< SET FLAG
                                    print("[DEBUG processed_data_ui] Set force_time_ui_snapshot_refresh_flag = True after applying individual rename rules.") # DEBUG ADDED
                                    st.rerun() # Rerun to reflect changes immediately
                                except ValueError as ve:
                                    st.error(f"应用重命名规则时出错: {ve}") # Was rename_expander.error
                                except Exception as e_apply_rules: # Catch any other exception during main apply
                                    st.error(f"应用重命名规则时发生未知错误: {e_apply_rules}") # Was rename_expander.error
                    with btn_col3:
                        if btn_col3.button("清除规则", key="clear_rules_btn_grouped"):
                            session_state.ts_tool_rename_rules = []
                            st.info("所有重命名规则已清除。") # Was rename_expander.info
                            st.rerun()
            
            with col_view_rules:
                if not session_state.ts_tool_rename_rules:
                    col_view_rules.caption("尚无重命名规则。")
                else:
                    rules_df_data = []
                    for i, rule in enumerate(session_state.ts_tool_rename_rules):
                        rules_df_data.append({
                            '序号': i + 1,
                            '原始列名': rule['original'],
                            '新列名': rule['new']
                        })
                    col_view_rules.dataframe(pd.DataFrame(rules_df_data), hide_index=True, use_container_width=True)
                
            st.markdown("---")

            st.markdown("##### 批量替换列名中的特定文本") # Was rename_expander.markdown, ALREADY H5

            if 'br_preview_df' not in session_state: session_state.br_preview_df = None
            if 'br_modified_map' not in session_state: session_state.br_modified_map = None
            if 'br_status_message' not in session_state: session_state.br_status_message = None
            if 'br_status_type' not in session_state: session_state.br_status_type = None

            batch_left_col, batch_right_col = st.columns([2, 3]) # Was rename_expander.columns

            with batch_left_col:
                find_text_br = batch_left_col.text_input("要查找的文本:", key="br_find_text_in_left_col", placeholder="例如：_当月同比")
                replace_text_br = batch_left_col.text_input("替换为 (留空则删除查找到的文本):", key="br_replace_text_in_left_col", placeholder="例如：_YoY")

                # --- MODIFIED: Button layout for Preview, Reset, Confirm ---
                # btn_preview_col, btn_reset_col, btn_confirm_col = batch_left_col.columns([2,1,2]) # Adjust ratios as needed
                # --- MODIFICATION: Remove the specific reset button, adjust columns for Preview and Confirm --- 
                btn_preview_col, btn_confirm_col = batch_left_col.columns([1,1]) # Adjusted to 2 columns

                with btn_preview_col:
                    if st.button("预览批量文本替换", key="batch_rename_preview_button_grouped"):
                        session_state.br_preview_df = None
                        session_state.br_modified_map = None
                        session_state.br_status_message = None
                        session_state.br_status_type = None
                        st.session_state.confirm_batch_rename_apply = False

                        if not find_text_br: # Use the widget's current value
                            st.warning("请输入要查找的文本。") # Changed from preview_confirm_buttons_col1.warning
                        elif session_state.ts_tool_data_processed is not None and not session_state.ts_tool_data_processed.empty:
                            time_col_name = session_state.get('ts_tool_time_col_info', {}).get('name')
                            _, status = batch_rename_columns_by_text_replacement(
                                session_state.ts_tool_data_processed, 
                                find_text_br, 
                                replace_text_br, 
                                time_col_name
                            )
                            session_state.br_status_message = status.get('message')
                            session_state.br_modified_map = status.get('modified_map')

                            if not status['success']:
                                session_state.br_status_type = 'error'
                                if status.get('conflicts'):
                                    st.error(f"{status.get('message')}") # Changed from preview_confirm_buttons_col1.error
                                    st.json(status.get('conflicts')) # Changed from preview_confirm_buttons_col1.json
                            elif status.get('modified_map') and any(k!=v for k,v in status['modified_map'].items()):
                                session_state.br_status_type = 'info'
                                preview_df_data = []
                                for old, new in status['modified_map'].items():
                                    if old != new:
                                        preview_df_data.append({"原始名称": old, "替换后名称": new})
                                if preview_df_data:
                                    session_state.br_preview_df = pd.DataFrame(preview_df_data)
                        else:
                            st.warning("没有已处理的数据可供操作。") # Changed from preview_confirm_buttons_col1.warning
                
                with btn_confirm_col:
                    if session_state.get('br_preview_df') is not None and not session_state.get('br_preview_df').empty:
                        if st.button("确认并应用替换", key="confirm_batch_rename_apply_button_grouped"):
                            st.session_state.confirm_batch_rename_apply = True 
                        st.caption("注意：此操作将直接修改列名。")
                    else:
                        # For now, it will just be empty if no preview_df.
                        pass
                # --- END MODIFIED Button layout ---
            
            with batch_right_col:
                if session_state.br_preview_df is not None and not session_state.br_preview_df.empty:
                    batch_right_col.markdown("**替换预览:**")
                    batch_right_col.dataframe(session_state.br_preview_df, use_container_width=True)
                    
            # Display status message if any (from preview or apply)
            if session_state.br_status_message:
                if session_state.br_status_type == 'error':
                    st.error(session_state.br_status_message) # Was rename_expander.error
                elif session_state.br_status_type == 'warning':
                    st.warning(session_state.br_status_message) # Was rename_expander.warning
                else: # 'info' or other
                    st.info(session_state.br_status_message) # Was rename_expander.info

            # Apply confirmed batch rename on rerun
            if st.session_state.get('confirm_batch_rename_apply'):
                find_text_confirmed = st.session_state.get("br_find_text_in_left_col", find_text_br) # Get current value
                replace_text_confirmed = st.session_state.get("br_replace_text_in_left_col", replace_text_br) # Get current value
                time_col_name = session_state.get('ts_tool_time_col_info', {}).get('name')
                
                df_renamed, status_final = batch_rename_columns_by_text_replacement(
                    session_state.ts_tool_data_processed, 
                    find_text_confirmed, 
                    replace_text_confirmed, 
                    time_col_name # Pass current time_col_name
                )

                if status_final['success'] and df_renamed is not None:
                    session_state.ts_tool_data_processed = df_renamed
                    final_rename_map = status_final.get('modified_map', {})
                    print(f"[DEBUG BATCH RENAME APPLIED] ts_tool_data_processed HEAD after rename:\\n{session_state.ts_tool_data_processed.head()}")
                    print(f"[DEBUG BATCH RENAME APPLIED] ts_tool_data_processed COLUMNS after rename: {session_state.ts_tool_data_processed.columns.tolist()}")

                    # Update ts_tool_cols_to_keep if it was based on old names
                    if 'ts_tool_cols_to_keep' in session_state and session_state.ts_tool_cols_to_keep:
                        session_state.ts_tool_cols_to_keep = [final_rename_map.get(col, col) for col in session_state.ts_tool_cols_to_keep]
                    print(f"[DEBUG BATCH RENAME APPLIED] ts_tool_cols_to_keep after rename: {session_state.ts_tool_cols_to_keep}")
                    
                    # Update time column name in session_state if it was renamed
                    current_time_col_name_in_state = session_state.get('ts_tool_time_col_info', {}).get('name')
                    if current_time_col_name_in_state and current_time_col_name_in_state in final_rename_map:
                        new_name_for_time_col = final_rename_map[current_time_col_name_in_state]
                        if new_name_for_time_col and new_name_for_time_col != current_time_col_name_in_state: # If actually renamed
                            if 'ts_tool_time_col_info' not in session_state or not isinstance(session_state.ts_tool_time_col_info, dict):
                                session_state.ts_tool_time_col_info = {} # Initialize if not present
                            session_state.ts_tool_time_col_info['name'] = new_name_for_time_col
                            # Also update manual selection if it was the one renamed
                            if session_state.get('ts_tool_manual_time_col') == current_time_col_name_in_state:
                                session_state.ts_tool_manual_time_col = new_name_for_time_col
                            st.info(f"提示：时间列 \'{current_time_col_name_in_state}\' 已通过批量操作重命名为 \'{new_name_for_time_col}\'。相关设置已同步更新。") # Was rename_expander.info
                    print(f"[DEBUG BATCH RENAME APPLIED] ts_tool_time_col_info after rename: {session_state.get('ts_tool_time_col_info')}")
                    print(f"[DEBUG BATCH RENAME APPLIED] ts_tool_manual_time_col after rename: {session_state.get('ts_tool_manual_time_col')}")
                    
                    # Clear individual rename rules as batch operation takes precedence
                    if session_state.ts_tool_rename_rules:
                         session_state.ts_tool_rename_rules = []
                         st.info("注意：由于执行了批量文本替换，已有的单条重命名规则已被清除。") # Was rename_expander.info
                    
                    # Update ts_tool_data_processed_FULL if it exists
                    if session_state.get('ts_tool_data_processed_FULL') is not None:
                        current_full_cols = session_state.ts_tool_data_processed_FULL.columns.tolist()
                        # Apply the same final_rename_map to the _FULL dataframe's columns
                        new_full_cols = [final_rename_map.get(col, col) for col in current_full_cols]
                        if list(current_full_cols) != list(new_full_cols): # If any column name actually changed
                            rename_dict_for_full = {old_col: new_name for old_col, new_name in zip(current_full_cols, new_full_cols) if old_col != new_name}
                            if rename_dict_for_full: # Check if there's anything to rename
                                session_state.ts_tool_data_processed_FULL = session_state.ts_tool_data_processed_FULL.rename(columns=rename_dict_for_full)
                                print(f"[DEBUG BATCH RENAME APPLIED] ts_tool_data_processed_FULL COLUMNS after rename: {session_state.ts_tool_data_processed_FULL.columns.tolist()}")
                    else:
                        print("[DEBUG BATCH RENAME APPLIED] ts_tool_data_processed_FULL was None, not updated.")

                    st.success("批量文本替换已成功应用。") # Was rename_expander.success
                    # Clear flags and preview data
                    session_state.br_preview_df = None
                    session_state.br_modified_map = None
                    session_state.br_status_message = None
                    session_state.br_status_type = None
                    st.session_state.confirm_batch_rename_apply = False

                    # MODIFIED: Conditional rerun
                    st.session_state.force_time_ui_snapshot_refresh_flag = True # <<< SET FLAG
                    print("[DEBUG processed_data_ui] Set force_time_ui_snapshot_refresh_flag = True after batch text replacement.") # DEBUG ADDED
                    st.rerun()
                elif not status_final['success']:
                    st.error(f"应用批量替换时出错: {status_final.get('message', '未知错误')}") # Was rename_expander.error
                    if status_final.get('conflicts'):
                        st.json(status_final.get('conflicts'))
                    st.session_state.confirm_batch_rename_apply = False # Reset flag on error

# Example of how this module might be called (for testing standalone)
if __name__ == '__main__':
    # Mock session_state for testing
    if 'session_state' not in st:
        st.session_state = {} # Simplistic mock, real Streamlit session_state is more complex
    
    # Initialize necessary session_state variables for this UI to run
    st.session_state.setdefault('ts_tool_processing_applied', True)
    st.session_state.setdefault('ts_tool_data_processed', pd.DataFrame({
        'A_numeric': [1, 2, 0, 4, 5], 
        'B_text': ['x', 'y', 'z', 'N/A', 'v'], 
        'C_mixed': [10, 'hello', 0, 30, '--'],
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    }))
    st.session_state.setdefault('ts_tool_data_processed_FULL', st.session_state.ts_tool_data_processed.copy())
    st.session_state.setdefault('ts_tool_rename_rules', [])
    st.session_state.setdefault('ts_tool_preview_rows', 5)
    st.session_state.setdefault('ts_tool_cols_to_keep', st.session_state.ts_tool_data_processed.columns.tolist())
    st.session_state.setdefault('ts_tool_manual_time_col', '(自动识别)')
    st.session_state.setdefault('ts_tool_time_col_info', {})
    st.session_state.setdefault('confirm_batch_rename_apply', False)
    st.session_state.setdefault('br_find_text_in_left_col', '')
    st.session_state.setdefault('br_replace_text_in_left_col', '')


    def mock_add_rename_rule():
        orig = st.session_state.get('rename_select_orig')
        new = st.session_state.get('rename_input_new')
        if orig and new and orig != new:
            # Avoid adding duplicate rules for the same original column
            if not any(rule['original'] == orig for rule in st.session_state.ts_tool_rename_rules):
                st.session_state.ts_tool_rename_rules.append({'original': orig, 'new': new})
            else:
                st.warning(f"列 '{orig}' 已存在重命名规则。")
        elif not orig or not new :
             st.warning("请同时选择原始列名并输入新列名。")
        else: # orig == new
            st.warning("新列名不能与原始列名相同。")


    st.title("测试 processed_data_ui 模块")
    
    # --- Test display_custom_nan_definition_ui ---
    st.header("测试：自定义缺失值 UI")
    display_custom_nan_definition_ui(st, st.session_state)
    st.dataframe(st.session_state.ts_tool_data_processed.head(), use_container_width=True)


    # --- Test display_column_selector_for_staging_ui ---
    st.header("测试：列选择器 UI (用于暂存)")
    display_column_selector_for_staging_ui(st, st.session_state)
    st.write("选定的列 (用于暂存):", st.session_state.get('ts_tool_staging_selected_columns'))


    st.header("测试：主数据处理与重命名 UI")
    display_processed_data_and_renaming(st, st.session_state, mock_add_rename_rule)
    
    st.subheader("Session State Inspector")
    st.json({k: str(v) if isinstance(v, pd.DataFrame) else v for k, v in st.session_state.items()})