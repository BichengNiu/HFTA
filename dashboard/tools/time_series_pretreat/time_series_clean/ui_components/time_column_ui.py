import streamlit as st
import pandas as pd
import numpy as np

# Assuming utils are in a sibling directory to ui_components
from ..utils.time_analysis import identify_time_column, generate_final_data, parse_time_column, infer_dataframe_frequency, align_dataframe_frequency
# Import the new UI function for custom NaN and the processing function
from .processed_data_ui import display_custom_nan_definition_ui, apply_custom_nan_values_processing
# Import the new backend logic function for frequency alignment
from ..utils.frequency_alignment_logic import perform_frequency_alignment

# --- FREQUENCY DEFINITIONS AND HELPER ---
FREQ_HIERARCHY = {
    'D': 0, 'B': 0, 'W': 1, 'M': 2, 'Q': 3, 'A': 4
}
def get_freq_level(pandas_freq_code):
    if not pandas_freq_code: return None
    base_code = pandas_freq_code.split('-')[0].upper()
    if base_code.startswith('W'): return FREQ_HIERARCHY['W']
    if base_code.startswith('M'): return FREQ_HIERARCHY['M']
    if base_code.startswith('Q'): return FREQ_HIERARCHY['Q']
    if base_code.startswith('A'): return FREQ_HIERARCHY['A']
    if base_code == 'D' or base_code == 'B': return FREQ_HIERARCHY['D']
    return None
# --- END OF FREQUENCY DEFINITIONS ---

# --- CALLBACK FUNCTION FOR RESETTING TIME SETTINGS (Restored) ---
def reset_time_related_settings_and_data():
    """
    Resets time-related settings and data to the state after 'Variable Processing and Preview'.
    It uses the snapshot 'ts_tool_df_at_time_ui_entry'.
    """
    print("[DEBUG TIME_RESET_CALLBACK] Entering reset_time_related_settings_and_data.")
    
    # --- DEBUG: Log current data before reset ---
    if 'ts_tool_data_processed' in st.session_state and st.session_state.ts_tool_data_processed is not None:
        print(f"[DEBUG TIME_RESET_CALLBACK] BEFORE reset - ts_tool_data_processed ID: {id(st.session_state.ts_tool_data_processed)}, Shape: {st.session_state.ts_tool_data_processed.shape}")
        print(f"[DEBUG TIME_RESET_CALLBACK] BEFORE reset - ts_tool_data_processed HEAD:\n{st.session_state.ts_tool_data_processed.head().to_string()}")
    else:
        print("[DEBUG TIME_RESET_CALLBACK] BEFORE reset - ts_tool_data_processed is None or not in session_state.")

    # --- DEBUG: Log snapshot data ---    
    if 'ts_tool_df_at_time_ui_entry' in st.session_state and st.session_state.ts_tool_df_at_time_ui_entry is not None:
        print(f"[DEBUG TIME_RESET_CALLBACK] SNAPSHOT ts_tool_df_at_time_ui_entry ID: {id(st.session_state.ts_tool_df_at_time_ui_entry)}, Shape: {st.session_state.ts_tool_df_at_time_ui_entry.shape}")
        print(f"[DEBUG TIME_RESET_CALLBACK] SNAPSHOT ts_tool_df_at_time_ui_entry HEAD:\n{st.session_state.ts_tool_df_at_time_ui_entry.head().to_string()}")
    else:
        print("[DEBUG TIME_RESET_CALLBACK] SNAPSHOT ts_tool_df_at_time_ui_entry is None or not in session_state.")

    if 'ts_tool_df_at_time_ui_entry' in st.session_state and st.session_state.ts_tool_df_at_time_ui_entry is not None:
        snapshot_df = st.session_state.ts_tool_df_at_time_ui_entry.copy() # Work with a copy of the snapshot
        print(f"[DEBUG TIME_RESET_CALLBACK] Copied snapshot_df ID: {id(snapshot_df)}")
        
        st.session_state.ts_tool_data_processed = snapshot_df # Assign the copy to ts_tool_data_processed
        
        # Also reset _FULL from the same snapshot logic
        if st.session_state.get('ts_tool_df_at_time_ui_entry') is not None: # Check again as it might be deleted above if None
             st.session_state.ts_tool_data_processed_FULL = st.session_state.ts_tool_df_at_time_ui_entry.copy()
             print(f"[DEBUG TIME_RESET_CALLBACK] Reset ts_tool_data_processed_FULL from snapshot. New ID: {id(st.session_state.ts_tool_data_processed_FULL)}")
        else: # Should not happen if snapshot existed initially for ts_tool_data_processed
            st.session_state.ts_tool_data_processed_FULL = None 
            print("[DEBUG TIME_RESET_CALLBACK] ts_tool_df_at_time_ui_entry became None, cannot reset _FULL from it.")

        # Update the source ID for snapshot comparison to reflect that we've reset to this state.
        # This source ID should ideally point to ts_tool_data_processed_FULL after it's reset.
        if st.session_state.ts_tool_data_processed_FULL is not None:
            st.session_state.ts_tool_df_at_time_ui_entry_source_id = id(st.session_state.ts_tool_data_processed_FULL)
            print(f"[DEBUG TIME_RESET_CALLBACK] Updated ts_tool_df_at_time_ui_entry_source_id to: {id(st.session_state.ts_tool_data_processed_FULL)}")
        else:
            st.session_state.ts_tool_df_at_time_ui_entry_source_id = None
            print("[DEBUG TIME_RESET_CALLBACK] ts_tool_data_processed_FULL is None after reset, source_id set to None.")
        
        # st.info("数据和时间设置已恢复到进入此部分之前的状态。") # Commented out potentially confusing message
    else:
        st.warning("未能找到用于重置的初始数据快照。可能需要重新加载或处理数据。")
    
    # --- DEBUG: Log current data AFTER reset attempt ---
    if 'ts_tool_data_processed' in st.session_state and st.session_state.ts_tool_data_processed is not None:
        print(f"[DEBUG TIME_RESET_CALLBACK] AFTER reset - ts_tool_data_processed ID: {id(st.session_state.ts_tool_data_processed)}, Shape: {st.session_state.ts_tool_data_processed.shape}")
        print(f"[DEBUG TIME_RESET_CALLBACK] AFTER reset - ts_tool_data_processed HEAD:\n{st.session_state.ts_tool_data_processed.head().to_string()}")
    else:
        print("[DEBUG TIME_RESET_CALLBACK] AFTER reset - ts_tool_data_processed is None or not in session_state.")

    st.session_state.ts_tool_manual_time_col = "(自动识别)"
    if 'ts_tool_manual_time_col_selectbox_val' in st.session_state:
        st.session_state.ts_tool_manual_time_col_selectbox_val = "(自动识别)"
    st.session_state.ts_tool_time_col_info = {
        'name': None, 'parsed_series': None, 
        'status_message': '时间相关设置已重置。请重新进行时间列选择和相关操作。', 'status_type': 'info' # Modified message for clarity
    }
    # Reset states for the OLD frequency completion UI (already mostly deprecated by commenting out its UI)
    # st.session_state.ts_tool_manual_frequency_fc = "自动" 
    # st.session_state.ts_tool_complete_time_index_fc = False 
    # st.session_state.ts_tool_frequency_completion_applied_flag = False
    # st.session_state.ts_tool_completion_message = None

    # Reset states for Frequency Analysis Results (right panel after custom NaN marking)
    st.session_state.ts_tool_frequency_analysis_results = None
    st.session_state.ts_tool_show_frequency_analysis_results = False

    # Reset states for Alignment Report (right panel after alignment operation)
    st.session_state.ts_tool_show_alignment_report = False
    st.session_state.ts_tool_alignment_report_items = [] # Reset to empty list
    st.session_state.ts_tool_alignment_results = None
    st.session_state.ts_tool_aligned_data_preview = None # If used for previewing aligned data separately

    # Reset custom NaN input string and selected columns for custom NaN
    if 'ts_tool_custom_nan_input_str' in st.session_state: # Check if it exists before resetting
        st.session_state['ts_tool_custom_nan_input_str'] = "0, N/A, --, NULL" 
    if 'custom_nan_input_text_global' in st.session_state: # Key for the text_input widget
        st.session_state['custom_nan_input_text_global'] = "0, N/A, --, NULL"

    if 'ts_tool_data_processed_cols_for_custom_nan' in st.session_state:
        del st.session_state['ts_tool_data_processed_cols_for_custom_nan']
    if 'custom_nan_selected_cols_global' in st.session_state: # Key for the multiselect widget
         # Setting to empty list to clear selection. Defaulting logic in display_custom_nan_definition_ui will handle it.
        st.session_state['custom_nan_selected_cols_global'] = []


    # Reset states for the new "数据频率对齐" section UI
    # Target alignment frequency
    alignment_freq_options_display_list_for_reset = [k for k, v in FREQUENCY_OPTIONS.items() if v != "auto"]
    if alignment_freq_options_display_list_for_reset: # Ensure list is not empty
        default_target_display = alignment_freq_options_display_list_for_reset[0]
        st.session_state.ts_tool_target_alignment_frequency_display = default_target_display
        st.session_state.ts_tool_target_alignment_frequency_code = FREQUENCY_OPTIONS.get(default_target_display)
    else: # Fallback if list is somehow empty (should not happen with current FREQUENCY_OPTIONS)
        st.session_state.ts_tool_target_alignment_frequency_display = None 
        st.session_state.ts_tool_target_alignment_frequency_code = None
    
    # Alignment mode and aggregation method
    st.session_state.ts_tool_alignment_mode = 'stat_align'
    st.session_state.ts_tool_selected_agg_method_display = "最后一个值 (Last)" # Default display name
    st.session_state.ts_tool_selected_agg_method_code = 'last' # Corresponding code
    
    # Checkbox for frequency completion in the "数据频率对齐" section
    st.session_state.ts_tool_complete_time_index_fc_moved = False

    # Clear snapshots of data taken before alignment
    if 'ts_tool_data_processed_before_alignment_or_completion' in st.session_state:
        del st.session_state.ts_tool_data_processed_before_alignment_or_completion
    if 'ts_tool_data_processed_FULL_before_alignment_or_completion' in st.session_state:
        del st.session_state.ts_tool_data_processed_FULL_before_alignment_or_completion

    st.session_state.ts_tool_filter_start_date = None
    st.session_state.ts_tool_filter_end_date = None
    st.session_state.ts_tool_active_filter_start = None 
    st.session_state.ts_tool_active_filter_end = None
    st.session_state["ts_tool_staging_active_time_filter_start_date"] = None
    st.session_state["ts_tool_staging_active_time_filter_end_date"] = None
    st.session_state.ts_tool_data_final = None 
    if 'ts_tool_manual_time_col_analyzed_for' in st.session_state:
        del st.session_state['ts_tool_manual_time_col_analyzed_for']
    print("[CALLBACK reset_time_related_settings_and_data] Time settings and data have been reset.")
    st.rerun()
# --- END OF CALLBACK FUNCTION ---

# Define a comprehensive list of frequency options with user-friendly names and their Pandas codes
# This controls what the user sees and what code is passed to backend functions.
FREQUENCY_OPTIONS = {
    "自动 (基于现有数据推断)": "auto",
    "日度 (D)": "D",
    "工作日度 (B)": "B",
    "周度 (W-SUN, 周日为始)": "W-SUN",
    "周度 (W-MON, 周一为始)": "W-MON",
    "周度 (W-FRI, 周五为始)": "W-FRI",
    "月度 (MS, 月初)": "MS",
    "月度 (M, 月末)": "M",
    "季度 (QS, 季初)": "QS",
    "季度 (Q, 季末)": "Q",
    "年度 (AS, 年初)": "AS",
    "年度 (A, 年末)": "A",
}

# Define the display order for the frequency options in the selectbox
# This ensures a logical and user-friendly presentation.
FREQ_DISPLAY_ORDER = [
    "自动 (基于现有数据推断)",
    "日度 (D)",
    "工作日度 (B)",
    "周度 (W-SUN, 周日为始)",
    "周度 (W-MON, 周一为始)",
    "周度 (W-FRI, 周五为始)",
    "月度 (MS, 月初)",
    "月度 (M, 月末)",
    "季度 (QS, 季初)",
    "季度 (Q, 季末)",
    "年度 (AS, 年初)",
    "年度 (A, 年末)"
]

# Ensure all keys in FREQ_DISPLAY_ORDER are in FREQUENCY_OPTIONS
for key in FREQ_DISPLAY_ORDER:
    if key not in FREQUENCY_OPTIONS:
        # This indicates a mismatch that needs to be fixed in the definitions above.
        # For now, remove it from order to prevent runtime errors if UI tries to use it.
        # A better approach might be to raise an error during development.
        print(f"Warning: Frequency '{key}' in FREQ_DISPLAY_ORDER not found in FREQUENCY_OPTIONS. Please check definitions.")
        # FREQ_DISPLAY_ORDER.remove(key) # Avoid modifying list while iterating like this
# Alternative check:
valid_ordered_keys = [key for key in FREQ_DISPLAY_ORDER if key in FREQUENCY_OPTIONS]
if len(valid_ordered_keys) != len(FREQ_DISPLAY_ORDER):
    print("Error: Mismatch between FREQ_DISPLAY_ORDER and FREQUENCY_OPTIONS keys. Review definitions.")
    # Potentially fall back to a default order or raise an error for dev
    # For now, we will use `valid_ordered_keys` for the selectbox to prevent crashes.
    # This means a misconfigured key in FREQ_DISPLAY_ORDER won't show up.

# --- NEW FUNCTION: For Time Filter (intended for staging prep) ---
def display_time_filter_for_staging_ui(st, session_state, df_key='ts_tool_data_processed', time_col_info_key='ts_tool_time_col_info', active_filter_prefix='ts_tool_staging_active_time_filter'):
    """
    UI for selecting a time range to filter data, result stored in session_state.
    It's intended to set session state variables that the staging process will use.
    It does NOT directly modify the DataFrame `session_state[df_key]` here.
    """
    st.markdown("##### **按时间范围筛选 (用于暂存)**")
    st.caption("选择一个时间范围。此范围将用于数据暂存或导出时的筛选。")

    time_info = session_state.get(time_col_info_key, {})
    time_col_name = time_info.get('name')

    if not time_col_name or df_key not in session_state or session_state[df_key] is None or session_state[df_key].empty:
        st.info("需要先在时间与频率设置中成功识别时间列，并有可用数据。")
        # Clear any previous filter settings if context is lost
        session_state[f"{active_filter_prefix}_start_date"] = None
        session_state[f"{active_filter_prefix}_end_date"] = None
        return

    df = session_state[df_key]
    if time_col_name not in df.columns:
        st.warning(f"时间列 '{time_col_name}' 不在当前数据中。筛选无法进行。")
        session_state[f"{active_filter_prefix}_start_date"] = None
        session_state[f"{active_filter_prefix}_end_date"] = None
        return

    try:
        # Ensure time column is datetime for min/max
        time_series = pd.to_datetime(df[time_col_name], errors='coerce').dropna()
        if time_series.empty:
            st.warning("时间列无法解析为有效日期，或解析后为空。无法确定筛选范围。")
            min_date, max_date = None, None
        else:
            min_date = time_series.min().date()
            max_date = time_series.max().date()
    except Exception as e:
        st.error(f"提取时间范围时出错: {e}")
        min_date, max_date = None, None

    # Get current filter values from session_state (these are what the staging logic will use)
    current_filter_start = session_state.get(f"{active_filter_prefix}_start_date")
    current_filter_end = session_state.get(f"{active_filter_prefix}_end_date")

    # UI for date input
    # The keys for these date_input widgets should be unique if this function is called multiple times
    # or if other date_inputs exist on the page. Adding a prefix related to staging.
    # Convert session state strings back to date objects for widget value if they exist
    widget_start_val = pd.to_datetime(current_filter_start).date() if current_filter_start else None
    widget_end_val = pd.to_datetime(current_filter_end).date() if current_filter_end else None

    # Adjust persisted session state values if they are outside the new min/max range
    if min_date and widget_start_val and widget_start_val < min_date:
        widget_start_val = min_date
        session_state[f"{active_filter_prefix}_start_date"] = min_date.isoformat()
        print(f"[STAGING_FILTER_ADJUST] Adjusted start_date in session to {min_date.isoformat()} as it was out of new data range.")

    if max_date and widget_end_val and widget_end_val > max_date:
        widget_end_val = max_date
        session_state[f"{active_filter_prefix}_end_date"] = max_date.isoformat()
        print(f"[STAGING_FILTER_ADJUST] Adjusted end_date in session to {max_date.isoformat()} as it was out of new data range.")
    
    # Further ensure that if only one bound exists from session, it's valid against the other if the other is from data min/max
    # (This specific scenario might be complex to handle perfectly without resetting user intent, so focusing on direct out-of-bounds for now)

    # If min_date/max_date are available, use them for the widgets
    # Otherwise, allow any date selection.
    filter_start_date_input_val = None
    filter_end_date_input_val = None

    # --- MODIFIED: Use columns for date inputs ---
    col_start_date, col_end_date = st.columns(2)

    with col_start_date:
        if min_date and max_date:
            filter_start_date_input_val = st.date_input(
                "选择开始日期 (包含):", 
                value=widget_start_val if widget_start_val else min_date, # Default to data min or current filter
                min_value=min_date, max_value=max_date, 
                key=f"{active_filter_prefix}_widget_start",
                help=f"数据最早日期: {min_date.strftime('%Y-%m-%d')}"
            )
        else: # Fallback if min_date/max_date couldn't be determined
            filter_start_date_input_val = st.date_input("选择开始日期 (包含):", value=widget_start_val, key=f"{active_filter_prefix}_widget_start_no_minmax")

    with col_end_date:
        if min_date and max_date:
            filter_end_date_input_val = st.date_input(
                "选择结束日期 (包含):", 
                value=widget_end_val if widget_end_val else max_date,   # Default to data max or current filter
                min_value=min_date, max_value=max_date, 
                key=f"{active_filter_prefix}_widget_end",
                help=f"数据最晚日期: {max_date.strftime('%Y-%m-%d')}"
            )
        else: # Fallback if min_date/max_date couldn't be determined
            filter_end_date_input_val = st.date_input("选择结束日期 (包含):", value=widget_end_val, key=f"{active_filter_prefix}_widget_end_no_minmax")
    # --- END MODIFICATION ---

    # Update session_state that staging logic will read
    # Convert date objects to ISO format strings for session state consistency
    session_state[f"{active_filter_prefix}_start_date"] = filter_start_date_input_val.isoformat() if filter_start_date_input_val else None
    session_state[f"{active_filter_prefix}_end_date"] = filter_end_date_input_val.isoformat() if filter_end_date_input_val else None
    
    if filter_start_date_input_val and filter_end_date_input_val and filter_start_date_input_val > filter_end_date_input_val:
        st.error("开始日期不能晚于结束日期。")
    elif session_state[f"{active_filter_prefix}_start_date"] or session_state[f"{active_filter_prefix}_end_date"]:
        start_display = session_state[f"{active_filter_prefix}_start_date"] or "最早"
        end_display = session_state[f"{active_filter_prefix}_end_date"] or "最晚"
        st.caption(f"暂存时将应用筛选范围: {start_display} 到 {end_display}")

    # if st.button("清除暂存筛选范围", key=f"{active_filter_prefix}_clear_button"):
    #     session_state[f"{active_filter_prefix}_start_date"] = None
    #     session_state[f"{active_filter_prefix}_end_date"] = None
    #     st.rerun()

def display_time_column_settings(st, session_state):
    """Displays UI for time column identification, its properties, and frequency completion."""

    # Custom CSS to reduce font size of st.metric values
    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important; /* Adjust size as needed */
    }
    </style>
    """, unsafe_allow_html=True)

    if not session_state.ts_tool_processing_applied or session_state.ts_tool_data_processed is None or session_state.ts_tool_data_processed.empty:
        return # Don't display if no processed data

    # --- Snapshot logic for reset functionality ---
    new_snapshot_needed = False
    
    # The ID for checking if a new snapshot is needed should be based on ts_tool_data_processed_FULL,
    # as this state variable is intended to be the stable baseline from upstream or after a full reset.
    # It is NOT changed by frequency completion itself, only by this snapshot logic or the reset button.
    # current_baseline_data_for_snapshot_id_check = session_state.get('ts_tool_data_processed_FULL') # This line can be removed or commented if current_snapshot_comparison_id is not used below for primary trigger
    # current_snapshot_comparison_id = id(current_baseline_data_for_snapshot_id_check) if current_baseline_data_for_snapshot_id_check is not None else None # This line can be removed or commented if not used below for primary trigger

    # --- REVISED Snapshot Trigger Logic ---
    # Snapshot is needed if:
    # 1. The force flag is set (coming from an upstream module).
    # 2. The snapshot itself ('ts_tool_df_at_time_ui_entry') or its source ID ('ts_tool_df_at_time_ui_entry_source_id') doesn't exist yet (first time loading this UI).
    if session_state.get('force_time_ui_snapshot_refresh_flag', False):
        new_snapshot_needed = True
        print("[DEBUG TIME_UI_SNAPSHOT] Snapshot needed: force_time_ui_snapshot_refresh_flag was True.")
    elif 'ts_tool_df_at_time_ui_entry' not in session_state or \
         session_state.ts_tool_df_at_time_ui_entry is None or \
         'ts_tool_df_at_time_ui_entry_source_id' not in session_state: # Check if snapshot or its source ID is missing
        new_snapshot_needed = True
        print("[DEBUG TIME_UI_SNAPSHOT] Snapshot needed: Snapshot or its source ID is missing (first time or lost state).")
    # REMOVED: The direct comparison of current_snapshot_comparison_id with ts_tool_df_at_time_ui_entry_source_id as a primary trigger
    # This prevents internal operations from overwriting the entry snapshot.
    # --- END OF REVISED Snapshot Trigger Logic ---
    
    if new_snapshot_needed:
        # If a new snapshot is needed, it means the baseline data (from upstream or reset) has changed,
        # or this is the first time the UI is being loaded for the current data.
        # The snapshot should be taken from the current session_state.ts_tool_data_processed_FULL,
        # as this should represent the state of the data upon entering this module.
        data_to_snapshot = session_state.get('ts_tool_data_processed_FULL')
        print(f"[DEBUG TIME_UI_SNAPSHOT] new_snapshot_needed is True. Data source for snapshot (ts_tool_data_processed_FULL) is None: {data_to_snapshot is None}")
        
        if data_to_snapshot is not None:
            print(f"[DEBUG TIME_UI_SNAPSHOT] ts_tool_data_processed_FULL to be snapshotted - Shape: {data_to_snapshot.shape}, Head:\\n{data_to_snapshot.head().to_string()}")
            
            session_state.ts_tool_df_at_time_ui_entry = data_to_snapshot.copy()
            # The 'ts_tool_data_processed_FULL' is already the source, so no need to copy to itself here.
            # 'ts_tool_data_processed' will be used for manipulations within this UI.
            print(f"[DEBUG TIME_UI_SNAPSHOT] Created ts_tool_df_at_time_ui_entry. Shape: {session_state.ts_tool_df_at_time_ui_entry.shape}")
            
            # Update the source ID for future reference, based on the data just snapshotted.
            session_state.ts_tool_df_at_time_ui_entry_source_id = id(data_to_snapshot)

            # Reset dependent UI states because the underlying baseline data has changed for this module's context
            # (This part might be excessive if only taking snapshot, review if these should always reset on new snapshot)
            # For now, keeping them as they were tied to the 'new_snapshot_needed' logic.
            session_state.ts_tool_frequency_completion_applied_flag = False
            session_state.ts_tool_active_filter_start = None 
            session_state.ts_tool_active_filter_end = None
            session_state.ts_tool_filter_start_date = None
            session_state.ts_tool_filter_end_date = None
            # session_state.ts_tool_complete_time_index_fc = False # This was for old UI, ts_tool_complete_time_index_fc_moved is current
            session_state.ts_tool_complete_time_index_fc_moved = False # Reset new checkbox state
            # session_state.ts_tool_manual_frequency_fc = "自动"    # This was for old UI
            
            # Reset parts of the new frequency alignment UI if the snapshot is being taken/refreshed
            # This ensures the UI defaults correctly when new data comes in or it's the first load.
            alignment_freq_options_display_list_for_reset_snapshot = [k for k, v in FREQUENCY_OPTIONS.items() if v != "auto"]
            if alignment_freq_options_display_list_for_reset_snapshot:
                default_target_display_snapshot = alignment_freq_options_display_list_for_reset_snapshot[0]
                session_state.ts_tool_target_alignment_frequency_display = default_target_display_snapshot
                session_state.ts_tool_target_alignment_frequency_code = FREQUENCY_OPTIONS.get(default_target_display_snapshot)
            session_state.ts_tool_alignment_mode = 'stat_align'
            session_state.ts_tool_selected_agg_method_display = "最后一个值 (Last)"
            session_state.ts_tool_selected_agg_method_code = 'last'
            session_state.ts_tool_show_alignment_report = False # Hide any previous report
            session_state.ts_tool_alignment_report_items = []
            session_state.ts_tool_frequency_analysis_results = None # Clear old freq analysis
            session_state.ts_tool_show_frequency_analysis_results = False


            if 'ts_tool_manual_time_col_analyzed_for' in session_state: # Force re-analysis of time col
                 del session_state['ts_tool_manual_time_col_analyzed_for']
            print(f"[DEBUG TIME_UI_SNAPSHOT] New snapshot taken for 'ts_tool_df_at_time_ui_entry'. Comparison ID recorded: {session_state.ts_tool_df_at_time_ui_entry_source_id}. UI states reset.")
        else: 
            session_state.ts_tool_df_at_time_ui_entry = None
            # session_state.ts_tool_data_processed_FULL = None # Don't nullify _FULL if it was the source and None
            session_state.ts_tool_df_at_time_ui_entry_source_id = None
            # Reset other states as above
            session_state.ts_tool_frequency_completion_applied_flag = False
            session_state.ts_tool_active_filter_start = None; session_state.ts_tool_active_filter_end = None
            session_state.ts_tool_filter_start_date = None; session_state.ts_tool_filter_end_date = None
            session_state.ts_tool_complete_time_index_fc_moved = False
            # ... (other UI state resets)
            if 'ts_tool_manual_time_col_analyzed_for' in session_state: del session_state['ts_tool_manual_time_col_analyzed_for']
            print("[DEBUG TIME_UI_SNAPSHOT] Input data_to_snapshot (ts_tool_data_processed_FULL) was None, snapshot and related UI states cleared/reset.")
        
        # Reset the force flag after use
        if session_state.get('force_time_ui_snapshot_refresh_flag', False):
            session_state.force_time_ui_snapshot_refresh_flag = False
            print("[DEBUG TIME_UI_SNAPSHOT] Reset force_time_ui_snapshot_refresh_flag to False after snapshot creation/update.")
    # --- End of Snapshot logic ---

    st.markdown("---")

    # --- MODIFIED: Title and Reset Button ---
    col_title_time, col_reset_button_time = st.columns([0.7, 0.3]) # Adjust ratio as needed
    with col_title_time:
        st.subheader("时间与频率设置")
    with col_reset_button_time:
        # --- MODIFIED: Removed CSS, simplified button ---
        if st.button("重置",  # Changed button text
                     key="reset_button_time_settings_main_title", 
                     on_click=reset_time_related_settings_and_data,
                     help='将所有时间相关设置（包括时间列选择、频率、补全、筛选）重置，并将数据恢复到刚完成"变量处理与预览"时的状态。请注意：自定义缺失值标记和频率分析结果（如有）也将被清除。如果已执行频率对齐，则对齐结果也将被重置。',
                     ):
            pass 
        # --- END OF MODIFICATION ---

    # Help text for time column selection
    st.caption(
        "工具会尝试自动识别代表时间或日期的列。"
        "如果自动识别不准确，或者您希望指定不同的列作为时间轴，请在此处手动选择。"
        "选定的时间列将用于后续的时间序列分析、频率推断和数据补全等操作。"
    )

    current_processed_cols = session_state.ts_tool_data_processed.columns.tolist()
    time_col_options = ["(自动识别)"] + current_processed_cols
    
    # Ensure the session_state.ts_tool_manual_time_col is valid, otherwise reset to auto
    if session_state.ts_tool_manual_time_col not in time_col_options:
        session_state.ts_tool_manual_time_col = "(自动识别)"
        session_state.ts_tool_time_col_info = {'name': None, 'parsed_series': None, 'status_message': '之前的选择失效，已重置为自动识别', 'status_type': 'info'}
        # Ensure analyzed_for is also reset or handled if it becomes inconsistent
        if 'ts_tool_manual_time_col_analyzed_for' in session_state:
            del session_state['ts_tool_manual_time_col_analyzed_for']

    selected_time_col_for_ui = session_state.ts_tool_manual_time_col
    try:
        current_selection_index = time_col_options.index(selected_time_col_for_ui)
    except ValueError:
        current_selection_index = 0 
        session_state.ts_tool_manual_time_col = "(自动识别)"

    # Store previous selection to detect change for automatic analysis
    prev_manual_time_col_selection = session_state.get('ts_tool_manual_time_col_selectbox_val', None)

    session_state.ts_tool_manual_time_col = st.selectbox(
        "选择时间列:",
        options=time_col_options,
        index=current_selection_index,
        key="manual_time_col_selector_main"
    )

    # --- Automatic Time Column Analysis ---
    # Analyze if the selection changed, or if info is missing/stale for the current selection
    needs_analysis = False
    if session_state.ts_tool_manual_time_col != prev_manual_time_col_selection:
        needs_analysis = True
        print(f"[DEBUG AUTO_TIME_ANALYSIS] Time col selection changed from '{prev_manual_time_col_selection}' to '{session_state.ts_tool_manual_time_col}'. Triggering analysis.")
    
    current_time_info = session_state.get('ts_tool_time_col_info', {})
    if not current_time_info or not current_time_info.get('name'): # No info at all
        needs_analysis = True
        print(f"[DEBUG AUTO_TIME_ANALYSIS] No existing time_col_info. Triggering analysis for '{session_state.ts_tool_manual_time_col}'.")
    elif session_state.ts_tool_manual_time_col == "(自动识别)" and current_time_info.get('name') is None: # Auto-detect selected but nothing found yet
        needs_analysis = True
        print(f"[DEBUG AUTO_TIME_ANALYSIS] Auto-detect selected and no time column identified yet. Triggering analysis.")
    elif session_state.ts_tool_manual_time_col != "(自动识别)" and current_time_info.get('name') != session_state.ts_tool_manual_time_col : # Manual selection doesn't match current info
        needs_analysis = True
        print(f"[DEBUG AUTO_TIME_ANALYSIS] Manual selection '{session_state.ts_tool_manual_time_col}' doesn't match current info name '{current_time_info.get('name')}'. Triggering analysis.")
    
    # Check against a marker to prevent re-analysis if already done for current selection in this run
    if session_state.get('ts_tool_manual_time_col_analyzed_for') == session_state.ts_tool_manual_time_col and not needs_analysis:
         # If it was already analyzed for this specific selection in this rerun cycle, and no other condition triggered 'needs_analysis', then skip.
         pass
    elif needs_analysis:
        print(f"[DEBUG AUTO_TIME_ANALYSIS] Proceeding with identify_time_column for: {session_state.ts_tool_manual_time_col}")
        session_state.ts_tool_time_col_info = identify_time_column(
            session_state.ts_tool_data_processed, 
            session_state.ts_tool_manual_time_col
        )
        session_state.ts_tool_manual_time_col_analyzed_for = session_state.ts_tool_manual_time_col # Mark as analyzed for this selection
        # If the selectbox value itself changed, Streamlit is already re-running.
        # If analysis was triggered for other reasons (e.g. stale info), a rerun ensures UI consistency.
        if session_state.ts_tool_manual_time_col != prev_manual_time_col_selection:
            # This was handled by selectbox causing rerun already
            pass
        else: # Analysis was triggered by stale info, not selectbox change
             st.rerun()


    # Update the stored selectbox value for next comparison
    session_state.ts_tool_manual_time_col_selectbox_val = session_state.ts_tool_manual_time_col

    # --- Display Time Column Info --- 
    time_info = session_state.get('ts_tool_time_col_info', {})
    
    if time_info.get('status_message'):
        if time_info.get('status_type') == 'success':
            st.success(time_info['status_message'])
        elif time_info.get('status_type') == 'warning':
            st.warning(time_info['status_message'])
        elif time_info.get('status_type') == 'error':
            st.error(time_info['status_message'])
        else:
            st.info(time_info['status_message'])

    if time_info.get('name'):
        st.markdown("**识别到的时间列信息:**")
        
        # Using Markdown with HTML for custom layout
        line1_html = f"""
        <div style='display: flex; flex-wrap: wrap; align-items: center; margin-bottom: 5px;'>
            <div style='margin-right: 20px;'><strong>时间列名称:</strong> {str(time_info['name'])}</div>
            <div style='margin-right: 20px;'><strong>推断频率:</strong> {str(time_info.get('freq', '未知'))}</div>
            <div><strong>解析方法:</strong> {str(time_info.get('parse_format', '未知'))}</div>
        </div>
        """
        line2_html = f"""
        <div style='display: flex; flex-wrap: wrap; align-items: center;'>
            <div style='margin-right: 20px;'><strong>开始时间:</strong> {str(time_info.get('start', '未知'))}</div>
            <div><strong>结束时间:</strong> {str(time_info.get('end', '未知'))}</div>
        </div>
        """
        st.markdown(line1_html + line2_html, unsafe_allow_html=True)

    # --- Frequency Completion Section (COMMENTING OUT THIS ENTIRE SECTION) ---
    # st.markdown("---")
    # st.markdown("##### **B. 时间序列频率补全与对齐 (可选)**", unsafe_allow_html=True)
    # st.caption("""
    #     如果您的时间序列数据在某些时间点上没有观测值（例如，月度数据缺少某几个月），
    #     或者您希望将不同原始频率的数据统一到一个新的目标频率（例如，都转换为月度），
    #     此功能可以帮助您。补全或对齐后，新插入的行/列的数据将填充为 NaN (缺失值)，
    #     您可以在后续的"缺失数据处理"步骤中对这些 NaN 值进行插值或其他方式的填充。
    #     如果您的数据已经是完整且等频率的，或者您不希望改变其固有频率，通常不需要启用此功能。
    # """)

    # # --- MODIFIED: Key for checkbox for frequency completion ---
    # # Key is now more specific: ts_tool_enable_frequency_completion_checkbox
    # # Default is False
    # enable_freq_completion = st.checkbox(
    #     "启用时间序列频率补全 (对齐到统一频率)",
    #     value=session_state.get('ts_tool_enable_frequency_completion_checkbox', False), # Default to False
    #     key='ts_tool_enable_frequency_completion_checkbox', # Specific key
    #     help="""
    #     如果您的时间序列数据在某些时间点上没有观测值（例如，月度数据缺少某几个月），
    #     此功能可以帮助您按照指定的频率补齐这些缺失的时间点。
    #     补全后，新插入的行的数据列将填充为 NaN (缺失值)，您可以在后续的"缺失数据处理"步骤中对这些 NaN 值进行插值或其他方式的填充。
    #     如果您的数据已经是完整且等频率的，通常不需要启用此功能。
    #     """
    # )
    # session_state.ts_tool_enable_frequency_completion_checkbox = enable_freq_completion
    # # --- END MODIFICATION ---

    # if session_state.get('ts_tool_enable_frequency_completion_checkbox', False): # Check session_state directly
    #     # ... (Existing logic for target frequency selection and apply button)
    #     # This entire block related to frequency completion UI and logic needs to be commented out or removed if the feature is removed.
        
    #     # --- Display UI for frequency selection and alignment --- 
    #     time_col_name = session_state.get('ts_tool_time_col_info', {}).get('name')
    #     inferred_freq_display = session_state.get('ts_tool_time_col_info', {}).get('inferred_freq_display', '未知')
    #     can_proceed_with_alignment = time_col_name and session_state.ts_tool_data_processed is not None and not session_state.ts_tool_data_processed.empty

    #     if not can_proceed_with_alignment:
    #         st.warning("需要先成功识别时间列并有可用数据才能进行频率对齐。")
    #     else:
    #         st.markdown("###### **1. 选择目标频率与对齐方式**")
            
    #         # Use valid_ordered_keys for options to avoid runtime errors from misconfiguration
    #         selected_freq_display_name = st.selectbox(
    #             label="选择补全/对齐的目标频率:",
    #             options=valid_ordered_keys, 
    #             index=0, # Default to "自动"
    #             key="ts_tool_manual_frequency_fc_selectbox",
    #             help=(
    #                 f"当前数据推断频率为: {inferred_freq_display}. "
    #                 "选择一个目标频率。'自动'将尝试补全至当前推断的频率。"
    #                 "其他选项会将数据对齐到所选频率，可能涉及升频（补NaN）或降频（聚合）。"
    #             )
    #         )
    #         session_state.ts_tool_manual_frequency_fc = FREQUENCY_OPTIONS[selected_freq_display_name]

    #         # --- AGGREGATION METHOD UI (Only if downsampling) --- 
    #         target_freq_code_fc = session_state.ts_tool_manual_frequency_fc
    #         original_freq_code_fc = session_state.get('ts_tool_time_col_info', {}).get('inferred_freq_code')

    #         # Determine if it's downsampling
    #         is_downsampling_fc = False
    #         if original_freq_code_fc and target_freq_code_fc and target_freq_code_fc != 'auto':
    #             original_level_fc = get_freq_level(original_freq_code_fc)
    #             target_level_fc = get_freq_level(target_freq_code_fc)
    #             if original_level_fc is not None and target_level_fc is not None and target_level_fc > original_level_fc:
    #                 is_downsampling_fc = True
            
    #         default_agg_method = session_state.get('ts_tool_agg_method_fc', 'mean')
    #         if is_downsampling_fc:
    #             agg_method_fc = st.selectbox(
    #                 "选择降频时的聚合方法:",
    #                 options=['mean', 'sum', 'median', 'min', 'max', 'first', 'last'],
    #                 index=['mean', 'sum', 'median', 'min', 'max', 'first', 'last'].index(default_agg_method),
    #                 key="ts_tool_agg_method_fc_selectbox",
    #                 help="当数据从高频率转换为低频率时（例如日度转月度），需要指定如何聚合原始时间点内的多个数据点。"
    #             )
    #             session_state.ts_tool_agg_method_fc = agg_method_fc
    #         else:
    #             # Ensure agg_method is cleared or not used if not downsampling
    #             if 'ts_tool_agg_method_fc' in session_state:
    #                 del session_state.ts_tool_agg_method_fc # Or set to None
            
    #         # --- Fill method for upsampling NaN values (optional, can be done in missing data step too) ---
    #         # For now, let's keep it simple and assume NaNs will be filled later.
    #         # fill_method_fc = st.selectbox(
    #         #     "选择升频时新产生NaN的填充方法 (可选，也可在后续缺失处理中操作):",
    #         #     options=['不填充 (保留NaN)', '向前填充 (ffill)', '向后填充 (bfill)'], # Consider 'interpolate' later
    #         #     index=0,
    #         #     key="ts_tool_fill_method_fc_selectbox"
    #         # )
    #         # session_state.ts_tool_fill_method_fc = fill_method_fc.split(' ')[0] # '不填充', '向前填充', etc.

    #         apply_freq_completion_button_key = "apply_frequency_completion_button_v2"
    #         if st.button("应用频率补全/对齐", key=apply_freq_completion_button_key, help="根据以上设置处理数据。处理后的数据将替换当前预览数据。"):
    #             if session_state.ts_tool_data_processed is not None and not session_state.ts_tool_data_processed.empty and time_col_name:
                    
    #                 processing_params = {
    #                     "df": session_state.ts_tool_data_processed.copy(), # Work on a copy
    #                     "time_col_name": time_col_name,
    #                     "target_freq_pd_code": session_state.ts_tool_manual_frequency_fc, # Pandas code from FREQUENCY_OPTIONS
    #                     "agg_method": session_state.get('ts_tool_agg_method_fc', 'mean'), # Default if not set
    #                     "fill_method": None # Not implemented yet, keep as None
    #                 }
                    
    #                 # Call the refactored backend logic
    #                 success, result_df, message = perform_frequency_alignment(processing_params, st.session_state)

    #                 if success and result_df is not None:
    #                     # Preserve the original (before alignment) FULL data if it exists
    #                     # The main 'ts_tool_data_processed' will be updated with the aligned data.
    #                     # 'ts_tool_data_processed_FULL' should ideally remain the version from before any frequency alignment.
    #                     # This is important if the user wants to reset frequency alignment or try different settings.
    #                     if 'ts_tool_data_processed_FULL' not in session_state or session_state.ts_tool_data_processed_FULL is None:
    #                         # If _FULL doesn't exist, or is None, create it from the current data BEFORE alignment.
    #                         # This should ideally be set when entering the time_ui section for the first time.
    #                         # Let's ensure it's based on the snapshot taken at UI entry.
    #                         if 'ts_tool_df_at_time_ui_entry' in st.session_state and st.session_state.ts_tool_df_at_time_ui_entry is not None:
    #                             st.session_state.ts_tool_data_processed_FULL = st.session_state.ts_tool_df_at_time_ui_entry.copy()
    #                             print("[DEBUG FreqAlign] ts_tool_data_processed_FULL (re)set from ts_tool_df_at_time_ui_entry before alignment.")
    #                         else: # Fallback, though less ideal
    #                             st.session_state.ts_tool_data_processed_FULL = session_state.ts_tool_data_processed.copy()
    #                             print("[DEBUG FreqAlign] ts_tool_data_processed_FULL set from current ts_tool_data_processed (fallback) before alignment.")
                        
    #                     # Update the main working DataFrame
    #                     st.session_state.ts_tool_data_processed = result_df
                        
    #                     # Update time column info based on the newly aligned data
    #                     new_time_col_info = identify_time_column(result_df, time_col_name) # Re-identify based on new df
    #                     st.session_state.ts_tool_time_col_info = new_time_col_info # Update session state
                        
    #                     st.session_state.ts_tool_frequency_completion_applied_flag = True
    #                     st.session_state.ts_tool_completion_message = message # Store success message
    #                     st.success(message)
    #                     # --- <<< 新增：设置标志位以强制刷新 missing_data_ui 中的快照 >>> ---
    #                     st.session_state.force_missing_ui_snapshot_refresh_flag = True
    #                     print("[DEBUG time_column_ui] Set force_missing_ui_snapshot_refresh_flag = True after frequency completion.")
    #                     # --- <<< 结束新增 >>> ---
    #                     st.rerun() # Rerun to update UI, especially time_col_info display
    #                 else:
    #                     st.error(f"频率补全/对齐失败: {message}")
    #                     st.session_state.ts_tool_frequency_completion_applied_flag = False
    #                     st.session_state.ts_tool_completion_message = f"错误: {message}"
    #             else:
    #                 st.warning("没有数据或未选择时间列，无法进行频率补全/对齐。")
        
    #     # Display message if completion has been applied in this session
    #     if session_state.get('ts_tool_frequency_completion_applied_flag') and session_state.get('ts_tool_completion_message'):
    #         # Check if the message is a success or error type for appropriate display (already handled by st.success/st.error above)
    #         # This is more for a persistent status if needed, but rerun already updates.
    #         # st.info(f"上次操作状态: {session_state.ts_tool_completion_message}") 
    #         pass

    # --- End of Frequency Completion Section ---

    # --- Custom NaN Definition UI (MOVED HERE) ---
    st.markdown("---")

    # --- Advanced Frequency Analysis and Alignment UI ---
    # This is where the user wants the custom NaN UI to be conceptually located.
    # The subheader and caption for "高级频率分析与规整" were previously removed by the user from this spot.
    # We will add the custom NaN UI before the logic that depends on `current_time_col_for_freq_ops` for this section.

    # MODIFICATION: Re-introduce columns for side-by-side layout of the two features
    col_left, col_right = st.columns(2)

    with col_left:
        # --- Custom NaN Definition UI ---
        nan_input_values = display_custom_nan_definition_ui(st, session_state) 

        # --- MODIFIED: Combined Button --- 
        if st.button("标记缺失值并重新分析频率", key="combined_nan_freq_button_v1"):
            # Step 1: Apply Custom NaN Values (without immediate rerun)
            apply_custom_nan_values_processing(
                st,
                session_state,
                df_key='ts_tool_data_processed',
                cols_to_process_key='ts_tool_data_processed_cols_for_custom_nan',
                nan_values_to_replace_list=nan_input_values["nan_values_to_replace"],
                selected_cols_list=nan_input_values["selected_cols_for_nan"],
                nan_values_str_input_for_warning=nan_input_values["nan_values_str_input"],
                rerun_after_processing=False # IMPORTANT: Defer rerun
            )
            # --- DEBUG PRINT AFTER CUSTOM NAN PROCESSING ---
            debug_col_name_after_nan = "中国:生产率:焦炉:国内独立焦化厂(230家)"
            if session_state.ts_tool_data_processed is not None and debug_col_name_after_nan in session_state.ts_tool_data_processed.columns:
                non_nan_count_after_nan = session_state.ts_tool_data_processed[debug_col_name_after_nan].notna().sum()
                print(f"DEBUG_AFTER_NAN_MARKING: Non-NaN count for '{debug_col_name_after_nan}' after custom NaN processing: {non_nan_count_after_nan}")
            elif session_state.ts_tool_data_processed is None:
                print(f"DEBUG_AFTER_NAN_MARKING: ts_tool_data_processed is None after custom NaN processing.")
            else:
                print(f"DEBUG_AFTER_NAN_MARKING: Column '{debug_col_name_after_nan}' not found in ts_tool_data_processed after custom NaN processing.")
            # --- END DEBUG PRINT ---
            
            # Assume apply_custom_nan_values_processing shows its own success/error messages.
            # Now, proceed to frequency analysis regardless of NaN marking success, 
            # as user might want to re-analyze even if no NaNs were marked (e.g. changed mind on input)
            # or if NaN marking failed but data is still valid for frequency analysis.
            
            # Step 2: Perform Frequency Analysis
            time_col_name = None # Initialize to avoid UnboundLocalError if first if is false
            if session_state.ts_tool_time_col_info and session_state.ts_tool_time_col_info.get('name'):
                time_col_name = session_state.ts_tool_time_col_info['name']
                if not time_col_name or time_col_name not in session_state.ts_tool_data_processed.columns:
                    st.warning("时间列未被正确识别或不在当前数据中，无法分析频率。")
                    session_state.ts_tool_frequency_analysis_results = None
                    session_state.ts_tool_show_frequency_analysis_results = False
                else: # Correctly placed else for "if not time_col_name..."
                    df_for_freq_analysis = session_state.ts_tool_data_processed
                    if df_for_freq_analysis is not None and not df_for_freq_analysis.empty:
                        with st.spinner("标记缺失值后，正在重新分析各数据列频率..."):
                            analysis_results = infer_dataframe_frequency(df_for_freq_analysis.copy(), time_col_name)
                        session_state.ts_tool_frequency_analysis_results = analysis_results
                        session_state.ts_tool_show_frequency_analysis_results = True
                        # Success message for frequency analysis will be shown in col_right based on these flags
                    else: # df_for_freq_analysis is None or empty
                        st.warning("数据为空，无法分析频率。")
                        session_state.ts_tool_frequency_analysis_results = None
                        session_state.ts_tool_show_frequency_analysis_results = False
            
            st.rerun() # Rerun once after both operations (or attempts) are done

        # <<< NEW FEATURE: Data Frequency Alignment >>>
        st.markdown("---") # Separator
        st.markdown("##### 数据频率对齐")
        st.caption("将数据聚合到指定的目标频率，或进行时间序列的频率补全。")

        # --- Create columns for a more compact layout ---
        col_align_setup1, col_align_setup2 = st.columns(2)

        with col_align_setup1:
            # --- Step 1: (MODIFIED) Display Base Frequency from Time Column Info ---
            base_freq_display = "未识别时间列或其频率"
            base_freq_code_from_time_col = None
            time_col_name_for_base_freq = "N/A"

            current_time_col_info = session_state.get('ts_tool_time_col_info')
            if current_time_col_info and current_time_col_info.get('name') and current_time_col_info.get('inferred_freq_code'):
                time_col_name_for_base_freq = current_time_col_info['name']
                base_freq_display = f"{current_time_col_info.get('freq', '未知')} (代码: {current_time_col_info['inferred_freq_code']})"
                base_freq_code_from_time_col = current_time_col_info['inferred_freq_code']
            
            st.markdown(f"**基准频率 (基于时间列 '{time_col_name_for_base_freq}')**")
            st.info(base_freq_display)
            session_state.ts_tool_selected_base_actual_pandas_code = base_freq_code_from_time_col

            # --- Step 2: (RE-ADD) Select Target Alignment Frequency ---
            st.markdown("**选择目标对齐频率:**")
            alignment_freq_options_display_list = [k for k, v in FREQUENCY_OPTIONS.items() if v != "auto"]
            alignment_freq_options_map_for_target = {k: v for k, v in FREQUENCY_OPTIONS.items() if v != "auto"}

            if 'ts_tool_target_alignment_frequency_display' not in session_state:
                session_state.ts_tool_target_alignment_frequency_display = alignment_freq_options_display_list[0]
            # ts_tool_target_alignment_frequency_code will be set after selectbox

            target_freq_selectbox_idx = 0
            if session_state.ts_tool_target_alignment_frequency_display in alignment_freq_options_display_list:
                try:
                    target_freq_selectbox_idx = alignment_freq_options_display_list.index(session_state.ts_tool_target_alignment_frequency_display)
                except ValueError:
                    target_freq_selectbox_idx = 0

            selected_target_align_freq_display = st.selectbox(
                label="选择目标频率",
                options=alignment_freq_options_display_list,
                index=target_freq_selectbox_idx,
                key="align_target_frequency_selector_main",
                label_visibility="collapsed"  # 使用label_visibility替代空label
            )
            session_state.ts_tool_target_alignment_frequency_display = selected_target_align_freq_display
            session_state.ts_tool_target_alignment_frequency_code = alignment_freq_options_map_for_target.get(selected_target_align_freq_display)

            # --- Frequency Comparison Logic (NOW INSIDE col_align_setup1, AFTER target selection) ---
            base_freq_code_for_comp = session_state.get('ts_tool_selected_base_actual_pandas_code')
            base_level = get_freq_level(base_freq_code_for_comp) # get_freq_level is now defined at a higher scope
            target_level = get_freq_level(session_state.ts_tool_target_alignment_frequency_code)

            session_state.ts_tool_is_high_to_low_conversion = False
            session_state.ts_tool_is_low_to_high_conversion = False
            session_state.ts_tool_is_same_frequency_conversion = False
            session_state.ts_tool_is_cross_two_levels = False
            valid_comparison = False
            if base_level is not None and target_level is not None and base_freq_code_for_comp:
                valid_comparison = True
                if base_level < target_level:
                    session_state.ts_tool_is_high_to_low_conversion = True
                    if target_level - base_level >= 2: session_state.ts_tool_is_cross_two_levels = True
                elif base_level > target_level: session_state.ts_tool_is_low_to_high_conversion = True
                else: session_state.ts_tool_is_same_frequency_conversion = True
            
            # --- Moved Conditional Operation Type Message to col_align_setup1 (uses above flags) ---
            if not valid_comparison or not base_freq_code_for_comp: 
                st.info("请选择有效的'基准频率'和'目标频率'以确定转换类型。") 
            elif session_state.ts_tool_is_high_to_low_conversion:
                st.write("检测到 **高频转低频** 操作。")
            elif session_state.ts_tool_is_low_to_high_conversion or session_state.ts_tool_is_same_frequency_conversion:
                st.write("检测到 **低频转高频** 或 **同频率调整** 操作。")

            # --- Initialize new session state variables for conditional UI (if not already present) ---
            if 'ts_tool_alignment_mode' not in session_state: session_state.ts_tool_alignment_mode = 'stat_align'
            if 'ts_tool_selected_agg_method_display' not in session_state: session_state.ts_tool_selected_agg_method_display = "最后一个值 (Last)"
            if 'ts_tool_selected_agg_method_code' not in session_state: session_state.ts_tool_selected_agg_method_code = 'last'
            if 'ts_tool_complete_time_index_fc_moved' not in session_state: session_state.ts_tool_complete_time_index_fc_moved = False

        with col_align_setup2:
            # --- Conditional UI for Alignment Mode and Aggregation Method ---
            if not valid_comparison or not base_freq_code_for_comp:
                st.info("请在左侧选择有效的'基准频率'和'目标频率'以确定对齐方式。") 
            elif session_state.ts_tool_is_high_to_low_conversion:
                # Only show Steps 3 & 4 for High-to-Low conversion
                st.markdown("**选择对齐模式:**")
                alignment_mode_options = {
                    "统计对齐 (使用聚合方法)": "stat_align",
                    "值对齐 (保留原始值，可能借调)": "value_align"
                }

                if session_state.ts_tool_is_cross_two_levels:
                    if "值对齐 (保留原始值，可能借调)" in alignment_mode_options:
                        del alignment_mode_options["值对齐 (保留原始值，可能借调)"]
                    if session_state.get('ts_tool_alignment_mode') == 'value_align':
                        session_state.ts_tool_alignment_mode = 'stat_align'
                        st.warning('由于频率转换跨度较大（如日转月），"值对齐"模式已禁用，自动切换到"统计对齐"。')

                current_alignment_mode_code = session_state.get('ts_tool_alignment_mode', 'stat_align')
                if current_alignment_mode_code not in alignment_mode_options.values(): 
                    if alignment_mode_options:
                        first_option_value = list(alignment_mode_options.values())[0]
                        session_state.ts_tool_alignment_mode = first_option_value
                        current_alignment_mode_code = first_option_value

                radio_options_keys = list(alignment_mode_options.keys())
                radio_options_values = list(alignment_mode_options.values())
                default_radio_idx = 0
                if current_alignment_mode_code in radio_options_values:
                    default_radio_idx = radio_options_values.index(current_alignment_mode_code)
                elif radio_options_values:
                    default_radio_idx = 0
                    session_state.ts_tool_alignment_mode = radio_options_values[0]

                selected_alignment_mode_display = st.radio(
                    label="选择对齐模式",
                    options=radio_options_keys,
                    index=default_radio_idx, 
                    key="alignment_mode_radio_high_to_low",
                    label_visibility="collapsed"
                )
                session_state.ts_tool_alignment_mode = alignment_mode_options[selected_alignment_mode_display]

                if session_state.ts_tool_alignment_mode == 'stat_align':
                    st.markdown("**选择聚合方法 (用于统计对齐):**")
                    aggregation_methods_map = {
                        "第一个值 (First)": "first", "最后一个值 (Last)": "last",
                        "平均值 (Mean)": "mean", "中位数 (Median)": "median",
                        "总和 (Sum)": "sum", "最小值 (Min)": "min", "最大值 (Max)": "max"
                    }
                    agg_method_display_options = list(aggregation_methods_map.keys())
                    current_agg_display = session_state.get('ts_tool_selected_agg_method_display', "最后一个值 (Last)")
                    if current_agg_display not in agg_method_display_options: 
                        current_agg_display = "最后一个值 (Last)"
                        session_state.ts_tool_selected_agg_method_display = current_agg_display
                    idx_agg = agg_method_display_options.index(current_agg_display)
                    selected_agg_method_display_val = st.selectbox(
                        label="选择聚合方法",
                        options=agg_method_display_options,
                        index=idx_agg,
                        key="align_aggregation_method_selector_high_to_low",
                        label_visibility="collapsed"
                    )
                    session_state.ts_tool_selected_agg_method_display = selected_agg_method_display_val
                    session_state.ts_tool_selected_agg_method_code = aggregation_methods_map[selected_agg_method_display_val]
                else: 
                    if 'ts_tool_selected_agg_method_code' in session_state: del session_state.ts_tool_selected_agg_method_code
                    if 'ts_tool_selected_agg_method_display' in session_state: del session_state.ts_tool_selected_agg_method_display
            
            # 频率补全选项，对所有情况都可见，无论是高频转低频、低频转高频还是同频率调整
            st.markdown("---") 
            step_title = "**额外操作: 频率补全**"
            
            st.markdown(step_title)
            
            # 根据频率转换类型提供不同的默认值和帮助信息
            is_low_to_high_or_same = session_state.get('ts_tool_is_low_to_high_conversion', False) or session_state.get('ts_tool_is_same_frequency_conversion', False)
            is_high_to_low = session_state.get('ts_tool_is_high_to_low_conversion', False)
            
            # 设置默认值：低频转高频和同频率默认勾选，高频转低频默认不勾选
            default_completion_value = session_state.get('ts_tool_complete_time_index_fc_moved', True if is_low_to_high_or_same else False)
            
            # 根据不同场景提供不同的帮助信息
            help_text_base = f"如果勾选，将尝试根据目标频率 ({session_state.get('ts_tool_target_alignment_frequency_display', '未指定')}) 补全时间序列中的缺失日期/时间点。"
            if is_high_to_low:
                help_text = f"{help_text_base} 注意：在高频转低频情况下，补全会在执行对齐操作后进行。"
            else:
                help_text = help_text_base
                
            session_state.ts_tool_complete_time_index_fc_moved = st.checkbox(
                "启用时间序列频率补全",
                value=default_completion_value,
                key="complete_time_index_checkbox_moved_v2",
                help=help_text
            )
            
            display_target_freq_for_caption = session_state.get('ts_tool_target_alignment_frequency_display') or '未设定'
            st.caption(
                f"补全将基于当前选择的'目标对齐频率' ({display_target_freq_for_caption}) 进行。 "
                f"新插入行的数据列将填充为 NaN（缺失值），您可以在后续的'缺失数据处理'步骤中处理。"
            )
            

        if st.button("计算对齐/补全结果", key="apply_frequency_alignment_or_completion_button"):
            if session_state.ts_tool_data_processed is not None and not session_state.ts_tool_data_processed.empty and \
               session_state.ts_tool_time_col_info and session_state.ts_tool_time_col_info.get('name') and \
               session_state.get('ts_tool_target_alignment_frequency_code'):

                time_col_name_align = session_state.ts_tool_time_col_info['name']
                original_freq_code_align = session_state.ts_tool_time_col_info.get('inferred_freq_code') # or 'freq' if that's the source of truth
                target_freq_code_align = session_state.get('ts_tool_target_alignment_frequency_code')
                alignment_mode_align = session_state.get('ts_tool_alignment_mode')
                stat_agg_method_align = session_state.get('ts_tool_selected_agg_method_code') if alignment_mode_align == 'stat_align' else None
                
                # --- DEBUG PRINT BEFORE SNAPSHOT ---
                debug_col_name_snapshot_check = "中国:生产率:焦炉:国内独立焦化厂(230家)"
                if debug_col_name_snapshot_check in session_state.ts_tool_data_processed.columns:
                    non_nan_count_before_snapshot = session_state.ts_tool_data_processed[debug_col_name_snapshot_check].notna().sum()
                    print(f"DEBUG_SNAPSHOT_CHECK: Non-NaN count for '{debug_col_name_snapshot_check}' in ts_tool_data_processed BEFORE snapshot: {non_nan_count_before_snapshot}")
                else:
                    print(f"DEBUG_SNAPSHOT_CHECK: Column '{debug_col_name_snapshot_check}' not found in ts_tool_data_processed BEFORE snapshot.")
                # --- END DEBUG PRINT ---

                session_state.ts_tool_data_processed_before_alignment_or_completion = session_state.ts_tool_data_processed.copy()
                if session_state.get('ts_tool_data_processed_FULL') is not None:
                    session_state.ts_tool_data_processed_FULL_before_alignment_or_completion = session_state.ts_tool_data_processed_FULL.copy()
                # else: # Ensure the snapshot for FULL is always created, even if empty or same as processed
                #     session_state.ts_tool_data_processed_FULL_before_alignment_or_completion = session_state.ts_tool_data_processed.copy() 

                # 准备进度提示信息
                operation_name = "频率对齐" if not session_state.get('ts_tool_complete_time_index_fc_moved', False) else "频率对齐和补全"
                with st.spinner(f"正在执行{operation_name}..."):
                    alignment_results = perform_frequency_alignment(
                        df_to_align=session_state.ts_tool_data_processed.copy(), # Main data to align
                        time_col_name=time_col_name_align,
                        original_pandas_freq_code=original_freq_code_align,
                        target_pandas_freq_code=target_freq_code_align,
                        alignment_mode=alignment_mode_align,
                        selected_stat_agg_method_code=stat_agg_method_align,
                        df_full_to_align=session_state.get('ts_tool_data_processed_FULL'), # Pass the FULL version
                        df_before_align_for_stats=session_state.ts_tool_data_processed_before_alignment_or_completion, 
                        time_col_info_for_stats=session_state.ts_tool_time_col_info.copy(),
                        selected_align_freq_display=session_state.get('ts_tool_target_alignment_frequency_display'),
                        selected_agg_method_display=session_state.get('ts_tool_selected_agg_method_display') if alignment_mode_align == 'stat_align' else alignment_mode_align,
                        # 添加频率补全参数
                        complete_time_index=session_state.get('ts_tool_complete_time_index_fc_moved', False)
                    )
                
                session_state.ts_tool_alignment_results = alignment_results
                session_state.ts_tool_show_alignment_report = True # Show the report area
                session_state.ts_tool_aligned_data_preview = alignment_results.get('aligned_df')

                if alignment_results:
                    if alignment_results.get('status_type') == 'success':
                        st.success(alignment_results.get('status_message', "频率对齐成功完成。"))
                        session_state.ts_tool_data_processed = alignment_results.get('aligned_df')
                        # Update _FULL only if alignment was successful and returned data
                        if alignment_results.get('aligned_df_full') is not None:
                             session_state.ts_tool_data_processed_FULL = alignment_results.get('aligned_df_full')
                        session_state.ts_tool_alignment_report_items = alignment_results.get('alignment_report_items', [])
                        session_state.ts_tool_show_frequency_analysis_results = False # Hide old freq analysis
                        # Potentially update time_col_info based on new_time_col_info from results
                        if alignment_results.get('new_time_col_info'):
                            session_state.ts_tool_time_col_info = alignment_results['new_time_col_info']
                        st.session_state.force_missing_ui_snapshot_refresh_flag = True # <<< SET FLAG
                        print("[TIME_COLUMN_UI] Set force_missing_ui_snapshot_refresh_flag = True after frequency alignment.")
                        st.rerun()

                        # --- MORE DETAILED DEBUG: After alignment, check session_state.ts_tool_aligned_data ---
                        if session_state.ts_tool_aligned_data is not None:
                            debug_col_name_ui_check = "中国:生产率:焦炉:国内独立焦化厂(230家)"
                            target_date_ui_check = pd.to_datetime('2020-01-31')
                            print(f"DEBUG_UI_ALIGNED_VAL_CHECK: Attempting to check {debug_col_name_ui_check} at {target_date_ui_check.date()} in session_state.ts_tool_aligned_data")
                            if debug_col_name_ui_check in session_state.ts_tool_aligned_data.columns:
                                print(f"DEBUG_UI_ALIGNED_VAL_CHECK: Column '{debug_col_name_ui_check}' FOUND in session_state.ts_tool_aligned_data.columns")
                                if target_date_ui_check in session_state.ts_tool_aligned_data.index:
                                    print(f"DEBUG_UI_ALIGNED_VAL_CHECK: Date '{target_date_ui_check.date()}' FOUND in session_state.ts_tool_aligned_data.index")
                                    aligned_val_check = session_state.ts_tool_aligned_data.loc[target_date_ui_check, debug_col_name_ui_check]
                                    print(f"DEBUG_UI_ALIGNED_VAL_CHECK for '{debug_col_name_ui_check}' at {target_date_ui_check.date()}: VALUE = {aligned_val_check}")
                                else:
                                    print(f"DEBUG_UI_ALIGNED_VAL_CHECK: Date '{target_date_ui_check.date()}' NOT FOUND in session_state.ts_tool_aligned_data.index")
                                    print(f"DEBUG_UI_ALIGNED_VAL_CHECK: Available index: {session_state.ts_tool_aligned_data.index}") # Log available index
                            else:
                                print(f"DEBUG_UI_ALIGNED_VAL_CHECK: Column '{debug_col_name_ui_check}' NOT FOUND in session_state.ts_tool_aligned_data.columns")
                                print(f"DEBUG_UI_ALIGNED_VAL_CHECK: Available columns: {session_state.ts_tool_aligned_data.columns.tolist()}") # Log available columns
                        else:
                            print(f"DEBUG_UI_ALIGNED_VAL_CHECK: session_state.ts_tool_aligned_data is None after alignment call.")
                        # --- END DEBUG ---

                    else:
                        st.error(alignment_results.get('status_message', "频率对齐时发生错误。"))
                        session_state.ts_tool_show_alignment_report = False 
                else:
                    st.error("频率对齐操作失败或未返回结果。")
                    session_state.ts_tool_show_alignment_report = False
            
            elif session_state.ts_tool_data_processed is None or session_state.ts_tool_data_processed.empty:
                st.warning("数据为空，无法执行频率对齐。")
            elif not session_state.ts_tool_time_col_info or not session_state.ts_tool_time_col_info.get('name'):
                st.error("未能识别有效的时间列，无法执行频率对齐。")
            elif not session_state.get('ts_tool_target_alignment_frequency_code'):
                st.error("未选择目标对齐频率，无法执行频率对齐。")
            else:
                st.info("请确保所有频率对齐的参数已正确设置。")


    with col_right:
        # --- Display Frequency Analysis Results (conditionally) ---
        if session_state.get('ts_tool_show_alignment_report') and session_state.get('ts_tool_alignment_report_items'):
            report_items = session_state.get('ts_tool_alignment_report_items')
            st.markdown("##### 频率对齐结果")
            if not report_items:
                st.info("频率对齐已执行，但未生成详细报告条目或没有需要报告的特殊事件。")
            else:  # Correctly indented else for "if not report_items"
                target_freq_overall = "N/A"
                agg_method_overall = "N/A"
                # Safety check before accessing report_items[0]
                if report_items and isinstance(report_items, list) and len(report_items) > 0:
                    if report_items[0].get('target_freq_display'):
                        target_freq_overall = report_items[0]['target_freq_display']
                    if report_items[0].get('agg_method_display'):
                        agg_method_overall = report_items[0]['agg_method_display']
                
                st.caption(f"对齐目标频率: **{target_freq_overall}**, 使用逻辑: **{agg_method_overall}**")
                
                markdown_report_list = []
                # <<< 新增外部循环调试打印 >>>
                print(f"DEBUG_MARKDOWN_RENDER: Preparing to render {len(report_items)} report items via Markdown.")
                for item_idx, item_data in enumerate(report_items): # This loop is fine due to the "else" handling empty report_items
                    # <<< 新增内部循环调试打印 >>>
                    print(f"DEBUG_MARKDOWN_RENDER: Item {item_idx} data for Markdown: {item_data}")

                    # <<< MODIFICATION: Only process items that are summary statistics >>>
                    if not (isinstance(item_data, dict) and \
                            'original_non_nan' in item_data and \
                            'aligned_non_nan' in item_data):
                        print(f"DEBUG_MARKDOWN_RENDER: Item {item_idx} is NOT a summary item or is malformed, skipping. Keys: {list(item_data.keys()) if isinstance(item_data, dict) else 'Not a dict'}")
                        continue
                    # <<< END MODIFICATION >>>

                    line = f"- **列 '{item_data.get('column', '未知列')}'**: "
                    original_val = item_data.get('original_non_nan', 0) # Already confirmed to exist by check above
                    aligned_val = item_data.get('aligned_non_nan', 0)   # Already confirmed to exist by check above
                    
                    # <<< 新增值获取调试打印 >>>
                    print(f"DEBUG_MARKDOWN_RENDER: Item {item_idx} - original_val: {original_val} (type: {type(original_val)}), aligned_val: {aligned_val} (type: {type(aligned_val)})")

                    line += f"原始有效值: {original_val}, "
                    line += f"对齐后有效值: {aligned_val}"
                    
                    if item_data.get('original_non_nan', 0) > 0 and item_data.get('aligned_non_nan', 0) == 0:
                         line += f" (注意: 此列聚合后在新频率下均为空值)"
                    elif item_data.get('original_non_nan', 0) == 0 and item_data.get('aligned_non_nan', 0) == 0:
                         line += f" (原始数据为空，对齐后仍为空)"
                    
                    custom_event = item_data.get('custom_event_info')
                    if custom_event:
                        event_time_str = custom_event.get('time', '未知时间')
                        reason_str = custom_event.get('reason', '未提供原因')
                        event_type_str = custom_event.get('event_type', '未知事件')
                        
                        # Customize message based on event_type for clarity
                        if event_type_str == 'value_align_pushed_to_prev_period':
                            line += f"\n  - <span style='color:DodgerBlue;'>推值事件</span>: {reason_str}" 
                        elif event_type_str == 'value_align_used_earliest_due_to_prev_empty_output':
                            line += f"\n  - <span style='color:orange;'>特殊调整</span>: {reason_str}"
                        elif event_type_str == 'value_align_used_latest_in_period': # Added based on previous logic
                             line += f"\n  - <span style='color:MediumSeaGreen;'>取值规则</span>: {reason_str}"
                        else: # Generic fallback for other potential custom events
                            line += f"\n  - <span style='color:green;'>自定义事件 ({event_type_str})</span>: {reason_str} (时间: {event_time_str})"

                    markdown_report_list.append(line)
                
                if markdown_report_list:
                    st.markdown("\n".join(markdown_report_list), unsafe_allow_html=True)
                else: 
                    st.info("已处理对齐，但未能从报告条目生成任何可显示的详细信息。")

        elif session_state.get('ts_tool_show_frequency_analysis_results') and session_state.get('ts_tool_frequency_analysis_results'):
            analysis_results_to_display = session_state.ts_tool_frequency_analysis_results
            
            if not analysis_results_to_display or not analysis_results_to_display.get('column_analysis'):
                st.info("频率分析未返回结果或未能分析任何数据列。")
            else: # Corrected indentation for line 928
                st.markdown("##### 各列频率分析结果")
                num_cols_analyzed = 0
                markdown_results_list = []
                for col, freq_info in analysis_results_to_display.get('column_analysis', {}).items():
                    display_freq_group = freq_info.get('freq_group', '未能推断')
                    display_pandas_code = freq_info.get('pandas_freq_code', '-')
                    result_line = f"- 列 '{col}' 推断频率: **{display_freq_group}** (Code: {display_pandas_code})"
                    markdown_results_list.append(result_line)
                    num_cols_analyzed += 1
                
                if markdown_results_list:
                    st.markdown("\n".join(markdown_results_list))
                
                if num_cols_analyzed > 0: 
                    st.success(f"已完成对 {num_cols_analyzed} 个数据列的频率分析。")
                    overall_group = analysis_results_to_display.get('overall_inferred_freq_group', 'N/A')
                    overall_code = analysis_results_to_display.get('overall_pandas_freq_code', 'N/A')
                    st.caption(f"整体推断频率组: {overall_group}, Pandas频率码: {overall_code}")
                elif not markdown_results_list: # Only show if no specific column results were displayed
                    st.info("频率分析结果中未找到具体列的频率信息。")
      