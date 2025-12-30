import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- Static configuration (must match with ble_rx.py) ---
EMG_WINDOW_SAMPLES = 400 
WINDOWS_PER_CAPTURE = 80 
REST_START_WINDOWS = 20  
GESTURE_END_WINDOWS = 60 

DATA_ROOT = 'data'
RAW_BASE_DIR = os.path.join(DATA_ROOT, 'raw_data')
LABEL_BASE_DIR = os.path.join(DATA_ROOT, 'labeled_dataset')

def plot_emg_capture(df_raw, raw_label, main_label):
    """
    Visualize a RAW capture of EMG wiht marked zones (rest - gesture - rest).
    """
    if df_raw.empty:
        print("Error: empty dataframe")
        return

    # 1. flat the dataframe
    numeric_data = df_raw.drop(columns=df_raw.columns[[0, 1]], errors='ignore') 
    numeric_data = numeric_data.select_dtypes(include=[np.number])
    
    y_values = numeric_data.values.flatten()
    x_values = np.arange(len(y_values))
    window_size = EMG_WINDOW_SAMPLES
    
    # 2. create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_values, y_values, label='Señal EMG', color='#1f77b4', linewidth=0.8)
    ax.set_ylabel("Valor ADC")
    
    title_text = f"QC: {raw_label} (Label Gesto: {main_label})."
    ax.set_title(title_text)
    ax.set_xlabel("Muestras")
    
    # 3. draw segmented zones (rest - gesture - rest)
    start_rest_end_x = REST_START_WINDOWS * window_size
    gesture_end_x = GESTURE_END_WINDOWS * window_size

    for x in range(window_size, len(x_values), window_size):
        ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5, alpha=0.4)
    
    ax.axvspan(0, start_rest_end_x, alpha=0.2, color='green', label='Reposo Inicial (2)')
    ax.axvspan(start_rest_end_x, gesture_end_x, alpha=0.2, color='orange', label=f'Gesture ({main_label})')
    ax.axvspan(gesture_end_x, WINDOWS_PER_CAPTURE * window_size, alpha=0.2, color='green',label='Reposo Inicial (2)')

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(x_values))
    plt.tight_layout()
    
    plt.show()

# =========================================================================
# Segment and save
# =========================================================================

def _save_emg_segment(df_segment, subject_output_dir, raw_label, timestamp_part, header, data_columns, prefix_label, name_suffix, description):
    """Guarda un segmento específico de datos en un archivo CSV individual."""
    
    # 1. create filename
    if name_suffix == 'gesture':
        # Gesto: Mantiene el RAW_LABEL original (ej: 1_20251212_140000.csv)
        filename = f"{raw_label}.csv"
    else:
        # Rest: label '2' with suffix _i (initial rest of the capture) o _t (final rest of the capture)
        # e.g: 0_20251212_140000_i.csv
        filename = f"{prefix_label}_{timestamp_part}_{name_suffix}.csv"
        
    output_path = os.path.join(subject_output_dir, filename)

    # 2. create the dataframe
    df_out = pd.DataFrame(columns=header)
    df_out['Label'] = prefix_label
    df_out['Timestamp'] = df_segment['Timestamp']
    df_out[data_columns] = df_segment[data_columns]
    
    # 3. save file
    try:
        df_out.to_csv(output_path, index=False)
        print(f"  -> [OK] file {description} saved in: {filename}")
    except Exception as e:
        print(f"  -> [ERROR] {filename}: {e}")

def segment_and_save_capture(df_raw, subject_name, raw_label, main_label):
    """
    Segment the capture in (0-20, 20-60, 60-80) and save them in three files 2 for rest and 1 for gesture
    """

    # 1. Prepare output path
    subject_output_dir = os.path.join(LABEL_BASE_DIR, subject_name, 'segmented_emg')
    if not os.path.exists(subject_output_dir): 
        os.makedirs(subject_output_dir)

    data_columns = df_raw.columns[2:] # ADC values of the windows
    header = ["Label", "Timestamp"] + data_columns.tolist()

    # split timestamp and label
    raw_label_parts = raw_label.split('_', 1)
    if len(raw_label_parts) < 2:
        print(f"Error: bad format ('{raw_label}')")
        return
        
    timestamp_part = raw_label_parts[1] # '20251212_140000'

    # --- 1. Initial rest segment (windows 0-20) ---
    df_rest_start = df_raw.iloc[0:REST_START_WINDOWS]
    _save_emg_segment(
        df_rest_start, 
        subject_output_dir, raw_label, timestamp_part, header, data_columns, 
        prefix_label='2', name_suffix='i', description="Initial rest (0-20)"
    )

    # --- 2. Gesture segment (windows 20-60) ---
    df_gesture = df_raw.iloc[REST_START_WINDOWS:GESTURE_END_WINDOWS]
    _save_emg_segment(
        df_gesture, 
        subject_output_dir, raw_label, timestamp_part, header, data_columns, 
        prefix_label=main_label, name_suffix='gesture', description=f"Gesto Activo ({main_label}, 20-60)"
    )
    
    # --- 3. Final rest segment (windows 60-80) ---
    df_rest_end = df_raw.iloc[GESTURE_END_WINDOWS:WINDOWS_PER_CAPTURE]
    _save_emg_segment(
        df_rest_end, 
        subject_output_dir, raw_label, timestamp_part, header, data_columns, 
        prefix_label='2', name_suffix='t', description="Final rest transition (60-80)"
    )

    print(f"\n[OK] The segments for the capture {raw_label} have been saved")

def rename_and_discard(current_path, raw_label):
    """Rename a RAW capture with prefix 'XX_' to mark it as discarded."""
    
    new_raw_label = f"XX_{raw_label}"
    new_path = os.path.join(os.path.dirname(current_path), f"{new_raw_label}.csv")
    
    try:
        os.rename(current_path, new_path)
        print(f"[Discarded] renamed_file: {new_raw_label}.csv")
        return True
    except OSError as e:
        print(f"Error: {e}")
        return False

# =========================================================================
# Main menu
# =========================================================================

def get_subject_name():
    """Get the subject name"""
    while True:
        subject_name = input("\n Enter a subject name (ej: S0) o 'q' to quit: ").upper().strip()
        if subject_name == 'Q':
            return None
        
        subject_emg_dir = os.path.join(RAW_BASE_DIR, subject_name, 'EMG')
        
        if os.path.isdir(subject_emg_dir):
            return subject_name
        else:
            print(f"Error: no subject '{subject_name}' en {subject_emg_dir}.")

def find_unvalidated_captures(subject_name):
    """Find files without the prefix 'XX_'."""
    emg_dir = os.path.join(RAW_BASE_DIR, subject_name, 'EMG')
    
    # Get all CSV
    all_files = glob.glob(os.path.join(emg_dir, "*.csv"))
    
    # Filter files with 'XX_'
    unvalidated_labels = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if not filename.startswith('XX_'):
            unvalidated_labels.append(filename.replace('.csv', ''))
            
    return sorted(unvalidated_labels)

def process_single_capture(subject_name, raw_label_to_plot):
    """Process a single file (Segment/Rename/Pass)."""
    
    subject_emg_dir = os.path.join(RAW_BASE_DIR, subject_name, 'EMG')
    raw_file_path = os.path.join(subject_emg_dir, f"{raw_label_to_plot}.csv")
    
    print(f"\n Capture: {raw_label_to_plot}")

    if not os.path.exists(raw_file_path):
        print(f"Error: file '{raw_label_to_plot}.csv' is missing in {raw_file_path}")
        return 'E' 

    try:
        df_raw = pd.read_csv(raw_file_path)
        
        main_label = '?'
        try:
             main_label = raw_label_to_plot.split('_')[0]
             if main_label not in ['1', '2'] and not raw_label_to_plot.startswith('XX_'):
                 print("Label not valid")
                 main_label = '?'
        except:
             main_label = '?'

        # Visualize the capture
        plot_emg_capture(df_raw, raw_label_to_plot, main_label)
        
        # process the capture
        while True:
            action = input(
                "\n Acción (S/R/P): "
                "S (Segment and save), "
                "R (Rename as no valid capture), "
                "P (pass): "
            ).upper().strip()
            
            if action == 'S':
                segment_and_save_capture(df_raw, subject_name, raw_label_to_plot, main_label)
                return 'S' # Segmented
            
            elif action == 'R':
                rename_and_discard(raw_file_path, raw_label_to_plot)
                return 'R' # Renamed
                
            elif action == 'P':
                print(f"Capture '{raw_label_to_plot}' passed")
                return 'P' # Passed
            
            else:
                print("No valid option")

    except Exception as e:
        print(f"Error al procesar la captura {raw_label_to_plot}: {e}")
        return 'E' # Error

def main():
    print("==========================================================")
    print("          EMG Visualizer & Segmentation TOOL       ")
    print("==========================================================")
    
    current_subject = None
    
    while True: # subject loop
        
        # 1. Subject selection
        if current_subject is None:
            current_subject = get_subject_name()
            if current_subject is None:
                print("Quit")
                return 

        # 2. Processing mode
        while True:
            print(f"\n[Current subject: {current_subject}]")
            mode_choice = input(
                "Mode selection:\n"
                " I (Iterative for all subject captures)\n"
                " E (Specific capture)\n"
                " C (Change the subject)\n"
                " Q (Quit)\n"
                "Option (I/E/C/Q): "
            ).upper().strip()

            if mode_choice == 'Q':
                return
            
            if mode_choice == 'C':
                current_subject = None
                break # Go to subject select

            # --- Iterative mode ---
            elif mode_choice == 'I':
                unvalidated_captures = find_unvalidated_captures(current_subject)
                
                if not unvalidated_captures:
                    print(f"\n All the captures have been discarded")
                    continue 

                print(f"\n[Iterative mode] Processing {len(unvalidated_captures)}")
                
                for i, raw_label_to_plot in enumerate(unvalidated_captures):
                    print(f"\n--- CAPTURE [{i+1}/{len(unvalidated_captures)}] ---")
                    process_single_capture(current_subject, raw_label_to_plot)
                
                print("\n[End interarive mode]")
                continue # go to mode selection

            # --- Specific mode ---
            elif mode_choice == 'E':
                while True:
                    raw_label_to_plot = input("\n[Specific mode] Enter filename (e.g: 1_2025...) or 'M' to return mode selection: ").strip()
                    if raw_label_to_plot.upper() == 'M':
                        break # # Go to mode selection

                    if raw_label_to_plot.startswith('XX_'):
                        print("Warning: discarded file, continue anyway? (y/n):")
                        if input().lower() != 'y': continue
                    
                    # Process file
                    process_single_capture(current_subject, raw_label_to_plot)
                    
                    # --- MENÚ DE ACCIÓN POST-ESPECÍFICA (Nivel 3) ---
                    post_action = input(
                        "\n[File processed]\n"
                        " S (Analize other file from same subject)\n"
                        " M (Back to mode selection)\n"
                        "Opción (S/M): "
                    ).upper().strip()
                    
                    if post_action == 'M':
                        break # Go to mode selection

            else:
                print("Invalid option")

if __name__ == "__main__":
    main()