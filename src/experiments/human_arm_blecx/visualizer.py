import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- CONFIGURACIÓN ---
DATA_FOLDER = 'data'
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'no_labels') # <<-- RUTA CORREGIDA
EMG_WINDOW_SAMPLES = 400
IMU_WINDOW_SAMPLES = 10
RAD_TO_DEG = 57.2957795

# --- PARÁMETROS DE CAPTURA/SEGMENTACIÓN (Deben coincidir con el script de captura) ---
WINDOWS_PER_CAPTURE = 80 
REST_START_WINDOWS = 20     # Ventanas 0 a 9 (Total 10)
GESTURE_END_WINDOWS = 60    # Ventanas 10 a 39 (Total 30)

def main():
    # 1. VALIDAR ARGUMENTOS
    if len(sys.argv) < 3:
        print("\n  Error de Uso.")
        print(f"   Sintaxis: python {sys.argv[0]} <LABEL_RAW> <MODO>")
        print(f"   NOTA: <LABEL_RAW> es el identificador único (ej: 1_20251210_091500)")
        print(f"   <MODO> es 'emg', 'accel', 'gyro', 'angles', etc.")
        print(f"   Ejemplos:")
        print(f"     python {sys.argv[0]} 1_20251210_091500 emg")
        print(f"     python {sys.argv[0]} 2_20251210_091500 accel")
        return

    # label_arg ahora es el identificador único RAW (ej: 1_20251210_091500)
    label_raw_arg = sys.argv[1] 
    mode_arg = sys.argv[2].lower()
    
    # El label principal (1 o 2) se extrae del label_raw_arg para el título
    try:
        main_label_arg = label_raw_arg.split('_')[0]
    except:
        main_label_arg = '?'

    slice_start = 0 
    slice_end = WINDOWS_PER_CAPTURE

    # Rutas de archivo CORREGIDAS
    # Usamos RAW_DATA_FOLDER para buscar el archivo capturado completo.
    emg_file = os.path.join(RAW_DATA_FOLDER, f"{label_raw_arg}_EMG.csv")
    imu_file = os.path.join(RAW_DATA_FOLDER, f"{label_raw_arg}_IMU.csv")

    
    x_values = np.array([])
    window_size = 0
    df_slice = pd.DataFrame()
    
    try:
        # ==========================================
        # MODO: EMG
        # ==========================================
        if mode_arg == 'emg':
            if not os.path.exists(emg_file): raise FileNotFoundError(f"No existe: {emg_file}")
            
            df = pd.read_csv(emg_file)
            
            # --- LÓGICA DE CORTE (SLICING) PARA EMG ---
            df_slice = df.iloc[slice_start:slice_end]

            # Procesar datos (Flatten)
            # Asumimos que las 2 primeras columnas son Raw_Label y Timestamp
            numeric_data = df_slice.drop(columns=df_slice.columns[[0, 1]], errors='ignore') 
            numeric_data = numeric_data.select_dtypes(include=[np.number])
            
            y_values = numeric_data.values.flatten()
            x_values = np.arange(len(y_values))
            window_size = EMG_WINDOW_SAMPLES
            
            # --- FIGURA 1: RAW CAPTURE ---
            fig_raw, ax_raw = plt.subplots(figsize=(14, 6))
            ax_raw.plot(x_values, y_values, label='Señal EMG', color='#1f77b4', linewidth=0.8)
            ax_raw.set_ylabel("Valor ADC")
            
            # Dibujar separadores de ventana y zonas de segmentación
            ax_raw.set_title(f"FIGURA 1: Captura RAW {label_raw_arg} (Total {WINDOWS_PER_CAPTURE} Ventanas)")
            ax_raw.set_xlabel("Muestras")
            
            # Dibujar separadores de ventana
            for x in range(window_size, len(x_values), window_size):
                ax_raw.axvline(x=x, color='red', linestyle='--', linewidth=0.5, alpha=0.4)
            
            # Dibujar zonas de segmentación (verticales)
            # Zona de Reposo Inicial (0)
            ax_raw.axvspan(0, REST_START_WINDOWS * window_size, alpha=0.2, color='green', label='Zona de Reposo Inicial (0)')
            # Zona de Gesto Principal (1/2)
            ax_raw.axvspan(REST_START_WINDOWS * window_size, GESTURE_END_WINDOWS * window_size, alpha=0.2, color='orange', label=f'Zona de Gesto Activo ({main_label_arg})')
            # Zona de Reposo Final (0)
            ax_raw.axvspan(GESTURE_END_WINDOWS * window_size, WINDOWS_PER_CAPTURE * window_size, alpha=0.2, color='green')

            ax_raw.grid(True, linestyle=':', alpha=0.6)
            ax_raw.legend(loc='upper right')
            ax_raw.set_xlim(0, len(x_values))
            plt.tight_layout()
            
            # --- FIGURA 2: SEGMENTACIÓN DE CHUNKS ---
            
            # Obtener las ventanas de EMG por segmento
            df_rest_start = df_slice.iloc[0:REST_START_WINDOWS]
            df_gesture = df_slice.iloc[REST_START_WINDOWS:GESTURE_END_WINDOWS]
            df_rest_end = df_slice.iloc[GESTURE_END_WINDOWS:WINDOWS_PER_CAPTURE]
            
            # Función auxiliar para aplanar y obtener valores
            def get_segment_data(segment_df):
                if segment_df.empty: return np.array([]), 0
                numeric_data = segment_df.drop(columns=segment_df.columns[[0, 1]], errors='ignore')
                numeric_data = numeric_data.select_dtypes(include=[np.number])
                y = numeric_data.values.flatten()
                x = np.arange(len(y))
                return x, y
            
            # Crear figura con 3 subplots
            fig_seg, axs_seg = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
            fig_seg.suptitle(f"FIGURA 2: Segmentación de la Captura RAW {label_raw_arg} (Label {main_label_arg})", fontsize=16)

            # 1. Reposo Inicial (0)
            x_start, y_start = get_segment_data(df_rest_start)
            axs_seg[0].plot(x_start, y_start, color='green', linewidth=0.8)
            axs_seg[0].set_title(f"Segmento 1: Reposo Inicial (Label 0) - Ventanas 0 a {REST_START_WINDOWS - 1}")
            axs_seg[0].set_ylabel("Valor ADC")
            axs_seg[0].grid(True, linestyle=':', alpha=0.6)
            
            # 2. Gesto Activo (main_label)
            x_gesture, y_gesture = get_segment_data(df_gesture)
            axs_seg[1].plot(x_gesture, y_gesture, color='orange', linewidth=0.8)
            axs_seg[1].set_title(f"Segmento 2: Gesto Principal (Label {main_label_arg}) - Ventanas {REST_START_WINDOWS} a {GESTURE_END_WINDOWS - 1}")
            axs_seg[1].set_ylabel("Valor ADC")
            axs_seg[1].grid(True, linestyle=':', alpha=0.6)
            
            # 3. Reposo Final (0)
            x_end, y_end = get_segment_data(df_rest_end)
            axs_seg[2].plot(x_end, y_end, color='green', linewidth=0.8)
            axs_seg[2].set_title(f"Segmento 3: Reposo Final (Label 0) - Ventanas {GESTURE_END_WINDOWS} a {WINDOWS_PER_CAPTURE - 1}")
            axs_seg[2].set_xlabel("Muestras")
            axs_seg[2].set_ylabel("Valor ADC")
            axs_seg[2].grid(True, linestyle=':', alpha=0.6)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para el suptitle
            

        # ==========================================
        # MODO: IMU (Accel, Gyro, Angles...)
        # ==========================================
        elif mode_arg in ['accel', 'gyro', 'angles', 'pitch', 'roll', 'temp']:
            if not os.path.exists(imu_file): raise FileNotFoundError(f"No existe: {imu_file}")
            
            df = pd.read_csv(imu_file)
            
            # --- LÓGICA DE CORTE (SLICING) PARA IMU ---
            start_row = slice_start * IMU_WINDOW_SAMPLES
            end_row = slice_end * IMU_WINDOW_SAMPLES
            
            df_slice = df.iloc[start_row:end_row]

            x_values = np.arange(len(df_slice))
            window_size = IMU_WINDOW_SAMPLES
            
            fig, ax = plt.subplots(figsize=(14, 6))

            # Graficar según sub-modo (lógica sin cambios)
            if mode_arg == 'accel':
                ax.plot(x_values, df_slice['accel_x'], 'r', label='Ax', alpha=0.8)
                ax.plot(x_values, df_slice['accel_y'], 'g', label='Ay', alpha=0.8)
                ax.plot(x_values, df_slice['accel_z'], 'b', label='Az', alpha=0.8)
                ax.set_ylabel("Fuerza G")

            elif mode_arg == 'gyro':
                ax.plot(x_values, df_slice['gyro_x'], 'r', label='Gx', alpha=0.8)
                ax.plot(x_values, df_slice['gyro_y'], 'g', label='Gy', alpha=0.8)
                ax.plot(x_values, df_slice['gyro_z'], 'b', label='Gz', alpha=0.8)
                ax.set_ylabel("Velocidad Angular (dps)")

            elif mode_arg in ['angles', 'pitch', 'roll']:
                ax.plot(x_values, df_slice['pitch'] * RAD_TO_DEG, 'purple', label='Pitch')
                ax.plot(x_values, df_slice['roll'] * RAD_TO_DEG, 'orange', label='Roll')
                ax.set_ylabel("Grados (°)")
            
            elif mode_arg == 'temp':
                ax.plot(x_values, df_slice['temp'], 'orange', label='Temp')
                ax.set_ylabel("Celsius")
            
            for x in range(window_size, len(x_values), window_size):
                ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5, alpha=0.4)
                
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            ax.set_xlim(0, len(x_values))
            plt.tight_layout()


        else:
            print(f" Modo '{mode_arg}' no reconocido.")
            return

    except Exception as e:
        print(f" Error: {e}")
        return
    
    # Mostrar todas las figuras
    plt.show()

if __name__ == "__main__":
    main()