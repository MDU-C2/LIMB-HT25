# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import sys
# import os

# # --- CONFIGURATION ---
# DATA_FOLDER = 'data'  # Folder where your CSVs are saved

# # Sample counts per window (Must match your capture script)
# EMG_WINDOW_SAMPLES = 400
# IMU_WINDOW_SAMPLES = 10

# RAD_TO_DEG = 57.2957795

# def main():
#     # 1. VALIDATE ARGUMENTS
#     # Usage: python visualizer.py [label_name] [variable]
#     if len(sys.argv) < 3:
#         print("\n Usage Error.")
#         print(f"   Correct syntax: python {sys.argv[0]} <LABEL> <MODE>")
#         print(f"   Examples:")
#         print(f"     python {sys.argv[0]} fist emg")
#         print(f"     python {sys.argv[0]} fist accel")
#         print(f"     python {sys.argv[0]} fist gyro")
#         print(f"     python {sys.argv[0]} fist angles")
#         return

#     label_arg = sys.argv[1]       # e.g., "fist", "rest"
#     mode_arg = sys.argv[2].lower() # e.g., "emg", "accel"

#     # Construct file paths based on the label
#     emg_file = os.path.join(DATA_FOLDER, f"{label_arg}_EMG.csv")
#     imu_file = os.path.join(DATA_FOLDER, f"{label_arg}_IMU.csv")

#     # 2. SETUP PLOT
#     fig, ax = plt.subplots(figsize=(14, 6))
#     x_values = np.array([])
#     window_size = 0
    
#     try:
#         # ==========================================
#         # MODE: EMG (Electromyography)
#         # ==========================================
#         if mode_arg == 'emg':
#             if not os.path.exists(emg_file):
#                 raise FileNotFoundError(f"File not found: {emg_file}")

#             print(f"Loading EMG data from: {emg_file}...")
#             df = pd.read_csv(emg_file)
            
#             # TRICKY PART: The CSV has rows of 400 columns. 
#             # We need to drop the 'Label' and "flatten" the rest into a single 1D array.
#             numeric_data = df.drop(columns=['Label'])
#             y_values = numeric_data.values.flatten() # Connects all rows head-to-tail
            
#             x_values = np.arange(len(y_values))
#             window_size = EMG_WINDOW_SAMPLES
            
#             ax.plot(x_values, y_values, label='EMG Signal', color='#1f77b4', linewidth=0.8)
#             ax.set_title(f"EMG Signal - Label: '{label_arg}'")
#             ax.set_ylabel("ADC Value (0-4095)")

#         # ==========================================
#         # MODE: ACCELEROMETER (X, Y, Z)
#         # ==========================================
#         elif mode_arg == 'accel':
#             if not os.path.exists(imu_file):
#                 raise FileNotFoundError(f"File not found: {imu_file}")

#             print(f"Loading IMU data from: {imu_file}...")
#             df = pd.read_csv(imu_file)
#             x_values = np.arange(len(df))
#             window_size = IMU_WINDOW_SAMPLES

#             ax.plot(x_values, df['accel_x'], label='Accel X', color='r', alpha=0.8)
#             ax.plot(x_values, df['accel_y'], label='Accel Y', color='g', alpha=0.8)
#             ax.plot(x_values, df['accel_z'], label='Accel Z', color='b', alpha=0.8)
            
#             ax.set_title(f"Accelerometer Data - Label: '{label_arg}'")
#             ax.set_ylabel("G-Force (g)")

#         # ==========================================
#         # MODE: GYROSCOPE (X, Y, Z)
#         # ==========================================
#         elif mode_arg == 'gyro':
#             if not os.path.exists(imu_file):
#                 raise FileNotFoundError(f"File not found: {imu_file}")

#             print(f"Loading IMU data from: {imu_file}...")
#             df = pd.read_csv(imu_file)
#             x_values = np.arange(len(df))
#             window_size = IMU_WINDOW_SAMPLES

#             ax.plot(x_values, df['gyro_x'], label='Gyro X', color='r', alpha=0.8)
#             ax.plot(x_values, df['gyro_y'], label='Gyro Y', color='g', alpha=0.8)
#             ax.plot(x_values, df['gyro_z'], label='Gyro Z', color='b', alpha=0.8)
            
#             ax.set_title(f"Gyroscope Data - Label: '{label_arg}'")
#             ax.set_ylabel("Angular Velocity (dps)")

#         # ==========================================
#         # MODE: ANGLES (Pitch, Roll)
#         # ==========================================
#         elif mode_arg in ['angles', 'pitch', 'roll']:
#             if not os.path.exists(imu_file):
#                 raise FileNotFoundError(f"File not found: {imu_file}")

#             print(f"Loading IMU data from: {imu_file}...")
#             df = pd.read_csv(imu_file)
#             x_values = np.arange(len(df))
#             window_size = IMU_WINDOW_SAMPLES

#             # Convert Radians to Degrees for visualization
#             pitch_deg = df['pitch'] * RAD_TO_DEG
#             roll_deg = df['roll'] * RAD_TO_DEG

#             ax.plot(x_values, pitch_deg, label='Pitch', color='purple')
#             ax.plot(x_values, roll_deg, label='Roll', color='orange')
            
#             ax.set_title(f"Euler Angles - Label: '{label_arg}'")
#             ax.set_ylabel("Degrees (°)")

#         # ==========================================
#         # MODE: TEMPERATURE
#         # ==========================================
#         elif mode_arg == 'temp':
#             if not os.path.exists(imu_file):
#                 raise FileNotFoundError(f"File not found: {imu_file}")
            
#             df = pd.read_csv(imu_file)
#             x_values = np.arange(len(df))
#             window_size = IMU_WINDOW_SAMPLES
            
#             ax.plot(x_values, df['temp'], label='Temperature', color='orange')
#             ax.set_title(f"Temperature - Label: '{label_arg}'")
#             ax.set_ylabel("Celsius (°C)")

#         else:
#             print(f"Mode '{mode_arg}' not recognized.")
#             print("   Options: emg, accel, gyro, angles, temp")
#             return

#     except Exception as e:
#         print(f"Error processing data: {e}")
#         return

#     # 3. DRAW WINDOW SEPARATORS
#     # This draws a vertical red line at the end of every window
#     if len(x_values) > 0:
#         print(f"   -> Drawing window separators (Size: {window_size})...")
#         for x in range(window_size, len(x_values), window_size):
#             ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

#     # 4. FINALIZE PLOT
#     ax.grid(True, linestyle=':', alpha=0.6)
#     ax.legend(loc='upper right')
#     ax.set_xlabel("Sample Index")
#     ax.set_xlim(0, len(x_values))
    
#     plt.tight_layout()
#     print("Plot generated.")
#     plt.show()

# if __name__ == "__main__":
#     main()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- CONFIGURACIÓN ---
DATA_FOLDER = 'data'
EMG_WINDOW_SAMPLES = 400
IMU_WINDOW_SAMPLES = 10
RAD_TO_DEG = 57.2957795

def parse_slice_arg(arg_str):
    """
    Convierte strings como "5" o "10:20" en índices de inicio y fin.
    Retorna (start, end, is_range)
    """
    if ':' in arg_str:
        parts = arg_str.split(':')
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else None
        return start, end, True
    else:
        idx = int(arg_str)
        return idx, idx + 1, False

def main():
    # 1. VALIDAR ARGUMENTOS
    if len(sys.argv) < 3:
        print("\n  Error de Uso.")
        print(f"   Sintaxis: python {sys.argv[0]} <LABEL> <MODO> [RANGO_OPCIONAL]")
        print(f"   Ejemplos:")
        print(f"     python {sys.argv[0]} 1 emg          (Ver todo)")
        print(f"     python {sys.argv[0]} 1 emg 5        (Ver solo ventana 5)")
        print(f"     python {sys.argv[0]} 1 emg 10:20    (Ver ventanas 10 a 20)")
        return

    label_arg = sys.argv[1]
    mode_arg = sys.argv[2].lower()
    
    # Manejo del rango opcional
    slice_start = 0
    slice_end = None
    is_specific_slice = False

    if len(sys.argv) > 3:
        try:
            slice_start, slice_end, is_range = parse_slice_arg(sys.argv[3])
            is_specific_slice = True
            print(f"🔍 Filtrando: Ventanas {slice_start} a {slice_end if slice_end else 'Final'}")
        except ValueError:
            print(" El rango debe ser un número (ej: 5) o rango (ej: 10:20)")
            return

    # Rutas de archivo
    emg_file = os.path.join(DATA_FOLDER, f"{label_arg}_EMG.csv")
    imu_file = os.path.join(DATA_FOLDER, f"{label_arg}_IMU.csv")

    fig, ax = plt.subplots(figsize=(14, 6))
    x_values = np.array([])
    window_size = 0
    
    try:
        # ==========================================
        # MODO: EMG
        # ==========================================
        if mode_arg == 'emg':
            if not os.path.exists(emg_file): raise FileNotFoundError(f"No existe: {emg_file}")
            
            df = pd.read_csv(emg_file)
            
            # --- LÓGICA DE CORTE (SLICING) ---
            # En EMG, 1 fila del CSV = 1 Ventana
            if is_specific_slice:
                # Si slice_end es None, va hasta el final
                df = df.iloc[slice_start:slice_end] if slice_end else df.iloc[slice_start:]

            if df.empty:
                print(" El rango seleccionado está vacío o fuera de límites.")
                return

            # Procesar datos (Flatten)
            numeric_data = df.drop(columns=['Label'], errors='ignore') # 'errors' por si ya se borró o cambió nombre
            # Ojo: Asegurarse de borrar columnas de texto si hay más
            numeric_data = numeric_data.select_dtypes(include=[np.number])
            
            y_values = numeric_data.values.flatten()
            x_values = np.arange(len(y_values))
            window_size = EMG_WINDOW_SAMPLES
            
            ax.plot(x_values, y_values, label='Señal EMG', color='#1f77b4', linewidth=0.8)
            ax.set_ylabel("Valor ADC")

        # ==========================================
        # MODO: IMU (Accel, Gyro, Angles...)
        # ==========================================
        elif mode_arg in ['accel', 'gyro', 'angles', 'pitch', 'roll', 'temp']:
            if not os.path.exists(imu_file): raise FileNotFoundError(f"No existe: {imu_file}")
            
            df = pd.read_csv(imu_file)
            
            # --- LÓGICA DE CORTE (SLICING) ---
            # En IMU, 10 filas del CSV = 1 Ventana. Multiplicamos índices por 10.
            if is_specific_slice:
                start_row = slice_start * IMU_WINDOW_SAMPLES
                end_row = slice_end * IMU_WINDOW_SAMPLES if slice_end else None
                
                df = df.iloc[start_row:end_row] if end_row else df.iloc[start_row:]

            if df.empty:
                print(" El rango seleccionado está vacío o fuera de límites.")
                return

            x_values = np.arange(len(df))
            window_size = IMU_WINDOW_SAMPLES

            # Graficar según sub-modo
            if mode_arg == 'accel':
                ax.plot(x_values, df['accel_x'], 'r', label='Ax', alpha=0.8)
                ax.plot(x_values, df['accel_y'], 'g', label='Ay', alpha=0.8)
                ax.plot(x_values, df['accel_z'], 'b', label='Az', alpha=0.8)
                ax.set_ylabel("Fuerza G")

            elif mode_arg == 'gyro':
                ax.plot(x_values, df['gyro_x'], 'r', label='Gx', alpha=0.8)
                ax.plot(x_values, df['gyro_y'], 'g', label='Gy', alpha=0.8)
                ax.plot(x_values, df['gyro_z'], 'b', label='Gz', alpha=0.8)
                ax.set_ylabel("Velocidad Angular (dps)")

            elif mode_arg in ['angles', 'pitch', 'roll']:
                ax.plot(x_values, df['pitch'] * RAD_TO_DEG, 'purple', label='Pitch')
                ax.plot(x_values, df['roll'] * RAD_TO_DEG, 'orange', label='Roll')
                ax.set_ylabel("Grados (°)")
            
            elif mode_arg == 'temp':
                ax.plot(x_values, df['temp'], 'orange', label='Temp')
                ax.set_ylabel("Celsius")

        else:
            print(f" Modo '{mode_arg}' no reconocido.")
            return

    except Exception as e:
        print(f" Error: {e}")
        return

    # 3. DIBUJAR SEPARADORES DE VENTANA
    if len(x_values) > 0:
        # Ajustamos el título para indicar qué estamos viendo
        rango_txt = f"Ventana {slice_start}" if (is_specific_slice and slice_end == slice_start +1) else "Rango Seleccionado"
        if not is_specific_slice: rango_txt = "Todos los datos"
        
        ax.set_title(f"Label: '{label_arg}' | Modo: {mode_arg.upper()} | {rango_txt}")
        
        # Líneas rojas
        for x in range(window_size, len(x_values), window_size):
            ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5, alpha=0.4)

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    ax.set_xlim(0, len(x_values))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()