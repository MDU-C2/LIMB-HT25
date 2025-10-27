import asyncio
import platform
import struct  
from bleak import BleakScanner, BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic

# --- Configuración ---
TARGET_NAME = "LIMBServer"

# Pega aquí los UUIDs que encontraste en el Paso 7.A
# Deben ser strings (cadenas de texto).
EMG_CHAR_UUID = "24011525-1212-efde-1523-785feabcd122"  
IMU_CHAR_UUID = "25011525-1212-efde-1523-785feabcd122"  
# --------------------------------------------------------------------


def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
    """
    Esta función se llama AUTOMÁTICAMENTE cada vez que el ESP32
    envía datos (una "notificación").
    
    Ahora, decodifica los datos basándose en el UUID del "buzón".
    """
    
    char_uuid = str(characteristic.uuid)

    # ----------------------------------------------------
    # --- Decodificador para el "Buzón" de EMG ---
    # ----------------------------------------------------
    if char_uuid == EMG_CHAR_UUID:
        if len(data) >= 2:
            try:
                emg_value = struct.unpack('<H', data[:2])[0]
                print(f"Datos EMG recibidos: {emg_value}\n")
            except Exception as e:
                print(f"Error al decodificar EMG: {e}\n")
        else:
            print(f"Paquete EMG demasiado corto ({len(data)} bytes)\n")

    # ----------------------------------------------------
    # --- Decodificador para el "Buzón" de IMU ---
    # ----------------------------------------------------
    elif char_uuid == IMU_CHAR_UUID:
        if len(data) >= 8:
            # Tu ESP32 envió dos floats (8 bytes).
            # Formato '<ff':
            #   < = Little-Endian
            #   f = float (4 bytes)
            #   f = float (4 bytes)
            try:
                pitch, roll = struct.unpack('<ff', data[:8])
                pitch_deg = pitch * 57.2957795
                roll_deg = roll * 57.2957795
                
                print(f"Datos IMU recibidos: Pitch = {pitch_deg:8.2f}°, Roll = {roll_deg:8.2f}°\n")
            except Exception as e:
                print(f"Error al decodificar IMU: {e}\n")
        else:
            print(f"Paquete IMU demasiado corto ({len(data)} bytes)\n")

    else:
        print(f"Datos de UUID desconocido [ {char_uuid} ]:")
        data_hex = data[:16].hex(sep='-', bytes_per_sep=1)
        print(f"  > {len(data)} bytes: {data_hex}\n")


async def main():
    """
    Función principal asíncrona (SIN CAMBIOS).
    """
    print(f"Buscando dispositivo llamado '{TARGET_NAME}'...")
    
    device = await BleakScanner.find_device_by_name(TARGET_NAME)
    
    if not device:
        print(f"Error: No se pudo encontrar el dispositivo '{TARGET_NAME}'.")
        print("Asegúrate de que el ESP32 esté encendido y anunciándose.")
        return

    print(f"¡Dispositivo encontrado! Dirección: {device.address}")

    async with BleakClient(device) as client:
        print(f"Conectado a {device.name}")
        
        print("Descubriendo servicios...")
        
        for service in client.services:
            if not service.uuid.startswith("0000180"):
                print(f"\n[Servicio Personalizado] UUID: {service.uuid}")
                
                for char in service.characteristics:
                    if "notify" in char.properties:
                        print(f"  > Encontrado 'buzón' (Característica): {char.uuid}")
                        
                        if str(char.uuid) in [EMG_CHAR_UUID, IMU_CHAR_UUID]:
                            try:
                                print(f"  > Suscribiéndose a notificaciones...")
                                await client.start_notify(char, notification_handler)
                                print(f"  > ¡Suscrito con éxito!")
                            except Exception as e:
                                print(f"  > Error al suscribirse: {e}")
                        else:
                            print(f"  > (Ignorando 'buzón' desconocido o Piezo)")

        print("\n=======================================================")
        print("¡Suscrito a los 'buzones' de EMG e IMU!")
        print("Esperando datos del ESP32... (Presiona Ctrl+C para salir)")
        print("=======================================================\n")
        
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma detenido por el usuario.")
