import asyncio
import threading
from typing import Optional, List
from queue import Queue, Empty
import time

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from .ble_interface import BLEInterface, BLESample
from ble_central.sensor_packet_serialization import decode_packet, deserialize_packet_data # TODO: Check if we should move this to hardware folder?



SERVICE_UUID = "23011525-1212-efde-1523-785feabcd122" # TODO: Maybe change this?

EMG_CHAR_UUID = "23011525-1212-efde-1523-785feabcd123" # TODO: Maybe change this?
IMU_CHAR_UUID = "23011525-1212-efde-1523-785feabcd124" # TODO: Maybe change this?
PIEZO_CHAR_UUID = "23011525-1212-efde-1523-785feabcd125" # TODO: Maybe change this?

# TODO Check that these correct below
EMG_BYTES_PER_VALUE = 2
EMG_VALUES_PER_SAMPLE = 16
EMG_SENSOR_COUNT = 8

IMU_BYTES_PER_VALUE = 4
IMU_VALUES_PER_SAMPLE = 6
IMU_SENSOR_COUNT = 3

PIEZO_BYTES_PER_VALUE = 2
PIEZO_VALUES_PER_SAMPLE = 1
PIEZO_SENSOR_COUNT = 1


class BleakBLEInterface(BLEInterface):
    """BLE interface using Bleak"""

    def __init__(self, device_name: str = "LIMBServer", scan_timeout: float = 10.0):
        self.device_name = device_name
        self.scan_timeout = scan_timeout
        self.running = False

        # Thread and event loop for async operations
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        
        # BLE Client
        self.client: Optional[BleakClient] = None

        # Sample buffer (thread-safe queue)
        self.sample_queue: Queue = Queue(maxsize=1000)

        # Stats?
        # self.rx_count = 0
        # self.error_count = 0

    
    def start(self) -> bool:
        """Start the BLE interface in a separate thread."""
        if self.running:
            return True

        try:
            # Create new event loop in thread
            self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.thread.start()

            max_wait = 15.0
            start_time = time.time()
            while not self.running and (time.time() - start_time) < max_wait:
                time.sleep(0.1)

            if self.running:
                print(f"BLE interface connected to {self.device_name}")
                return True
            else:
                print(f"Failed to connect to {self.device_name} after {max_wait} seconds")
                return False
       
        except Exception as e:
            print(f"Failed to start BLE interface: {e}")
            self.running = False
            return False

    def stop(self) -> bool:
        """Stop the BLE interface"""
        if not self.running:
            return True

        self.running = False

        # Stop async operations
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._disconnect(), self.loop) # TODO: Implement _disconnect

        if self.thread:
            self.thread.join(timeout=2.0)

        print("BLE interface stopped")
        return True
    
    def read(self, timeout: Optional[float] = None) -> List[BLESample]:
        """Read sample from buffer (non-blocking)"""
        if not self.running:
            return []

        samples = []
        while True:
            try:
                sample = self.sample_queue.get_nowait()
                samples.append(sample)
                self.rx_count += 1
            except Empty:
                break

        return samples

    def is_running(self) -> bool:
        """Check if BLE interface is running"""
        return self.running

    def _run_event_loop(self):
        """Run async event loop in separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect_and_subscribe()) # TODO: Implement _connect_and_subscribe

    async def _connect_and_subscribe(self):
        """Connect to BLE device and subscrive to characteristics"""
        try:
            # Scan for device
            print(f"Scanning for BLE device '{self.device_name}'...")
            device = await BleakScanner.find_device_by_name(self.device_name, timeout=self.scan_timeout)
            if not device:
                print(f"Count not find BLE device '{self.device_name}'")
                return
            
            print(f"Found device: {device}")

            # Connect
            self.client = BleakClient(device)
            await self.client.connect()

            if not self.client.is_connected:
                print(f"Failed to connect to BLE device")
                return

            # Get service and characterstics
            service = self.client.services.get_service(SERVICE_UUID)
            if not service:
                print("Could not find sensor service")
                await self.client.disconnect()
                return
            
            emg_chr = service.get_characteristic(EMG_CHAR_UUID)
            imu_chr = service.get_characteristic(IMU_CHAR_UUID)
            piezo_chr = service.get_characteristic(PIEZO_CHAR_UUID)

            if not all([emg_chr, imu_chr, piezo_chr]):
                print("Could not find all required characteristics")
                await self.client.disconnect()
                return
            
            # Set up notifications handlers
            await self.client.start_notify(emg_chr, self._emg_handler)
            await self.client.start_notify(imu_chr, self._imu_handler)
            await self.client.start_notify(piezo_chr, self._piezo_handler)

            self.running = True
            print("BLE notifications started.")

            # Keep connection alive
            while self.running:
                await asyncio.sleep(0.1)
                if not self.client.is_connected:
                    print("BLE connection lost")
                    self.running = False
                    break

        except Exception as e:
            print(f"Error in BLE connection: {e}")
            self.error_count += 1
            self.running = False


    async def _disconnect(self):
        """Disconnect from BLE device"""
        if self.client and self.client.is_connected:
            await self.client.disconnect()

    def _emg_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray):
        """Handle EMG notifications"""
        try:
            sequence_number, sensor_data = decode_packet(memoryview(data))
            sensors = deserialize_packet_data(
                sensor_data,
                bytes_per_value=EMG_BYTES_PER_VALUE,
                values_per_sample=EMG_VALUES_PER_SAMPLE,
                sensor_count=EMG_SENSOR_COUNT,
            )

            timestamp = time.time()
            for sensor_id, sensor_samples in enumerate(sensors):
                for sample in sensor_samples:
                    ble_sample = BLESample(
                        message_type="EMG",
                        data={"channels": sample.tolist(), "sensor_id": sensor_id},
                        timestamp=timestamp
                    )
                    try: 
                        self.sample_queue.put_nowait(ble_sample)
                    except:
                        pass

        except Exception as e:
            #self.error_count += 1
            print(f"Error processing EMG data: {e}")


    def _imu_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray):
        """Handle IMU notifications"""
        try:
            sequence_number, sensor_data = decode_packet(memoryview(data))
            sensors = deserialize_packet_data(
                sensor_data,
                bytes_per_value=IMU_BYTES_PER_VALUE,
                values_per_sample=IMU_VALUES_PER_SAMPLE,
                sensor_count=IMU_SENSOR_COUNT,
            )

            timestamp = time.time()
            for sensor_id, sensor_samples in enumerate(sensors):
                for sample in sensor_samples:
                    # IMU: 6 values in list #TODO: Check if I should split them into two parts

                    ble_sample = BLESample(
                        message_type="IMU",
                        data={"data": sample.tolist(), "sensor_id": sensor_id},
                        timestamp=timestamp
                    )
                    try:
                        self.sample_queue.put_nowait(ble_sample)
                    except:
                        pass

        except Exception as e:
            #self.error_count += 1
            print(f"Error processing IMU data: {e}")

    def _piezo_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray):
        """Handle PIEZO notifications"""
        try:
            sequence_number, sensor_data = decode_packet(memoryview(data))
            sensors = deserialize_packet_data(
                sensor_data,
                bytes_per_value=PIEZO_BYTES_PER_VALUE,
                values_per_sample=PIEZO_VALUES_PER_SAMPLE,
                sensor_count=PIEZO_SENSOR_COUNT,
            )
            
            timestamp = time.time()
            for sensor_id, sensor_samples in enumerate(sensors):
                for sample in sensor_samples:
                    ble_sample = BLESample(
                        message_type="PIEZO",
                        data={"value": float(sample[0]), "sensor_id": sensor_id}, # TODO: Check if float() is correct
                        timestamp=timestamp
                    )
                    try:
                        self.sample_queue.put_nowait(ble_sample)
                    except:
                        pass
        except Exception as e:
            #self.error_count += 1
            print(f"Error processing PIEZO data: {e}")