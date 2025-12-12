#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "driver/twai.h"
#include "esp_timer.h"

// BLE includes
#include "esp_bt.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "esp_gatt_common_api.h"
#include "esp_gatt_defs.h"
#include "esp_bt_main.h"

#include "adc_emg_driver.h"
#include "imu_driver.h"

// =============================================================================
// 1. CAN IDs DEFINITION (RECEIVE FROM UPPER ARM ONLY)
// =============================================================================
#define ID_UPPER_EMG   0x100 // Upper ARM EMG
#define ID_UPPER_IMU   0x101 // Upper ARM IMU (legacy: ax, ay only)
#define ID_UPPER_IMU_AXY  0x105 // Upper ARM IMU accel.x, accel.y
#define ID_UPPER_IMU_AZGX 0x106 // Upper ARM IMU accel.z, gyro.x
#define ID_UPPER_IMU_GYGZ 0x107 // Upper ARM IMU gyro.y, gyro.z

// =============================================================================
// 2. CONFIGURATION
// =============================================================================
#define CAN_TX_PIN GPIO_NUM_7 
#define CAN_RX_PIN GPIO_NUM_6 
static const char *TAG = "LOWER_ARM";

// RTOS Sync
static EventGroupHandle_t s_sync_event_group;
const int RX_DATA_READY_BIT = BIT0;
const int LOCAL_DATA_READY_BIT = BIT1;
static SemaphoreHandle_t g_rx_data_mutex;
static SemaphoreHandle_t g_local_data_mutex;

// Buffer to store data received from Upper ARM
typedef struct {
    float emg_val;
    lsm6dso32_data_t imu_data;
} upper_arm_data_t;

// Buffer to store local sensor data from Lower ARM
typedef struct {
    float emg_val;
    float piezo_val;
    lsm6dso32_data_t imu_data;
} lower_arm_data_t;

static upper_arm_data_t g_upper_data = {0};
static lower_arm_data_t g_lower_data = {0};

// =============================================================================
// 3. BLE FUSED DATA STRUCTURE
// =============================================================================
typedef struct {
    uint32_t timestamp;        // Milliseconds when packet created
    
    // Upper ARM data (received via CAN)
    float upper_emg;           // 4 bytes
    float upper_imu_ax;        // 4 bytes
    float upper_imu_ay;        // 4 bytes
    float upper_imu_az;        // 4 bytes
    float upper_imu_gx;        // 4 bytes
    float upper_imu_gy;        // 4 bytes
    float upper_imu_gz;        // 4 bytes
    
    // Lower ARM data (local sensors)
    float lower_emg;           // 4 bytes
    float lower_piezo;         // 4 bytes
    float lower_imu_ax;        // 4 bytes
    float lower_imu_ay;        // 4 bytes
    float lower_imu_az;        // 4 bytes
    float lower_imu_gx;        // 4 bytes
    float lower_imu_gy;        // 4 bytes
    float lower_imu_gz;        // 4 bytes
} BLE_FUSED_PACKET_t;  // Total: 64 bytes (1 uint32 + 15 floats)

// BLE state
static uint16_t g_ble_conn_handle = 0xFFFF;
static uint16_t g_ble_char_handle = 0;
static uint16_t g_ble_app_id = 0x55;
static bool g_ble_ready = false;
static bool g_ble_service_created = false;

// UUIDs for BLE Service
#define BLE_SERVICE_UUID_LEN (ESP_UUID_LEN_128)
static uint8_t ble_service_uuid[16] = {
    0xde, 0xad, 0xbe, 0xef,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x50, 0x52, 0x4f, 0x53
};

#define BLE_CHAR_UUID_LEN (ESP_UUID_LEN_16)
static uint16_t ble_char_uuid = 0xFFE1;

// Attribute table indices
enum {
    IDX_SVC,
    IDX_CHAR_DECL,
    IDX_CHAR_VAL,
    IDX_CHAR_CFG,
    IDX_NB
};

// Forward declaration for attribute table
static uint8_t char_prop_notify = ESP_GATT_CHAR_PROP_BIT_NOTIFY;
static uint16_t character_client_config_uuid = ESP_GATT_UUID_CHAR_CLIENT_CONFIG;
static uint16_t char_cfg = 0x0000;

// =============================================================================
// 4. BLE GATT SERVER IMPLEMENTATION
// =============================================================================

// GATT Attribute Table
static const esp_gatts_attr_db_t gatt_db[IDX_NB] = {
    // Service Definition
    [IDX_SVC] = {
        {ESP_GATT_AUTO_RSP},
        {ESP_UUID_LEN_128, (uint8_t *)ble_service_uuid},
        ESP_GATT_PERM_READ,
        2, NULL, NULL
    },
    
    // Characteristic Declaration
    [IDX_CHAR_DECL] = {
        {ESP_GATT_AUTO_RSP},
        {ESP_UUID_LEN_16, (uint8_t *)&ble_char_uuid},
        ESP_GATT_PERM_READ,
        sizeof(uint8_t), sizeof(esp_gatt_char_prop_t), (uint8_t *)&char_prop_notify
    },
    
    // Characteristic Value (64 bytes of sensor data)
    [IDX_CHAR_VAL] = {
        {ESP_GATT_AUTO_RSP},
        {ESP_UUID_LEN_16, (uint8_t *)&ble_char_uuid},
        ESP_GATT_PERM_READ,
        sizeof(BLE_FUSED_PACKET_t), sizeof(BLE_FUSED_PACKET_t), NULL
    },
    
    // Client Characteristic Configuration Descriptor (for notifications)
    [IDX_CHAR_CFG] = {
        {ESP_GATT_AUTO_RSP},
        {ESP_UUID_LEN_16, (uint8_t *)&character_client_config_uuid},
        ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
        2, 2, (uint8_t *)&char_cfg
    }
};

// BLE event handler with complete GATT lifecycle
static void gatts_event_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if,
                                esp_ble_gatts_cb_param_t *param) {
    switch (event) {
        case ESP_GATTS_REG_EVT: {
            ESP_LOGI(TAG, "GATT server app registered (app_id=%d)", param->reg.app_id);
            if (param->reg.app_id == g_ble_app_id) {
                // Create attribute table for this app
                esp_ble_gatts_create_attr_tab(gatt_db, gatts_if, IDX_NB, 0);
            }
            break;
        }
        
        case ESP_GATTS_CREATE_ATTR_TAB_EVT: {
            if (param->add_attr_tab.status == ESP_GATT_OK && param->add_attr_tab.num_handle == IDX_NB) {
                ESP_LOGI(TAG, "GATT attribute table created successfully (%d handles)", param->add_attr_tab.num_handle);
                
                // Save characteristic handle for sending notifications
                for (int i = 0; i < IDX_NB; i++) {
                    if (gatt_db[i].att_desc.uuid.len == ESP_UUID_LEN_16 &&
                        gatt_db[i].att_desc.uuid.uuid.uuid16 == ble_char_uuid) {
                        g_ble_char_handle = param->add_attr_tab.handles[i];
                        break;
                    }
                }
                
                g_ble_service_created = true;
                // Start the service
                esp_ble_gatts_start_service(param->add_attr_tab.handles[IDX_SVC]);
            } else {
                ESP_LOGE(TAG, "GATT attribute table creation failed, status=%d, num_handle=%d", 
                         param->add_attr_tab.status, param->add_attr_tab.num_handle);
            }
            break;
        }
        
        case ESP_GATTS_START_EVT:
            ESP_LOGI(TAG, "GATT service started successfully");
            break;
        
        case ESP_GATTS_CONNECT_EVT: {
            g_ble_conn_handle = param->connect.conn_id;
            g_ble_ready = true;
            ESP_LOGI(TAG, "BLE Client connected (conn_id=%d, addr=" ESP_BD_ADDR_STR ")", 
                     g_ble_conn_handle, ESP_BD_ADDR(&param->connect.remote_bda[0]));
            break;
        }
        
        case ESP_GATTS_DISCONNECT_EVT: {
            g_ble_conn_handle = 0xFFFF;
            g_ble_ready = false;
            ESP_LOGI(TAG, "BLE Client disconnected (reason=%d)", param->disconnect.reason);
            // Restart advertising
            esp_ble_gap_start_advertising(NULL);
            break;
        }
        
        case ESP_GATTS_WRITE_EVT: {
            if (param->write.handle == gatt_db[IDX_CHAR_CFG].attr_handle) {
                uint16_t descr_value = param->write.value[1] << 8 | param->write.value[0];
                if (descr_value == 0x0001) {
                    ESP_LOGI(TAG, "Notifications ENABLED by client");
                } else if (descr_value == 0x0000) {
                    ESP_LOGI(TAG, "Notifications DISABLED by client");
                }
            }
            break;
        }
        
        default:
            break;
    }
}

// GAP event handler
static void gap_event_handler(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
    switch (event) {
        case ESP_GAP_BLE_ADV_START_COMPLETE_EVT:
            if (param->adv_start_cmpl.status == ESP_BT_STATUS_SUCCESS) {
                ESP_LOGI(TAG, "BLE advertising started successfully");
            } else {
                ESP_LOGE(TAG, "BLE advertising start failed (status=%d)", param->adv_start_cmpl.status);
            }
            break;
            
        case ESP_GAP_BLE_ADV_STOP_COMPLETE_EVT:
            ESP_LOGI(TAG, "BLE advertising stopped");
            break;
            
        case ESP_GAP_BLE_UPDATE_CONN_PARAMS_EVT:
            ESP_LOGI(TAG, "BLE connection parameters updated");
            break;
        
        default:
            break;
    }
}

// Initialize BLE GATT Server with full advertising setup
static void ble_init(void) {
    esp_err_t ret;
    
    // 1. Release classic BT memory
    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT));
    
    // 2. Initialize BT controller
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ret = esp_bt_controller_init(&bt_cfg);
    if (ret) {
        ESP_LOGE(TAG, "BT controller init failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 3. Enable BT controller (BLE only)
    ret = esp_bt_controller_enable(ESP_BT_MODE_BLE);
    if (ret) {
        ESP_LOGE(TAG, "BT controller enable failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 4. Initialize Bluedroid stack
    ret = esp_bluedroid_init();
    if (ret) {
        ESP_LOGE(TAG, "Bluedroid init failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 5. Enable Bluedroid
    ret = esp_bluedroid_enable();
    if (ret) {
        ESP_LOGE(TAG, "Bluedroid enable failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 6. Register GAP callback
    ret = esp_ble_gap_register_callback(gap_event_handler);
    if (ret) {
        ESP_LOGE(TAG, "GAP register failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 7. Register GATT server callback
    ret = esp_ble_gatts_register_callback(gatts_event_handler);
    if (ret) {
        ESP_LOGE(TAG, "GATT register failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 8. Register GATT application (triggers ESP_GATTS_REG_EVT)
    ret = esp_ble_gatts_app_register(g_ble_app_id);
    if (ret) {
        ESP_LOGE(TAG, "GATT app register failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 9. Configure GAP advertising parameters
    esp_ble_adv_params_t adv_params = {
        .adv_int_min = 0x0020,  // 20ms minimum
        .adv_int_max = 0x0040,  // 40ms maximum
        .adv_type = ADV_TYPE_IND,  // Connectable undirected advertising
        .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
        .channel_map = ADV_CHNL_ALL,  // All channels (37, 38, 39)
        .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY
    };
    
    ret = esp_ble_gap_config_adv_params(&adv_params);
    if (ret) {
        ESP_LOGE(TAG, "GAP config adv params failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 10. Configure advertising data
    esp_ble_adv_data_t adv_data = {
        .set_scan_rsp = false,
        .include_name = true,
        .include_txpower = true,
        .appearance = 0x00,
        .manufacturer_len = 0,
        .p_manufacturer_data = NULL,
        .service_data_len = 0,
        .p_service_data = NULL,
        .service_uuid_len = 0,
        .p_service_uuid = NULL,
        .flag = (ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT)
    };
    
    ret = esp_ble_gap_config_adv_data(&adv_data);
    if (ret) {
        ESP_LOGE(TAG, "GAP config adv data failed: %s", esp_err_to_name(ret));
        return;
    }
    
    // 11. Start advertising (will continue after GATT service is ready)
    ret = esp_ble_gap_start_advertising(&adv_params);
    if (ret) {
        ESP_LOGE(TAG, "GAP start advertising failed: %s", esp_err_to_name(ret));
        return;
    }
    
    ESP_LOGI(TAG, "BLE initialization started - waiting for service creation");
}

// Send BLE notification with fused data
static void ble_send_fused_packet(BLE_FUSED_PACKET_t *packet) {
    if (!g_ble_ready || g_ble_conn_handle == 0xFFFF || !g_ble_service_created) {
        return;  // Not ready to send
    }
    
    // Send as GATT notification on characteristic value
    esp_ble_gatts_send_indicate(0,  // gatts_if (0 = all interfaces)
                               g_ble_conn_handle,
                               g_ble_char_handle,
                               sizeof(BLE_FUSED_PACKET_t),
                               (uint8_t *)packet,
                               false);  // false = notification (no ACK)
}

// =============================================================================
// =============================================================================
void acquisition_task(void *pvParameters) {
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10)); // ~100Hz

        // 1. Read local EMG/Piezo
        emg_driver_process_data();
        emg_data_packet_t packet;
        
        if (emg_driver_is_window_ready()) {
            emg_driver_get_packet(&packet);
            xSemaphoreTake(g_local_data_mutex, portMAX_DELAY);
            g_lower_data.emg_val = (float)packet.emg_ch0_window[0];   // Local EMG
            g_lower_data.piezo_val = (float)packet.emg_ch1_window[0]; // Local Piezo
            xSemaphoreGive(g_local_data_mutex);
        }

        // 2. Read local IMU
        lsm6dso32_data_t imu_tmp;
        if (imu_read_data(&imu_tmp) == ESP_OK) {
            xSemaphoreTake(g_local_data_mutex, portMAX_DELAY);
            g_lower_data.imu_data = imu_tmp;
            xSemaphoreGive(g_local_data_mutex);
        }

        xEventGroupSetBits(s_sync_event_group, LOCAL_DATA_READY_BIT);
    }
}

// =============================================================================
// 4. RECEIVE TASK (RX) - RECEIVE DATA FROM UPPER ARM
// =============================================================================
void can_rx_task(void *pvParameters) {
    twai_message_t rx_msg;
    ESP_LOGI(TAG, "RX Task Started - Listening to Upper ARM");

    while (1) {
        if (twai_receive(&rx_msg, portMAX_DELAY) == ESP_OK) {
            
            xSemaphoreTake(g_rx_data_mutex, portMAX_DELAY);
            
            switch (rx_msg.identifier) {
                
                case ID_UPPER_EMG: // 0x100 - Upper EMG
                    if (rx_msg.data_length_code >= 4) {
                        memcpy(&g_upper_data.emg_val, rx_msg.data, 4);
                        ESP_LOGI(TAG, "RX > UPPER EMG: %.2f", g_upper_data.emg_val);
                    }
                    break;

                case ID_UPPER_IMU: // 0x101 - Upper IMU (legacy: ax, ay only)
                    if (rx_msg.data_length_code >= 8) {
                        memcpy(&g_upper_data.imu_data.accel.x, &rx_msg.data[0], 4);
                        memcpy(&g_upper_data.imu_data.accel.y, &rx_msg.data[4], 4);
                        ESP_LOGI(TAG, "RX > UPPER IMU 0x101: ax=%.2f ay=%.2f", 
                                 g_upper_data.imu_data.accel.x, g_upper_data.imu_data.accel.y);
                    }
                    break;

                case ID_UPPER_IMU_AXY: // 0x105 - Upper IMU accel.x, accel.y
                    if (rx_msg.data_length_code >= 8) {
                        memcpy(&g_upper_data.imu_data.accel.x, &rx_msg.data[0], 4);
                        memcpy(&g_upper_data.imu_data.accel.y, &rx_msg.data[4], 4);
                        ESP_LOGI(TAG, "RX > UPPER IMU 0x105: ax=%.2f ay=%.2f", 
                                 g_upper_data.imu_data.accel.x, g_upper_data.imu_data.accel.y);
                    }
                    break;

                case ID_UPPER_IMU_AZGX: // 0x106 - Upper IMU accel.z, gyro.x
                    if (rx_msg.data_length_code >= 8) {
                        memcpy(&g_upper_data.imu_data.accel.z, &rx_msg.data[0], 4);
                        memcpy(&g_upper_data.imu_data.gyro.x, &rx_msg.data[4], 4);
                        ESP_LOGI(TAG, "RX > UPPER IMU 0x106: az=%.2f gx=%.2f", 
                                 g_upper_data.imu_data.accel.z, g_upper_data.imu_data.gyro.x);
                    }
                    break;

                case ID_UPPER_IMU_GYGZ: // 0x107 - Upper IMU gyro.y, gyro.z
                    if (rx_msg.data_length_code >= 8) {
                        memcpy(&g_upper_data.imu_data.gyro.y, &rx_msg.data[0], 4);
                        memcpy(&g_upper_data.imu_data.gyro.z, &rx_msg.data[4], 4);
                        ESP_LOGI(TAG, "RX > UPPER IMU 0x107: gy=%.2f gz=%.2f", 
                                 g_upper_data.imu_data.gyro.y, g_upper_data.imu_data.gyro.z);
                        
                        xEventGroupSetBits(s_sync_event_group, RX_DATA_READY_BIT);
                    }
                    break;
            }
            
            xSemaphoreGive(g_rx_data_mutex);
        }
    }
}

// =============================================================================
// 5. BLUETOOTH RELAY TASK - SEND ALL DATA TO AGX
// =============================================================================
void bluetooth_relay_task(void *pvParameters) {
    static uint32_t packet_count = 0;
    
    while (1) {
        // Wait for both local data AND upper arm data to be ready
        EventBits_t uxBits = xEventGroupWaitBits(s_sync_event_group, 
                                                  (LOCAL_DATA_READY_BIT | RX_DATA_READY_BIT), 
                                                  pdTRUE, pdFALSE, pdMS_TO_TICKS(100));

        if (!(uxBits & RX_DATA_READY_BIT)) {
            continue;  // Skip if no Upper ARM data received
        }

        // Retrieve local sensor data
        xSemaphoreTake(g_local_data_mutex, portMAX_DELAY);
        lower_arm_data_t local_copy = g_lower_data;
        xSemaphoreGive(g_local_data_mutex);

        // Retrieve data from Upper ARM
        xSemaphoreTake(g_rx_data_mutex, portMAX_DELAY);
        upper_arm_data_t upper_copy = g_upper_data;
        xSemaphoreGive(g_rx_data_mutex);

        // === CREATE FUSED BLE PACKET ===
        BLE_FUSED_PACKET_t fused_packet;
        
        // Timestamp in milliseconds
        fused_packet.timestamp = (uint32_t)(esp_timer_get_time() / 1000);
        
        // Fill Upper ARM data (7 floats)
        fused_packet.upper_emg = upper_copy.emg_val;
        fused_packet.upper_imu_ax = upper_copy.imu_data.accel.x;
        fused_packet.upper_imu_ay = upper_copy.imu_data.accel.y;
        fused_packet.upper_imu_az = upper_copy.imu_data.accel.z;
        fused_packet.upper_imu_gx = upper_copy.imu_data.gyro.x;
        fused_packet.upper_imu_gy = upper_copy.imu_data.gyro.y;
        fused_packet.upper_imu_gz = upper_copy.imu_data.gyro.z;
        
        // Fill Lower ARM data (8 floats)
        fused_packet.lower_emg = local_copy.emg_val;
        fused_packet.lower_piezo = local_copy.piezo_val;
        fused_packet.lower_imu_ax = local_copy.imu_data.accel.x;
        fused_packet.lower_imu_ay = local_copy.imu_data.accel.y;
        fused_packet.lower_imu_az = local_copy.imu_data.accel.z;
        fused_packet.lower_imu_gx = local_copy.imu_data.gyro.x;
        fused_packet.lower_imu_gy = local_copy.imu_data.gyro.y;
        fused_packet.lower_imu_gz = local_copy.imu_data.gyro.z;
        
        // === SEND VIA BLUETOOTH ===
        ble_send_fused_packet(&fused_packet);
        
        // Debug log (every 10th packet to avoid spam)
        if (++packet_count % 10 == 0) {
            ESP_LOGI(TAG, "BLE TX: pkt#%u U_EMG=%.2f L_EMG=%.2f L_PIEZO=%.2f (rdy=%d)", 
                     packet_count, fused_packet.upper_emg, fused_packet.lower_emg, 
                     fused_packet.lower_piezo, g_ble_ready);
        }
    }
}// =============================================================================
// MAIN - System Initialization
// =============================================================================
void app_main(void) {
    ESP_LOGI(TAG, "=== Lower ARM System Starting ===");
    
    // 1. Initialize CAN Bus
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);
    twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();
    
    ESP_ERROR_CHECK(twai_driver_install(&g_config, &t_config, &f_config));
    ESP_ERROR_CHECK(twai_start());
    ESP_LOGI(TAG, "CAN Bus initialized (500kbps)");

    // 2. Initialize Sensors (drivers for local sensors)
    emg_driver_init();
    emg_driver_start();
    imu_init();
    ESP_LOGI(TAG, "Local sensors initialized (EMG + IMU)");
    
    // 3. Initialize BLE (Bluetooth Low Energy)
    ble_init();
    ESP_LOGI(TAG, "BLE initialized - waiting for AGX connection");
    
    // 4. Initialize RTOS Synchronization
    s_sync_event_group = xEventGroupCreate();
    g_rx_data_mutex = xSemaphoreCreateMutex();
    g_local_data_mutex = xSemaphoreCreateMutex();
    ESP_LOGI(TAG, "RTOS synchronization ready");

    // 5. Create tasks
    xTaskCreate(acquisition_task, "Acq", 4096, NULL, 5, NULL);      // Read local sensors
    xTaskCreate(can_rx_task, "RX", 4096, NULL, 6, NULL);            // Listen on CAN (high priority)
    xTaskCreate(bluetooth_relay_task, "BLE", 4096, NULL, 4, NULL);  // Bluetooth relay
    ESP_LOGI(TAG, "All tasks created");

    ESP_LOGI(TAG, "Lower ARM Ready!");
    ESP_LOGI(TAG, "- Listening on CAN Bus (0x100, 0x101, 0x105, 0x106, 0x107)");
    ESP_LOGI(TAG, "- BLE advertising Fused Data Packet (64 bytes)");
    ESP_LOGI(TAG, "- Ready to send Upper + Lower IMU/EMG/Piezo to AGX");
}