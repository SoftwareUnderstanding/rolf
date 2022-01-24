# esp32 xtensa docs and experiments

- included experimental data: no c++ optimization on-of delay @80,160,240 MHz




links :
oficial:
https://github.com/espressif/esp-who
- http://arxiv.org/abs/1604.02878 Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
- https://arxiv.org/abs/1801.04381 MobileNetV2: Inverted Residuals and Linear Bottlenecks


https://github.com/espressif/esp-face This is a component which provides functions of face detection and face recognition, and also operations of neurual network.
- https://github.com/espressif/esp-face/blob/master/face_detection/README.md face detection
- https://github.com/espressif/esp-face/blob/master/face_recognition/README.md face recognition
- https://arxiv.org/abs/1801.07698 ArcFace: Additive Angular Margin Loss for Deep Face Recognition
- https://github.com/espressif/esp-face/blob/master/lib/README.md DL lib


quick template:

git clone --recursive https://github.com/espressif/esp-idf.git

git clone https://github.com/espressif/crosstool-NG.git

git clone https://github.com/tamnil/esp-at


##########################################
other stuff:

https://github.com/hsoft/collapseos


## headers
gpio lib

```
typedef intr_handle_t gpio_isr_handle_t;
esp_err_t gpio_config(const gpio_config_t *pGPIOConfig);
esp_err_t gpio_reset_pin(gpio_num_t gpio_num);
esp_err_t gpio_set_intr_type(gpio_num_t gpio_num, gpio_int_type_t intr_type);
esp_err_t gpio_intr_enable(gpio_num_t gpio_num);
esp_err_t gpio_intr_disable(gpio_num_t gpio_num);
esp_err_t gpio_set_level(gpio_num_t gpio_num, uint32_t level);
int gpio_get_level(gpio_num_t gpio_num);
esp_err_t gpio_set_direction(gpio_num_t gpio_num, gpio_mode_t mode);
esp_err_t gpio_set_pull_mode(gpio_num_t gpio_num, gpio_pull_mode_t pull);
esp_err_t gpio_wakeup_enable(gpio_num_t gpio_num, gpio_int_type_t intr_type);
esp_err_t gpio_wakeup_disable(gpio_num_t gpio_num);
esp_err_t gpio_isr_register(void (*fn)(void *), void *arg, int intr_alloc_flags, gpio_isr_handle_t *handle);
esp_err_t gpio_pullup_en(gpio_num_t gpio_num);
esp_err_t gpio_pullup_dis(gpio_num_t gpio_num);
esp_err_t gpio_pulldown_en(gpio_num_t gpio_num);
esp_err_t gpio_pulldown_dis(gpio_num_t gpio_num);
esp_err_t gpio_install_isr_service(int intr_alloc_flags);
void gpio_uninstall_isr_service(void);
esp_err_t gpio_isr_handler_add(gpio_num_t gpio_num, gpio_isr_t isr_handler, void *args);
esp_err_t gpio_isr_handler_remove(gpio_num_t gpio_num);
esp_err_t gpio_set_drive_capability(gpio_num_t gpio_num, gpio_drive_cap_t strength);
esp_err_t gpio_get_drive_capability(gpio_num_t gpio_num, gpio_drive_cap_t *strength);
esp_err_t gpio_hold_en(gpio_num_t gpio_num);
esp_err_t gpio_hold_dis(gpio_num_t gpio_num);
void gpio_deep_sleep_hold_en(void);
void gpio_deep_sleep_hold_dis(void);
void gpio_iomux_in(uint32_t gpio_num, uint32_t signal_idx);
void gpio_iomux_out(uint8_t gpio_num, int func, bool oen_inv);
esp_err_t gpio_force_hold_all(void);
```

i2c defaults

```

I2C_ENTER_CRITICAL_ISR(mux)    portENTER_CRITICAL_ISR(mux)
I2C_EXIT_CRITICAL_ISR(mux)     portEXIT_CRITICAL_ISR(mux)
I2C_ENTER_CRITICAL(mux)        portENTER_CRITICAL(mux)
I2C_EXIT_CRITICAL(mux)         portEXIT_CRITICAL(mux)

I2C_DRIVER_ERR_STR             "i2c driver install error"
I2C_DRIVER_MALLOC_ERR_STR      "i2c driver malloc error"
I2C_NUM_ERROR_STR              "i2c number error"
I2C_TIMEING_VAL_ERR_STR        "i2c timing value error"
I2C_ADDR_ERROR_STR             "i2c null address error"
I2C_DRIVER_NOT_INSTALL_ERR_STR "i2c driver not installed"
I2C_SLAVE_BUFFER_LEN_ERR_STR   "i2c buffer size too small for slave mode"
I2C_EVT_QUEUE_ERR_STR          "i2c evt queue error"
I2C_SEM_ERR_STR                "i2c semaphore error"
I2C_BUF_ERR_STR                "i2c ringbuffer error"
I2C_MASTER_MODE_ERR_STR        "Only allowed in master mode"
I2C_MODE_SLAVE_ERR_STR         "Only allowed in slave mode"
I2C_CMD_MALLOC_ERR_STR         "i2c command link malloc error"
I2C_TRANS_MODE_ERR_STR         "i2c trans mode error"
I2C_MODE_ERR_STR               "i2c mode error"
I2C_SDA_IO_ERR_STR             "sda gpio number error"
I2C_SCL_IO_ERR_STR             "scl gpio number error"
I2C_CMD_LINK_INIT_ERR_STR      "i2c command link error"
I2C_GPIO_PULLUP_ERR_STR        "this i2c pin does not support internal pull-up"
I2C_ACK_TYPE_ERR_STR           "i2c ack type error"
I2C_DATA_LEN_ERR_STR           "i2c data read length error"
I2C_PSRAM_BUFFER_WARN_STR      "Using buffer allocated from psram"
I2C_LOCK_ERR_STR               "Power lock creation error"
I2C_FIFO_FULL_THRESH_VAL       (28)
I2C_FIFO_EMPTY_THRESH_VAL      (5)
I2C_IO_INIT_LEVEL              (1)
I2C_CMD_ALIVE_INTERVAL_TICK    (1000 / portTICK_PERIOD_MS)
I2C_CMD_EVT_ALIVE              (0)
I2C_CMD_EVT_DONE               (1)
I2C_EVT_QUEUE_LEN              (1)
I2C_SLAVE_TIMEOUT_DEFAULT      (32000)     /* I2C slave timeout value, APB clock cycle number */
I2C_SLAVE_SDA_SAMPLE_DEFAULT   (10)        /* I2C slave sample time after scl positive edge default value */
I2C_SLAVE_SDA_HOLD_DEFAULT     (10)        /* I2C slave hold time after scl negative edge default value */
I2C_MASTER_TOUT_CNUM_DEFAULT   (8)         /* I2C master timeout cycle number of I2C clock, after which the timeout interrupt will be triggered */
I2C_ACKERR_CNT_MAX             (10)
I2C_FILTER_CYC_NUM_DEF         (7)         /* The number of apb cycles filtered by default*/
I2C_CLR_BUS_SCL_NUM            (9)
I2C_CLR_BUS_HALF_PERIOD_US     (5)
```

mqtt lib

```

enum mqtt_message_type {
    MQTT_MSG_TYPE_CONNECT = 1,
    MQTT_MSG_TYPE_CONNACK = 2,
    MQTT_MSG_TYPE_PUBLISH = 3,
    MQTT_MSG_TYPE_PUBACK = 4,
    MQTT_MSG_TYPE_PUBREC = 5,
    MQTT_MSG_TYPE_PUBREL = 6,
    MQTT_MSG_TYPE_PUBCOMP = 7,
    MQTT_MSG_TYPE_SUBSCRIBE = 8,
    MQTT_MSG_TYPE_SUBACK = 9,
    MQTT_MSG_TYPE_UNSUBSCRIBE = 10,
    MQTT_MSG_TYPE_UNSUBACK = 11,
    MQTT_MSG_TYPE_PINGREQ = 12,
    MQTT_MSG_TYPE_PINGRESP = 13,
    MQTT_MSG_TYPE_DISCONNECT = 14
};

typedef struct mqtt_message {
    uint8_t *data;
    uint32_t length;
    uint32_t fragmented_msg_total_length;       /*!< total len of fragmented messages (zero for all other messages) */
    uint32_t fragmented_msg_data_offset;        /*!< data offset of fragmented messages (zero for all other messages) */
} mqtt_message_t;

typedef struct mqtt_connection {
    mqtt_message_t message;

    uint16_t message_id;
    uint8_t *buffer;
    uint16_t buffer_length;

} mqtt_connection_t;

typedef struct mqtt_connect_info {
    char *client_id;
    char *username;
    char *password;
    char *will_topic;
    char *will_message;
    int keepalive;
    int will_length;
    int will_qos;
    int will_retain;
    int clean_session;

} mqtt_connect_info_t;


static inline int mqtt_get_type(uint8_t *buffer) { return (buffer[0] & 0xf0) >> 4; }
static inline int mqtt_get_connect_session_present(uint8_t *buffer) { return buffer[2] & 0x01; }
static inline int mqtt_get_connect_return_code(uint8_t *buffer) { return buffer[3]; }
static inline int mqtt_get_dup(uint8_t *buffer) { return (buffer[0] & 0x08) >> 3; }
static inline void mqtt_set_dup(uint8_t *buffer) { buffer[0] |= 0x08; }
static inline int mqtt_get_qos(uint8_t *buffer) { return (buffer[0] & 0x06) >> 1; }
static inline int mqtt_get_retain(uint8_t *buffer) { return (buffer[0] & 0x01); }
void mqtt_msg_init(mqtt_connection_t *connection, uint8_t *buffer, uint16_t buffer_length);
bool mqtt_header_complete(uint8_t *buffer, uint16_t buffer_length);
uint32_t mqtt_get_total_length(uint8_t *buffer, uint16_t length, int *fixed_size_len);
char *mqtt_get_publish_topic(uint8_t *buffer, uint32_t *length);
char *mqtt_get_publish_data(uint8_t *buffer, uint32_t *length);
uint16_t mqtt_get_id(uint8_t *buffer, uint16_t length);
int mqtt_has_valid_msg_hdr(uint8_t *buffer, uint16_t length);
mqtt_message_t *mqtt_msg_connect(mqtt_connection_t *connection, mqtt_connect_info_t *info);
mqtt_message_t *mqtt_msg_publish(mqtt_connection_t *connection, const char *topic, const char *data, int data_length, int qos, int retain, uint16_t *message_id);
mqtt_message_t *mqtt_msg_puback(mqtt_connection_t *connection, uint16_t message_id);
mqtt_message_t *mqtt_msg_pubrec(mqtt_connection_t *connection, uint16_t message_id);
mqtt_message_t *mqtt_msg_pubrel(mqtt_connection_t *connection, uint16_t message_id);
mqtt_message_t *mqtt_msg_pubcomp(mqtt_connection_t *connection, uint16_t message_id);
mqtt_message_t *mqtt_msg_subscribe(mqtt_connection_t *connection, const char *topic, int qos, uint16_t *message_id);
mqtt_message_t *mqtt_msg_unsubscribe(mqtt_connection_t *connection, const char *topic, uint16_t *message_id);
mqtt_message_t *mqtt_msg_pingreq(mqtt_connection_t *connection);
mqtt_message_t *mqtt_msg_pingresp(mqtt_connection_t *connection);
mqtt_message_t *mqtt_msg_disconnect(mqtt_connection_t *connection);


```


