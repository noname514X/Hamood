from machine import Pin, I2C
import ssd1306
import time
from dht import DHT11  # 实际物理设备使用 DHT11，而不是 DHT22

# 设置 I2C
i2c = I2C(0, scl=Pin(4), sda=Pin(5))

# 等待一段时间，确保总线初始化
time.sleep(0.5)

# 扫描 I2C 设备
devices = i2c.scan()


if not devices:
    print('No I2C devices found')
    raise RuntimeError('OLED not detected on I2C bus, check wiring')
print('I2C devices found:', devices)

oled_width = 128
oled_height = 64

# 使用正确的 I2C 地址，根据实际情况选择第一个
addr = devices[0]
oled = ssd1306.SSD1306_I2C(oled_width, oled_height, i2c, addr=addr)

dht = DHT11(Pin(6))  # 初始化 DHT11 传感器，连接到 GPIO 6 引脚
while True:
    dht.measure()
    temp = dht.temperature()  # 获取温度
    hum = dht.humidity()      # 获取湿度
    # 将读数显示到控制台
    oled.fill(0)
    msg1 = f"Temp: {temp} "
    msg2 = f"Humid: {hum}%"
    print(msg1)
    oled.text(msg1, 10, 10)
    oled.text(msg2, 10, 30)
    oled.show()
    time.sleep(2)
