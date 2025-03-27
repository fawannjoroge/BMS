import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

PORT = 'COM4'
BAUD_RATE = 115200
TIMEOUT = 1
MAX_POINTS = 300
UPDATE_INTERVAL = 50

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)

start_time = time.time()

class SensorData:
    def __init__(self, max_size):
        self.time = deque(maxlen=max_size)
        self.temp = deque(maxlen=max_size)
        self.voltage = deque(maxlen=max_size)
        self.current = deque(maxlen=max_size)
        self.power = deque(maxlen=max_size)
        self.speed = deque(maxlen=max_size)
        self.soc = deque(maxlen=max_size)
        self.predicted_range = deque(maxlen=max_size)

    def add_reading(self, timestamp, values):
        self.time.append(timestamp)
        self.temp.append(values[0])
        self.voltage.append(values[1])
        self.current.append(values[2])
        self.power.append(values[3])
        self.speed.append((values[4]))
        self.soc.append(values[5])
        self.predicted_range.append(values[6])

sensor_data = SensorData(MAX_POINTS)

fig, axes = plt.subplots(7, 1, sharex=True, figsize=(10, 12))
titles = [
    "Temperature (C)",
    "Voltage (V)",
    "Current (mA)",
    "Power (mW)",
    "Speed (km/h)",
    "SoC (%)",
    "Predicted Range (Km)"
]
y_limits = [
    (-10, 100),  # Temperature (assuming sensor range)
    (0, 12),     # Voltage (0 to 12V)
    (0, 500),   # Current (adjust based on expected range)
    (0, 2000),  # Power (mW)
    (0, ),     # Speed (km/h)
    (0, 100),    # SoC (0 to 100%)
    (0, 100)     # Predicted Range (adjust based on expected values)
]

for ax, title, ylim in zip(axes, titles, y_limits):
    ax.set_ylabel(title)
    ax.set_ylim(ylim)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Real-Time Sensor Data from ESP32", fontsize=16)

lines = [ax.plot([], [], linewidth=2)[0] for ax in axes]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def read_serial_data():
    data_updated = False
    while ser.in_waiting:
        try:
            raw_line = ser.readline().decode('utf-8', errors='replace').strip()
            parts = raw_line.split(',')
            if len(parts) != 7:
                continue 

            values = list(map(float, parts)) 
            current_elapsed = time.time() - start_time
            sensor_data.add_reading(current_elapsed, values)

            data_updated = True
        except ValueError as e:
            print(f"Error parsing data: {raw_line} - {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    return data_updated

def update(frame):
    if read_serial_data():
        lines[0].set_data(sensor_data.time, sensor_data.temp)
        lines[1].set_data(sensor_data.time, sensor_data.voltage)
        lines[2].set_data(sensor_data.time, sensor_data.current)
        lines[3].set_data(sensor_data.time, sensor_data.power)
        lines[4].set_data(sensor_data.time, sensor_data.speed)
        lines[5].set_data(sensor_data.time, sensor_data.soc)
        lines[6].set_data(sensor_data.time, sensor_data.predicted_range)

        if sensor_data.time:
            for ax in axes:
                ax.set_xlim(sensor_data.time[0], sensor_data.time[-1] + 1)

    return lines

ani = animation.FuncAnimation(
    fig, update, init_func=init, interval=UPDATE_INTERVAL, blit=True, cache_frame_data=False
)

plt.tight_layout()
plt.show()
