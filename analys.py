import pandas as pd
import matplotlib.pyplot as plt
import config

# Load data (replace with your file 
# or paste the data)
data_path = config.CONFIG['data_path']
data = pd.read_csv(data_path)  # Assuming CSV format

# Plot 1: Time vs. SOC and Range
fig, ax1 = plt.subplots()
ax1.plot(data['Time'], data['SOC (%)'], color='blue', label='SOC')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('SOC (%)', color='blue')
ax2 = ax1.twinx()
ax2.plot(data['Time'], data['Range'], color='green', label='Range')
ax2.set_ylabel('Range (km)', color='green')
plt.title('Time vs. SOC and Range')
plt.show()

# Plot 2: Speed vs. Current
plt.scatter(data['Speed'], data['current'], alpha=0.5)
plt.xlabel('Speed (km/h)')
plt.ylabel('Current (A)')
plt.title('Speed vs. Current')
plt.show()

# Plot 3: Time vs. Temperature
# plt.plot(data['time'], data['temperature'], color='red')
# plt.xlabel('Time (s)')
# plt.ylabel('Temperature (°C)')
# plt.title('Time vs. Temperature')
# plt.show()

# Plot 4: SOC vs. Range
plt.scatter(data['soc'], data['range'], alpha=0.5, color='purple')
plt.xlabel('SOC (%)')
plt.ylabel('Range (km)')
plt.title('SOC vs. Range')
plt.show()