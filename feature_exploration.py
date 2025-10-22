# A feture exploration conducting graphs of the differnet factors that are recorded in the given locations. To be give to understand some of the variations in the results

#Wind direction graphs - histograms that see the correlaton between each individual one and frequency to see how these might affect wind power
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Import Necessary Libraries
# These imports are required for data manipulation and plotting

# Step 2: Load the Data from the CSV File
# Assume the CSV file is named 'Location1.csv' and is in the same directory as the script
df = pd.read_csv('Location1.csv') # change it with the other locations and will give the 

# Step 3: Extract the 'winddirection_100m' column
wind_direction = df['winddirection_100m']  # Removed the space before 'winddirection_100m'

# Step 4: Plot the Histogram for 'winddirection_100m'
plt.figure(figsize=(10, 6))
plt.hist(wind_direction, bins=30, edgecolor='black')  # Adjust bins as necessary
plt.title('Wind direction at 100m')
plt.xlabel('Wind direction (degree)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



#Wind speed graphs 
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Import Necessary Libraries
# These imports are required for data manipulation and plotting

# Step 2: Load the Data from the CSV File
# Assume the CSV file is named 'Location3.csv' and is in the same directory as the script
df = pd.read_csv('Location1.csv') # change it with the other locations and will give the 

# Step 3: Extract the 'windspeed_10m' column
wind_speed = df['windspeed_10m']

# Step 4: Plot the Histogram for 'windspeed_10m'
plt.figure(figsize=(10, 6))
plt.hist(wind_speed, bins=30, edgecolor='black')  # Adjust bins as necessary
plt.title('Wind Speed at 10m')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Dew point

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Import Necessary Libraries
# These imports are required for data manipulation and plotting

# Step 2: Load the Data from the CSV File
# Assume the CSV file is named 'Location1.csv' and is in the same directory as the script
df = pd.read_csv('Location1.csv') # change it with the other locations and will give the 

# Step 3: Extract the 'dewpoint_2m' column
dew_point = df['dewpoint_2m']

# Step 4: Plot the Histogram for 'dewpoint_2m'
plt.figure(figsize=(10, 6))
plt.hist(dew_point, bins=30, edgecolor='black')  # Adjust bins as necessary
plt.title('Dew Point at 2m')
plt.xlabel('Dew Point (°C)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Relative humidiy
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Import Necessary Libraries
# These imports are required for data manipulation and plotting

# Step 2: Load the Data from the CSV File
# Assume the CSV file is named 'Location1.csv' and is in the same directory as the script
df = pd.read_csv('Location2.csv') # change it with the other locations and will give the 

# Step 3: Extract the 'relativehumidity_2m' column
relative_humidity = df['relativehumidity_2m']

# Step 4: Plot the Histogram for 'relativehumidity_2m'
plt.figure(figsize=(10, 6))
plt.hist(relative_humidity, bins=30, edgecolor='black')  # Adjust bins as necessary
plt.title('Relative Humidity at 2m')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 

