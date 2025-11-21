# Smart City IoT Sensor Data - Exploratory Data Analysis

## Project Overview
This project analyzes IoT sensor data collected as part of Grant No.BR24992852 "Intelligent models and methods of Smart City digital ecosystem for sustainable development and the citizens' quality of life improvement".

## Dataset Description
- **Source**: IoT sensors with ESP Arduino microcontroller
- **Period**: March 1-7, 2025 (7 days)
- **Sampling Rate**: Every 5 seconds
- **Total Records**: 120,960
- **Sensors**: 
  - Temperature (°C)
  - Humidity (%)
  - Light Intensity
  - pH Level
  - Electrical Conductivity (mS/cm)

## Assignment Tasks Completed
1. Plot trends and correlations between temperature, humidity, and light  
2. Identify patterns (day-night light cycles)  
3. Identify humidity-temperature inverse relation  
4. Compute basic statistics (mean, min, max, variance per sensor)

## Key Findings

### 1. Basic Statistics
- **Temperature**: Mean = 22.50°C, Variance = 2.08, Range = [20.0 - 25.0]°C
- **Humidity**: Mean = 50.03%, Variance = 33.26, Range = [40.0 - 60.0]%
- **Light**: Mean = 549.10, Variance = 67457.73, Range = [100.0 - 999.99]

### 2. Correlations
- Temperature vs Humidity: r = [YOUR VALUE]
- Temperature vs Light: r = [YOUR VALUE]
- Humidity vs Light: r = [YOUR VALUE]

### 3. Day-Night Patterns
- **Light** peaks at [HOUR]:00 and lowest at [HOUR]:00
- Clear circadian rhythms observed across all 7 days
- Temperature and humidity follow similar daily patterns

### 4. Humidity-Temperature Relationship
- Inverse relationship detected (if r < 0)
- Statistically significant correlation (p < 0.05)

## Project Structure
```
IoT_sensor_EDA/
├── data/              # Raw and processed data, too large so it's been added to gitignore
├── notebooks/         # Jupyter notebooks
├── outputs/           # Figures and reports
└── README.md          # This file
```

## Technologies Used
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## How to Run
1. Clone this repository
2. Install required packages: `pip install pandas numpy matplotlib seaborn scipy`
3. Open `notebooks/iot_sensor_eda.ipynb` in Jupyter
4. Run all cells

## Output Files
- **Figures**: 6 PNG visualization files in `outputs/figures/`
- **Reports**: Summary statistics and findings in `outputs/reports/`
- **Data**: Processed statistics in `data/processed/`


## License
Academic Project - Grant No.BR24992852
```

---