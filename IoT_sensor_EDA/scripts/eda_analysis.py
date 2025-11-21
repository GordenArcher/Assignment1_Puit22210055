"""
Smart City IoT Sensor Data - Exploratory Data Analysis
Grant No.BR24992852

This script performs comprehensive EDA on IoT sensor data including:
- Basic statistics computation
- Trend visualization
- Correlation analysis
- Day-night pattern identification
- Humidity-temperature relationship analysis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_sensor_data(folder_path):
    """Load and combine all sensor CSV files from a folder."""
    all_files = glob.glob(os.path.join(folder_path, "sensor_data_2025-03-*.csv"))
    
    print(f"Found {len(all_files)} files")
    
    df_list = []
    for file in sorted(all_files):
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
        print(f"Loaded {os.path.basename(file)}: {len(temp_df):,} records")
    
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add time-based features
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    return df


def compute_statistics(df, sensors):
    """Compute comprehensive statistics for all sensors."""
    stats_summary = pd.DataFrame({
        'Sensor': sensors,
        'Mean': [df[s].mean() for s in sensors],
        'Median': [df[s].median() for s in sensors],
        'Min': [df[s].min() for s in sensors],
        'Max': [df[s].max() for s in sensors],
        'Range': [df[s].max() - df[s].min() for s in sensors],
        'Variance': [df[s].var() for s in sensors],
        'Std Dev': [df[s].std() for s in sensors]
    })
    
    return stats_summary


def plot_trends(df, output_dir='outputs/figures'):
    """Plot time series trends for temperature, humidity, and light."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('IoT Sensor Trends: Temperature, Humidity, and Light\n(7-Day Period)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Temperature
    axes[0].plot(df['timestamp'], df['temperature'], 
                 color='red', linewidth=0.5, alpha=0.7)
    axes[0].axhline(df['temperature'].mean(), color='darkred', 
                    linestyle='--', linewidth=2, 
                    label=f'Mean: {df["temperature"].mean():.2f}°C')
    axes[0].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    axes[0].set_title('Temperature Trend', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Humidity
    axes[1].plot(df['timestamp'], df['humidity'], 
                 color='blue', linewidth=0.5, alpha=0.7)
    axes[1].axhline(df['humidity'].mean(), color='darkblue', 
                    linestyle='--', linewidth=2, 
                    label=f'Mean: {df["humidity"].mean():.2f}%')
    axes[1].set_ylabel('Humidity (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Humidity Trend', fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Light
    axes[2].plot(df['timestamp'], df['light'], 
                 color='orange', linewidth=0.5, alpha=0.7)
    axes[2].axhline(df['light'].mean(), color='darkorange', 
                    linestyle='--', linewidth=2, 
                    label=f'Mean: {df["light"].mean():.2f}')
    axes[2].set_ylabel('Light Intensity', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[2].set_title('Light Intensity Trend', fontsize=13, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensor_trends_temp_humidity_light.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Trend plots saved")


def plot_correlations(df, sensors, output_dir='outputs/figures'):
    """Create correlation heatmap and scatter plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    correlation_matrix = df[sensors].corr()
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=2, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Sensor Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Scatter Plots: Key Relationships', fontsize=14, fontweight='bold')
    
    axes[0].scatter(df['temperature'], df['humidity'], alpha=0.2, s=1, color='purple')
    axes[0].set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Humidity (%)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Temp vs Humidity\nr = {correlation_matrix.loc["temperature", "humidity"]:.3f}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(df['temperature'], df['light'], alpha=0.2, s=1, color='green')
    axes[1].set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Light', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Temp vs Light\nr = {correlation_matrix.loc["temperature", "light"]:.3f}')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(df['humidity'], df['light'], alpha=0.2, s=1, color='orange')
    axes[2].set_xlabel('Humidity (%)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Light', fontsize=11, fontweight='bold')
    axes[2].set_title(f'Humidity vs Light\nr = {correlation_matrix.loc["humidity", "light"]:.3f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plots_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Correlation plots saved")
    return correlation_matrix


def analyze_day_night_cycles(df, output_dir='outputs/figures'):
    """Analyze and plot day-night patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    hourly_avg = df.groupby('hour')[['temperature', 'humidity', 'light']].mean()
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Day-Night Patterns: Hourly Averages', fontsize=15, fontweight='bold')
    
    # Light
    axes[0].plot(hourly_avg.index, hourly_avg['light'], 
                 marker='o', linewidth=2, markersize=8, color='orange')
    axes[0].axvspan(0, 6, alpha=0.2, color='blue', label='Night')
    axes[0].axvspan(18, 24, alpha=0.2, color='blue')
    axes[0].axvspan(6, 18, alpha=0.2, color='yellow', label='Day')
    axes[0].set_ylabel('Light Intensity', fontsize=11, fontweight='bold')
    axes[0].set_title('Light by Hour', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24))
    
    # Temperature
    axes[1].plot(hourly_avg.index, hourly_avg['temperature'], 
                 marker='o', linewidth=2, markersize=8, color='red')
    axes[1].axvspan(0, 6, alpha=0.2, color='blue')
    axes[1].axvspan(18, 24, alpha=0.2, color='blue')
    axes[1].axvspan(6, 18, alpha=0.2, color='yellow')
    axes[1].set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    axes[1].set_title('Temperature by Hour', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(0, 24))
    
    # Humidity
    axes[2].plot(hourly_avg.index, hourly_avg['humidity'], 
                 marker='o', linewidth=2, markersize=8, color='blue')
    axes[2].axvspan(0, 6, alpha=0.2, color='blue')
    axes[2].axvspan(18, 24, alpha=0.2, color='blue')
    axes[2].axvspan(6, 18, alpha=0.2, color='yellow')
    axes[2].set_ylabel('Humidity (%)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    axes[2].set_title('Humidity by Hour', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(range(0, 24))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'day_night_cycles.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Day-night cycle plots saved")
    return hourly_avg


def analyze_humidity_temperature(df, output_dir='outputs/figures'):
    """Analyze humidity-temperature inverse relationship."""
    os.makedirs(output_dir, exist_ok=True)
    
    corr, p_value = pearsonr(df['temperature'], df['humidity'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Humidity-Temperature Relationship', fontsize=14, fontweight='bold')
    
    # Scatter with regression
    axes[0].scatter(df['temperature'], df['humidity'], alpha=0.3, s=2, color='purple')
    z = np.polyfit(df['temperature'], df['humidity'], 1)
    p = np.poly1d(z)
    temp_sorted = np.sort(df['temperature'])
    axes[0].plot(temp_sorted, p(temp_sorted), "r-", linewidth=3, 
                 label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    axes[0].set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Humidity (%)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Correlation: r = {corr:.3f}\np-value = {p_value:.6f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Density plot
    hexbin = axes[1].hexbin(df['temperature'], df['humidity'], 
                             gridsize=30, cmap='YlOrRd', mincnt=1)
    axes[1].set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Humidity (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Density Plot')
    plt.colorbar(hexbin, ax=axes[1], label='Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'humidity_temperature_inverse.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Humidity-temperature analysis saved")
    return corr, p_value


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("SMART CITY IoT SENSOR DATA - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Configuration
    data_folder = 'data/raw/'
    sensors = ['temperature', 'humidity', 'light', 'pH', 'electrical_conductivity']
    
    # Load data
    print("\n1. Loading data...")
    df = load_sensor_data(data_folder)
    print(f"✅ Loaded {len(df):,} records")
    
    # Compute statistics
    print("\n2. Computing statistics...")
    stats = compute_statistics(df, sensors)
    stats.to_csv('data/processed/sensor_statistics.csv', index=False)
    print(stats)
    print("✅ Statistics saved")
    
    # Plot trends
    print("\n3. Plotting trends...")
    plot_trends(df)
    
    # Correlation analysis
    print("\n4. Analyzing correlations...")
    corr_matrix = plot_correlations(df, sensors)
    print(corr_matrix)
    
    # Day-night cycles
    print("\n5. Analyzing day-night patterns...")
    hourly_avg = analyze_day_night_cycles(df)
    
    # Humidity-temperature relationship
    print("\n6. Analyzing humidity-temperature relationship...")
    corr, p_value = analyze_humidity_temperature(df)
    print(f"Correlation: {corr:.4f}, p-value: {p_value:.6f}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()