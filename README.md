# Handling missing values 
Python tutorial for handling missing values

## Overview
This repository contains a beginner‑friendly tutorial on how to handle missing values in tabular datasets using Python and pandas.

The main example is a small sensor dataset (`sensor_log.csv`) with missing readings. You will work through a Jupyter notebook that shows, step by step, how to detect, summarise, and fix missing values using different strategies.

## Learning goals
By the end of the tutorial, you should be able to:

- Explain what a missing value is and why it matters in data analysis.
- Load data into a pandas `DataFrame` and check for missing values.
- Summarise how many missing values each column has (counts and percentages).
- Decide when it is reasonable to drop rows or columns with missing values.
- Fill (impute) missing values using simple strategies:
  - Constants (e.g. 0 or a special code)
  - Mean or median for numeric columns
  - Most frequent value (mode) for categorical columns
- Use time‑series methods such as forward fill, backward fill, and interpolation when the data is ordered by time.
- Reflect on the advantages and disadvantages of each approach.

## Files in this repository

- `missing_values_tutorial.ipynb`  
  The main Jupyter notebook. This is the step‑by‑step tutorial you should open and run.

- `sensor_log.csv`  
  A small sensor dataset with missing values. The notebook loads this file using `pandas.read_csv`.

- `README.md`  
  This file, giving an overview and guidance for students.

## Requirements and setup

To run the notebook, you will need:

- Python 3.x
- `pandas`
- `numpy`
- Jupyter Notebook or JupyterLab

One simple way to get started is:

1. Create and activate a virtual environment (optional but recommended).
2. Install the required packages:
   ```bash
   pip install pandas numpy jupyter
   ```
3. In the folder containing this project, start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `missing_values_tutorial.ipynb` from the Jupyter interface.

## How to work through the tutorial

1. **Read the explanations** in each section of the notebook.
2. **Run the code cells** in order, so you can see how the outputs change as different strategies for handling missing data are applied.
3. **Complete the exercises**:
   - Filtering rows with missing values in specific columns.
   - Comparing mean‑based and median‑based imputation.
   - Trying out forward fill, backward fill, and interpolation on time‑series data.
4. At the end, there is a **mini‑project** where you:
   - Design and justify your own missing‑data strategy for the sensor dataset.
   - Compare summary statistics before and after imputation.

## Suggested use

- **For students**: work through the notebook at your own pace, making notes on which strategies you prefer and why.
- **For instructors**: you can use the notebook as a live demonstration, followed by the exercises and mini‑project as homework or in‑class activities.
