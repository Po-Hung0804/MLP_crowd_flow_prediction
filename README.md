# MLP_crowd_flow_prediction
## Introducton
### predict the crowd flow with using MLP and compare with the LSTM model build with Keras

---

## Data Analysis 

### Data Format
- `uid`: User ID
- `d`: Days
- `t`: Time of a specific day
- `x`: X-axis coordinate
- `y`: Y-axis coordinate

### Data Preprocessing
1. Filter out data points where `x` or `y` exceeds the range of 0-200 (`filter_data.py`).
2. Count the occurrence of each grid point over 75 days and visualize the city's outline (`grid_total.py`).

---

## Analyzing Trends
- Counted the number of people passing through each grid point in `grid_total.csv` over 75 days.
- Generated a heatmap to display and identify the five grids with the highest traffic:
  - `(80,95)`, `(80,92)`, `(80,96)`, `(79,93)`, `(79,92)`
- Analyzed these five grids individually and selected three for prediction.

---

