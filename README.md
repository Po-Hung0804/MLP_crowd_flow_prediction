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
  -![image](https://github.com/Po-Hung0804/MLP_crowd_flow_prediction/blob/main/flow_map.png)
- Generated a heatmap to display and identify the five grids with the highest traffic:
  - ![image](https://github.com/Po-Hung0804/MLP_crowd_flow_prediction/blob/main/heatmap.png)
  - `(80,95)`, `(80,92)`, `(80,96)`, `(79,93)`, `(79,92)`
- Analyzed these five grids individually and selected three for prediction.

### Selected Grids for Prediction
Chosen grids:
- `(80,95)`
- `(80,96)`
- `(79,93)`

---
## Custom MLP Prediction Results

| Grid      | Models | Epochs | MAE(best) |  MSE(best) | Loss(best) | MAPE(best) |
|-----------|--------|--------|-----------|------------|------------|------------|
| `(80,95)` | 100    | 1000   |   12.83   |   337.50   |   ~0.012   |   38.62%   |
| `(90,96)` | 100    | 1000   |   10.27   |   204.28   |   ~0.006   |   38.65%   |
| `(79,93)` | 100    | 1000   |   11.54   |   245.86   |   ~0.008   |   44.04%   |

---

## Comparison with Keras LSTM Model

| Grid      | Epochs | MAE   | MSE    | MAPE   |
|-----------|--------|-------|--------|--------|
| `(80,95)` | 1000   | 8.16  | 130.24 | 18.77% |
| `(80,96)` | 1000   | 7.18  | 98.96  | 22.12% |
| `(79,93)` | 1000   | 9.23  | 182.94 | 27.65% |

---
## Conclusion
1. Custom MLP models show lower performance, with MAPE around 40%, compared to LSTM models at 20%-30%.
2. The selected grids `(80,95)`, `(80,96)`, and `(79,93)` performed well, with `(80,95)` achieving the best results, proving the effectiveness of the selection strategy.
3. Grid `(80,95)` exhibits a seven-day cycle in population distribution, likely due to lower weekend traffic.
