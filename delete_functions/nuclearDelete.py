import os
import glob

data_output = glob.glob('data_output_csv/*')
for data in data_output:
    os.remove(data)

initial_visualizations = glob.glob('visualizations_output/initial_visualizations/*')
for visualizations in initial_visualizations:
    os.remove(visualizations)

xgbVisualizations = glob.glob('visualizations_output/final_visualizations/xgbModel/*')
for visualizations in xgbVisualizations:
    os.remove(visualizations)

rfVisualizations = glob.glob('visualizations_output/final_visualizations/rfModel/*')
for visualizations in rfVisualizations:
    os.remove(visualizations)

