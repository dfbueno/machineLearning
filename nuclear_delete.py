import os
import glob

data_output = glob.glob('data_output_csv/*')
for data in data_output:
    os.remove(data)

initial_visualizations = glob.glob('visualizations_output/*')
for visualizations in initial_visualizations:
    os.remove(visualizations)

