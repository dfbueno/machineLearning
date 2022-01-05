import os
import glob

initial_visualizations = glob.glob('visualizations_output/initial_visualizations/*')
for visualizations in initial_visualizations:
    os.remove(visualizations)