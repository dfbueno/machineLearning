import os
import glob

final_visualizations = glob.glob('visualizations_output/final_visualizations/*')
for visualizations in final_visualizations:
    os.remove(visualizations)
