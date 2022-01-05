import os
import glob

xgbVisualizations = glob.glob('visualizations_output/final_visualizations/xgbModel/*')
for visualizations in xgbVisualizations:
    os.remove(visualizations)

rfVisualizations = glob.glob('visualizations_output/final_visualizations/rfModel/*')
for visualizations in rfVisualizations:
    os.remove(visualizations)

