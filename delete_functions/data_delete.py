import os
import glob

data_output = glob.glob('data_output_csv/*')
for data in data_output:
    os.remove(data)