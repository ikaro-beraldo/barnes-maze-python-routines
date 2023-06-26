# barnes-maze-python-routines
Routines created in Python for Barnes Maze analysis based on DeepLabCut outputs

# Instructions
# 0 - Download
Download the content of this repository
# 1 - Creating a conda-environment
 - Download anaconda (https://www.anaconda.com/)
 - Open the conda-navigator on Windows start
 - Navigate to where the file 'environment.yml' is
 - Enter:
     conda env create -f environment.yml
# 2 - Activate the virtual environment and open Spyder
 - On conda-navigator enter:
     conda activate Ikaro
 - Open Spyder (Spyder is a data-science GUI IDE heavily inspired by MATLAB but for Python):
     spyder

# 3 - Start running the routines
- For 1 trial
  The first and simplest routine is the 'main_routine.py'. Run it by pressing F5 and select the DLC output file regarding the trial you want to analyze (eg. 'C38_1_G3_D4_T2DLC_resnet50_BM_GPUMar26shuffle2_700000.h5')

- For multiple trials
  Run the 'main_batch.py' routine and select all the trials you want to analyze. 

# 4 - Statistics
Run the 'final_statistics.py' routine and select the file 'Final_results.h5' generated from the multiple trial analysis.
