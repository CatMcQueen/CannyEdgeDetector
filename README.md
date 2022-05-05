Note: For ease of reading, markdown formatted instructions can be found at
https://github.com/CatMcQueen/CannyEdgeDetector

# CannyEdgeDetector
Canny Edge Detector for CUDA implementation using C++, with Otsu's method for 
threshold values

# Build Instructions
This project uses cmake for building. 

## Pre-build setup
Paths to nvcc need to be established prior to building. If running on the UofA 
HPC, you will need to do the following to setup your environment
```
module load cuda11/11.0
```
While this project does make use of the libwb library, it is using 
modifications and was forked into this project

## Compile
Run the following from the top-level directory
```
cd canny/build_dir/
cmake ../labs
make
```
This should generate the following executables 
```
canny/build_dir/CannyImage_Serial
canny/build_dir/CannyImage_Solution
canny/build_dir/CannyImage_Solution_Opt
canny/build_dir/CannyImage_Solution_Best
```

## Experiment Execution
A bash/slurm script has been maintained for executing all of the runs used for 
this experiment. To execute all of the runs on the UofA HPC system, do the 
following from the top-level directory
```
cd canny/build_dir
srun run_canny.slurm
```
# Project organization
## Image dataset for this project
Images used for this experiment are stored in the git repository and can be
found at
```
canny/build_dir/CannyImage/Dataset/<img_num>/input.ppm
```
There are 16 images stored for the experiments. Simply replace ```<img_num>```
above with the digit coresponding to the image you wish to view.

## Output Data
Outputs of the ```run_canny.slurm``` file will be written to 
```
canny/build_dir/CannyImage_output/
```
Each of the outputs for the executables run using ```run_canny.slurm``` are prefixed
with the following

| Binary being run                         | output prefix |
|------------------------------------------|---------------|
| canny/build_dir/CannyImage_Serial        | serial        |
| canny/build_dir/CannyImage_Solution      | gpu           |
| canny/build_dir/CannyImage_Solution_Opt  | opt           |
| canny/build_dir/CannyImage_Solution_Best | best          |

The following files are output from the ```run_canny.slurm``` file and referencing 
the prefixes and images above
```
canny/build_dir/CannyImage_output/<prefix>_0.83_output<img_num>.txt
canny/build_dir/CannyImage_output/<prefix>_0.83_outputImg<img_num>.ppm
canny/build_dir/CannyImage_output/timing-report-0.83-<img_num>.txt
```

The functional verification should be run on MatLab. Using ```CannyDiff.m``` and 
```CannyVerify.m``` from the ```build_dir``` folder, change the filepaths in 
each to fit the absolute file path associated with the output images. 

These file paths will be for serial, gpu, opt, and best. These file paths should
look like 
```
$path/canny/build_dir/CannyImage_output/serial_0.83_outputImg<img_num>.ppm
$path/canny/build_dir/CannyImage_output/gpu_0.83_outputImg<img_num>.ppm
$path/canny/build_dir/CannyImage_output/opt_0.83_outputImg<img_num>.ppm
$path/canny/build_dir/CannyImage_output/best_0.83_outputImg<img_num>.ppm
```
Then running in matlab will give the verification plots.

