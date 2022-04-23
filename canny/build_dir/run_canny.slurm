#!/bin/bash

### Set the job name
#SBATCH --job-name=devicequery

### Specify the group for this job
### List of PI groups available to each user can be found with "va" command

####ece569 students can also replace ece569 with another group id if sponsored by another faculty. 
#SBATCH --account=ece569

### Set the queue for this job as windfall or standard (adjust ### and #)
#SBATCH --partition=standard

#SBATCH -e run.error
#SBATCH -o run.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time", hhh:mm:ss. Required field
#SBATCH --time 00:00:50

### Load required modules/libraries if needed (openmpi example)
### Use "module avail" command to list all available modules
###module load openmpi3
###module load cuda11/11.0

### set directory for job execution, replace the following path with your own path
#hw_path=~/ece569_hpc/proj/build_dir
###akoglu/class/ece569/build_dir
hw_path=${PWD}
filter_size=3

cd $hw_path
###
PATHS[0]=$hw_path/CannyImage/Dataset/0;
PATHS[1]=$hw_path/CannyImage/Dataset/1;
PATHS[2]=$hw_path/CannyImage/Dataset/2;
PATHS[3]=$hw_path/CannyImage/Dataset/3;
PATHS[4]=$hw_path/CannyImage/Dataset/4;
PATHS[5]=$hw_path/CannyImage/Dataset/5;
PATHS[6]=$hw_path/CannyImage/Dataset/6;
PATHS[7]=$hw_path/CannyImage/Dataset/7;
PATHS[8]=$hw_path/CannyImage/Dataset/8;
PATHS[9]=$hw_path/CannyImage/Dataset/9;

## Uncomment vars below and run for memory report
memcheck=""
#memcheck="valgrind --tool=memcheck --leak-check=full "

## Uncomment vars below and run for memory report
cuda_memcheck=""
#cuda_memcheck="cuda-memcheck "

count=0
for i in ${PATHS[@]}
do
   file=output$((count)).txt
   imgOutFile=outputImg$((count)).ppm
   echo "cmd: ${cuda_memcheck} ./CannyImage_Solution -f ${filter_size} -o ${hw_path}/CannyImage_output/gpu_$imgOutFile -i $i/input.ppm -t image &> ${hw_path}/CannyImage_output/gpu_$file"
   ${cuda_memcheck} ./CannyImage_Solution -f ${filter_size} -o ${hw_path}/CannyImage_output/gpu_$imgOutFile -i $i/input.ppm -t image &> ${hw_path}/CannyImage_output/gpu_$file 
   
   echo "cmd: ${cuda_memcheck} ./CannyImage_Solution_Opt -f ${filter_size} -o ${hw_path}/CannyImage_output/opt_$imgOutFile -i $i/input.ppm -t image &> ${hw_path}/CannyImage_output/opt_$file"
   ${cuda_memcheck} ./CannyImage_Solution_Opt -f ${filter_size} -o ${hw_path}/CannyImage_output/opt_$imgOutFile -i $i/input.ppm -t image &> ${hw_path}/CannyImage_output/opt_$file 

   echo "cmd: ${memcheck} ./CannyImage_Serial -f ${filter_size} -o ${hw_path}/CannyImage_output/serial_$imgOutFile -i $i/input.ppm -t image &> ${hw_path}/CannyImage_output/serial_$file"
   ${memcheck} ./CannyImage_Serial -f ${filter_size} -o ${hw_path}/CannyImage_output/serial_$imgOutFile -i $i/input.ppm -t image &> ${hw_path}/CannyImage_output/serial_$file

   echo
   python timingReport.py ${hw_path}/CannyImage_output/serial_$file ${hw_path}/CannyImage_output/gpu_$file ${hw_path}/CannyImage_output/opt_$file &> ${hw_path}/CannyImage_output/timing-report-img$((count)).txt
   count=$((count+1))
done


