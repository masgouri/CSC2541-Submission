#!/bin/bash

module load python/2.7.12

cd /hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array

log_dir="/hpf/largeprojects/ccm/devin/plastics-data/logs/preprocessing"
log_file="color_3.log"
#log_file="grayscale.log"
mkdir -p $log_dir

starting_filename="/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_3.h5"
#starting_filename="/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_grayscaled_4.h5"
echo "Starting Filename: $starting_filename" | tee -a "${log_dir}/${log_file}"
echo "" | tee -a "${log_dir}/${log_file}"


for ds_scale in 4 8 12 16 20 24
do
	echo "Current Downscaling Scale: $ds_scale" | tee -a "${log_dir}/${log_file}"
	filename=$starting_filename
	echo "Starting Filename: $filename" | tee -a "${log_dir}/${log_file}"

	#Downscale
	python downscale.py "$filename" $ds_scale >> "${log_dir}/${log_file}"
	filename=${filename/.h5/_ds${ds_scale}_uni.h5}
	echo "Filename after downscaling: $filename" | tee -a "${log_dir}/${log_file}"

	#Split Data
	python split_data.py "$filename" >> "${log_dir}/${log_file}"
	filename=${filename/.h5/_split.h5}
	echo "Filename after splitting data: $filename" | tee -a "${log_dir}/${log_file}"

	#Apply Flips and Rotations
	python flips_and_rotations.py "$filename" >> "${log_dir}/${log_file}"
	filename=${filename/.h5/_flips.h5}
	echo "Filename after applying flips and rotations: $filename" | tee -a "${log_dir}/${log_file}"

	echo "" | tee -a "${log_dir}/${log_file}"
	echo "" | tee -a "${log_dir}/${log_file}"
	echo "" | tee -a "${log_dir}/${log_file}"
done
