#!/bin/bash

#PBS -j oe
#PBS -o /hpf/largeprojects/ccm/devin/plastics-data/logs/convert_to_vox
#PBS -l mem=6gb,vmem=6gb,nodes=1:ppn=2
#PBS -l walltime=5:00:00

#Script doesn't work with openstack nodes, so re-submit if you end up there (keeps doing it until it finally doesn't land on an openstack node)
if [[ $(hostname) == *"node"* ]]
then
	echo "Error, running on openstack nodes, try again"
	qsub -F "$1 $2 $3" /hpf/largeprojects/ccm/devin/plastics-data/general/convert_to_vox/convert_to_vox.sh
	exit 1
fi

module load x11/1.6.1.2

disp_id=$1
path_to_load=$2
path_to_save=$3

Xvfb $disp_id -screen 0 640x480x24 &
export DISPLAY=$disp_id
/hpf/largeprojects/ccm/devin/plastics-data/modules/binvox -d 128 -cb -pb $path_to_load

#xvfb-run -s "-screen 0 640x480x24" /hpf/largeprojects/ccm/devin/plastics-data/modules/binvox -pb $path_to_load

mkdir -p $path_to_save

binvox_file="${path_to_load%.obj}.binvox"
mv $binvox_file $path_to_save
