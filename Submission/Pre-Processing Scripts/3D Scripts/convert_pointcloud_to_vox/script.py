import sqlite3
import os
import os.path
import math
import sys

prepath = "/hpf/largeprojects/ccm/devin/plastics-data/"
conn = sqlite3.connect(prepath + "db/squishy.sql")
vox_script_path = prepath + "general/convert_to_vox/convert_to_vox.sh"

labels_id = 3
num_dimensions = 3

input_data_images = conn.execute("select subject_id, folder, filename from images where labels_id = %d and num_dimensions = %d order by subject_id, folder, filename" % (labels_id, num_dimensions)).fetchall()

d_id=101

for data in input_data_images:
	input_path = prepath + data[1] + "/" + data[2]
	output_path = prepath + "converted_data/3D/" + "/".join(data[1].split("/")[1:]) + "/"

	if os.path.isfile(output_path + ".".join(data[2].split(".")[:-1]) + ".binvox"):
		continue

	disp_id = ":" + str(d_id)
	d_id += 1

	print input_path
	print output_path
	print disp_id
	#os.system(vox_script_path + " '" + disp_id + "' " + input_path + " " + output_path)
	os.system("qsub -F '" + disp_id + " " + input_path + " " + output_path + "' " + vox_script_path)
	print ""
