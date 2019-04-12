
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')           #To disable X rendering so it works on terminal only
import matplotlib.pyplot as plt
from collections import Counter
import sys
import math
import copy
#from PIL import Image
import h5py

prepath = "/hpf/largeprojects/ccm/devin/plastics-data/"

sys.path.insert(0, prepath + 'modules/binvox-rw-py')
import binvox_rw



path_to_save = sys.argv[1] + "/"
data_type = int(sys.argv[2])
angle_id = int(sys.argv[3])

conn = sqlite3.connect(prepath + "db/squishy.sql")

label_ids = -1
num_dimensions = -1
filename = ""

if data_type == 0:
	#load 2D info (color)
	label_ids = [['3','3'], ['6','6'], ['9','9'], ['12','12'], ['15','15']]
	num_dimensions = 2
	filename = "2D_data_color"
	#dataset = {'3': {'input_images': [], 'input_info': [], 'output': []}, '6': {'input_images': [], 'input_info': [], 'output': []}, '9': {'input_images': [], 'input_info': [], 'output': []}, '12': {'input_images': [], 'input_info': [], 'output': []}, '15': {'input_images': [], 'input_info': [], 'output': []}}
	
if data_type == 1:
	#load 2D info (depth)
	label_ids = [['1','2'], ['4','5'], ['7','8'], ['10','11'], ['13','14']]
	num_dimensions = 2
	filename = "2D_data_depth"
	

input_d = []
output_d = []

input_data_subjects = []
output_data = []

def load_2D_image(path):
	np_arr = np.load(path).item()['img_data']
	return np_arr


def load_3D_image(path):
	f = open(path, 'rb')
	model = binvox_rw.read_as_3d_array(f).data
	f.close()
	return model


for l_index, labels_id in enumerate(label_ids):

	if l_index != angle_id:
		continue

	dataset = {'input_images': [], 'input_info': [], 'output': []}
	curr_dataset = {}
	
	subjects_with_no_missing_data = "select sis from (select subject_id as sis, num_dimensions, folder, count(distinct filename) as cis from images group by subject_id, num_dimensions, folder) where num_dimensions = %s and cis = 15" % (num_dimensions);
	input_data_images = conn.execute("select subject_id, folder, filename from images where (labels_id = %s or labels_id = %s) and num_dimensions = %d and subject_id in (%s) order by subject_id, folder, filename" % (labels_id[0], labels_id[1], num_dimensions, subjects_with_no_missing_data)).fetchall()

	for data in input_data_images:
		if data[0] not in curr_dataset:
			curr_dataset[data[0]] = {'image_locations': [], 'images': [], 'subject_info': [], 'output': []}

		image_path = prepath + "converted_data/2D/" + "/".join(data[1].split("/")[1:]) + "/" + data[2] + ".npy"
		curr_dataset[data[0]]['image_locations'].append(image_path)
		curr_dataset[data[0]]['images'].append(load_2D_image(image_path))


	output_data = conn.execute("select subject_id, diagnosis_id from subjects").fetchall()
	print output_data[:5]
	
	for data in input_data_subjects:
		if data[0] not in curr_dataset:
			continue

		##Need to do this later
	
	
	for data in output_data:
		if data[0] not in curr_dataset:
			continue	

		curr_dataset[data[0]]['output'].append(data[1])
		
	for data in curr_dataset:
	
		curr_input_im_d = []
		curr_input_si_d = []
		curr_output_d = []

		if len(curr_dataset[data]['output']) == 0 or len(curr_dataset[data]['images']) == 0 or len(curr_dataset[data]['subject_info']) < 0:
			continue
	
		for i in range(len(curr_dataset[data]['images'])):
			curr_input_im_d.append(curr_dataset[data]['images'][i])
			curr_input_si_d.append(curr_dataset[data]['subject_info'])
			curr_output_d.append(curr_dataset[data]['output'])

		#print dataset[labels_id]
	
		dataset['input_images'] += curr_input_im_d
		dataset['input_info'] += curr_input_si_d
		dataset['output'] += curr_output_d

	input_im_d = dataset['input_images']
	input_si_d = dataset['input_info']
	output_d = dataset['output']

	del dataset
	del curr_input_im_d
	del curr_dataset

	Data_im = np.array(input_im_d)
	Data_si = np.array(input_si_d)
	output_d = np.array(output_d)

	print len(input_im_d)
	print Data_im.shape
	print ""


	print len(input_im_d[0])
	print Data_im[0].shape
	print ""

	print len(output_d)
	print np.array(output_d).shape


	final_input_im = Data_im
	final_input_si = Data_si
	final_output = output_d

	data = { 'data_im': final_input_im, 'data_si': final_input_si, 'target': final_output}

	h5py_file = h5py.File(filename + "_" + str(l_index) + ".h5", 'w')
	for dd in data:
		h5py_file.create_dataset(dd, data=data[dd])
	h5py_file.close()

