import sys
import os
import numpy as np
import torch
import pydicom
import warnings
import matplotlib.pyplot as plt
import json
import nibabel as nib
import tifffile
import configparser

from tcia_utils import nbia
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, LoadImaged, Orientation, Orientationd, EnsureChannelFirst,EnsureTyped,EnsureChannelFirstd, Compose, AddChanneld,NormalizeIntensityd,ScaleIntensity, Lambda, ScaleIntensityd,Spacingd,Activationsd,AsDiscreted,Invertd,SaveImaged
from rt_utils import RTStructBuilder
from scipy.ndimage import label, measurements
from PIL import Image
from monai.transforms import Compose
from monai.data.meta_tensor import MetaTensor 
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_numpy, convert_to_tensor
from scipy import ndimage



# Ensure at least 4 arguments (1 directory, minimum of 2 files, and 1 yes/no)
if len(sys.argv) < 5:
	print("Insufficient arguments. Please provide directory, at least two segmentation folders, and an option to output pngs ('yes' or 'no').")
	sys.exit(1)

# Extract name and extension of the files
NAME = sys.argv[1]
NAME_ext = '_of_' + NAME + '.nii.gz'
NAME_pref = '0_'

dir_structure = os.path.join(NAME, '02_LABEL_training_data', '0')

# Add directory path and correct file names
files = [os.path.join(dir_structure, NAME_pref + i + NAME_ext) for i in sys.argv[2:-1]]

output_png = sys.argv[-1].lower()

# Confirm whether PNG output was selected
if output_png == 'yes':
	png_output = True
elif output_png == 'no':
	png_output = False
else:
	print("Invalid final argument. Please specify 'yes' or 'no' to indicate whether y'all want .png outputs.")
	sys.exit(1)

# MONAI LoadImage --> load Nifti files of individual channels. 
from monai.transforms import LoadImage
loader = LoadImage(image_only=True)

# Initialize the master track with zeros from the first image.
master_track = loader(files[0]).astype(np.int32)

### next line defines objects to store channel ID number and name to associate with that number
channels = []

# Loop over the rest of the images, processing them similarly.
for i, file in enumerate(files[1:], start=2):
	image = loader(file)
	image = image.astype(np.int32) * i
	master_track += (image * (master_track == 0))  # Add image values where master_track is still zero


	# Append channel number with formatted string
	#channels.append(f'{sys.argv[i]}')


channels.extend(sys.argv[2:-1])
	

print('Master channel max =', np.max(master_track), ', min =', np.min(master_track))



# Perform 3D connected components analysis on the filtered 3D data
data = master_track
labeled_array, num_features = ndimage.label(data)
unique_labels, label_counts = np.unique(labeled_array, return_counts=True)
# Count and print the number of remaining shapes after filtering
labeled_array_3d2, num_features_3d = ndimage.label(data, structure=ndimage.generate_binary_structure(3, 1))
remaining_shapes = np.unique(labeled_array_3d2)
total_3D_shapes = ((len(remaining_shapes))-1)
total_3D_shapes =np.array(total_3D_shapes)
print("Total connected 3D shapes:", (len(remaining_shapes))-1)


#channel_meta_info = '\n'.join(channels)  # Joins all channel descriptions with a newline character



xtek_file = [os.path.join(NAME, f) for f in os.listdir(NAME) if f.endswith('.xtekVolume')]

# Read the Xtek file to extract metadata that could be useful
config = configparser.ConfigParser()
config.read(xtek_file)

xyz = (((float(config['XTekCT']['VoxelsX'])), (float(config['XTekCT']['VoxelsY'])), (float(config['XTekCT']['VoxelsZ']))))
xyz = np.array(xyz)

vox_xyz = (((float(config['XTekCT']['VoxelSizeX'])), (float(config['XTekCT']['VoxelSizeY'])), (float(config['XTekCT']['VoxelSizeZ']))))
vox_xyz = np.array(vox_xyz)

origin = (((float(config['XTekCT']['OffsetX'])), (float(config['XTekCT']['OffsetY'])), (float(config['XTekCT']['OffsetZ']))))
origin = np.array(origin)

samp = (config['XTekCT']['Name'])

label_seg_name = (f'MULTI_CH_LABEL_of_{samp}')


# Compile image with relevant metadata from the Xtek file


CT_with_metadata = {'image': master_track,'image_meta_dict' : {
	'channels':channels,
	'total_3d_shapes': total_3D_shapes,
	'spacing': vox_xyz,
	'origin': origin,
	'units': (config['XTekCT']['Units']),
	'sample_name': label_seg_name,
	'white_level': (config['XTekCT']['WhiteLevel']),
	'mask_radius': (config['XTekCT']['MaskRadius']),
	'spatial_shape': xyz,
	'original_channel_dim': xyz
	}
}


data = convert_to_tensor(data=CT_with_metadata,track_meta=True, wrap_sequence=True)
data['image'] = data['image'].unsqueeze(0)

old_directory_name = (f'{NAME}/02_LABEL_training_data/0')
new_directory_name = (f'{NAME}/02_LABEL_training_data/single_channel_labels')

os.rename(old_directory_name, new_directory_name)


# Create a new Nifti1Image and save it
nifti_img = nib.Nifti1Image(data['image'].numpy(), np.eye(4))
nifti_img.header['descrip'] = str(data['image_meta_dict'])[:80]

multi_dir = (f'{NAME}/02_LABEL_training_data/0/')
if not os.path.exists(multi_dir):
	os.mkdir(multi_dir)

nib.save(nifti_img, os.path.join(multi_dir, '0_' + label_seg_name + '.nii.gz'))

old_directory_name = (f'{NAME}/02_LABEL_training_data/0')
new_directory_name = (f'{NAME}/02_LABEL_training_data/multi_channel_labels')

os.rename(old_directory_name, new_directory_name)

print("multi-channel label saved as NiFti (.nii) in --> / ",NAME,"/ 02_LABEL_training_data / multi_channel_labels /",)



###############################################################################





	
if png_output:
	png_dir = (f'{NAME}/pngs/')
	if not os.path.exists(png_dir):
		os.mkdir(png_dir)
	sample = (config['XTekCT']['Name'])
	image = '_IMAGE'
	samplename = f'{sample}{image}'
		
	print("Preparing slice check pngs.")
	
	mid_x = ((float(config['XTekCT']['VoxelsX'])))
	mid_x = np.array(mid_x).astype(int)
	mid_x = np.multiply(mid_x,(0.5)).astype(int)
		
	mid_y = ((float(config['XTekCT']['VoxelsY'])))
	mid_y = np.array(mid_y).astype(int)
	mid_y = np.multiply(mid_y,(0.5)).astype(int)
		
	mid_z = ((float(config['XTekCT']['VoxelsZ'])))
	mid_z = np.array(mid_z).astype(int)
	mid_z = np.multiply(mid_z,(0.5)).astype(int)
		
		
		
	onethird_x = ((float(config['XTekCT']['VoxelsX'])))
	onethird_x = np.array(onethird_x).astype(int)
	onethird_x = np.multiply(onethird_x,(0.333)).astype(int)
		
	onethird_y = ((float(config['XTekCT']['VoxelsY'])))
	onethird_y = np.array(onethird_y).astype(int)
	onethird_y = np.multiply(onethird_y,(0.333)).astype(int)
		
	onethird_z = ((float(config['XTekCT']['VoxelsZ'])))
	onethird_z = np.array(onethird_z).astype(int)
	onethird_z = np.multiply(onethird_z,(0.333)).astype(int)
		
		
	twothird_x = ((float(config['XTekCT']['VoxelsX'])))
	twothird_x = np.array(twothird_x).astype(int)
	twothird_x = np.subtract(twothird_x,onethird_x).astype(int)
		
	twothird_y = ((float(config['XTekCT']['VoxelsY'])))
	twothird_y = np.array(twothird_y).astype(int)
	twothird_y = np.subtract(twothird_y,onethird_y).astype(int)
		
	twothird_z = ((float(config['XTekCT']['VoxelsZ'])))
	twothird_z = np.array(twothird_z).astype(int)
	twothird_z = np.subtract(twothird_z,onethird_z).astype(int)
		
		
		
	fig_size_xz = ((float(config['XTekCT']['VoxelsX'])), (float(config['XTekCT']['VoxelsZ'])))
	fig_size_xz = np.array(fig_size_xz).astype(int)
	fig_size_xz = np.divide(fig_size_xz, (120,120))
	fig_size_xz = np.add(fig_size_xz, (1,0))
	fig_size_xz = np.multiply(fig_size_xz, (3,1))
	#fig_size_xz = np.add(fig_size_xz, (1.01,3.03))
		
	fig_size_yz = ((float(config['XTekCT']['VoxelsY'])), (float(config['XTekCT']['VoxelsZ'])))
	fig_size_yz = np.array(fig_size_yz).astype(int)
	fig_size_yz = np.divide(fig_size_yz, (120,120))
	fig_size_yz = np.add(fig_size_yz, (1,0))
	fig_size_yz = np.multiply(fig_size_yz, (3,1))
	#fig_size_yz = np.add(fig_size_yz, (1.01,3.03))
		
	fig_size_xy = ((float(config['XTekCT']['VoxelsX'])), (float(config['XTekCT']['VoxelsY'])))
	fig_size_xy = np.array(fig_size_xy).astype(int)
	fig_size_xy = np.divide(fig_size_xy, (120,120))
	fig_size_xy = np.add(fig_size_xy, (1,0))
	fig_size_xy = np.multiply(fig_size_xy, (3,1))
	#fig_size_xy = np.add(fig_size_xy, (1.01,3.03))
		
		
		
	CT_onethird_xz_slice = CT_with_metadata['image'][:,(onethird_y),:]
	CT_mid_xz_slice = CT_with_metadata['image'][:,(mid_y),:]
	CT_twothird_xz_slice = CT_with_metadata['image'][:,(twothird_y),:]
		
	plt.subplots(1,3,figsize=fig_size_xz)
	plt.subplot(131)
	plt.pcolormesh(CT_onethird_xz_slice.T, cmap='nipy_spectral')
	plt.title('xz slice @ 1/3 y',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
	plt.subplot(132)
	plt.pcolormesh(CT_mid_xz_slice.T, cmap='nipy_spectral')
	plt.title('xz slice @ 1/2 y',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
	plt.subplot(133)
	plt.pcolormesh(CT_twothird_xz_slice.T, cmap='nipy_spectral')
	plt.title('xz slice @ 2/3 y',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
	plt.savefig(f'{NAME}/pngs/0_MULTI_CH_LABEL_xz_slice_check.png', format='png')
	
	print("Preparing slice check pngs..")
	
	
	CT_onethird_yz_slice = CT_with_metadata['image'][(onethird_x),:,:]
	CT_mid_yz_slice = CT_with_metadata['image'][(mid_x),:,:]
	CT_twothird_yz_slice = CT_with_metadata['image'][(twothird_x),:,:]
	
	plt.subplots(1,3,figsize=fig_size_yz)
	plt.subplot(131)
	plt.pcolormesh(CT_onethird_yz_slice.T, cmap='nipy_spectral')
	plt.title('yz slice @ 1/3 x',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
	plt.subplot(132)
	plt.pcolormesh(CT_mid_yz_slice.T, cmap='nipy_spectral')
	plt.title('yz slice @ 1/2 x',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
	plt.subplot(133)
	plt.pcolormesh(CT_twothird_yz_slice.T, cmap='nipy_spectral')
	plt.title('yz slice @ 2/3 x',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
	plt.savefig(f'{NAME}/pngs/0_MULTI_CH_LABEL_yz_slice_check.png', format='png')
		
	print("Preparing slice check pngs...")
		
		
	
	
	CT_onethird_xy_slice = CT_with_metadata['image'][:,:,(onethird_z)]
	CT_mid_xy_slice = CT_with_metadata['image'][:,:,(mid_z)]
	CT_twothird_xy_slice = CT_with_metadata['image'][:,:,(twothird_z)]
		
	plt.subplots(1,3,figsize=fig_size_xy)
	plt.subplot(131)
	plt.pcolormesh(CT_onethird_xy_slice.T, cmap='nipy_spectral')
	plt.title('xy slice @ 1/3 z',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching 
	plt.subplot(132)
	plt.pcolormesh(CT_mid_xy_slice.T, cmap='nipy_spectral')
	plt.title('xy slice @ 1/2 z',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
	plt.subplot(133)
	plt.pcolormesh(CT_twothird_xy_slice.T, cmap='nipy_spectral')
	plt.title('xy slice @ 2/3 z',fontweight='bold',fontsize=25)
	plt.colorbar(label='PRESENCE / ABSENCE')
	plt.axis('on')
	plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
	plt.savefig(f'{NAME}/pngs/0_MULTI_CH_LABEL_xy_slice_check.png', format='png')




	############################### txt output with volume info

	# Define the channel names and numbers based on command line arguments
	# Command line arguments are considered starting from 2 up to second to last
	# As 0 is already assumed to be background, channel numbers start from 1
	channel_names = ["background"] + sys.argv[2:-1]
	channel_numbers = list(range(len(channel_names)))

	multi_channel_volumes = []
	#voxel_volume_cm3 = np.prod(CT_with_metadata['image_meta_dict']['spacing'] / 10)
	voxel_volume_mm3 = np.prod(CT_with_metadata['image_meta_dict']['spacing'])

	for channel_num, channel_name in zip(channel_numbers, channel_names):
		number_voxels = (master_track == channel_num).sum().item()
		channel_volume = number_voxels * voxel_volume_mm3
		out = f'(ch_{channel_num:03d})	{channel_name}	volume_mm^3:	{channel_volume:.6f}' # change the :.#f to number of sig figs after decimal place
		multi_channel_volumes.append(out)

	# Write all voxel measurements to a text file
	with open(f'{NAME}/ref_seg_volumes_for_{NAME}.txt', 'w') as f:
		for item in multi_channel_volumes:
			f.write("%s\n" % item)










