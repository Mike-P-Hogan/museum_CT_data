import os
import sys
import numpy as np
import torch
import pydicom
import warnings
import matplotlib.pyplot as plt
import json
import nibabel as nib
import tifffile
import configparser
import glob

from tcia_utils import nbia
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, LoadImaged, Orientation, Orientationd, EnsureChannelFirst,EnsureTyped,EnsureChannelFirstd, Compose, AddChanneld,NormalizeIntensityd,ScaleIntensity, Lambda, ScaleIntensityd,Spacingd,Activationsd,AsDiscreted,Invertd,SaveImaged
from rt_utils import RTStructBuilder
from scipy.ndimage import label, measurements
import json
from PIL import Image
from monai.transforms import Compose
from monai.data.meta_tensor import MetaTensor 
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_numpy, convert_to_tensor
from scipy import ndimage


def main(sp_data_directory, segmentation, make_png):

	datadir = sp_data_directory
	seg_name = segmentation
	
	
	
	LABEL_folder = (f'{datadir}/zzz_temp/{seg_name}')
	IMAGE_ref = (f'{datadir}/01_IMAGE_training_data/0/')
	IMAGE_ref = glob.glob(f"{IMAGE_ref}/*.nii.gz")
	
	
	
	####################################################################################### read in IMAGE as nii:
	#  MONAI LoadImage --> load Nifti files of individual channels. 
	# We'll use this as the reference to compare the inverted mask tiff stack against
	from monai.transforms import LoadImage
	loader = LoadImage(image_only=True)
	
	# load the image
	IMAGE_ref = loader(IMAGE_ref)
	
	# image is a numpy array
	IMAGE_ref = np.array(IMAGE_ref)
	print('IMAGE shape =', IMAGE_ref.shape)
	
	####################################################################### read in inverted LABEL from tiff stack:
	# Directory containing TIFF image slices
	#data_directory = LABEL_folder
	
	# List all TIFF files in the directory
	#tiff_files = sorted([os.path.join(data_directory, filename) for filename in os.listdir(data_directory) if filename.endswith('.tif')],reverse=True)
	tiff_files = sorted([os.path.join(LABEL_folder, filename) for filename in os.listdir(LABEL_folder) if filename.endswith('.tif')])
	
	# Create a list to store loaded slices
	loaded_slices = []
	
	# Load each slice and append it to the list
	for tiff_file in tiff_files:
		loaded_slice = np.array(Image.open(tiff_file))
		loaded_slices.append(loaded_slice)
	
	
	# Convert the list of loaded slices to a NumPy array and transpose to get X, Y, Z
	LABEL = np.stack(loaded_slices, axis=0)
	LABEL = np.transpose(LABEL, (2, 1, 0))
	# LABEL = np.flip(LABEL, axis=1) # flip not needed for segmentation stack
	
	
	LABEL = LABEL.astype(np.float32)
	print("LABEL shape =", LABEL.shape)
	
	
	####################################################################### now compare, make differences 1 matches 0
	
	# Ensure both arrays have the same shape
	if IMAGE_ref.shape != LABEL.shape:
		raise ValueError("IMAGE and LABEL arrays must have the same shape for comparison!!!!")
	
	mismatch_mask = (IMAGE_ref != LABEL)
	mask_array = np.zeros_like(IMAGE_ref, dtype=np.int32)
	mask_array[mismatch_mask] = 1
	
	
	LABEL = mask_array
	
	
	print(f'New {seg_name} single channel mask value =',np.max(LABEL),', background value =', np.min(LABEL))
	
	
	# Perform 3D connected components analysis on the filtered 3D data
	data = LABEL
	labeled_array, num_features = ndimage.label(data)
	unique_labels, label_counts = np.unique(labeled_array, return_counts=True)
	# Count and print the number of remaining shapes after filtering
	labeled_array_3d2, num_features_3d = ndimage.label(data, structure=ndimage.generate_binary_structure(3, 1))
	remaining_shapes = np.unique(labeled_array_3d2)
	total_3D_shapes = ((len(remaining_shapes))-1)
	total_3D_shapes =np.array(total_3D_shapes)
	print("Total connected 3D shapes:", (len(remaining_shapes))-1)
	
	
	
	
	xtek_file = [os.path.join(sp_data_directory, f) for f in os.listdir(sp_data_directory) if f.endswith('.xtekVolume')]
	
	# Read the Xtek file to extract metadata that could be useful
	config = configparser.ConfigParser()
	config.read(xtek_file)
	
	xyz = (((float(config['XTekCT']['VoxelsX'])), (float(config['XTekCT']['VoxelsY'])), (float(config['XTekCT']['VoxelsZ']))))
	xyz = np.array(xyz).astype(int)
	
	vox_xyz = (((float(config['XTekCT']['VoxelSizeX'])), (float(config['XTekCT']['VoxelSizeY'])), (float(config['XTekCT']['VoxelSizeZ']))))
	vox_xyz = np.array(xyz).astype(int)
	
	origin = (((float(config['XTekCT']['OffsetX'])), (float(config['XTekCT']['OffsetY'])), (float(config['XTekCT']['OffsetZ']))))
	origin = np.array(origin)
	
	samp = (config['XTekCT']['Name'])
	
	label_seg_name = (f'{seg_name}_of_{samp}')
	
	
	# Compile image with relevant metadata from the Xtek file
	
	
	LABEL_with_metadata = {'image': LABEL,'image_meta_dict' : {
		'spacing': vox_xyz,
		'origin': origin,
		'units': (config['XTekCT']['Units']),
		'sample_name': label_seg_name,
		'white_level': (config['XTekCT']['WhiteLevel']),
		'mask_radius': (config['XTekCT']['MaskRadius']),
		'spatial_shape': xyz,
		'original_channel_dim': xyz,
		'total_3d_shapes': total_3D_shapes,
		}
	}
	
	
	LABEL = convert_to_tensor(data=LABEL_with_metadata,track_meta=True, wrap_sequence=True)
	LABEL['image'] = LABEL['image'].unsqueeze(0)
	
	################ save new single channel LABEL
	
	export_raw_image_nii = Compose([
		SaveImaged(keys='image',meta_keys='image_meta_dict', output_dir=os.path.join(sp_data_directory, '02_LABEL_training_data'), output_postfix=(label_seg_name),resample=False)
	])
	
	export_raw_image_nii(LABEL)

	print("single channel label saved as NiFti (.nii) in --> / ",sp_data_directory," / 02_LABEL_training_data / 0 /",)
	
	# Run additional code if specified TO MAKE PNGS!!!!!!!!!!!
	
	
	if make_png == "yes":
		png_dir = (f'{sp_data_directory}/pngs/')
		if not os.path.exists(png_dir):
			os.mkdir(png_dir)

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
		
		
		
		LABEL_onethird_xz_slice = LABEL_with_metadata['image'][:,(onethird_y),:]
		LABEL_mid_xz_slice = LABEL_with_metadata['image'][:,(mid_y),:]
		LABEL_twothird_xz_slice = LABEL_with_metadata['image'][:,(twothird_y),:]
		
		plt.subplots(1,3,figsize=fig_size_xz)
		plt.subplot(131)
		plt.pcolormesh(LABEL_onethird_xz_slice.T, cmap='Greys_r')
		plt.title('xz slice @ 1/3 y',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(132)
		plt.pcolormesh(LABEL_mid_xz_slice.T, cmap='Greys_r')
		plt.title('xz slice @ 1/2 y',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(133)
		plt.pcolormesh(LABEL_twothird_xz_slice.T, cmap='Greys_r')
		plt.title('xz slice @ 2/3 y',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.savefig(f'{datadir}/pngs/{seg_name}_xz_slice_check.png', format='png')
		
		print("Preparing slice check pngs..")
		
		
		LABEL_onethird_yz_slice = LABEL_with_metadata['image'][(onethird_x),:,:]
		LABEL_mid_yz_slice = LABEL_with_metadata['image'][(mid_x),:,:]
		LABEL_twothird_yz_slice = LABEL_with_metadata['image'][(twothird_x),:,:]
		
		plt.subplots(1,3,figsize=fig_size_yz)
		plt.subplot(131)
		plt.pcolormesh(LABEL_onethird_yz_slice.T, cmap='Greys_r')
		plt.title('yz slice @ 1/3 x',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(132)
		plt.pcolormesh(LABEL_mid_yz_slice.T, cmap='Greys_r')
		plt.title('yz slice @ 1/2 x',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(133)
		plt.pcolormesh(LABEL_twothird_yz_slice.T, cmap='Greys_r')
		plt.title('yz slice @ 2/3 x',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.savefig(f'{datadir}/pngs/{seg_name}_yz_slice_check.png', format='png')
		
		print("Preparing slice check pngs...")
			
			
		
		
		LABEL_onethird_xy_slice = LABEL_with_metadata['image'][:,:,(onethird_z)]
		LABEL_mid_xy_slice = LABEL_with_metadata['image'][:,:,(mid_z)]
		LABEL_twothird_xy_slice = LABEL_with_metadata['image'][:,:,(twothird_z)]
		
		plt.subplots(1,3,figsize=fig_size_xy)
		plt.subplot(131)
		plt.pcolormesh(LABEL_onethird_xy_slice.T, cmap='Greys_r')
		plt.title('xy slice @ 1/3 z',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching 
		plt.subplot(132)
		plt.pcolormesh(LABEL_mid_xy_slice.T, cmap='Greys_r')
		plt.title('xy slice @ 1/2 z',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(133)
		plt.pcolormesh(LABEL_twothird_xy_slice.T, cmap='Greys_r')
		plt.title('xy slice @ 2/3 z',fontweight='bold',fontsize=25)
		plt.colorbar(label='PRESENCE / ABSENCE')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.savefig(f'{datadir}/pngs/{seg_name}_xy_slice_check.png', format='png')




if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Usage: python3 STEP_2_invert_seg_x_ref_to_nii.py [sp_data_directory] [segmentation (this should also be the name of the directory holding the inverted tif stack)] [make midpoint pngs? (yes/no)]")
	else:
		sp_data_directory = sys.argv[1]
		segmentation = sys.argv[2]
		make_png = sys.argv[3].lower()  # Convert to lowercase for case-insensitive check
		if make_png not in ["yes", "no"]:
			print("Third argument must be 'yes' or 'no'")
		else:
			main(sp_data_directory, segmentation, make_png)



