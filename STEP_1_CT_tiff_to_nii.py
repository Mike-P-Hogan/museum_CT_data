import os
import numpy as np
import torch
import sys
import configparser
import pydicom
import json
import matplotlib.pyplot as plt
import nibabel
import tifffile
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, LoadImaged, Orientation, Orientationd, EnsureChannelFirst,EnsureTyped,EnsureChannelFirstd, Compose, AddChanneld,NormalizeIntensityd,ScaleIntensityd,Spacingd,Activationsd,AsDiscreted,Invertd,SaveImaged
from rt_utils import RTStructBuilder
from scipy.ndimage import label, measurements
from PIL import Image
from monai.transforms import Compose
from monai.data.meta_tensor import MetaTensor 
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_numpy, convert_to_tensor
from tcia_utils import nbia
from rt_utils import RTStructBuilder
from scipy.ndimage import label, measurements

def main(sp_data_directory, make_png):

	datadir = (f'{sp_data_directory}/zzz_temp/CT/')

	# Directory containing TIFF image slices
	data_directory = datadir

	# List all TIFF files in the directory
	#tiff_files = sorted([os.path.join(data_directory, filename) for filename in os.listdir(data_directory) if filename.endswith('.tif')],reverse=True)
	tiff_files = sorted([os.path.join(data_directory, filename) for filename in os.listdir(data_directory) if filename.endswith('.tif')])

	# Create a list to store loaded slices
	loaded_slices = []

	# Load each slice and append it to the list
	for tiff_file in tiff_files:
		loaded_slice = np.array(Image.open(tiff_file))
		loaded_slices.append(loaded_slice)

	# Convert the list of loaded slices to a NumPy array and transpose to get X, Y, Z
	CT = np.stack(loaded_slices, axis=0)
	CT = np.transpose(CT, (2, 1, 0))
	CT = np.flip(CT, axis=1)


	print("Loaded TIFF stack shape = ", CT.shape)


	# Directory containing TIFF image slices and Xtek file = data_directory
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


	# Compile image with relevant metadata from the Xtek file


	CT_with_metadata = {'image': CT,'image_meta_dict' : {
		'spacing': vox_xyz,
		'origin': origin,
		'units': (config['XTekCT']['Units']),
		'sample_name': (config['XTekCT']['Name']),
		'white_level': (config['XTekCT']['WhiteLevel']),
		'mask_radius': (config['XTekCT']['MaskRadius']),
		'spatial_shape': xyz,
		'original_channel_dim': xyz,
		'filename': (config['XTekCT']['Name'])
		}
	}


	metatensor_with_metadata = convert_to_tensor(data=CT_with_metadata,track_meta=True, wrap_sequence=True)

	data = metatensor_with_metadata
	data['image'] = data['image'].unsqueeze(0)

	#print("MetaTensor shape:", metatensor_with_metadata['image'].shape)

	#config = ConfigParser()

	sample = (config['XTekCT']['Name'])
	#image = '_IMAGE'
	samplename = f'CT_of_{sample}'
	#samplename

	export_raw_image_nii = Compose([
		SaveImaged(keys='image',meta_keys='image_meta_dict', output_dir=os.path.join(sp_data_directory, '01_IMAGE_training_data'), output_postfix=(samplename),resample=False)
	])
	
	export_raw_image_nii(data)

	print("CT scan image saved as NiFti (.nii) in --> / ",sp_data_directory," / 01_IMAGE_training_data / 0 /",)
	
	# Run additional code if specified 
	
	if make_png == "yes":
		png_dir = (f'{sp_data_directory}/pngs/')
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
		plt.pcolormesh(CT_onethird_xz_slice.T, cmap='Greys_r')
		plt.title('xz slice @ 1/3 y',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(132)
		plt.pcolormesh(CT_mid_xz_slice.T, cmap='Greys_r')
		plt.title('xz slice @ 1/2 y',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(133)
		plt.pcolormesh(CT_twothird_xz_slice.T, cmap='Greys_r')
		plt.title('xz slice @ 2/3 y',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.savefig(f'{sp_data_directory}/pngs/0_CT_xz_slice_check.png', format='png')
		
		print("Preparing slice check pngs..")
		
		
		CT_onethird_yz_slice = CT_with_metadata['image'][(onethird_x),:,:]
		CT_mid_yz_slice = CT_with_metadata['image'][(mid_x),:,:]
		CT_twothird_yz_slice = CT_with_metadata['image'][(twothird_x),:,:]
		
		plt.subplots(1,3,figsize=fig_size_yz)
		plt.subplot(131)
		plt.pcolormesh(CT_onethird_yz_slice.T, cmap='Greys_r')
		plt.title('yz slice @ 1/3 x',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(132)
		plt.pcolormesh(CT_mid_yz_slice.T, cmap='Greys_r')
		plt.title('yz slice @ 1/2 x',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(133)
		plt.pcolormesh(CT_twothird_yz_slice.T, cmap='Greys_r')
		plt.title('yz slice @ 2/3 x',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.savefig(f'{sp_data_directory}/pngs/0_CT_yz_slice_check.png', format='png')
		
		print("Preparing slice check pngs...")
			
			
		
		
		CT_onethird_xy_slice = CT_with_metadata['image'][:,:,(onethird_z)]
		CT_mid_xy_slice = CT_with_metadata['image'][:,:,(mid_z)]
		CT_twothird_xy_slice = CT_with_metadata['image'][:,:,(twothird_z)]
		
		plt.subplots(1,3,figsize=fig_size_xy)
		plt.subplot(131)
		plt.pcolormesh(CT_onethird_xy_slice.T, cmap='Greys_r')
		plt.title('xy slice @ 1/3 z',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching 
		plt.subplot(132)
		plt.pcolormesh(CT_mid_xy_slice.T, cmap='Greys_r')
		plt.title('xy slice @ 1/2 z',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		plt.subplot(133)
		plt.pcolormesh(CT_twothird_xy_slice.T, cmap='Greys_r')
		plt.title('xy slice @ 2/3 z',fontweight='bold',fontsize=25)
		plt.colorbar(label='HU')
		plt.axis('on')
		plt.gca().set_aspect(1)  # set aspect ratio to 1, so no stretching
		
		plt.savefig(f'{sp_data_directory}/pngs/0_CT_xy_slice_check.png', format='png')


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: python3 STEP_1_CT_tiff_to_nii.py [specimen data directory; tiff stack inside 'zzz_temp/CT'] [make midpoint pngs? (yes/no)]")
	else:
		sp_data_directory = sys.argv[1]
		make_png = sys.argv[2].lower()  # Convert to lowercase for case-insensitive check
		if make_png not in ["yes", "no"]:
			print("Second argument must be 'yes' or 'no'")
		else:
			main(sp_data_directory, make_png)


