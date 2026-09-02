# Before running this .py code in terminal, set the path as
# $ export PATH=~/miniforge3/envs/capillarywave/bin:$PATH

# To RUN:
# $ 
# vpp=0p25
# letters=(k l m n o p)

# start_rep=11

# for i in "${!letters[@]}"; do
#     rep=$((start_rep + i))
#     dir="Ca_ac_0p001250_rep${rep}"
#     letter=${letters[$i]}

#     out="Q_1D_${vpp}vpp_${letter}.h5"

#     python 2Dto1D_conversion.py "$vpp" "$dir" "$out" &
# done

# wait
# echo "All done"


import numpy as np
import h5py
import os
import sys

# READ specific hdf5 file
# if len(sys.argv) < 5:
#     print("Usage: python script.py <input_file> <output_file> <input_dir> <vpp_dir>")
#     sys.exit(1)

# # Get filenames from command line arguments
# vpp_dir        = sys.argv[1]   # e.g., 0p25
# input_dir      = sys.argv[2]   # e.g., Ca_ac_0p002080_rep11
# read_file_name = sys.argv[3]  # e.g., 11252025_b_0.3vpp_data_roi-none_cal-true.hdf5
# save_file_name = sys.argv[4]  # e.g., Q_1D_0p30vpp_b.h5

# base_input = '/disk/hyk049/DHM_new_experiment'
# base_output = '/home/jonas/ucsd_thesis/DHM_new_1Dcenter'

# read_file_path = os.path.join(base_input, vpp_dir, input_dir, read_file_name)
# save_file_path = os.path.join(base_output, vpp_dir, save_file_name)


# READ only one hdf5 file in the input directory
if len(sys.argv) < 4:
    print("Usage: python script.py <vpp_dir> <input_dir> <output_file>")
    sys.exit(1)

vpp_dir   = sys.argv[1]
input_dir = sys.argv[2]
save_file_name = sys.argv[3]

base_input = '/disk/hyk049/DHM_new_experiment'
base_output = '/home/jonas/ucsd_thesis/DHM_new_1Dcenter'

folder_path = os.path.join(base_input, vpp_dir, input_dir)

files = [
    f for f in os.listdir(folder_path)
    if f.endswith('.hdf5') and os.path.isfile(os.path.join(folder_path, f))
]

if len(files) == 0:
    raise RuntimeError(f"No .hdf5 file found in {folder_path}")
if len(files) > 1:
    raise RuntimeError(f"Multiple .hdf5 files found in {folder_path}: {files}")

read_file_name = files[0]
read_file_path = os.path.join(folder_path, read_file_name)
print(f"Found input file: {read_file_name}")

# ensure output dir exists
os.makedirs(os.path.join(base_output, vpp_dir), exist_ok=True)
save_file_path = os.path.join(base_output, vpp_dir, save_file_name)


data_dict = {}

# Read file & save it in a new dictionary
print(f"Reading the file {read_file_name}...")
with h5py.File(read_file_path, 'r') as f:
    for key in f['main'].keys():
        data_dict[key] = f['main'][key][:]
        
with h5py.File(read_file_path, 'r') as f:
    t = f['meta']['t'][:]
    x = f['meta']['x'][:]
    y = f['meta']['y'][:]
    
# time_steps = len(data_dict)
time_steps = len(sorted(data_dict.keys(), key=int))
rows, cols = 200, 200

Q = np.zeros((rows * cols, time_steps))

print("Converting data to 2D matrix...")
for i, key in enumerate(sorted(data_dict.keys(), key=int)):
    Q[:, i] = data_dict[key].reshape(-1)
    
del data_dict
Q_subspace = Q.reshape(200, 200, Q.shape[1])            

Q_1D = Q_subspace[:,100,:]  # center point

print(f"Saving 1D data to {save_file_name}...")
with h5py.File(save_file_path, "w") as f:
    f.create_dataset("Q_1D", data=Q_1D)
    f.create_dataset("t", data=t.astype(np.float64))
    f.create_dataset("x", data=x.astype(np.float64))
    
print("Data conversion complete.")
