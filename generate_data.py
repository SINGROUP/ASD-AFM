import os
import time
import numpy as np
from data_generation import construct_generator

# Settings
PPMdir = './ProbeParticleModel/'    # Path to the ProbeParticleModel repository
Moldir = './Molecules3/'            # Path to the molecule geometry files
dataset = 'light'                   # Or 'heavy'. Which dataset to generate
data_dir = './Data_%s/' % dataset   # Directory where to save data

# Generator options
gen_args = {
    'PPMdir'            : PPMdir,       # Probe Particle Model repository directory
    'preName'           : Moldir,       # Molecules are read from filename = preName + molecules[imol] + postName
    'postName'          : '.xyz',
    'Ymode'             : 'D-S-H',      # Generator descriptor output mode
    'batch_size'        : 30,           # Batch size
    'nBestRotations'    : 30,           # Number of rotations for each molecule
    'distAbove'         : 5.0,          # Distance between the scan start height and highest atom in the scan
    'distAboveDelta'    : 0.25,         # Random variation range(+-) for distAbove
    'scan_dim'          : (128,128,20), # Scan size (voxels), final z-size = scan_dim[2]-df_weight_steps (df_weight_steps = 10 by default)
    'iZPPs'             : [8],          # Probe particle(s) (8=O, 54=Xe)
    'Qs'                : [-0.1],       # Probe particle charges
    'tip_type'          : 'monopole',   # Tip electrostatics model (monopole, dipole, quadrupole)
    'maxTilt0'          : 0.5,
    'diskMode'          : 'sphere',
    'dzmax'             : 1.2,
    'dzmax_s'           : 1.2
}

# Molecules
if dataset == 'light':
    N_train        = 5000      # Number of training molecules from light molecule set
    N_val          = 1500      # Number of validation molecules from light molecule set
    N_test         = 2500      # Number of test molecules from light molecule set
    N_train_h      = 0         # Number of training molecules from heavy molecule set
    N_val_h        = 0         # Number of validation molecules from heavy molecule set
    N_test_h       = 0         # Number of test molecules from heavy molecule set
elif dataset == 'heavy':
    N_train        = 3500
    N_val          = 900
    N_test         = 1200
    N_train_h      = 2500
    N_val_h        = 600
    N_test_h       = 1200
else:
    raise ValueError('Invalid dataset')

# Heavy molecules
train_molecules = ['heavy/'+str(n) for n in range(N_train_h)]
val_molecules = ['heavy/'+str(n) for n in range(N_train_h,N_train_h+N_val_h)]
test_molecules = ['heavy/'+str(n) for n in range(N_train_h+N_val_h,N_train_h+N_val_h+N_test_h)]

# Light molecules
train_molecules += ['light/'+str(n) for n in range(N_train)]
val_molecules += ['light/'+str(n) for n in range(N_train,N_train+N_val)]
test_molecules += ['light/'+str(n) for n in range(N_train+N_val,N_train+N_val+N_test)]

# Make sure save directory exists
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Generate dataset
start_time = time.time()
counter = 1
total_len = len(train_molecules)+len(val_molecules)+len(test_molecules)
for mode, molecules in zip(['train', 'val', 'test'], [train_molecules, val_molecules, test_molecules]):

    # Make generator
    gen = construct_generator(molecules=molecules, **gen_args)

    # Make sure target directory exists
    target_dir = data_dir+mode+'/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Generate batches
    for i in range(len(gen)):
        batch = gen.next()
        np.savez(target_dir+'batch_%d.npz'%i, batch[0].astype(np.float32), batch[1].astype(np.float32))
        eta = (time.time() - start_time)/counter * (total_len - counter)
        print('Generated %s batch %d/%d - ETA: %ds' % (mode, i+1, len(gen), eta))
        counter += 1

print('Total time taken: %d' % (time.time() - start_time))
