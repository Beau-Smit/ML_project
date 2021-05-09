import h5py

fpath = "C:/Users/Beau Smit/Documents/wsl/precipitation_data/3B-HHR.MS.MRG.3IMERG.20000601-S020000-E022959.0120.V06B.HDF5"
f = h5py.File(fpath, 'r')
import pdb; pdb.set_trace()

dset = f['Grid']

dset.shape

print(f)
print(dir(f))
print(dset)
print(dir(dset))
