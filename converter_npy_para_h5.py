"""Converte os .npy agrupados em um unico campos.h5 (rode na sua maquina: pip install h5py)."""
import glob, os, numpy as np, h5py
with h5py.File("campos.h5", "w") as h:
    for f in sorted(glob.glob("campos_agrupados/*.npy")):
        h.create_dataset(os.path.basename(f)[:-4], data=np.load(f), compression="gzip")
print("campos.h5 criado")
