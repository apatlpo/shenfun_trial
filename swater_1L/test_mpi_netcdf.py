from mpi4py import MPI
import numpy as np
from netCDF4 import Dataset
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

nc = Dataset('parallel_tst.nc','w',parallel=True)
d = nc.createDimension('dim',4)
v = nc.createVariable('var', np.int, 'dim')
v[rank] = rank
nc.close()


