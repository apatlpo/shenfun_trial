# shenfun_trial

Play/test with shenfun library that may be found [here](https://github.com/spectralDNS/shenfun).



## Installation

```
conda create --name shenfun -c conda-forge -c spectralDNS python=3.6 shenfun h5py-parallel matplotlib
source activate shenfun
```

## Run

```
mpirun -np 4  python code.py
```

