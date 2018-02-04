# shenfun_trial

Play/test with shenfun library that may be found [here](https://github.com/spectralDNS/shenfun).



## Installation

```
conda create --name shenfun -c conda-forge -c spectralDNS python=3.6 shenfun h5py-parallel matplotlib
source activate shenfun
conda install -c conda-forge jupyter 
pip install line_profiler
pip install numba
```

## Run

```
mpirun -np 4  python code.py
```

## Profiling

Add `@profile` decorators and run:
```
kernprof -lv myscript.py
```

