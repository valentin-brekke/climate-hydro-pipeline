#!/bin/bash -l
# Build the hydro/processing env for climate-hydro-pipeline on Myriad (no hiroace).
set -euo pipefail
ROOT=~/Scratch/climate-hydro
ENV=$ROOT/envs/hydro
cd $ROOT/deps
for r in DiffHydro DiffRoute xtensor; do
  [[ -d $r ]] || git clone -q https://github.com/TristHas/$r.git
  echo "$r $(git -C $r rev-parse HEAD)"
done
conda create -y -p $ENV --override-channels -c conda-forge \
  python=3.11 numpy pandas xarray zarr dask netcdf4 h5netcdf scipy \
  geopandas shapely contextily networkx tqdm joblib matplotlib \
  hvplot holoviews geoviews panel ipykernel ipywidgets jupyter_bokeh pytest pip
$ENV/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$ENV/bin/pip install -e $ROOT/deps/xtensor -e $ROOT/deps/DiffRoute -e $ROOT/deps/DiffHydro
$ENV/bin/pip freeze > $ROOT/envs/hydro_freeze.txt
$ENV/bin/python -c "import torch, triton, xtensor, diffroute, diffhydro; print(\"IMPORT OK torch\", torch.__version__, \"triton\", triton.__version__)"
echo INSTALL_DONE
