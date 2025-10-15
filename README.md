# MLP-ResNet (GRAIL) Fortran Inference API

Here, we put together a small Fortran API to load our folded MLP-ResNet (GRAIL) from HDF5 and run inference. We also include a minimal example app.

This is a living document / WIP, so some things might not work perfectly out of the gate. However, the API is fairly basic, and should integrate fairly easy with current models. You might need to tweak things for different fortran versions etc., but the underlying structure should work.

In the example app, we show how to provide your own data to the model to get the corresponding 12 GMI channel output. Inference is quite fast. 
Also, no need to worry about input scaling, we prebake this into the A0 front affine.

## Files
- `grail_api.f90` - API (`load_model`, `forward_into`, `predict`, `describe_model`, `print_outputs`)
- `example_app.f90` - demo (self-test + one preset input)
- `grail_export.h5` - exported model (from Python)

## Requirements
- **Fortran compiler:** gfortran 9+ (or ifx/ifort/nvfortran).
- **HDF5 with Fortran bindings:**
  - Prefer wrapper **`h5fc`**; or
  - `pkg-config` for `hdf5-fortran`; or
  - Manual `-I` / `-L -lhdf5_fortran -lhdf5`.

## Model file contents
HDF5 datasets (names must match) - this is provided:
- Front: `A0`, `c0`
- ResBlock1: `bn1_gamma`, `bn1_beta`, `bn1_mean`, `bn1_var`, `bn1_eps`, `W1`, `b1`
- ResBlock2: `bn2_gamma`, `bn2_beta`, `bn2_mean`, `bn2_var`, `bn2_eps`, `W2`, `b2`
- Skip/Narrow/Out: `P`, `p`, `An`, `cn`, `Ao`, `co`
- Optional test vectors: `probe_x` + `probe_y_py` (preferred) or `test_x` + `test_y`

## Build
```bash
h5fc -c -O3 -std=f2008 -Wall -Wextra grail_api.f90
h5fc -O3 -std=f2008 -o example_app grail_api.o example_app.f90
```

## Run
./example_app

## Questions
Reach out to Sarah Ringerud (sarah.e.ringerud@nasa.gov) or Fraser King (fraser.d.king@nasa.gov) if you have further questions# grail-fortran
