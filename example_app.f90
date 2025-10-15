! Example app for our GRAIL MLP-ResNet inference (Fortran)
! Loads model from HDF5, runs a self-test, then predicts 12 outputs from 7 inputs
! Adapt as needed for operational use

program example_app
  use, intrinsic :: iso_fortran_env, only: real64
  use mlp_resnet_api
  implicit none ! safety first

  ! CHANGE PATH TO .h5 MODEL LOCATION
  character(len=*), parameter :: H5PATH = "grail_model.h5"

  type(mlp_model) :: m
  real(real64), allocatable :: x_ref(:), y_ref(:),y(:)
  logical :: used_probe
  real(real64) :: x_preset(7)
  integer :: din, dout

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Load model + reference vectors
  call load_model(H5PATH, m, x_ref, y_ref, used_probe)
  din = size(m%A0,2)
  dout = size(m%Ao,1)
  if (din /= 7) then
    write(*,*) "Error: expected 7 inputs; model has ", din
    stop 1
  end if

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Brief description of model structure
  write(*,*)
  call describe_model(m)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Self-test vs embedded reference (should match to precalculated python output)
  write(*,*)
  write(*,*) "Testing model accuracy to known predictions in Python (err should be ~E-08 from float precision)..."
  if (allocated(y)) deallocate(y)
  call forward_into(m, x_ref, y)
  call print_outputs(y, y_ref)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Example inputs (play with these as you wish, or feed your own in)
  ! This predict structure is what you would likely be using in practice operationally
  ! Recall the inputs: gmi_skin_temp, cell_elev, clay, sand, LAI, sm_surface, emiss_c1
  x_preset = (/ 295.0d0, 150.0d0, 0.25d0, 0.45d0, 2.0d0, 0.25d0, 0.95d0 /)

  if (allocated(y)) deallocate(y)
  call predict(m, x_preset, y)

  write(*, *)
  write(*,'(A)') 'Preset input (gmi_skin_temp, elev, clay, sand, LAI, sm, emiss_c1):'
  write(*,'(7(ES12.5,1X))') x_preset
  write(*,'(A)')'Outputs (12 GMI channels):'

  ! print the result
  call print_outputs(y)

  write(*,*)
  write(*,*) "Execution complete!"

end program example_app
