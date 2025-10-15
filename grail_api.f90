! Fortrain Grail_API main file
! Contains all the helper functions etc. to load the model and run data through it
! Use in tandem with example_app.f90 to see how to hook this

module mlp_resnet_api
  use, intrinsic :: iso_fortran_env, only: real64
  use hdf5
  implicit none
  private

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! This is the fortran-ified version of our model with baked affines. 
  ! All reduced to a few matrix multiplications
  type, public :: mlp_model
    ! Front affine: h0 = ReLU(A0*x + c0)
    real(real64), allocatable :: A0(:,:), c0(:)

    ! Residual block (pre-activation)- first BN + linear
    real(real64), allocatable :: bn1_gamma(:), bn1_beta(:), bn1_mean(:), bn1_var(:)
    real(real64) :: bn1_eps
    real(real64), allocatable :: W1(:,:), b1(:)

    ! Residual block - second BN + linear
    real(real64), allocatable :: bn2_gamma(:), bn2_beta(:),bn2_mean(:), bn2_var(:)
    real(real64) :: bn2_eps
    real(real64), allocatable :: W2(:,:), b2(:)

    ! Skip/shortcut projection
    real(real64), allocatable :: skipW(:,:), skipb(:)

    ! Narrowing + output
    real(real64), allocatable :: An(:,:), cn(:)
    real(real64), allocatable :: Ao(:,:), co(:)
  end type mlp_model

  public :: load_model
  public :: forward_into
  public :: predict
  public :: run_self_test_from_path
  public :: describe_model
  public :: print_outputs

contains
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Math subroutines (e.g., a basic relu implementation)
  pure subroutine relu_inplace(x)
    real(real64), intent(inout) :: x(:)
    x = merge(x, 0.0_real64, x >0.0_real64)
  end subroutine

  pure subroutine lin_into(W, b, x, y)
    real(real64), intent(in) :: W(:,:), b(:), x(:)
    real(real64), intent(out) :: y(:)
    y = matmul(W, x) + b
  end subroutine

  pure subroutine batchnorm_inplace(x, g, beta, meanv, varv, eps)
    real(real64), intent(inout) :: x(:)
    real(real64), intent(in) :: g(:), beta(:),meanv(:),varv(:)
    real(real64), intent(in)  :: eps
    x = ((x - meanv) / sqrt(varv + eps)) * g + beta
  end subroutine

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Model forward pass
  pure subroutine forward_into(m, x, y)
    type(mlp_model), intent(in) :: m
    real(real64), intent(in) :: x(:)
    real(real64), allocatable, intent(out) :: y(:)

    real(real64), allocatable :: h0(:), z(:), u1(:), u2(:), skipv(:), h1(:), h2(:)
    integer :: w0, wres, wnarrow, dout

    ! sizes (centralized so we only allocate once)
    w0      = size(m%A0, 1)   ! front input width
    wres    = size(m%W1, 1)   ! residual width
    wnarrow = size(m%An, 1)   ! narrow width
    dout    = size(m%Ao, 1)   ! outputs

    allocate(h0(w0),z(wres),u1(wres),u2(wres),skipv(wres),h1(wres),h2(wnarrow))
    allocate(y(dout))

    ! Front affine + ReLU
    call lin_into(m%A0, m%c0, x, h0)
    call relu_inplace(h0)

    ! Residual block pre-act #1
    z = h0
    call batchnorm_inplace(z, m%bn1_gamma, m%bn1_beta, m%bn1_mean, m%bn1_var, m%bn1_eps)
    call relu_inplace(z)
    call lin_into(m%W1, m%b1, z, u1)

    ! Residual block pre-act #2
    z = u1
    call batchnorm_inplace(z, m%bn2_gamma, m%bn2_beta, m%bn2_mean, m%bn2_var, m%bn2_eps)
    call relu_inplace(z)
    call lin_into(m%W2, m%b2, z, u2)

    ! Skip + add (projection makes dims match)
    call lin_into(m%skipW, m%skipb, h0, skipv)
    h1 = skipv + u2

    ! Narrow + relu we made earlier
    call lin_into(m%An, m%cn, h1, h2)
    call relu_inplace(h2)

    ! Output (linear)
    call lin_into(m%Ao, m%co, h2, y)
  end subroutine forward_into

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! HDF helper files for readinging scalars/1d/2d vars (needed for filling the model)
  subroutine read_scalar(fid, name, val)
    integer(HID_T), intent(in) :: fid
    character(*), intent(in) :: name
    real(real64), intent(out):: val
    integer(HID_T) :: dset
    integer :: ierr
    integer(HSIZE_T) :: dims(1)
    dims = (/ 1_HSIZE_T /)
    call h5dopen_f(fid, name, dset, ierr); if (ierr/=0) stop "read_scalar: open"
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, val, dims, ierr)
    call h5dclose_f(dset, ierr)
  end subroutine

  subroutine read_1d(fid, name, arr)
    integer(HID_T), intent(in) :: fid
    character(*), intent(in) :: name
    real(real64), allocatable, intent(out) :: arr(:)
    integer(HID_T) :: dset, space
    integer :: ierr, rank
    integer(HSIZE_T) :: dims(1), maxdims(1)
    call h5dopen_f(fid, name, dset, ierr); if (ierr/=0) stop "read_1d: open"
    call h5dget_space_f(dset, space, ierr)
    call h5sget_simple_extent_ndims_f(space, rank, ierr)
    if (rank /= 1) stop "read_1d: rank mismatch"
    call h5sget_simple_extent_dims_f(space, dims, maxdims, ierr)
    allocate(arr(dims(1)))
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, arr, dims, ierr)
    call h5sclose_f(space, ierr); call h5dclose_f(dset, ierr)
  end subroutine

  ! TODO: combine these reads into 1 func for cohesion
  subroutine read_2d(fid, name, arr)
    integer(HID_T), intent(in) :: fid
    character(*), intent(in) :: name
    real(real64), allocatable, intent(out) :: arr(:,:)
    integer(HID_T) :: dset, space
    integer :: ierr, rank
    integer(HSIZE_T) :: dims(2), maxdims(2)
    real(real64), allocatable :: tmp(:,:)

    call h5dopen_f(fid, name, dset, ierr); if (ierr/=0) stop "read_2d: open"
    call h5dget_space_f(dset, space, ierr)
    call h5sget_simple_extent_ndims_f(space, rank, ierr)
    if (rank /= 2) stop "read_2d: rank mismatch"
    call h5sget_simple_extent_dims_f(space, dims, maxdims, ierr)

    allocate(tmp(dims(1), dims(2)))
    call h5dread_f(dset, H5T_NATIVE_DOUBLE, tmp, dims, ierr)
    call h5sclose_f(space, ierr); call h5dclose_f(dset, ierr)

    ! Transpose C-order (nout,nin) to Fortran layout
    allocate(arr(dims(2), dims(1)))
    arr = transpose(tmp)
  end subroutine

  subroutine try_read_1d(fid, name, arr, ok)
    integer(HID_T), intent(in) :: fid
    character(*), intent(in) :: name
    real(real64), allocatable, intent(out) :: arr(:)
    logical, intent(out) :: ok
    integer(HID_T) :: dset
    integer :: ierr
    ok = .false. ! sponge
    call h5dopen_f(fid, name, dset, ierr)
    if (ierr == 0) then
       call h5dclose_f(dset, ierr)
       call read_1d(fid, name, arr)
       ok = .true.
    end if
  end subroutine

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Top-level I/O for interacting with the model more easily
  subroutine load_model(h5_path, m, x_ref, y_ref, used_probe)
    character(*), intent(in) :: h5_path
    type(mlp_model),  intent(out) :: m
    real(real64), allocatable,intent(out), optional :: x_ref(:), y_ref(:)
    logical, intent(out), optional :: used_probe

    integer :: ierr
    integer(HID_T) :: fid
    logical :: have_probe ! we can also save version without probe data if needed, so check

    call h5open_f(ierr); if (ierr/=0) stop "HDF5 open failed"
    call h5fopen_f(trim(h5_path), H5F_ACC_RDONLY_F, fid, ierr)
    if (ierr/=0) stop "HDF5 file open failed"

    ! Core params
    call read_2d(fid, "A0",  m%A0); call read_1d(fid, "c0",  m%c0)
    call read_1d(fid, "bn1_gamma", m%bn1_gamma); call read_1d(fid, "bn1_beta", m%bn1_beta)
    call read_1d(fid, "bn1_mean", m%bn1_mean); call read_1d(fid, "bn1_var",  m%bn1_var)
    call read_scalar(fid, "bn1_eps", m%bn1_eps)
    call read_2d(fid, "W1", m%W1); call read_1d(fid, "b1",  m%b1)
    call read_1d(fid, "bn2_gamma", m%bn2_gamma); call read_1d(fid, "bn2_beta", m%bn2_beta)
    call read_1d(fid, "bn2_mean", m%bn2_mean);  call read_1d(fid, "bn2_var",  m%bn2_var)
    call read_scalar(fid, "bn2_eps",m%bn2_eps)
    call read_2d(fid, "W2", m%W2);  call read_1d(fid, "b2",  m%b2)
    call read_2d(fid, "P", m%skipW); call read_1d(fid, "p",   m%skipb)
    call read_2d(fid, "An", m%An); call read_1d(fid, "cn",  m%cn)
    call read_2d(fid, "Ao", m%Ao); call read_1d(fid, "co",  m%co)

    ! Optional reference vectors (probe preferred though)
    if (present(x_ref) .or. present(y_ref) .or. present(used_probe)) then
       have_probe = .false.
       if (present(used_probe)) used_probe = .false.
       if (present(x_ref)) call try_read_1d(fid, "probe_x", x_ref, have_probe)
       if (have_probe) then
          if (present(y_ref)) call read_1d(fid, "probe_y_py", y_ref)
          if (present(used_probe)) used_probe = .true.
       else
          if (present(x_ref)) call read_1d(fid, "test_x", x_ref)
          if (present(y_ref)) call read_1d(fid, "test_y", y_ref)
       end if
    end if

    call h5fclose_f(fid, ierr)
    call h5close_f(ierr)

    write(*,*) "Model successfully loaded!"
  end subroutine

  subroutine predict(m, x, y)
    type(mlp_model), intent(in)  :: m
    real(real64),intent(in) :: x(:)
    real(real64), allocatable, intent(out) :: y(:)
    call forward_into(m, x, y) ! this should allocate y
  end subroutine

  subroutine run_self_test_from_path(h5_path, max_abs_err, verbose)
    character(*), intent(in) :: h5_path
    real(real64), intent(out) :: max_abs_err
    logical, intent(in), optional :: verbose

    type(mlp_model) :: m
    real(real64), allocatable :: x(:), y(:), y_ref(:)
    logical :: used_probe_
    logical :: chatty
    integer :: i

    ! flag to allow for more detailed prints
    chatty = .false.; if (present(verbose)) chatty = verbose

    call load_model(h5_path, m, x, y_ref, used_probe_)
    call forward_into(m, x, y)
    max_abs_err = maxval(abs(y - y_ref))

    ! TODO: This formatting could be improved?
    if (chatty) then
       call describe_model(m)
       write(*,'(A,ES12.5)') "max |y - ref| = ", max_abs_err
       write(*,*)
       write(*,*) "idx        y_fortran            y_ref               |diff|"
       do i = 1, size(y)
          write(*,'(I3,2X,ES16.8,2X,ES16.8,2X,ES12.4)') i, y(i), y_ref(i), abs(y(i)-y_ref(i))
       end do
    end if
  end subroutine

  ! Basic model overview (so you can keep track without going to README)
  subroutine describe_model(m, unit)
    type(mlp_model), intent(in) :: m
    integer, intent(in), optional :: unit
    integer :: u
    u = 6; if (present(unit)) u = unit
    write(u,*) "=== Model Description (MLP with pre-activated residual block) ==="
    write(u,'(A,I0)') "Input dimension:            ", size(m%A0,2)
    write(u,'(A,I0)') "Front affine width:         ", size(m%A0,1)
    write(u,'(A,I0)') "Residual block width:       ", size(m%W1,1)
    write(u,'(A,I0)') "Narrowing width:            ", size(m%An,1)
    write(u,'(A,I0)') "Output dimension:           ", size(m%Ao,1)
    write(u,*) "Layers:"
    write(u,*) "  1) Affine (A0,c0) + ReLU"
    write(u,*) "  2) Residual path:"
    write(u,*) "     - BN1 -> ReLU -> Linear(W1,b1)"
    write(u,*) "     - BN2 -> ReLU -> Linear(W2,b2)"
    write(u,*) "     - Skip projection: skipW*h0 + skipb"
    write(u,*) "     - Residual add -> h1"
    write(u,*) "  3) Narrow: Affine(An,cn) + ReLU"
    write(u,*) "  4) Output: Affine(Ao,co)"
    write(u,*) "-----------------------------------------------------------------"
  end subroutine

  ! basic print
  subroutine print_outputs(y, y_ref)
    real(real64), intent(in) :: y(:)
    real(real64), intent(in), optional :: y_ref(:)
    integer :: i
    if (present(y_ref)) then
      write(*,*) "idx        y_pred               y_ref               |diff|"
      do i=1,size(y)
        write(*,'(I3,2X,ES16.8,2X,ES16.8,2X,ES12.4)') i, y(i), y_ref(i), abs(y(i)-y_ref(i))
      end do
      write(*,'(A,ES12.5)') "max |y - ref| = ", maxval(abs(y - y_ref))
    else
      write(*,*) "idx        y_pred"
      do i=1,size(y)
        write(*,'(I3,2X,ES16.8)') i, y(i)
      end do
    end if
  end subroutine

end module mlp_resnet_api
