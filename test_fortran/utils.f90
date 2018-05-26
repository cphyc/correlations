module mod_integration
  use fgsl
  use, intrinsic :: iso_c_binding
  implicit none

  private
  integer, parameter :: dp = c_double
  real(dp), allocatable, dimension(:) :: k, Pk, k2Pk
  integer :: Nk
  real(dp), parameter :: pi = atan(1._dp) * 4, twopi = 2*pi, twopi2 = 2*pi**2, eightpi3 = 8*pi**3

  public :: init, integrate, sigma

  !------------------------------------------------------------
  ! Type definition to communicate between/from C
  !------------------------------------------------------------
  type params1
     real(dp) :: dx, dy, dz, R1, R2
     integer :: ikx, iky, ikz, ikk, ii
  end type params1

  type, extends(params1) :: params2
     real(dp) :: sin_theta, cos_theta
  end type params2

  ! Workspace for GSL
  type(fgsl_integration_workspace) :: wk1, wk2
  integer(fgsl_size_t), parameter :: nmax=1000

contains
  
  subroutine init(k_, Pk_, N)
    real(dp), dimension(N), intent(in) :: k_, Pk_
    integer, intent(in) :: N

    if (allocated(k)) deallocate(k, Pk, k2Pk)
    allocate(k(N), Pk(N), k2Pk(N))

    k = k_
    Pk = Pk_
    k2Pk = k2Pk
    Nk = size(k, 1)
  end subroutine init

  subroutine sigma(i, R, integrand)
    integer, intent(in) :: i
    real(dp), intent(in) :: R

    real(dp), intent(out) :: integrand

    real(dp) :: prev, cur, kk, Pkk, kprev

    integer :: j

    integrand = 0
    prev = 0
    kprev = 0

    do j = 1, Nk
       kk = k(i)
       Pkk = Pk(i)
       cur = kk**(2+2*i) * Pkk * exp(-(kk*R)**2)

       if (j > 0) then
          integrand = integrand + (prev + cur) / 2 * (kk - kprev)
       end if
       prev = cur
       kprev = kk
    end do

    integrand = sqrt(integrand / twopi2)
  end subroutine sigma

  real(dp) function integrate(dx, dy, dz, R1, R2, ikx, iky, ikz, ikk)
    real(dp), intent(in) :: dx, dy, dz, R1, R2
    integer, intent(in) :: ikx, iky, ikz, ikk

    ! GSL stuff
    real(fgsl_double) :: result, error
    integer(fgsl_int) :: status
    type(c_ptr) :: ptr
    type(fgsl_function) :: f1_obj

    type(params1), target :: params

    ! Fill parameters
    params%dx  = dx
    params%dy  = dy
    params%dz  = dz
    params%R1  = R1
    params%R2  = R2
    params%ikx = ikx
    params%iky = iky
    params%ikz = ikz
    params%ikk = ikk
    params%ii  = ikx + iky + ikz - ikk
    
    wk1 = fgsl_integration_workspace_alloc(nmax)
    wk2 = fgsl_integration_workspace_alloc(nmax)

    ! Get C pointers
    ptr = c_loc(params)
    f1_obj = fgsl_function_init(f1, ptr)

    ! Integrate over theta
    status = fgsl_integration_qag(f1_obj, 0.0_fgsl_double, pi * 1.0_fgsl_double, &
         0.0_fgsl_double, 1.0e-7_fgsl_double, nmax, FGSL_INTEG_GAUSS61, wk1, result, error)
    call fsgl_function_free(f1_obj)
    call fgsl_integration_workspace_free(wk2)
    call fgsl_integration_workspace_free(wk1)

    integrate = result
    
  end function integrate

  !------------------------------------------------------------
  ! Integrand stuff
  !------------------------------------------------------------
  function f1(theta, params) bind(c)
    real(c_double), value :: theta
    type(c_ptr), value :: params
    real(c_double) :: f1

    type(params1), pointer :: p
    type(params2), target  :: p2

    ! GSL stuff
    type(c_ptr) :: ptr
    type(fgsl_function) :: f2_obj
    real(fgsl_double) :: result, error
    integer(fgsl_int) :: status

    call c_f_pointer(params, p)

    ! Copy data over
    p2%dx  = p%dx
    p2%dy  = p%dy
    p2%dz  = p%dz
    p2%R1  = p%R1
    p2%R2  = p%R2
    p2%ikx = p%ikx
    p2%iky = p%iky
    p2%ikz = p%ikz
    p2%ikk = p%ikk
    p2%ii  = p%ii

    ! Compute theta-specific values
    p2%sin_theta = sin(theta)
    p2%cos_theta = cos(theta)

    ! Get C pointer
    ptr = c_loc(p2)
    f2_obj = fgsl_function_init(f2, ptr)
    
    ! Integrate over phi
    status = fgsl_integration_qag(f2_obj, 0.0_fgsl_double, twopi * 1.0_fgsl_double, &
         0.0_fgsl_double, 1.0e-7_fgsl_double, nmax, FGSL_INTEG_GAUSS61, wk2, result, error)

    call fsgl_function_free(f2_obj)

    f1 = result

  end function f1

  function f2(phi, params) bind(c)
    real(c_double), value :: phi
    type(c_ptr), value :: params
    real(c_double) :: f2

    type(params2), pointer :: p

    real(c_double) :: kx, ky, kz
    real(c_double) :: sincos, sinsin, cos, iipio2
    real(c_double) :: prev, cur, kk, kprev, res
    integer :: i

    ! Get data from C pointer
    call c_f_pointer(params, p)

    ! Precompute some stuff
    sincos = p%sin_theta * cos(phi)
    sinsin = p%sin_theta * sin(phi)
    iipio2 = p%ii * pi / 2

    ! Initialize variables
    res = 0
    kprev = 0
    prev = 0

    ! Integrate in k direction using trapezoidal rule
    do i = 1, Nk
       kk = k(i)
       kx = kk * sincos
       ky = kk * sinsin
       kz = kk * p%cos_theta

       cur = k2Pk(i) * cos((kx * p%dx + ky * p%dy + kz * p%dz) - iipio2) * &
            exp(-kk**2*(p%R1**2 + p%R2**2)/2)

       if (p%ikx /= 0) cur = cur * kx**p%ikx
       if (p%iky /= 0) cur = cur * ky**p%iky
       if (p%ikz /= 0) cur = cur * kz**p%ikz
       if (p%ikk /= 0) cur = cur / kk**p%ikk

       if (i > 0) &
            res = res + (kk - kprev) * (prev + cur) / 2

       kprev = kk
       prev = cur
    end do

    f2 = res * p%sin_theta / eightpi3
  end function f2

end module mod_integration
