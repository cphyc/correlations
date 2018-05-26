module mod_utils
  use fgsl
  use, intrinsic :: iso_c_binding

  use mod_constants
  implicit none

  private
  real(dp), allocatable, dimension(:) :: k, Pk, k2Pk, tmp, dk, kpart
  integer :: Nk

  public :: init, integrate, sigma

  !------------------------------------------------------------
  ! Type definition to communicate between/from C
  !------------------------------------------------------------
  ! type :: params1
  !    real(dp) :: dx, dy, dz, R1, R2
  !    integer :: ikx, iky, ikz, ikk, ii
  ! end type params1

  type :: params2
     real(dp) :: dx, dy, dz, R1, R2
     integer :: ikx, iky, ikz, ikk, ii
     real(dp) :: sin_theta, cos_theta
  end type params2

  ! Workspace for GSL
  type(fgsl_integration_workspace) :: wk1, wk2
  integer(fgsl_size_t), parameter :: nmax=1000

contains

  subroutine init(k_, Pk_, N)
    ! Allocate and store the power spectrum
    real(dp), dimension(N), intent(in) :: k_, Pk_
    integer, intent(in) :: N

    integer :: i

    if (allocated(k)) deallocate(k, Pk, k2Pk, dk, kpart)
    allocate(k(N), Pk(N), k2Pk(N), tmp(N), dk(N-1), kpart(N))

    k = k_
    Pk = Pk_
    k2Pk = k**2 * Pk
    Nk = size(k, 1)
    kpart = 0

    do i = 1, Nk-1
       dk(i) = k(i+1) - k(i)
    end do

  end subroutine init

  subroutine sigma(ii, R, res)
    ! Evaluate the variance of the i-th derivative (or antiderivative
    ! if i < 0) of the field, smoothed at scale R
    integer, intent(in) :: ii
    real(dp), intent(in) :: R

    real(dp), intent(out) :: res

    real(dp) :: prev, cur, kk, Pkk, k2Pkk, kprev, integrand

    integer :: i

    integrand = 0
    prev = 0
    kprev = 0

    tmp = k2Pk * k**(2*ii) * exp(-(k*R)**2)
    integrand = sum((tmp(1:) + tmp(:-1)) * dk) / 2

    res = sqrt(integrand / twopi2)
  end subroutine sigma

  subroutine integrate(dx, dy, dz, R1, R2, ikx, iky, ikz, ikk, res) bind(c)
    ! Evaluate the integral
    ! int_0^\infty dk
    !   int_0^pi dtheta
    !     int_0^2pi dphi
    !       k^2 Pk W(kR1) W(kR2) exp(-ik.r) kx^ikx ky^iky kz^ikz / k*ikk

    real(dp), intent(in) :: dx, dy, dz, R1, R2
    integer, intent(in) :: ikx, iky, ikz, ikk

    real(dp), intent(out) :: res

    ! GSL stuff
    real(fgsl_double) :: result, error
    integer(fgsl_int) :: status
    type(c_ptr) :: ptr
    type(fgsl_function) :: f1_obj

    type(params2), target :: params

    ! Detect when the result should be 0
    if ( (mod(ikx, 2) == 1 .and. dx == 0) .or. &
         (mod(iky, 2) == 1 .and. dy == 0) .or. &
         (mod(ikz, 2) == 1 .and. dz == 0)) then
       res = 0
       return
    end if

    ! Precompute quantities
    kpart = k2Pk * exp(-k**2 * (R1**2 + R2**2) / 2) * k**(ikx + iky + ikz - ikk)

    ! Fill parameters
    params%dx  = dx
    params%dy  = dy
    params%dz  = dz
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
         1.0e-3_fgsl_double, 1.0e-5_fgsl_double, nmax, FGSL_INTEG_GAUSS15, wk1, result, error)
    call fgsl_function_free(f1_obj)
    call fgsl_integration_workspace_free(wk2)
    call fgsl_integration_workspace_free(wk1)

    res = result

  end subroutine integrate

  !------------------------------------------------------------
  ! Integrand stuff
  !------------------------------------------------------------
  function f1(theta, params) bind(c)
    real(c_double), value :: theta
    type(c_ptr), value :: params
    real(c_double) :: f1

    type(params2), pointer :: p

    ! GSL stuff
    type(c_ptr) :: ptr
    type(fgsl_function) :: f2_obj
    real(fgsl_double) :: result, error
    integer(fgsl_int) :: status

    call c_f_pointer(params, p)

    ! Compute theta-specific values
    p%sin_theta = sin(theta)
    p%cos_theta = cos(theta)

    f2_obj = fgsl_function_init(f2, params)

    ! Integrate over phi
    status = fgsl_integration_qag(f2_obj, 0.0_fgsl_double, twopi * 1.0_fgsl_double, &
         1.0e-3_fgsl_double, 1.0e-5_fgsl_double, nmax, FGSL_INTEG_GAUSS15, wk2, result, error)

    call fgsl_function_free(f2_obj)

    f1 = result

  end function f1

  function f2(phi, params) bind(c)
    real(c_double), value :: phi
    type(c_ptr), value :: params
    real(c_double) :: f2

    type(params2), pointer :: p

    real(c_double) :: sincos, sinsin, cos, iipio2
    real(c_double) :: prev, cur, kk, kprev, res

    ! real(c_double) :: kx, ky, kz, foo
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

       cur = kpart(i) * cos(kk * (sincos * p%dx + sinsin * p%dy + p%cos_theta * p%dz) - iipio2)

       if (p%ikx /= 0) cur = cur * sincos**p%ikx
       if (p%iky /= 0) cur = cur * sinsin**p%iky
       if (p%ikz /= 0) cur = cur * p%cos_theta**p%ikz

       if (i > 0) &
            res = res + (kk - kprev) * (prev + cur) / 2

       kprev = kk
       prev = cur
    end do

    f2 = res * p%sin_theta / eightpi3
  end function f2

end module mod_utils
