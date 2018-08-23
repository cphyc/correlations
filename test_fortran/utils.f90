module mod_utils
  use fgsl
  use, intrinsic :: iso_c_binding

  use mod_constants
  use hashtbl
  implicit none

  private
  real(dp), allocatable, dimension(:) :: k, Pk, k2Pk, tmp, dk
  integer :: Nk
  real(fgsl_double) :: epsrel, epsabs

  public :: init, integrate, sigma, compute_covariance

  !------------------------------------------------------------
  ! Type definition to communicate between/from C
  !------------------------------------------------------------
  type :: params_t
     real(dp) :: dx, dy, dz, R1, R2
     integer :: ikx, iky, ikz, ikk, ii
     ! real(dp) :: sin_theta, cos_theta
     real(dp) :: sin_phi, cos_phi
     real(dp), allocatable :: kpart(:)
  end type params_t

  ! Workspace for GSL
  integer(fgsl_size_t), parameter :: nmax=1000

  ! Hashtable to prevent recomputation of same things multiple times
  type(hash_tbl_sll) :: table
  integer, parameter :: tbl_length = 100
  logical :: table_allocated = .false.

contains

  subroutine init(k_, Pk_, N, epsrel_, epsabs_) bind(c, name='init')
    ! Allocate and store the power spectrum
    real(dp), dimension(N), intent(in) :: k_, Pk_
    real(dp), intent(in), value :: epsrel_, epsabs_
    integer(c_int), intent(in), value :: N

    integer :: i

    if (table_allocated) call table%free
    if (allocated(k)) deallocate(k, Pk, k2Pk, tmp, dk)
    allocate(k(N), Pk(N), k2Pk(N), tmp(N), dk(N-1))

    call table%init(tbl_length)
    table_allocated = .true.

    k = k_
    Pk = Pk_
    k2Pk = k**2 * Pk
    Nk = size(k, 1)

    do i = 1, Nk-1
       dk(i) = k(i+1) - k(i)
    end do

    epsabs = epsabs_
    epsrel = epsrel_
  end subroutine init

  subroutine sigma(ii, R, res) bind(c)
    ! Evaluate the variance of the i-th derivative (or antiderivative
    ! if i < 0) of the field, smoothed at scale R
    integer(c_int), intent(in), value :: ii
    real(dp), intent(in), value :: R

    real(dp), intent(out) :: res

    real(dp) :: integrand

    tmp = Pk * k**(2+2*ii) * exp(-min((k*R)**2, 250._dp))
    integrand = sum((tmp(2:Nk) + tmp(1:Nk-1)) * dk) / 2

    res = sqrt(integrand / twopi2)
  end subroutine sigma

  subroutine compute_covariance(x, y, z, R, iikx, iiky, iikz, iikk, signs, covariance, npt) bind(c)
    real(dp), intent(in), dimension(npt) :: x, y, z
    real(dp), intent(in), dimension(npt) :: R(npt)
    integer(c_int), intent(in), dimension(npt) :: iikx, iiky, iikz, iikk, signs
    integer(c_int), intent(in), value :: npt

    real(dp), intent(out), dimension(npt, npt) :: covariance

    integer :: i, i1, i2
    real(dp) :: dx, dy, dz, R1, R2, res, s, sigma1, sigma2
    integer :: ikx, iky, ikz, ikk

    real(dp), dimension(npt) :: sigmas

    ! Precompute sigma values
    do i = 1, npt
       call sigma(iikx(i)+iiky(i)+iikz(i)-iikk(i), R(i), sigmas(i))
    end do

    covariance = 0

    ! $OMP PARALLEL DO default(shared) firstprivate(sigmas)                                &
    ! $OMP private(i1, i2, dx, dy, dz, R1, R2, ikx, iky, ikz, ikk, res, s, sigma1, sigma2) &
    ! $OMP reduction(+:covariance) schedule(static, npt/8)
    do i1 = 1, npt
       sigma1 = sigmas(i1)
       do i2 = i1, npt
          sigma2 = sigmas(i2)

          ikx = iikx(i1) + iikx(i2)
          iky = iiky(i1) + iiky(i2)
          ikz = iikz(i1) + iikz(i2)
          ikk = iikk(i1) + iikk(i2)

          R1 = R(i1)
          R2 = R(i2)

          ! Compute sign of output
          s = (-1)**(iikx(i2)+iiky(i2)+iikz(i2)-iikk(i2)) * signs(i1) * signs(i2)
          dx = x(i2) - x(i1)
          dy = y(i2) - y(i1)
          dz = z(i2) - z(i1)

          ! Perform integration
          call integrate(dx, dy, dz, R1, R2, ikx, iky, ikz, ikk, res)

          ! Scale result by corresponding sigma + sign
          res = s * res / sigma1 / sigma2
          covariance(i1, i2) = res
          covariance(i2, i1) = res

       end do
    end do
    ! $OMP END PARALLEL DO

  end subroutine compute_covariance

  subroutine integrate(dx, dy, dz, R1, R2, ikx, iky, ikz, ikk, res) bind(c)
    ! Evaluate the integral
    ! int_0^\infty dk
    !   int_0^pi dtheta
    !     int_0^2pi dphi
    !       k^2 Pk W(kR1) W(kR2) exp(-ik.r) kx^ikx ky^iky kz^ikz / k*ikk
    real(dp), intent(in) :: dx, dy, dz, R1, R2
    integer(c_int), intent(in) :: ikx, iky, ikz, ikk

    real(dp), intent(out) :: res

    ! Hashtable stuff
    character(len=20) :: key

    ! GSL stuff
    type(fgsl_integration_workspace) :: wk1
    real(fgsl_double) :: result, error
    integer(fgsl_int) :: status
    type(c_ptr) :: ptr
    type(fgsl_function) :: f1_obj

    type(params_t), target :: params

    ! Lookup in the table
    write(key, '')
    ! Detect when the result should be 0
    if ( (mod(ikx, 2) == 1 .and. dx == 0) .or. &
         (mod(iky, 2) == 1 .and. dy == 0) .or. &
         (mod(ikz, 2) == 1 .and. dz == 0)) then
       res = 0
       return
    end if

    ! Fill parameters
    allocate(params%kpart(Nk))
    params%dx  = dx
    params%dy  = dy
    params%dz  = dz
    params%ikx = ikx
    params%iky = iky
    params%ikz = ikz
    params%ikk = ikk
    params%ii  = ikx + iky + ikz - ikk

    ! Note: e^{-250} < 1e^-100
    params%kpart = Pk * exp(-min(k**2 * (R1**2 + R2**2) / 2, 250._dp)) * k**(2 + ikx + iky + ikz - ikk)

    wk1 = fgsl_integration_workspace_alloc(nmax)

    ! Get C pointers
    ptr = c_loc(params)
    f1_obj = fgsl_function_init(f1, ptr)

    ! Integrate over theta
    status = fgsl_integration_qag(f1_obj, 0.0_fgsl_double, twopi, &
         epsrel, epsabs, nmax, FGSL_INTEG_GAUSS15, wk1, result, error)
    call fgsl_function_free(f1_obj)
    ! call fgsl_integration_workspace_free(wk2)
    call fgsl_integration_workspace_free(wk1)

    res = result

  end subroutine integrate

  !------------------------------------------------------------
  ! Integrand stuff
  !------------------------------------------------------------
  function f1(phi, params) bind(c)
    real(c_double), value :: phi
    type(c_ptr), value :: params
    real(c_double) :: f1

    type(params_t), pointer :: p

    ! GSL stuff
    type(fgsl_integration_workspace) :: wk2
    type(fgsl_function) :: f2_obj
    real(fgsl_double) :: result, error
    integer(fgsl_int) :: status

    call c_f_pointer(params, p)
    wk2 = fgsl_integration_workspace_alloc(nmax)

    ! Compute theta-specific values
    p%sin_phi = sin(phi)
    p%cos_phi = cos(phi)

    f2_obj = fgsl_function_init(f2, params)

    ! Integrate over phi
    status = fgsl_integration_qag(f2_obj, 0.0_fgsl_double, pi, &
         epsrel, epsabs, nmax, FGSL_INTEG_GAUSS15, wk2, result, error)

    call fgsl_function_free(f2_obj)
    call fgsl_integration_workspace_free(wk2)

    f1 = result

  end function f1

  function f2(theta, params) bind(c)
    real(c_double), value :: theta
    type(c_ptr), value :: params
    real(c_double) :: f2

    type(params_t), pointer :: p

    real(c_double) :: sincos, sinsin, cos_theta, sin_theta, iipio2
    real(c_double) :: prev, cur, kk, res

    integer :: i

    ! Get data from C pointer
    call c_f_pointer(params, p)

    sin_theta = sin(theta)
    cos_theta = cos(theta)

    ! Precompute some stuff
    sincos = sin_theta * p%cos_phi
    sinsin = sin_theta * p%sin_phi
    iipio2 = p%ii * pi / 2

    ! Initialize variables
    res = 0
    prev = 0

    ! Integrate in k direction using trapezoidal rule
    do i = 1, Nk
       kk = k(i)

       cur = p%kpart(i) * cos(kk * (sincos * p%dx + sinsin * p%dy + cos_theta * p%dz) - iipio2)

       if (p%ikx /= 0) cur = cur * sincos**p%ikx
       if (p%iky /= 0) cur = cur * sinsin**p%iky
       if (p%ikz /= 0) cur = cur * cos_theta**p%ikz

       if (i > 1) &
            res = res + dk(i-1) * (prev + cur) / 2

       prev = cur
    end do

    f2 = res * sin_theta / eightpi3
  end function f2

end module mod_utils
