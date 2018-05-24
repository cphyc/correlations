module compute_correlations
  use iso_fortran_env, only: real64
  use iso_c_binding, only: c_double, c_int
  implicit none

  private
  integer, parameter :: dp = real64
  integer, parameter :: GAUSSIAN = 0, TOP_HAT = 1
  integer :: filter = GAUSSIAN

  complex :: j = cmplx(0, 1)

  real(dp), allocatable :: k(:), Pk(:), integrand(:), k2Pk(:), tmp(:), kx(:), ky(:), kz(:), ksintheta(:)
  real(dp), parameter :: pi = atan(1.0_dp) * 4._dp, twopi = 2*pi, twopi2 = 2*pi**2

  integer, parameter :: NPHI = 50, NTHETA = 50
  real(dp) :: phigrid(NPHI), thetagrid(NTHETA)

  public :: covariance, c_covariance

contains

  subroutine c_covariance(positions, kfactor, filter, R, knew, Pknew, cov, ndim, npt, nk) bind(c)
    integer(c_int), intent(in) :: ndim, npt, nk
    real(c_double), intent(in) :: positions(npt, ndim), R(npt)
    real(c_double), intent(in) :: knew(nk), Pknew(nk)
    integer(c_int), intent(in) :: kfactor(npt, ndim+1), filter(npt)

    real(c_double), intent(out) :: cov(npt, npt)
    call covariance(positions, kfactor, filter, R, knew, Pknew, cov, ndim, npt, nk)
  end subroutine c_covariance

  elemental real(dp) function WTH(x)
    ! Top-Hat filter
    real(dp), intent(in) :: x

    if (x < 0.2_dp) then
       ! Taylor expansion around 0 with relative error of ≤ 1e-8
       WTH = 1 - x**2/10 + x**4/280
    else
       WTH = 3*(sin(x)/x**2 - cos(x)) / x**2
    end if

  end function WTH

  elemental real(dp) function WG(x)
    real(dp), intent(in) :: x
    WG = exp(-x**2/2)
  end function WG

  elemental real(dp) function W(x, ifilter)
    real(dp), intent(in) :: x
    integer, intent(in) :: ifilter

    if (ifilter == GAUSSIAN) then
       W = WG(x)
    else if (ifilter == TOP_HAT) then
       W = WTH(x)
    end if

  end function W

  subroutine covariance(positions, kfactor, filter, R, knew, Pknew, cov, ndim, npt, nk)
    ! Compute the covariance matrix
    !
    ! Parameters
    ! ----------
    ! positions, float, (ndim, N)
    ! kfactor, integer, (ndim, N)
    ! filter, integer, (ndim, ): 0 for Gaussian, 1 for TOP_HAT
    ! R, float, (N, ): smoothing scale

    integer, intent(in)  :: ndim, npt, nk
    real(dp), intent(in) :: positions(npt, ndim), R(npt)
    integer, intent(in)  :: kfactor(npt, ndim+1)
    integer, intent(in)  :: filter(npt)
    real(dp), intent(in) :: knew(nk), Pknew(nk)

    real(dp), intent(out) :: cov(npt, npt)
    integer :: i1, i2
    real(dp) :: dX(3), d, res

    if (allocated(k)) then
       deallocate(k, Pk, integrand, k2Pk, tmp, kx, ky, kz, ksintheta)
    end if

    associate(n=>nk)
      allocate(k(n), Pk(n), integrand(n), k2Pk(n), tmp(n), kx(n), ky(n), kz(n), ksintheta(n))
    end associate

    k = knew
    Pk = Pknew
    k2Pk = k**2 * Pk / twopi2
    integrand = 0
    tmp = 0

    ! Precompute grids
    do i1 = 0, NPHI-1
       phigrid(i1) = twopi * i1 / NPHI
    end do
    do i1 = 0, NTHETA-1
       thetagrid(i1) = pi * i1 / NTHETA
    end do
        
    ! Loop over the positions
    do i1 = 1, npt
       do i2 = i1, npt
          dX = positions(i1, :) - positions(i2, :)
          d = norm2(dX)
          call integrate(dX, d, kfactor(i1, :), kfactor(i2, :), filter(i1), filter(i2), R(i1), R(i2), res, nk)
          cov(i1, i2) = res
          cov(i2, i1) = res
       end do
    end do
  end subroutine covariance

  subroutine integrate(dX, d, k1, k2, filter1, filter2, R1, R2, res, nk)
    real(dp), intent(in) :: dX(3), d, R1, R2
    integer, intent(in) :: k1(4), k2(4), filter1, filter2, nk
    real(dp), intent(out) :: res

    integer :: itheta, iphi, ikx, iky, ikz, ikk
    real(dp) :: theta, phi, sintheta
    real(dp), save :: restheta(NTHETA), resphi(NPHI)

    ! Precompute some numerical factors
    tmp(:) = k2Pk(:) * W(k*R1, filter1) * W(k*R2, filter2)

    ikk = sum(k1) + sum(k2)
    ikx = k1(1) + k2(1)
    iky = k1(2) + k2(2)
    ikz = k1(3) + k2(3)

    do itheta = 1, NTHETA
       theta = thetagrid(itheta)
       sintheta = sin(theta)
       ksintheta = k * sintheta
       resphi = 0
       do iphi = 1, NPHI
          phi = phigrid(iphi)

          kx = ksintheta * cos(phi)
          ky = ksintheta * sin(phi)
          kz = k * cos(theta)

          ! Compute the integrand: k2Pk W(kR1) W(kR2) kx**i ky**j kz**k / k**ikk
          integrand = tmp * sintheta * kx**ikx * ky**iky * kz**ikz / k**ikk

          if (d > 0) then
             if (mod(ikk, 2) == 0) then
                integrand = integrand * (-1)**ikk * cos(kx*dX(1) + ky*dX(2) + kz*dX(3))
             else
                integrand = integrand * (-1)**(ikk-1) * sin(kx*dX(1) + ky*dX(2) + kz*dX(3))
             end if
          end if

          ! Integrate over k using trapezoidal rule
          associate(n => nk)
            resphi(iphi) = sum((integrand(2:n) + integrand(1:n-1))*(k(2:n) - k(1:n-1)))/2
          end associate

       end do

       ! Integrate over phi using trapezoidal rule
       associate(n => NPHI)
         restheta(itheta) = sum((resphi(2:n) + resphi(1:n-1))*(phigrid(2:n) - phigrid(1:n-1)))/2
       end associate
    end do

    ! Integrate over theta using trapezoidal rule
    associate(n => NTHETA)
      res = sum((restheta(2:n) + restheta(1:n-1))*(thetagrid(2:n) - thetagrid(1:n-1)))/2
    end associate

    ! Take care of derivative on the first member
    res = res * (-1)**(k2(1)+k2(2)+k2(3)-k2(4))
  end subroutine integrate

end module compute_correlations
