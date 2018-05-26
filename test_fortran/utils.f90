module utils
  implicit none

  private
  integer, parameter :: dp = selected_real_kind(15)
  real(dp), allocatable, dimension(:) :: k, Pk, k2Pk
  integer :: Nk

  real(dp), parameter :: pi = atan(1._dp) * 4, twopi = 2*pi, twopi2 = 2*pi**2

  public :: init
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

    real(dp) :: prev, cur, k, Pk, kprev

    integer :: j

    integrand = 0
    do j = 1, Nk
       cur = k**(2+2*i) * Pk * exp(-(k*R)**2)

       if (j > 0) then
          integrand = integrand + (prev + cur) / 2 * (k - kprev)
       end if
       prev = cur
       kprev = k
    end do

    integrand = sqrt(integrand / twopi2)
  end subroutine sigma


end module utils
