program test
  use mod_utils
  use mod_constants
  implicit none

  real(dp), allocatable :: k(:), Pk(:)
  ! real(dp) :: kmin, kmax
  real(dp) :: tmp
  real(dp) :: epsrel = 1d-3, epsabs = 1d-5

  integer :: i, N, stat, Nk

  real(4) :: values(2), time

  open(10, file='../correlations/data/power.dat', form='formatted')
  read(10, *)
  Nk = 0
  do while (.true.)
     read(10, *, iostat=stat)
     if (stat < 0) exit ! End of file
     Nk = Nk + 1
  end do

  rewind(10)
  read(10, *)

  allocate(k(Nk), Pk(Nk))
  do i = 1, Nk
     read(10, *) k(i), Pk(i), tmp
  end do

  Pk = Pk * 2*pi**2 * 4*pi
  
  ! kmin = 1e-4
  ! kmax = 1e4

  ! do i = 1, Nk
  !    k(i)  = (kmax/kmin)**((i-1._dp)/Nk) * kmin
  !    Pk(i) = k(i)**(-2)
  ! end do

  call init(k, Pk, Nk, epsrel, epsabs)

  ! ! Test sigma
  ! N = 5000
  ! call dtime(values, time)
  ! do i = 1, N
  !    call sigma(0, 8._dp, tmp)
  ! end do
  ! call dtime(values, time)

  ! write(*, '(a10,es14.5,a,f5.2,a)') 'sigma=', tmp, ' t= ', time / N * 1e6, ' µs/call'

  ! ! Test integration
  ! block
  !   integer :: ikx, iky, ikz, ikk
  !   do ikx = 0, 2
  !      do iky = 0, 2
  !         do ikz = 0, 2
  !            do ikk = 0, 2, 2
  !               N = 1
  !               call dtime(values, time)
  !               do i = 1, N
  !                  call integrate(1._dp, 0._dp, 0._dp, 1._dp, 1._dp, ikx, iky, ikz, ikk, tmp)
  !               end do
  !               call dtime(values, time)
  !               write(*, '("ikx=",i2," iky=",i2," ikz=",i2," ikk=",i2,"  =",es14.5,a,f6.2,a)') &
  !                    ikx, iky, ikz, ikk, tmp, ' t= ', time / N * 1e3, ' ms/call'
  !            end do
  !         end do
  !      end do
  !   end do

  ! end block

  ! Test computation of covariance
  block
    integer, parameter :: npt = 12
    real(dp), dimension(npt) :: x, y, z
    real(dp), dimension(npt) :: R
    integer, dimension(npt) :: iikx, iiky, iikz, iikk, signs
    real(dp), dimension(npt, npt) :: covariance

    integer :: i

    x = 0
    x(6:) = 1
    y = 0
    z = 0

    R(:) = 1
    iikx(:) = [2, 0, 0, 1, 1, 0, 2, 0, 0, 1, 1, 0]
    iiky(:) = [0, 2, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1]
    iikz(:) = [0, 0, 2, 0, 1, 1, 0, 0, 2, 0, 1, 1]
    iikk(:) = 0
    signs(:)= 1

    call dtime(values, time)
    call compute_covariance(x, y, z, R, iikx, iiky, iikz, iikk, signs, covariance, npt)
    call dtime(values, time)

    do i = 1, npt
       write (*, '(*(f10.5))') covariance(i, :) * 15
    end do
    write(*, '("t= ",f6.2," s")') time
    
  end block
  

end program test
