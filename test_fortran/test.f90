program test
  use mod_utils
  use mod_constants
  implicit none

  integer, parameter :: Nk = 3000
  real(dp) :: k(Nk), Pk(Nk), kmin, kmax, tmp

  integer :: i, N

  real(4) :: values(2), time

  kmin = 1e-4
  kmax = 1e4

  do i = 1, Nk
     k(i)  = (kmax/kmin)**((i-1._dp)/Nk) * kmin
     Pk(i) = k(i)**(-2)
  end do

  call init(k, Pk, Nk)

  ! Test sigma
  N = 5000
  call dtime(values, time)
  do i = 1, N
     call sigma(0, 8._dp, tmp)
  end do
  call dtime(values, time)

  write(*, '(a10,es14.5,a,f5.2,a)') 'sigma=', tmp, ' t= ', time / N * 1e6, ' µs/call'

  ! Test integration
  block
    integer :: ikx, iky, ikz, ikk
    do ikx = 0, 2
       do iky = 0, 2
          do ikz = 0, 2
             do ikk = 0, 2, 2
                N = 1
                call dtime(values, time)
                do i = 1, N
                   call integrate(1._dp, 0._dp, 0._dp, 1._dp, 1._dp, ikx, iky, ikz, ikk, tmp)
                end do
                call dtime(values, time)
                write(*, '("ikx=",i2," iky=",i2," ikz=",i2," ikk=",i2,"  =",es14.5,a,f6.2,a)') &
                     ikx, iky, ikz, ikk, tmp, ' t= ', time / N * 1e3, ' ms/call'
             end do
          end do
       end do
    end do

  end block

end program test
