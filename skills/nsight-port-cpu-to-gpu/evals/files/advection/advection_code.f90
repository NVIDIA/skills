! SPDX-License-Identifier: Apache-2.0
! Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

! 1D linear advection, u_t + a u_x = 0, on a periodic domain with a
! first-order upwind scheme.
module advection_module
  use iso_fortran_env, only: dp => real64
  implicit none
contains
  function periodic_index(i, n) result(idx)
    ! Returns a periodic index (0 <= idx < n)
    integer, intent(in) :: i, n
    integer :: idx
    idx = modulo(i, n)
  end function periodic_index

  subroutine advection_kernel(u_in, u_out, n, dt, dx, a)
    integer, intent(in) :: n
    real(dp), intent(in)  :: dt, dx, a
    real(dp), intent(in)  :: u_in(0:n-1)
    real(dp), intent(out) :: u_out(0:n-1)
    integer :: i, im, ip
    real(dp) :: c

    c = a * dt / dx
    do i = 0, n-1
       if (a >= 0.0_dp) then
          im = periodic_index(i - 1, n)
          u_out(i) = u_in(i) - c * (u_in(i) - u_in(im))
       else
          ip = periodic_index(i + 1, n)
          u_out(i) = u_in(i) - c * (u_in(ip) - u_in(i))
       end if
    end do
  end subroutine advection_kernel
end module advection_module

program advection_code
  use iso_fortran_env, only: dp => real64
  use advection_module, only: advection_kernel
  implicit none

  integer, parameter :: n = 256
  real(dp), parameter :: pi = 3.141592653589793_dp
  real(dp), parameter :: length  = 1.0_dp
  real(dp), parameter :: a       = 1.0_dp     ! advection speed
  real(dp), parameter :: cfl     = 0.5_dp
  real(dp), parameter :: t_final = 0.5_dp

  real(dp) :: dx, dt, t, err
  real(dp), allocatable :: u(:), u_new(:), x(:)
  integer :: i, nsteps

  dx = length / real(n, dp)
  dt = cfl * dx / abs(a)

  allocate(u(0:n-1), u_new(0:n-1), x(0:n-1))

  ! Initial condition: one period of a sine wave
  do i = 0, n-1
     x(i) = i * dx
     u(i) = sin(2.0_dp * pi * x(i) / length)
  end do

  t = 0.0_dp
  nsteps = 0
  do while (t < t_final)
     if (t + dt > t_final) dt = t_final - t
     call advection_kernel(u, u_new, n, dt, dx, a)
     u = u_new
     t = t + dt
     nsteps = nsteps + 1
  end do

  ! Exact solution is the initial wave shifted by a*t_final
  err = 0.0_dp
  do i = 0, n-1
     err = max(err, abs(u(i) - sin(2.0_dp * pi * (x(i) - a * t_final) / length)))
  end do

  print '(A, I0, A, I0)', 'cells: ', n, '  steps: ', nsteps
  print '(A, ES14.6)', 'sum(u):        ', sum(u)
  print '(A, ES14.6)', 'max |u-exact|: ', err
end program advection_code
