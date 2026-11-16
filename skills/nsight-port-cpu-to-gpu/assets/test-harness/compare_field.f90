! SPDX-License-Identifier: Apache-2.0
! Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

!> Numerical comparison harness for Nsight AI Porting Agent standalone tests.
!>
!> Copy this file into the test directory of the function under test and
!> `use nsight_port_compare`. The gate is `wmad`, the normalized weighted
!> mean absolute difference against the reference field; `n_exceed` and the
!> worst element are reported so an outlier failure is reviewable rather
!> than merely a number to loosen. See references/verification-guide.md.
!>
!> Every tolerance is read from the environment (WMAD_TOL, ABS_TOL,
!> REL_TOL, REL_FLOOR, MAX_REL_CHECK) so a test never hardcodes one.
module nsight_port_compare

   use, intrinsic :: iso_fortran_env,  only: real32, real64, int64, error_unit
   use, intrinsic :: iso_c_binding,    only: c_loc, c_f_pointer
   use, intrinsic :: ieee_arithmetic,  only: ieee_is_finite

   implicit none
   private

   public :: compare_result, compare_tolerances
   public :: tolerances_for, compare_field, report_and_gate

   integer, parameter :: dp = real64
   integer, parameter :: sp = real32

   !> Outcome of one field comparison. `wmad` gates; the rest is diagnosis.
   type :: compare_result
      real(dp)       :: wmad      = 0.0_dp
      real(dp)       :: max_rel   = 0.0_dp
      integer(int64) :: n_exceed  = 0_int64
      integer(int64) :: n_total   = 0_int64
      integer(int64) :: worst     = 1_int64      !< flat index, 1-based
      integer(int64) :: worst_sub(3) = 1_int64   !< subscripts, unused ranks = 1
      integer        :: field_rank = 1
      logical        :: nonfinite = .false.
      !> Candidate, reference and weights must agree in shape. A mismatch is
      !> a bug in the test, not a numerical result, so no metric is computed.
      logical        :: shape_error = .false.
      !> A zero-element comparison. Every metric is trivially 0 over an empty
      !> field, so it would otherwise pass while proving nothing.
      logical        :: empty_field = .false.
   end type compare_result

   !> Gate thresholds. Defaults depend on the field's numeric kind and are
   !> then overridden from the environment.
   type :: compare_tolerances
      real(dp) :: wmad          = 1.0e-12_dp
      real(dp) :: tol_abs       = 1.0e-12_dp
      real(dp) :: tol_rel       = 1.0e-10_dp
      real(dp) :: rel_floor     = 1.0e-10_dp
      logical  :: max_rel_check = .true.
   end type compare_tolerances

   interface tolerances_for
      module procedure tolerances_for_r4, tolerances_for_r8
   end interface tolerances_for

   interface compare_field
      module procedure cmp_r4_1d, cmp_r4_2d, cmp_r4_3d
      module procedure cmp_r8_1d, cmp_r8_2d, cmp_r8_3d
   end interface compare_field

contains

   ! ---------------------------------------------------------------- core

   !> The single implementation of the metric. Everything else adapts
   !> shapes and kinds onto this.
   function compare_core(a, ref, tol, w) result(c)
      real(dp),                 intent(in)           :: a(:), ref(:)
      type(compare_tolerances), intent(in)           :: tol
      real(dp),                 intent(in), optional :: w(:)  !< mesh weights
      type(compare_result) :: c

      real(dp)       :: peak, floor_v, num, den, ad, rd, wi
      integer(int64) :: i, n
      logical        :: weighted

      n = size(a, kind=int64)
      c%n_total = n
      weighted  = present(w)

      ! Non-finite anywhere is an unconditional failure — never waivable.
      peak = 0.0_dp
      do i = 1_int64, n
         if (.not. ieee_is_finite(a(i)) .or. .not. ieee_is_finite(ref(i))) then
            c%nonfinite = .true.
            c%worst     = i
            c%wmad      = huge(0.0_dp)
            c%max_rel   = huge(0.0_dp)
            c%n_exceed  = -1_int64
            return
         end if
         peak = max(peak, abs(ref(i)))
      end do

      ! Magnitude floor: cells far below the field peak must not manufacture
      ! a huge relative error out of numerical dust.
      floor_v = tol%rel_floor * peak
      num     = 0.0_dp
      den     = 0.0_dp

      do i = 1_int64, n
         if (weighted) then
            wi = w(i)
         else
            wi = 1.0_dp
         end if
         ad  = abs(a(i) - ref(i))
         num = num + wi * ad
         den = den + wi * abs(ref(i))
         rd  = ad / (abs(ref(i)) + floor_v)
         if (rd > c%max_rel) then
            c%max_rel = rd
            c%worst   = i
         end if
         ! An element passes if EITHER tolerance is met, so it counts as an
         ! exceedance only when both are exceeded.
         if (ad > tol%tol_abs .and. rd > tol%tol_rel) then
            c%n_exceed = c%n_exceed + 1_int64
         end if
      end do

      if (den > 0.0_dp) then
         c%wmad = num / den
      else
         c%wmad = num
      end if
   end function compare_core

   !> Fill worst_sub from the flat worst index and the field shape.
   subroutine set_worst_sub(c, shp)
      type(compare_result), intent(inout) :: c
      integer(int64),       intent(in)    :: shp(:)
      integer(int64) :: rem, d
      integer        :: k

      c%field_rank   = size(shp)
      c%worst_sub    = 1_int64
      rem            = c%worst - 1_int64
      do k = 1, size(shp)
         d = max(shp(k), 1_int64)
         c%worst_sub(k) = mod(rem, d) + 1_int64
         rem = rem / d
      end do
   end subroutine set_worst_sub

   ! ------------------------------------------------------- rank-1 leaves

   function cmp_r8_1d(a, ref, tol, w) result(c)
      real(dp),                 intent(in)           :: a(:), ref(:)
      type(compare_tolerances), intent(in)           :: tol
      real(dp),                 intent(in), optional :: w(:)
      type(compare_result) :: c

      ! compare_core trusts size(a) as its loop bound, so a mismatched
      ! reference or weight array is refused here rather than truncated.
      if (size(a) /= size(ref)) then
         c%shape_error = .true.
         return
      end if
      if (size(a) == 0) then
         c%empty_field = .true.
         return
      end if
      if (present(w)) then
         if (size(w) /= size(a)) then
            c%shape_error = .true.
            return
         end if
         c = compare_core(a, ref, tol, w)
      else
         c = compare_core(a, ref, tol)
      end if
      call set_worst_sub(c, [size(a, kind=int64)])
   end function cmp_r8_1d

   function cmp_r4_1d(a, ref, tol, w) result(c)
      real(sp),                 intent(in)           :: a(:), ref(:)
      type(compare_tolerances), intent(in)           :: tol
      real(sp),                 intent(in), optional :: w(:)
      type(compare_result) :: c

      if (size(a) /= size(ref)) then
         c%shape_error = .true.
         return
      end if
      if (size(a) == 0) then
         c%empty_field = .true.
         return
      end if
      if (present(w)) then
         if (size(w) /= size(a)) then
            c%shape_error = .true.
            return
         end if
      end if

      ! Single-precision fields are promoted so the metric always
      ! accumulates in double, whatever the field kind.
      if (present(w)) then
         c = compare_core(real(a, dp), real(ref, dp), tol, real(w, dp))
      else
         c = compare_core(real(a, dp), real(ref, dp), tol)
      end if
      call set_worst_sub(c, [size(a, kind=int64)])
   end function cmp_r4_1d

   ! ------------------------------------------------ rank-2/3 flatteners
   ! c_loc + c_f_pointer gives a rank-1 view with no copy. `contiguous`
   ! makes the compiler supply a contiguous actual if the caller's slice
   ! was not one. Shape is checked before flattening, because a (100,50)
   ! field and a (50,100) one are indistinguishable once flat.

   function cmp_r8_2d(a, ref, tol, w) result(c)
      real(dp), target, contiguous, intent(in)           :: a(:,:), ref(:,:)
      type(compare_tolerances),     intent(in)           :: tol
      real(dp), target, contiguous, intent(in), optional :: w(:,:)
      type(compare_result) :: c
      real(dp), pointer :: pa(:), pr(:), pw(:)

      if (any(shape(a) /= shape(ref))) then
         c%shape_error = .true.
         return
      end if
      if (size(a) == 0) then
         c%empty_field = .true.
         return
      end if
      if (present(w)) then
         if (any(shape(w) /= shape(a))) then
            c%shape_error = .true.
            return
         end if
      end if

      call c_f_pointer(c_loc(a),   pa, [size(a)])
      call c_f_pointer(c_loc(ref), pr, [size(ref)])
      if (present(w)) then
         call c_f_pointer(c_loc(w), pw, [size(w)])
         c = cmp_r8_1d(pa, pr, tol, pw)
      else
         c = cmp_r8_1d(pa, pr, tol)
      end if
      call set_worst_sub(c, int(shape(a), int64))
   end function cmp_r8_2d

   function cmp_r8_3d(a, ref, tol, w) result(c)
      real(dp), target, contiguous, intent(in)           :: a(:,:,:), ref(:,:,:)
      type(compare_tolerances),     intent(in)           :: tol
      real(dp), target, contiguous, intent(in), optional :: w(:,:,:)
      type(compare_result) :: c
      real(dp), pointer :: pa(:), pr(:), pw(:)

      if (any(shape(a) /= shape(ref))) then
         c%shape_error = .true.
         return
      end if
      if (size(a) == 0) then
         c%empty_field = .true.
         return
      end if
      if (present(w)) then
         if (any(shape(w) /= shape(a))) then
            c%shape_error = .true.
            return
         end if
      end if

      call c_f_pointer(c_loc(a),   pa, [size(a)])
      call c_f_pointer(c_loc(ref), pr, [size(ref)])
      if (present(w)) then
         call c_f_pointer(c_loc(w), pw, [size(w)])
         c = cmp_r8_1d(pa, pr, tol, pw)
      else
         c = cmp_r8_1d(pa, pr, tol)
      end if
      call set_worst_sub(c, int(shape(a), int64))
   end function cmp_r8_3d

   function cmp_r4_2d(a, ref, tol, w) result(c)
      real(sp), target, contiguous, intent(in)           :: a(:,:), ref(:,:)
      type(compare_tolerances),     intent(in)           :: tol
      real(sp), target, contiguous, intent(in), optional :: w(:,:)
      type(compare_result) :: c
      real(sp), pointer :: pa(:), pr(:), pw(:)

      if (any(shape(a) /= shape(ref))) then
         c%shape_error = .true.
         return
      end if
      if (size(a) == 0) then
         c%empty_field = .true.
         return
      end if
      if (present(w)) then
         if (any(shape(w) /= shape(a))) then
            c%shape_error = .true.
            return
         end if
      end if

      call c_f_pointer(c_loc(a),   pa, [size(a)])
      call c_f_pointer(c_loc(ref), pr, [size(ref)])
      if (present(w)) then
         call c_f_pointer(c_loc(w), pw, [size(w)])
         c = cmp_r4_1d(pa, pr, tol, pw)
      else
         c = cmp_r4_1d(pa, pr, tol)
      end if
      call set_worst_sub(c, int(shape(a), int64))
   end function cmp_r4_2d

   function cmp_r4_3d(a, ref, tol, w) result(c)
      real(sp), target, contiguous, intent(in)           :: a(:,:,:), ref(:,:,:)
      type(compare_tolerances),     intent(in)           :: tol
      real(sp), target, contiguous, intent(in), optional :: w(:,:,:)
      type(compare_result) :: c
      real(sp), pointer :: pa(:), pr(:), pw(:)

      if (any(shape(a) /= shape(ref))) then
         c%shape_error = .true.
         return
      end if
      if (size(a) == 0) then
         c%empty_field = .true.
         return
      end if
      if (present(w)) then
         if (any(shape(w) /= shape(a))) then
            c%shape_error = .true.
            return
         end if
      end if

      call c_f_pointer(c_loc(a),   pa, [size(a)])
      call c_f_pointer(c_loc(ref), pr, [size(ref)])
      if (present(w)) then
         call c_f_pointer(c_loc(w), pw, [size(w)])
         c = cmp_r4_1d(pa, pr, tol, pw)
      else
         c = cmp_r4_1d(pa, pr, tol)
      end if
      call set_worst_sub(c, int(shape(a), int64))
   end function cmp_r4_3d

   ! -------------------------------------------------------- tolerances

   function tolerances_for_r8(sample) result(tol)
      real(dp), intent(in) :: sample
      type(compare_tolerances) :: tol
      real(dp) :: ignored
      ignored = sample                 ! kind selection only
      tol%wmad      = 1.0e-12_dp
      tol%tol_abs   = 1.0e-12_dp
      tol%tol_rel   = 1.0e-10_dp
      call apply_tolerance_env(tol)
   end function tolerances_for_r8

   function tolerances_for_r4(sample) result(tol)
      real(sp), intent(in) :: sample
      type(compare_tolerances) :: tol
      real(sp) :: ignored
      ignored = sample                 ! kind selection only
      tol%wmad      = 1.0e-6_dp
      tol%tol_abs   = 1.0e-5_dp
      tol%tol_rel   = 1.0e-4_dp
      call apply_tolerance_env(tol)
   end function tolerances_for_r4

   subroutine apply_tolerance_env(tol)
      type(compare_tolerances), intent(inout) :: tol
      call get_real_env('WMAD_TOL',      tol%wmad)
      call get_real_env('ABS_TOL',       tol%tol_abs)
      call get_real_env('REL_TOL',       tol%tol_rel)
      call get_real_env('REL_FLOOR',     tol%rel_floor)
      call get_flag_env('MAX_REL_CHECK', tol%max_rel_check)
   end subroutine apply_tolerance_env

   subroutine get_real_env(name, value)
      character(len=*), intent(in)    :: name
      real(dp),         intent(inout) :: value
      character(len=64) :: buf, s
      integer  :: length, status, ios, k, n
      real(dp) :: tmp

      call get_environment_variable(name, buf, length, status)
      if (status /= 0 .or. length == 0 .or. length > len(buf)) return

      ! List-directed input stops at the first separator, so '1e-3 oops'
      ! and '1,5' would otherwise read as 1e-3 and 1. Require a single
      ! complete token, matching env_real in the C++ harness — the same
      ! environment has to mean the same tolerance in either language.
      s = buf(1:length)
      do k = 1, length
         if (s(k:k) == achar(9)) s(k:k) = ' '   ! tabs pad like blanks
      end do
      s = adjustl(s)
      n = len_trim(s)
      if (n == 0) return
      if (index(s(1:n), ' ') > 0 .or. index(s(1:n), ',') > 0 .or. &
          index(s(1:n), '/') > 0) return

      read (s(1:n), *, iostat=ios) tmp
      if (ios == 0) value = tmp
   end subroutine get_real_env

   subroutine get_flag_env(name, value)
      character(len=*), intent(in)    :: name
      logical,          intent(inout) :: value
      character(len=16) :: buf
      integer :: length, status, k

      call get_environment_variable(name, buf, length, status)
      if (status /= 0 .or. length == 0 .or. length > len(buf)) return
      do k = 1, length
         if (buf(k:k) >= 'A' .and. buf(k:k) <= 'Z') then
            buf(k:k) = achar(iachar(buf(k:k)) + 32)
         end if
      end do
      select case (trim(buf(1:length)))
      case ('0', 'off', 'false', 'no')
         value = .false.
      case ('1', 'on', 'true', 'yes')
         value = .true.
      end select
   end subroutine get_flag_env

   ! ------------------------------------------------------ report + gate

   !> Print the four required numbers and apply the gate. Returns .true.
   !> when the comparison passes. The caller decides how to fail (typically
   !> `if (.not. report_and_gate(...)) stop 1`).
   function report_and_gate(c, tol, label) result(ok)
      type(compare_result),     intent(in) :: c
      type(compare_tolerances), intent(in) :: tol
      character(len=*),         intent(in) :: label
      logical :: ok

      if (c%shape_error) then
         write (*, '(a,a)') 'SHAPE MISMATCH between candidate, reference ', &
            'or weights'
         write (*, '(a,a)') 'VERIFICATION FAILED (shape mismatch): ', label
         ok = .false.
         return
      end if

      if (c%empty_field) then
         write (*, '(a,a)') 'EMPTY FIELD: zero elements compared, so the ', &
            'result would prove nothing'
         write (*, '(a,a)') 'VERIFICATION FAILED (empty field): ', label
         ok = .false.
         return
      end if

      if (c%nonfinite) then
         write (*, '(a,a,a,i0)') 'NON-FINITE VALUES in ', label, &
            ' at flat index ', c%worst
         write (*, '(a,a)') 'VERIFICATION FAILED (non-finite): ', label
         ok = .false.
         return
      end if

      write (*, '(a,a,a,es12.5,a,es12.5)') 'compare ', label, &
         ': wmad ', c%wmad, '  max_rel ', c%max_rel
      write (*, '(a,i0,a,i0,a,3(1x,i0))') '  n_exceed ', c%n_exceed, &
         ' of ', c%n_total, '  worst', c%worst_sub(1:c%field_rank)

      ok = (c%wmad <= tol%wmad)
      if (.not. ok) then
         write (*, '(a,a)') 'VERIFICATION FAILED (wmad): ', label
         return
      end if

      ! The outlier bound IS reviewable — see verification-guide.md,
      ! "Reviewing an outlier failure".
      if (tol%max_rel_check .and. c%n_exceed > 0_int64) then
         write (*, '(a,a)') &
            'VERIFICATION FAILED (outliers) - review before waiving: ', label
         ok = .false.
         return
      end if

      write (*, '(a,a)') 'VERIFICATION PASSED: ', label
   end function report_and_gate

end module nsight_port_compare
