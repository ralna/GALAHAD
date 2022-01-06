! THIS VERSION: 29/12/2021 AT 15:40:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M C 6 4 _ C I F A C E   M O D U L E  -*-*-

module hsl_mc64_single_ciface
   use iso_c_binding
   use hsl_mc64_single, only:                &
      f_mc64_control       => mc64_control,  &
      f_mc64_info          => mc64_info,     &
      f_mc64_matching      => mc64_matching
   implicit none

   type, bind(C) :: mc64_control
      integer(C_INT) :: f_arrays
      integer(C_INT) :: lp
      integer(C_INT) :: wp
      integer(C_INT) :: sp
      integer(C_INT) :: ldiag
      integer(C_INT) :: checking
   end type mc64_control

   type, bind(C) :: mc64_info
      integer(C_INT) :: flag
      integer(C_INT) :: more
      integer(C_INT) :: strucrank
      integer(C_INT) :: stat
   end type mc64_info

contains
   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(mc64_control), intent(in) :: ccontrol
      type(f_mc64_control), intent(out) :: fcontrol
      logical, intent(out) :: f_arrays

      f_arrays          = (ccontrol%f_arrays.ne.0)
      fcontrol%lp       = ccontrol%lp
      fcontrol%wp       = ccontrol%wp
      fcontrol%sp       = ccontrol%sp
      fcontrol%ldiag    = ccontrol%ldiag
      fcontrol%checking = ccontrol%checking
   end subroutine copy_control_in

   subroutine copy_info_out(finfo, cinfo)
      type(f_mc64_info), intent(in) :: finfo
      type(mc64_info), intent(out) :: cinfo

      cinfo%flag        = finfo%flag
      cinfo%more        = finfo%more
      cinfo%strucrank   = finfo%strucrank
      cinfo%stat        = finfo%stat
   end subroutine copy_info_out
end module hsl_mc64_single_ciface
