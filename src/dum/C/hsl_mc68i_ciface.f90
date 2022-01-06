! THIS VERSION: 29/12/2021 AT 15:45:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M C 6 8 _ C I F A C E   M O D U L E  -*-*-

module hsl_mc68_integer_ciface
   use iso_c_binding
   use hsl_mc68_integer, only: &
      f_mc68_control       => mc68_control,     &
      f_mc68_info          => mc68_info,        &
      f_mc68_order         => mc68_order
   implicit none

   type, bind(C) :: mc68_control
      integer(C_INT) :: f_array_in ! 0 is false, otherwise true
      integer(C_INT) :: f_array_out! 0 is false, otherwise true
      integer(C_INT) :: min_l_workspace
      integer(C_INT) :: lp
      integer(C_INT) :: wp
      integer(C_INT) :: mp
      integer(C_INT) :: nemin
      integer(C_INT) :: print_level
      integer(C_INT) :: row_full_thresh
      integer(C_INT) :: row_search
   end type mc68_control

   type, bind(C) :: mc68_info
     integer(C_INT) :: flag
     integer(C_INT) :: iostat
     integer(C_INT) :: stat
     integer(C_INT) :: out_range
     integer(C_INT) :: duplicate
     integer(C_INT) :: n_compressions
     integer(C_INT) :: n_zero_eigs
     integer(C_LONG) :: l_workspace
     integer(C_INT) :: zb01_info
     integer(C_INT) :: n_dense_rows
   end type mc68_info
contains
   subroutine copy_control_in(ccontrol, fcontrol, f_array_in, f_array_out, &
         min_l_workspace)
      type(mc68_control), intent(in) :: ccontrol
      type(f_mc68_control), intent(out) :: fcontrol
      logical, intent(out) :: f_array_in
      logical, intent(out) :: f_array_out
      integer, intent(out) :: min_l_workspace

      f_array_in                 = (ccontrol%f_array_in .ne. 0)
      f_array_out                = (ccontrol%f_array_out .ne. 0)
      min_l_workspace            = ccontrol%min_l_workspace
      fcontrol%lp                = ccontrol%lp
      fcontrol%wp                = ccontrol%wp
      fcontrol%mp                = ccontrol%mp
      fcontrol%nemin             = ccontrol%nemin
      fcontrol%print_level       = ccontrol%print_level
      fcontrol%row_full_thresh   = ccontrol%row_full_thresh
      fcontrol%row_search        = ccontrol%row_search
   end subroutine copy_control_in

   subroutine copy_info_out(finfo, cinfo)
      type(f_mc68_info), intent(in) :: finfo
      type(mc68_info), intent(out) :: cinfo

      cinfo%flag              = finfo%flag
      cinfo%iostat            = finfo%iostat
      cinfo%stat              = finfo%stat
      cinfo%out_range         = finfo%out_range
      cinfo%duplicate         = finfo%duplicate
      cinfo%n_compressions    = finfo%n_compressions
      cinfo%n_zero_eigs       = finfo%n_zero_eigs
      cinfo%l_workspace       = finfo%l_workspace
      cinfo%zb01_info         = finfo%zb01_info
      cinfo%n_dense_rows      = finfo%n_dense_rows
   end subroutine copy_info_out
end module hsl_mc68_integer_ciface
