! THIS VERSION: GALAHAD 5.0 - 2024-03-27 AT 09:10 GMT.

#include "hsl_subset.h"
#include "hsl_subset_ciface.h"

!-*-*-  G A L A H A D  -  D U M M Y   M A 7 7 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma77_real_ciface
   use hsl_kinds_real
   use hsl_ma77_real, only :                       &
      f_ma77_keep          => ma77_keep,           &
      f_ma77_control       => ma77_control,        &
      f_ma77_info          => ma77_info,           &
      f_ma77_open          => ma77_open,           &
      f_ma77_input_vars    => ma77_input_vars,     &
      f_ma77_input_reals   => ma77_input_reals,    &
      f_ma77_analyse       => ma77_analyse,        &
      f_ma77_factor        => ma77_factor,         &
      f_ma77_factor_solve  => ma77_factor_solve,   &
      f_ma77_solve         => ma77_solve,          &
      f_ma77_resid         => ma77_resid,          &
      f_ma77_scale         => ma77_scale,          &
      f_ma77_enquire_posdef=> ma77_enquire_posdef, &
      f_ma77_enquire_indef => ma77_enquire_indef,  &
      f_ma77_alter         => ma77_alter,          &
      f_ma77_restart       => ma77_restart,        &
      f_ma77_finalise      => ma77_finalise,       &
      f_ma77_solve_fredholm=> ma77_solve_fredholm, &
      f_ma77_lmultiply     => ma77_lmultiply
   implicit none

   ! Data type for user controls
   type, bind(C) :: ma77_control
      ! C/Fortran interface related controls
      integer(ipc_) :: f_arrays ! 0 is false, otherwise is true

      ! Printing controls
      integer(ipc_)  :: print_level
      integer(ipc_)  :: unit_diagnostics
      integer(ipc_)  :: unit_error
      integer(ipc_)  :: unit_warning

      ! Controls used by MA77_open
      integer(ipc_)  :: bits
      integer(ipc_)  :: buffer_lpage(2)
      integer(ipc_)  :: buffer_npage(2)
      integer(longc_) :: file_size
      integer(longc_) :: maxstore
      integer(longc_) :: storage(3)

      ! Controls used by MA77_analyse
      integer(ipc_)  :: nemin

      ! Controls used by MA77_scale
      integer(ipc_)  :: maxit
      integer(ipc_)  :: infnorm
      real(rpc_)  :: thresh

      ! Controls used by MA77_factor with posdef true
      integer(ipc_)  :: nb54

      ! Controls used by MA77_factor with posdef false
      integer(ipc_)  :: action ! 0 is false, otherwise is true
      real(rpc_)  :: multiplier
      integer(ipc_)  :: nb64
      integer(ipc_)  :: nbi
      real(rpc_)  :: small
      real(rpc_)  :: static
      integer(longc_) :: storage_indef
      real(rpc_)  :: u
      real(rpc_)  :: umin

      ! Controls used by ma77_solve_fredholm
      real(rpc_) :: consist_tol

      ! Padding for future growth
      integer(ipc_) :: ispare(5)
      integer(longc_) :: lspare(5)
      real(rpc_) :: rspare(5)
   end type ma77_control

   !*************************************************

   ! data type for returning information to user.
   type, bind(C) :: ma77_info
      real(rpc_)  :: detlog
      integer(ipc_)  :: detsign
      integer(ipc_)  :: flag
      integer(ipc_)  :: iostat
      integer(ipc_)  :: matrix_dup
      integer(ipc_)  :: matrix_rank
      integer(ipc_)  :: matrix_outrange
      integer(ipc_)  :: maxdepth
      integer(ipc_)  :: maxfront
      integer(longc_) :: minstore
      integer(ipc_)  :: ndelay
      integer(longc_) :: nfactor
      integer(longc_) :: nflops
      integer(ipc_)  :: niter
      integer(ipc_)  :: nsup
      integer(ipc_)  :: num_neg
      integer(ipc_)  :: num_nothresh
      integer(ipc_)  :: num_perturbed
      integer(ipc_)  :: ntwo
      integer(ipc_)  :: stat
      integer(ipc_)  :: index(4)
      integer(longc_) :: nio_read(2)
      integer(longc_) :: nio_write(2)
      integer(longc_) :: nwd_read(2)
      integer(longc_) :: nwd_write(2)
      integer(ipc_)  :: num_file(4)
      integer(longc_) :: storage(4)
      integer(ipc_)  :: tree_nodes
      integer(ipc_)  :: unit_restart
      integer(ipc_)  :: unused
      real(rpc_)  :: usmall

      ! Padding for future growth
      integer(ipc_) :: ispare(5)
      integer(longc_) :: lspare(5)
      real(rpc_) :: rspare(5)
   end type ma77_info

   interface
      integer(C_SIZE_T) pure function strlen(cstr) bind(C)
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(in) :: cstr
      end function strlen
   end interface

contains

   function cstr_to_fchar(cstr) result(fchar)
      type(C_PTR) :: cstr
      character(kind=C_CHAR,len=strlen(cstr)) :: fchar

      integer(ip_) :: i
      character(C_CHAR), dimension(:), pointer :: temp

      call C_F_POINTER(cstr, temp, shape = (/ strlen(cstr) /) )

      do i = 1, size(temp)
         fchar(i:i) = temp(i)
      end do
   end function cstr_to_fchar

   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(ma77_control), intent(in) :: ccontrol
      type(f_ma77_control), intent(out) :: fcontrol
      logical(lp_), intent(out) :: f_arrays

      f_arrays                   = (ccontrol%f_arrays .ne. 0)
      fcontrol%action            = (ccontrol%action .ne. 0)
      fcontrol%bits              = ccontrol%bits
      fcontrol%buffer_lpage(1:2) = ccontrol%buffer_lpage(1:2)
      fcontrol%buffer_npage(1:2) = ccontrol%buffer_npage(1:2)
      fcontrol%consist_tol       = ccontrol%consist_tol
      fcontrol%file_size         = ccontrol%file_size
      fcontrol%infnorm           = ccontrol%infnorm
      fcontrol%maxit             = ccontrol%maxit
      fcontrol%maxstore          = ccontrol%maxstore
      fcontrol%multiplier        = ccontrol%multiplier
      fcontrol%nb54              = ccontrol%nb54
      fcontrol%nb64              = ccontrol%nb64
      fcontrol%nbi               = ccontrol%nbi
      fcontrol%nemin             = ccontrol%nemin
      fcontrol%print_level       = ccontrol%print_level
      fcontrol%small             = ccontrol%small
      fcontrol%static            = ccontrol%static
      fcontrol%storage(1:3)      = ccontrol%storage(1:3)
      fcontrol%storage_indef     = ccontrol%storage_indef
      fcontrol%thresh            = ccontrol%thresh
      fcontrol%unit_diagnostics  = ccontrol%unit_diagnostics
      fcontrol%unit_error        = ccontrol%unit_error
      fcontrol%unit_warning      = ccontrol%unit_warning
      fcontrol%u                 = ccontrol%u
      fcontrol%umin              = ccontrol%umin
   end subroutine copy_control_in

   subroutine copy_info_out(finfo, cinfo)
      type(f_ma77_info), intent(in) :: finfo
      type(ma77_info), intent(out) :: cinfo

      cinfo%detlog         = finfo%detlog
      cinfo%detsign        = finfo%detsign
      cinfo%flag           = finfo%flag
      cinfo%iostat         = finfo%iostat
      cinfo%matrix_dup     = finfo%matrix_dup
      cinfo%matrix_rank    = finfo%matrix_rank
      cinfo%matrix_outrange= finfo%matrix_outrange
      cinfo%maxdepth       = finfo%maxdepth
      cinfo%maxfront       = finfo%maxfront
      cinfo%minstore       = finfo%minstore
      cinfo%ndelay         = finfo%ndelay
      cinfo%nfactor        = finfo%nfactor
      cinfo%nflops         = finfo%nflops
      cinfo%niter          = finfo%niter
      cinfo%nsup           = finfo%nsup
      cinfo%num_neg        = finfo%num_neg
      cinfo%num_nothresh   = finfo%num_nothresh
      cinfo%num_perturbed  = finfo%num_perturbed
      cinfo%ntwo           = finfo%ntwo
      cinfo%stat           = finfo%stat
      cinfo%index(1:4)     = finfo%index(1:4)
      cinfo%nio_read(1:2)  = finfo%nio_read(1:2)
      cinfo%nio_write(1:2) = finfo%nio_write(1:2)
      cinfo%nwd_read(1:2)  = finfo%nwd_read(1:2)
      cinfo%nwd_write(1:2) = finfo%nwd_write(1:2)
      cinfo%num_file(1:4)  = finfo%num_file(1:4)
      cinfo%storage(1:4)   = finfo%storage(1:4)
      cinfo%tree_nodes     = finfo%tree_nodes
      cinfo%unit_restart   = finfo%unit_restart
      cinfo%unused         = finfo%unused
      cinfo%usmall         = finfo%u
   end subroutine copy_info_out

   subroutine ma77_open_main(n, cfname1, cfname2, cfname3, cfname4, ckeep, &
         ccontrol, cinfo, nelt)

      integer(ipc_), intent(in) :: n
      type(C_PTR), intent(in) :: cfname1
      type(C_PTR), intent(in) :: cfname2
      type(C_PTR), intent(in) :: cfname3
      type(C_PTR), intent(in) :: cfname4
      type(C_PTR), intent(out) :: ckeep
      type(ma77_control), intent(in) :: ccontrol
      type(ma77_info), intent(inout) :: cinfo
      integer(ipc_), optional, intent(in) :: nelt

      type(f_ma77_keep), pointer :: fkeep
      type(f_ma77_control) :: fcontrol
      type(f_ma77_info) :: finfo
      character( kind = C_CHAR, len = max( &
         strlen(cfname1),strlen(cfname2),strlen(cfname3),strlen(cfname4) ) &
         ), dimension(4) :: fname
      logical(lp_) :: f_arrays

      ! Copy data in and associate pointers correctly
      call copy_control_in(ccontrol, fcontrol, f_arrays)
      fname(1) = cstr_to_fchar(cfname1)
      fname(2) = cstr_to_fchar(cfname2)
      fname(3) = cstr_to_fchar(cfname3)
      fname(4) = cstr_to_fchar(cfname4)

      ! Allocate space to store keep and arrange a C pointer to it
      allocate(fkeep)
      ckeep = c_loc(fkeep)

      ! Call the Fortran routine
      call f_ma77_open(n, fname, fkeep, fcontrol, finfo, nelt=nelt)

      ! Copy information out to C structure
      call copy_info_out(finfo, cinfo)

   end subroutine ma77_open_main

end module hsl_ma77_real_ciface
