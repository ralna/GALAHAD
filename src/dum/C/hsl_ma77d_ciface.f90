! THIS VERSION: 29/12/2021 AT 15:35:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M A 7 7 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma77_double_ciface
   use iso_c_binding
   USE GALAHAD_common_ciface
   use hsl_ma77_double, only :                     &
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
      integer(C_INT) :: f_arrays ! 0 is false, otherwise is true

      ! Printing controls
      integer(C_INT)  :: print_level
      integer(C_INT)  :: unit_diagnostics
      integer(C_INT)  :: unit_error
      integer(C_INT)  :: unit_warning

      ! Controls used by MA77_open
      integer(C_INT)  :: bits
      integer(C_INT)  :: buffer_lpage(2)
      integer(C_INT)  :: buffer_npage(2)
      integer(C_LONG) :: file_size
      integer(C_LONG) :: maxstore
      integer(C_LONG) :: storage(3)

      ! Controls used by MA77_analyse
      integer(C_INT)  :: nemin

      ! Controls used by MA77_scale
      integer(C_INT)  :: maxit
      integer(C_INT)  :: infnorm
      real(C_DOUBLE)  :: thresh

      ! Controls used by MA77_factor with posdef true
      integer(C_INT)  :: nb54

      ! Controls used by MA77_factor with posdef false
      integer(C_INT)  :: action ! 0 is false, otherwise is true
      real(C_DOUBLE)  :: multiplier
      integer(C_INT)  :: nb64
      integer(C_INT)  :: nbi
      real(C_DOUBLE)  :: small
      real(C_DOUBLE)  :: static
      integer(C_LONG) :: storage_indef
      real(C_DOUBLE)  :: u
      real(C_DOUBLE)  :: umin

      ! Controls used by ma77_solve_fredholm
      real(C_DOUBLE) :: consist_tol
      
      ! Padding for future growth
      integer(C_INT) :: ispare(5)
      integer(C_LONG) :: lspare(5)
      real(C_DOUBLE) :: rspare(5)
   end type ma77_control

   !*************************************************

   ! data type for returning information to user.
   type, bind(C) :: ma77_info
      real(C_DOUBLE)  :: detlog
      integer(C_INT)  :: detsign
      integer(C_INT)  :: flag
      integer(C_INT)  :: iostat
      integer(C_INT)  :: matrix_dup
      integer(C_INT)  :: matrix_rank
      integer(C_INT)  :: matrix_outrange
      integer(C_INT)  :: maxdepth
      integer(C_INT)  :: maxfront
      integer(C_LONG) :: minstore
      integer(C_INT)  :: ndelay
      integer(C_LONG) :: nfactor
      integer(C_LONG) :: nflops
      integer(C_INT)  :: niter
      integer(C_INT)  :: nsup
      integer(C_INT)  :: num_neg
      integer(C_INT)  :: num_nothresh
      integer(C_INT)  :: num_perturbed
      integer(C_INT)  :: ntwo
      integer(C_INT)  :: stat
      integer(C_INT)  :: index(4)
      integer(C_LONG) :: nio_read(2)
      integer(C_LONG) :: nio_write(2)
      integer(C_LONG) :: nwd_read(2)
      integer(C_LONG) :: nwd_write(2)
      integer(C_INT)  :: num_file(4)
      integer(C_LONG) :: storage(4)
      integer(C_INT)  :: tree_nodes
      integer(C_INT)  :: unit_restart
      integer(C_INT)  :: unused
      real(C_DOUBLE)  :: usmall
      
      ! Padding for future growth
      integer(C_INT) :: ispare(5)
      integer(C_LONG) :: lspare(5)
      real(C_DOUBLE) :: rspare(5)
   end type ma77_info

contains

   subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
      type(ma77_control), intent(in) :: ccontrol
      type(f_ma77_control), intent(out) :: fcontrol
      logical, intent(out) :: f_arrays

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

      integer(C_INT), intent(in) :: n
      type(C_PTR), intent(in) :: cfname1
      type(C_PTR), intent(in) :: cfname2
      type(C_PTR), intent(in) :: cfname3
      type(C_PTR), intent(in) :: cfname4
      type(C_PTR), intent(out) :: ckeep
      type(ma77_control), intent(in) :: ccontrol
      type(ma77_info), intent(inout) :: cinfo
      integer(C_INT), optional, intent(in) :: nelt

      type(f_ma77_keep), pointer :: fkeep
      type(f_ma77_control) :: fcontrol
      type(f_ma77_info) :: finfo
      character( kind=C_CHAR, len = max( &
         strlen(cfname1),strlen(cfname2),strlen(cfname3),strlen(cfname4) ) &
         ), dimension(4) :: fname
      logical :: f_arrays

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

end module hsl_ma77_double_ciface
