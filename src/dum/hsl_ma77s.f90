! THIS VERSION: GALAHAD 4.3 - 2024-01-27 AT 13:30 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 7 7   M O D U L E  -*-*-*-

module hsl_MA77_single

   use GALAHAD_KINDS
   use hsl_of01_single, of01_rdata => of01_data
   use hsl_of01_integer, of01_idata => of01_data

  implicit none

  real (sp_), parameter, private :: one = 1.0_sp_
  real (sp_), parameter, private :: zero = 0.0_sp_
  integer(ip_), parameter, private :: nemin_default = 8

  interface MA77_open
      module procedure MA77_open_single
  end interface

  interface MA77_input_vars
      module procedure MA77_input_vars_single
  end interface

  interface MA77_input_reals
      module procedure MA77_input_reals_single
  end interface

  interface MA77_analyse
      module procedure MA77_analyse_single
  end interface

  interface MA77_factor
      module procedure MA77_factor_single
  end interface

  interface MA77_factor_solve
      module procedure MA77_factor_solve_single
  end interface

  interface MA77_solve
      module procedure MA77_solve_single
  end interface

  interface MA77_solve_fredholm
      module procedure MA77_solve_fredholm_single
  end interface

  interface MA77_resid
      module procedure MA77_resid_single
  end interface

  interface MA77_scale
      module procedure MA77_scale_single
  end interface

  interface MA77_enquire_posdef
    module procedure MA77_enquire_posdef_single
  end interface

  interface MA77_enquire_indef
    module procedure MA77_enquire_indef_single
  end interface

  interface MA77_alter
    module procedure MA77_alter_single
  end interface

  interface MA77_restart
      module procedure MA77_restart_single
  end interface

  interface MA77_lmultiply
      module procedure MA77_lmultiply_single
  end interface

  interface MA77_finalise
      module procedure MA77_finalise_single
  end interface

  type MA77_control

    logical(lp_) :: action = .true.
    integer(ip_) :: bits = 32
    integer(ip_) :: buffer_lpage(2) = 2**12
    integer(ip_) :: buffer_npage(2) = 1600
    real(sp_) :: consist_tol = epsilon(one)
    integer(long_) :: file_size = 2**21
    integer(ip_) :: infnorm = 0
    integer(ip_) :: maxit = 1
    integer(long_) :: maxstore = 0_long_
    real(sp_) :: multiplier = 1.1
    integer(ip_) :: nb54 = 150
    integer(ip_) :: nb64 = 120
    integer(ip_) :: nbi = 40
    integer(ip_) :: nemin = nemin_default
    integer(ip_) :: p = 4
    integer(ip_) :: print_level = 0
    real(sp_) :: small = tiny(one)
    real (sp_) :: static = zero
    integer(long_) :: storage(3) = 0_long_
    integer(long_) :: storage_indef = 0
    real(sp_) :: thresh = 0.5
    integer(ip_) :: unit_diagnostics = 6
    integer(ip_) :: unit_error = 6
    integer(ip_) :: unit_warning = 6
    real (sp_) :: u = 0.01
    real (sp_) :: umin = 0.01
  end type MA77_control

  type MA77_info
    real (sp_) :: detlog = 0.0
    integer(ip_) :: detsign = 1
    integer(ip_) :: flag = 0
    integer(ip_) :: iostat = 0
    integer(ip_) :: matrix_dup = 0
    integer(ip_) :: matrix_rank = 0
    integer(ip_) :: matrix_outrange = 0
    integer(ip_) :: maxdepth = 0
    integer(ip_) :: maxfront = 0
    integer(long_)  :: minstore = 0_long_
    integer(ip_) :: ndelay = 0
    integer(long_) :: nfactor = 0
    integer(long_) :: nflops = 0
    integer(ip_)  :: niter = 0
    integer(ip_) :: nsup = 0
    integer(ip_) :: num_neg = 0
    integer(ip_) :: num_nothresh = 0
    integer(ip_) :: num_perturbed = 0
    integer(ip_) :: ntwo = 0
    integer(ip_) :: stat = 0
    integer(ip_) :: index(1:4) = -1
    integer(long_) :: nio_read(1:2) = 0
    integer(long_) :: nio_write(1:2) = 0
    integer(long_) :: nwd_read(1:2) = 0
    integer(long_) :: nwd_write(1:2) = 0
    integer(ip_) :: num_file(1:4) = 0

    integer(long_) :: storage(1:4) = 0_long_
    integer(ip_) :: tree_nodes = 0
    integer(ip_) :: unit_restart = -1
    integer(ip_) :: unused = 0
    real(sp_) :: u = zero
  end type MA77_info

  type MA77_node
    integer(ip_), allocatable :: child(:)
    integer(ip_) :: nelim
  end type MA77_node

  type MA77_keep
    integer(long_) :: dfree
    logical(lp_) :: element_input
    integer(long_) :: file_size
    integer(ip_) :: flag = 0
    integer(long_) :: ifree
    integer(ip_) :: index(1:4) = -1
    integer(ip_) :: inelrs
    integer(ip_) :: inelrn
    integer(ip_) :: lpage(2)
    integer(ip_) :: ltree
    integer(long_)  :: lup
    integer(ip_) :: l1,l2
    integer(ip_) :: matrix_dup
    integer(ip_) :: matrix_outrange
    integer(ip_) :: maxelim
    integer(ip_) :: maxelim_actual
    integer(long_)  :: maxfa
    integer(ip_) :: maxfront
    integer(ip_) :: maxdepth
    integer(ip_) :: maxfrontb
    integer(ip_) :: maxlen
    integer(long_) :: maxstore
    integer(ip_) :: mvar
    integer(ip_) :: mmvar
    integer(long_)  :: mx_ifree
    integer(ip_) :: n
    character(50)  :: name
    integer(ip_) :: nb
    integer(ip_) :: nbi
    integer(ip_) :: nelt
    integer(ip_) :: npage(2)
    integer(ip_) :: nsup
    integer(ip_) :: ntwo
    integer(ip_) :: null
    logical(lp_) :: pos_def
    integer(long_)  :: posfac
    integer(long_)  :: posint
    integer(long_)  :: rfree
    integer(long_)  :: rtopmx
    integer(long_)  :: rtopdmx
    integer(ip_) :: scale = 0
    integer(ip_) :: status = 0
    integer(long_) :: used
    integer(ip_) :: tnode

    real(sp_), allocatable :: aelt(:)
    real(sp_), allocatable :: arow(:)
    integer(ip_), allocatable :: clist(:)
    integer(long_), allocatable :: ifile(:)
    integer(ip_), allocatable :: iptr(:)
    integer(ip_), allocatable :: map(:)
    integer(ip_), allocatable :: new(:)
    integer(long_), allocatable :: rfile(:)
    integer(ip_), allocatable :: roots(:)
    integer(ip_), allocatable :: size(:)
    integer(ip_), allocatable  :: size_ind(:)
    integer(ip_), allocatable :: splitp(:)
    integer(ip_), allocatable :: svar(:)
    integer(ip_), allocatable :: vars(:)
    integer(ip_), allocatable :: varflag(:)
    integer(long_) :: size_imain
    integer(long_) :: size_rmain
    integer(long_) :: size_rwork
    integer(long_) :: size_rwdelay
    integer(ip_),allocatable :: imain(:)
    real(sp_),allocatable :: rmain(:)
    real(sp_),allocatable :: rwork(:)
    real(sp_),allocatable :: rwdelay(:)
    character,allocatable :: file1(:)
    character,allocatable :: file2(:)
    character,allocatable :: file3(:)
    character,allocatable :: file4(:)
    character,allocatable :: file5(:)
    type (MA77_node),allocatable :: tree(:)
    type (of01_rdata) :: rdata
    type (of01_idata) :: idata
  end type MA77_keep

contains

  subroutine MA77_open_single(n,filename,keep,control,info,nelt,path)
    USE GALAHAD_SYMBOLS
    integer(ip_), intent (in) :: n
    character (len=*), intent (in) :: filename(4)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info) :: info
    integer(ip_), optional, intent (in) :: nelt
    character (len=*), optional, intent (in) :: path(:)
    call MA77_unavailable( info, control, 'ma77_open' )
  end subroutine MA77_open_single

  subroutine MA77_input_vars_single(index,nvar,list,keep,control,info)
    USE GALAHAD_SYMBOLS
    integer(ip_), intent (in) :: index
    integer(ip_), intent (in) :: nvar
!            in the incoming element/row. Must be >= 0.
    integer(ip_), intent (in) :: list(nvar)
!            incoming element/row.
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    call MA77_unavailable( info, control, 'ma77_input_vars' )
  end subroutine MA77_input_vars_single

  subroutine MA77_analyse_single(order,keep,control,info)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    integer(ip_), intent (inout), dimension(keep%n) :: order
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    call MA77_unavailable( info, control, 'ma77_analyse' )
  end subroutine MA77_analyse_single

!****************************************************************************

  subroutine MA77_input_reals_single(index,length,reals,keep,control,info)
    USE GALAHAD_SYMBOLS
    integer(ip_), intent (in) :: index
    integer(ip_), intent (in) :: length
    real (sp_), intent (in) :: reals(length)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    call MA77_unavailable( info, control, 'ma77_input_reals' )
  end subroutine MA77_input_reals_single

  subroutine MA77_factor_single(pos_def,keep,control,info,scale)
    USE GALAHAD_SYMBOLS
    logical(lp_) :: pos_def
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_factor' )
  end subroutine MA77_factor_single

  subroutine MA77_factor_solve_single(pos_def,keep,control,info,nrhs, &
     lx,x,scale)
    USE GALAHAD_SYMBOLS
    integer(ip_), parameter :: nb_default = 150
    logical(lp_), intent (in) :: pos_def
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_), intent(in), optional :: scale(:)
    integer(ip_) :: lx
    integer(ip_) :: nrhs
    real (sp_), intent(inout) :: x(lx,nrhs)
    call MA77_unavailable( info, control, 'ma77_factor_solve' )
  end subroutine MA77_factor_solve_single

   subroutine MA77_resid_single(nrhs,lx,x,lresid,resid,keep,control,info,anorm)
    USE GALAHAD_SYMBOLS
    integer(ip_) :: nrhs
    integer(ip_) :: lx
    integer(ip_) :: lresid
    real(sp_), intent(in) :: x(lx,nrhs)
    real(sp_), intent(inout) :: resid(lresid,nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_),optional :: anorm
    call MA77_unavailable( info, control, 'ma77_resid' )
  end subroutine MA77_resid_single

  subroutine MA77_solve_single(nrhs,lx,x,keep,control,info,scale,job)
    USE GALAHAD_SYMBOLS
    integer(ip_), intent (in) :: nrhs
    integer(ip_), intent (in) :: lx
    real (sp_), intent (inout) :: x(lx,nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_), intent(in), optional :: scale(:)
    integer(ip_), optional, intent (in) :: job
    call MA77_unavailable( info, control, 'ma77_solve' )
  end subroutine MA77_solve_single

  subroutine MA77_solve_fredholm_single( nrhs, flag_out, lx, x,                &
                                         keep, control, info, scale )
   USE GALAHAD_SYMBOLS
    integer(ip_), intent (in) :: nrhs
    logical(lp_), intent(out) :: flag_out(nrhs)
    integer(ip_), intent (in) :: lx
    real (sp_), intent (inout) :: x(lx,2*nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_solve_fredholm' )
  end subroutine MA77_solve_fredholm_single

  subroutine MA77_lmultiply_single(trans,k,lx,x,ly,y,keep,control,info,scale)
    logical(lp_), intent (in) :: trans
    integer(ip_), intent (in) :: k
    integer(ip_), intent (in) :: lx, ly
    real (sp_), intent (inout) :: x(lx,k) ! On entry, x must
    real (sp_), intent (out) :: y(ly,k) ! On exit,
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_lmultiply' )
  end subroutine MA77_lmultiply_single

  subroutine MA77_enquire_posdef_single(d,keep,control,info)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_) :: d(:)
    call MA77_unavailable( info, control, 'ma77_enquire_posdef' )
  end subroutine ma77_enquire_posdef_single

  subroutine MA77_enquire_indef_single(piv_order,d,keep,control,info)
    USE GALAHAD_SYMBOLS
    real(sp_) :: d(:,:)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    integer(ip_) :: piv_order(:)
    call MA77_unavailable( info, control, 'ma77_enquire_indef' )
  end subroutine ma77_enquire_indef_single

  subroutine MA77_alter_single(d,keep,control,info)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_), intent (inout) :: d(:,:)
    call MA77_unavailable( info, control, 'ma77_alter' )
  end subroutine ma77_alter_single

  subroutine MA77_scale_single(scale,keep,control,info,anorm)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(sp_),optional :: anorm
    real (sp_) :: scale(:)
    call MA77_unavailable( info, control, 'ma77_scale' )
  end subroutine MA77_scale_single

  subroutine MA77_print_iflag(keep,nout,iflag,ie,st,ios)
    integer(ip_), intent (in) :: iflag, nout
    integer(ip_), intent (in), optional :: ie, st, ios
    type (MA77_keep), intent (in) :: keep
  end subroutine MA77_print_iflag

  subroutine MA77_read_integer(ifile,keep_array,loc,length,read_array, &
             flag,data,lp)
    USE GALAHAD_SYMBOLS
   integer(ip_), intent(in) :: ifile
   integer(ip_), allocatable :: keep_array(:)
   integer(long_), intent(in) :: loc
   integer(ip_), intent(in) :: length
   integer(ip_), dimension(:) :: read_array
   integer(ip_) :: flag
   type (of01_idata), intent(inout) :: data
   integer(ip_), intent(in) :: lp
   flag = GALAHAD_unavailable_option
  end subroutine MA77_read_integer

!************************************************************************

  subroutine MA77_read_real(rfile,keep_array,loc,length,read_array, &
             flag,data,lp,map)
    USE GALAHAD_SYMBOLS
   integer(ip_), intent(in) :: rfile
   real(sp_), allocatable :: keep_array(:)
   integer(long_), intent(in) :: loc
   integer(ip_), intent(in) :: length
   real(sp_), dimension(:), intent(inout) :: read_array
   integer(ip_) :: flag
   type (of01_rdata), intent(inout) :: data
   integer(ip_), intent(in) :: lp
   integer(ip_), optional, intent (in) :: map(length)
   flag = GALAHAD_unavailable_option
  end subroutine MA77_read_real

  subroutine MA77_read_discard_real(rfile,keep_array,loc,length,read_array, &
             flag,data,lp,map)
    USE GALAHAD_SYMBOLS
   integer(ip_), intent(in) :: rfile
   real(sp_), allocatable :: keep_array(:)
   integer(long_), intent(in) :: loc
   integer(ip_), intent(in) :: length
   real(sp_), dimension(:), intent(inout) :: read_array
   integer(ip_) :: flag
   type (of01_rdata), intent(inout) :: data
   integer(ip_), intent(in) :: lp
   integer(ip_), optional, intent (in) :: map(length)
   flag = GALAHAD_unavailable_option
  end subroutine MA77_read_discard_real

  subroutine MA77_write_real(rfile,size_array,keep_array,loc,length, &
             write_array,flag,data,lp,maxstore,used,inactive)
    USE GALAHAD_SYMBOLS
   integer(ip_), intent(inout) :: rfile
   real(sp_), allocatable :: keep_array(:)
   integer(long_), intent(inout) :: size_array
   integer(long_), intent(in) :: loc
   integer(ip_), intent(in) :: length
   real(sp_), intent(in) :: write_array(:)
   integer(ip_) :: flag
   type (of01_rdata), intent(inout) :: data
   integer(ip_), intent(in) :: lp
   integer(long_), intent(in) :: maxstore
   integer(long_), intent(inout) :: used
   integer(long_), optional, intent(in) :: inactive
   flag = GALAHAD_unavailable_option
  end subroutine MA77_write_real

  subroutine MA77_write_integer(ifile,size_array,keep_array,loc,length, &
             write_array,flag,data,lp,maxstore,used)
    USE GALAHAD_SYMBOLS
   integer(ip_), intent(inout) :: ifile
   integer(ip_), allocatable :: keep_array(:)
   integer(long_), intent(inout) :: size_array
   integer(long_), intent(in) :: loc
   integer(ip_), intent(in) :: length
   integer(ip_), intent(in) :: write_array(:)
   integer(ip_) :: flag
   type (of01_idata), intent(inout) :: data
   integer(ip_), intent(in) :: lp
   integer(long_), intent(in) :: maxstore
   integer(long_), intent(inout) :: used
   flag = GALAHAD_unavailable_option
  end subroutine MA77_write_integer

  subroutine MA77_finalise_single(keep,control,info,restart_file)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info) :: info
    character (len=*), optional, intent (in) :: restart_file
    call MA77_unavailable( info, control, 'ma77_finalize' )
  end subroutine MA77_finalise_single

  subroutine MA77_restart_single(restart_file,filename,keep,control,info,path)
    USE GALAHAD_SYMBOLS
    type (MA77_keep) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    character (len=*), intent (in) :: restart_file
    character (len=*), optional, intent (in) :: path(:)
    character (len=*), intent (in) :: filename(4)
    call MA77_unavailable( info, control, 'ma77_restart' )
  end subroutine MA77_restart_single

  subroutine MA77_unavailable( info, control, name )
    USE GALAHAD_SYMBOLS
    type (MA77_info), intent (inout) :: info
    type (MA77_control), intent (in) :: control
    character ( len = * ), intent( in ) :: name
        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine ', A, ' HSL namesake ', /,                                &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" ) name
    info%flag = GALAHAD_unavailable_option
    info%detlog = 0.0
    info%detsign = 1
    info%iostat = 0
    info%matrix_dup = 0
    info%matrix_rank = 0
    info%matrix_outrange = 0
    info%maxdepth = 0
    info%maxfront = 0
    info%minstore = 0_long_
    info%ndelay = 0
    info%nfactor = 0
    info%nflops = 0
    info%niter = 0
    info%nsup = 0
    info%num_neg = 0
    info%num_nothresh = 0
    info%num_perturbed = 0
    info%ntwo = 0
    info%stat = 0
    info%index(1:4) = -1
    info%nio_read(1:2) = 0
    info%nio_write(1:2) = 0
    info%nwd_read(1:2) = 0
    info%nwd_write(1:2) = 0
    info%num_file(1:4) = 0
    info%storage(1:4) = 0_long_
    info%tree_nodes = 0
    info%unit_restart = -1
    info%unused = 0
    info%u = zero
  end subroutine MA77_unavailable

end module hsl_MA77_single
