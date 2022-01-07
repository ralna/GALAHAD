! THIS VERSION: GALAHAD 4.0 - 2022-01-07 AT 12:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 7 7   M O D U L E  -*-*-*-

module hsl_MA77_single

!  use hsl_ma54_single
!  use hsl_ma64_single
   use hsl_of01_single, of01_rdata => of01_data
   use hsl_of01_integer, of01_idata => of01_data

  implicit none

  integer, parameter, private  :: wp = kind(0.0)
  integer, parameter, private  :: long = selected_int_kind(18)
  integer, parameter, private  :: short = kind(0)
  real (wp), parameter, private :: one = 1.0_wp
  real (wp), parameter, private :: zero = 0.0_wp
  real (wp), parameter, private :: half = 0.5_wp
  integer(short), parameter, private :: nemin_default = 8
  integer(short), parameter, private :: lup1 = huge(0_short)/8
  integer(short), parameter, private :: nb54_default = 150
  integer(short), parameter, private :: nb64_default = 120
  integer(short), parameter, private :: nbi_default = 40

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

    logical :: action = .true.
    integer(short) :: bits = 32
    integer(short) :: buffer_lpage(2) = 2**12 
    integer(short) :: buffer_npage(2) = 1600
    real(wp) :: consist_tol = epsilon(one) 
    integer(long) :: file_size = 2**21
    integer(short) :: infnorm = 0
    integer(short) :: maxit = 1
    integer(long) :: maxstore = 0_long
    real(wp) :: multiplier = 1.1
    integer(short) :: nb54 = 150
    integer(short) :: nb64 = 120
    integer(short) :: nbi = 40
    integer(short) :: nemin = nemin_default
    integer(short) :: p = 4
    integer(short) :: print_level = 0
    real(wp) :: small = tiny(one)
    real (wp) :: static = zero
    integer(long) :: storage(3) = 0_long
    integer(long) :: storage_indef = 0
    real(wp) :: thresh = 0.5
    integer(short) :: unit_diagnostics = 6
    integer(short) :: unit_error = 6
    integer(short) :: unit_warning = 6
    real (wp) :: u = 0.01
    real (wp) :: umin = 0.01
  end type MA77_control

  type MA77_info
    real (wp) :: detlog = 0.0
    integer(short) :: detsign = 1 
    integer(short) :: flag = 0    
    integer(short) :: iostat = 0
    integer(short) :: matrix_dup = 0
    integer(short) :: matrix_rank = 0
    integer(short) :: matrix_outrange = 0
    integer(short) :: maxdepth = 0
    integer(short) :: maxfront = 0
    integer(long)  :: minstore = 0_long
    integer(short) :: ndelay = 0
    integer(long) :: nfactor = 0
    integer(long) :: nflops = 0
    integer(short)  :: niter = 0
    integer(short) :: nsup = 0
    integer(short) :: num_neg = 0
    integer(short) :: num_nothresh = 0
    integer(short) :: num_perturbed = 0
    integer(short) :: ntwo = 0
    integer(short) :: stat = 0
    integer(short) :: index(1:4) = -1
    integer(long) :: nio_read(1:2) = 0
    integer(long) :: nio_write(1:2) = 0
    integer(long) :: nwd_read(1:2) = 0
    integer(long) :: nwd_write(1:2) = 0
    integer(short) :: num_file(1:4) = 0

    integer(long) :: storage(1:4) = 0_long
    integer(short) :: tree_nodes = 0
    integer(short) :: unit_restart = -1
    integer(short) :: unused = 0
    real(wp) :: u = zero
  end type MA77_info

  type MA77_node
    integer(short), allocatable :: child(:)
    integer(short) :: nelim
  end type MA77_node

  type MA77_keep
    integer(long) :: dfree
    logical :: element_input
    integer(long) :: file_size
    integer(short) :: flag = 0
    integer(long) :: ifree
    integer(short) :: index(1:4) = -1
    integer(short) :: inelrs
    integer(short) :: inelrn
    integer(short) :: lpage(2)
    integer(short) :: ltree
    integer(long)  :: lup
    integer(short) :: l1,l2
    integer(short) :: matrix_dup
    integer(short) :: matrix_outrange
    integer(short) :: maxelim
    integer(short) :: maxelim_actual
    integer(long)  :: maxfa
    integer(short) :: maxfront
    integer(short) :: maxdepth
    integer(short) :: maxfrontb
    integer(short) :: maxlen  
    integer(long) :: maxstore
    integer(short) :: mvar
    integer(short) :: mmvar
    integer(long)  :: mx_ifree
    integer(short) :: n
    character(50)  :: name
    integer(short) :: nb
    integer(short) :: nbi
    integer(short) :: nelt
    integer(short) :: npage(2)
    integer(short) :: nsup
    integer(short) :: ntwo
    integer(short) :: null
    logical :: pos_def
    integer(long)  :: posfac
    integer(long)  :: posint
    integer(long)  :: rfree
    integer(long)  :: rtopmx 
    integer(long)  :: rtopdmx
    integer(short) :: scale = 0
    integer(short) :: status = 0
    integer(long) :: used
    integer(short) :: tnode

    real(wp), allocatable :: aelt(:)
    real(wp), allocatable :: arow(:)
    integer(short), allocatable :: clist(:)
    integer(long), allocatable :: ifile(:)
    integer(short), allocatable :: iptr(:)
    integer(short), allocatable :: map(:)
    integer(short), allocatable :: new(:)
    integer(long), allocatable :: rfile(:)
    integer(short), allocatable :: roots(:)
    integer(short), allocatable :: size(:)
    integer(short), allocatable  :: size_ind(:)
    integer(short), allocatable :: splitp(:)
    integer(short), allocatable :: svar(:)
    integer(short), allocatable :: vars(:)
    integer(short), allocatable :: varflag(:)
    integer(long) :: size_imain
    integer(long) :: size_rmain
    integer(long) :: size_rwork
    integer(long) :: size_rwdelay
    integer(short),allocatable :: imain(:)
    real(wp),allocatable :: rmain(:)
    real(wp),allocatable :: rwork(:)
    real(wp),allocatable :: rwdelay(:)
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
    integer(short), intent (in) :: n
    character (len=*), intent (in) :: filename(4)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info) :: info
    integer(short), optional, intent (in) :: nelt
    character (len=*), optional, intent (in) :: path(:)
    call MA77_unavailable( info, control, 'ma77_open' )
  end subroutine MA77_open_single

  subroutine MA77_input_vars_single(index,nvar,list,keep,control,info)
    USE GALAHAD_SYMBOLS
    integer(short), intent (in) :: index
    integer(short), intent (in) :: nvar
!            in the incoming element/row. Must be >= 0.
    integer(short), intent (in) :: list(nvar)
!            incoming element/row.
    type (MA77_keep), intent (inout) :: keep   
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info   
    call MA77_unavailable( info, control, 'ma77_input_vars' )
  end subroutine MA77_input_vars_single

  subroutine MA77_analyse_single(order,keep,control,info)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    integer(short), intent (inout), dimension(keep%n) :: order
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    call MA77_unavailable( info, control, 'ma77_analyse' )
  end subroutine MA77_analyse_single

!****************************************************************************

  subroutine MA77_input_reals_single(index,length,reals,keep,control,info)
    USE GALAHAD_SYMBOLS
    integer(short), intent (in) :: index
    integer(short), intent (in) :: length
    real (wp), intent (in) :: reals(length)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    call MA77_unavailable( info, control, 'ma77_input_reals' )
  end subroutine MA77_input_reals_single

  subroutine MA77_factor_single(pos_def,keep,control,info,scale)
    USE GALAHAD_SYMBOLS
    logical :: pos_def
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(wp), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_factor' )
  end subroutine MA77_factor_single

  subroutine MA77_factor_solve_single(pos_def,keep,control,info,nrhs, &
     lx,x,scale)
    USE GALAHAD_SYMBOLS
    integer(short), parameter :: nb_default = 150
    logical, intent (in) :: pos_def
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(wp), intent(in), optional :: scale(:) 
    integer(short) :: lx
    integer(short) :: nrhs
    real (wp), intent(inout) :: x(lx,nrhs)
    call MA77_unavailable( info, control, 'ma77_factor_solve' )
  end subroutine MA77_factor_solve_single

   subroutine MA77_resid_single(nrhs,lx,x,lresid,resid,keep,control,info,anorm)
    USE GALAHAD_SYMBOLS
    integer(short) :: nrhs
    integer(short) :: lx
    integer(short) :: lresid
    real(wp), intent(in) :: x(lx,nrhs)
    real(wp), intent(inout) :: resid(lresid,nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(wp),optional :: anorm
    call MA77_unavailable( info, control, 'ma77_resid' )
  end subroutine MA77_resid_single

  subroutine MA77_solve_single(nrhs,lx,x,keep,control,info,scale,job)
    USE GALAHAD_SYMBOLS
    integer(short), intent (in) :: nrhs
    integer(short), intent (in) :: lx
    real (wp), intent (inout) :: x(lx,nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(wp), intent(in), optional :: scale(:)
    integer(short), optional, intent (in) :: job 
    call MA77_unavailable( info, control, 'ma77_solve' )
  end subroutine MA77_solve_single

  subroutine MA77_solve_fredholm_single( nrhs, flag_out, lx, x,                &
                                         keep, control, info, scale )
   USE GALAHAD_SYMBOLS
    integer(short), intent (in) :: nrhs
    logical, intent(out) :: flag_out(nrhs) 
    integer(short), intent (in) :: lx
    real (wp), intent (inout) :: x(lx,2*nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(wp), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_solve_fredholm' )
  end subroutine MA77_solve_fredholm_single

  subroutine MA77_lmultiply_single(trans,k,lx,x,ly,y,keep,control,info,scale)
    logical, intent (in) :: trans
    integer(short), intent (in) :: k
    integer(short), intent (in) :: lx, ly
    real (wp), intent (inout) :: x(lx,k) ! On entry, x must
    real (wp), intent (out) :: y(ly,k) ! On exit,
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(wp), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_lmultiply' )
  end subroutine MA77_lmultiply_single

  subroutine MA77_enquire_posdef_single(d,keep,control,info)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    real(wp) :: d(:)
    call MA77_unavailable( info, control, 'ma77_enquire_posdef' )
  end subroutine ma77_enquire_posdef_single

  subroutine MA77_enquire_indef_single(piv_order,d,keep,control,info)
    USE GALAHAD_SYMBOLS
    real(wp) :: d(:,:)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    integer(short) :: piv_order(:)
    call MA77_unavailable( info, control, 'ma77_enquire_indef' )
  end subroutine ma77_enquire_indef_single

  subroutine MA77_alter_single(d,keep,control,info)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    real(wp), intent (inout) :: d(:,:)
    call MA77_unavailable( info, control, 'ma77_alter' )
  end subroutine ma77_alter_single

  subroutine MA77_scale_single(scale,keep,control,info,anorm)
    USE GALAHAD_SYMBOLS
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(wp),optional :: anorm
    real (wp) :: scale(:)
    call MA77_unavailable( info, control, 'ma77_scale' )
  end subroutine MA77_scale_single

  subroutine MA77_print_iflag(keep,nout,iflag,ie,st,ios)
    integer(short), intent (in) :: iflag, nout
    integer(short), intent (in), optional :: ie, st, ios
    type (MA77_keep), intent (in) :: keep
  end subroutine MA77_print_iflag

  subroutine MA77_read_integer(ifile,keep_array,loc,length,read_array, &
             flag,data,lp)
    USE GALAHAD_SYMBOLS
   integer(short), intent(in) :: ifile
   integer(short), allocatable :: keep_array(:)
   integer(long), intent(in) :: loc
   integer(short), intent(in) :: length
   integer(short), dimension(:) :: read_array
   integer(short) :: flag
   type (of01_idata), intent(inout) :: data
   integer(short), intent(in) :: lp
   flag = GALAHAD_unavailable_option
  end subroutine MA77_read_integer

!************************************************************************

  subroutine MA77_read_real(rfile,keep_array,loc,length,read_array, &
             flag,data,lp,map)
    USE GALAHAD_SYMBOLS
   integer(short), intent(in) :: rfile
   real(wp), allocatable :: keep_array(:)
   integer(long), intent(in) :: loc
   integer(short), intent(in) :: length
   real(wp), dimension(:), intent(inout) :: read_array
   integer(short) :: flag
   type (of01_rdata), intent(inout) :: data
   integer(short), intent(in) :: lp
   integer(short), optional, intent (in) :: map(length)
   flag = GALAHAD_unavailable_option
  end subroutine MA77_read_real

  subroutine MA77_read_discard_real(rfile,keep_array,loc,length,read_array, &
             flag,data,lp,map)
    USE GALAHAD_SYMBOLS
   integer(short), intent(in) :: rfile
   real(wp), allocatable :: keep_array(:)
   integer(long), intent(in) :: loc
   integer(short), intent(in) :: length
   real(wp), dimension(:), intent(inout) :: read_array
   integer(short) :: flag
   type (of01_rdata), intent(inout) :: data
   integer(short), intent(in) :: lp
   integer(short), optional, intent (in) :: map(length)
   flag = GALAHAD_unavailable_option
  end subroutine MA77_read_discard_real

  subroutine MA77_write_real(rfile,size_array,keep_array,loc,length, &
             write_array,flag,data,lp,maxstore,used,inactive)
    USE GALAHAD_SYMBOLS
   integer(short), intent(inout) :: rfile
   real(wp), allocatable :: keep_array(:)
   integer(long), intent(inout) :: size_array
   integer(long), intent(in) :: loc    
   integer(short), intent(in) :: length
   real(wp), intent(in) :: write_array(:)
   integer(short) :: flag   
   type (of01_rdata), intent(inout) :: data
   integer(short), intent(in) :: lp      
   integer(long), intent(in) :: maxstore
   integer(long), intent(inout) :: used 
   integer(long), optional, intent(in) :: inactive 
   flag = GALAHAD_unavailable_option
  end subroutine MA77_write_real

  subroutine MA77_write_integer(ifile,size_array,keep_array,loc,length, &
             write_array,flag,data,lp,maxstore,used)
    USE GALAHAD_SYMBOLS
   integer(short), intent(inout) :: ifile
   integer(short), allocatable :: keep_array(:)
   integer(long), intent(inout) :: size_array
   integer(long), intent(in) :: loc
   integer(short), intent(in) :: length  
   integer(short), intent(in) :: write_array(:)
   integer(short) :: flag   
   type (of01_idata), intent(inout) :: data
   integer(short), intent(in) :: lp      
   integer(long), intent(in) :: maxstore 
   integer(long), intent(inout) :: used  
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
    info%minstore = 0_long
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
    info%storage(1:4) = 0_long
    info%tree_nodes = 0
    info%unit_restart = -1
    info%unused = 0
    info%u = zero
  end subroutine MA77_unavailable

end module hsl_MA77_single
