! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 15:00 GMT.

#include "hsl_subset.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 7 7   M O D U L E  -*-*-*-

module hsl_ma77_real

   use hsl_kinds_real, only: ip_, long_, lp_, rp_
   use hsl_of01_real, of01_rdata => of01_data
   use hsl_of01_integer, of01_idata => of01_data
#ifdef INTEGER_64
   USE GALAHAD_SYMBOLS_64, ONLY: GALAHAD_unavailable_option
#else
   USE GALAHAD_SYMBOLS, ONLY: GALAHAD_unavailable_option
#endif

  implicit none

  private :: ip_, long_, lp_, rp_
  LOGICAL, PUBLIC, PARAMETER :: ma77_available = .FALSE.

  real (rp_), parameter, private :: one = 1.0_rp_
  real (rp_), parameter, private :: zero = 0.0_rp_
  integer(ip_), parameter, private :: nemin_default = 8

  interface MA77_open
      module procedure MA77_open_real
  end interface

  interface MA77_input_vars
      module procedure MA77_input_vars_real
  end interface

  interface MA77_input_reals
      module procedure MA77_input_reals_real
  end interface

  interface MA77_analyse
      module procedure MA77_analyse_real
  end interface

  interface MA77_factor
      module procedure MA77_factor_real
  end interface

  interface MA77_factor_solve
      module procedure MA77_factor_solve_real
  end interface

  interface MA77_solve
      module procedure MA77_solve_real
  end interface

  interface MA77_solve_fredholm
      module procedure MA77_solve_fredholm_real
  end interface

  interface MA77_resid
      module procedure MA77_resid_real
  end interface

  interface MA77_scale
      module procedure MA77_scale_real
  end interface

  interface MA77_enquire_posdef
    module procedure MA77_enquire_posdef_real
  end interface

  interface MA77_enquire_indef
    module procedure MA77_enquire_indef_real
  end interface

  interface MA77_alter
    module procedure MA77_alter_real
  end interface

  interface MA77_restart
      module procedure MA77_restart_real
  end interface

  interface MA77_lmultiply
      module procedure MA77_lmultiply_real
  end interface

  interface MA77_finalise
      module procedure MA77_finalise_real
  end interface

  type MA77_control

    logical(lp_) :: action = .true.
    integer(ip_) :: bits = 32
    integer(ip_) :: buffer_lpage(2) = 2**12
    integer(ip_) :: buffer_npage(2) = 1600
    real(rp_) :: consist_tol = epsilon(one)
    integer(long_) :: file_size = 2**21
    integer(ip_) :: infnorm = 0
    integer(ip_) :: maxit = 1
    integer(long_) :: maxstore = 0_long_
    real(rp_) :: multiplier = 1.1
    integer(ip_) :: nb54 = 150
    integer(ip_) :: nb64 = 120
    integer(ip_) :: nbi = 40
    integer(ip_) :: nemin = nemin_default
    integer(ip_) :: p = 4
    integer(ip_) :: print_level = 0
    real(rp_) :: small = tiny(one)
    real (rp_) :: static = zero
    integer(long_) :: storage(3) = 0_long_
    integer(long_) :: storage_indef = 0
    real(rp_) :: thresh = 0.5
    integer(ip_) :: unit_diagnostics = 6
    integer(ip_) :: unit_error = 6
    integer(ip_) :: unit_warning = 6
    real (rp_) :: u = 0.01
    real (rp_) :: umin = 0.01
  end type MA77_control

  type MA77_info
    real (rp_) :: detlog = 0.0
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
    real(rp_) :: u = zero
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

    real(rp_), allocatable :: aelt(:)
    real(rp_), allocatable :: arow(:)
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
    real(rp_),allocatable :: rmain(:)
    real(rp_),allocatable :: rwork(:)
    real(rp_),allocatable :: rwdelay(:)
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

  subroutine MA77_open_real(n,filename,keep,control,info,nelt,path)
    integer(ip_), intent (in) :: n
    character (len=*), intent (in) :: filename(4)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info) :: info
    integer(ip_), optional, intent (in) :: nelt
    character (len=*), optional, intent (in) :: path(:)
    call MA77_unavailable( info, control, 'ma77_open' )
  end subroutine MA77_open_real

  subroutine MA77_input_vars_real(index,nvar,list,keep,control,info)
    integer(ip_), intent (in) :: index
    integer(ip_), intent (in) :: nvar
!            in the incoming element/row. Must be >= 0.
    integer(ip_), intent (in) :: list(nvar)
!            incoming element/row.
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    call MA77_unavailable( info, control, 'ma77_input_vars' )
  end subroutine MA77_input_vars_real

  subroutine MA77_analyse_real(order,keep,control,info)
    type (MA77_keep), intent (inout) :: keep
    integer(ip_), intent (inout), dimension(keep%n) :: order
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    call MA77_unavailable( info, control, 'ma77_analyse' )
  end subroutine MA77_analyse_real

!****************************************************************************

  subroutine MA77_input_reals_real(index,length,reals,keep,control,info)
    integer(ip_), intent (in) :: index
    integer(ip_), intent (in) :: length
    real (rp_), intent (in) :: reals(length)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    call MA77_unavailable( info, control, 'ma77_input_reals' )
  end subroutine MA77_input_reals_real

  subroutine MA77_factor_real(pos_def,keep,control,info,scale)
    logical(lp_) :: pos_def
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_factor' )
  end subroutine MA77_factor_real

  subroutine MA77_factor_solve_real(pos_def,keep,control,info,nrhs, &
     lx,x,scale)
    integer(ip_), parameter :: nb_default = 150
    logical(lp_), intent (in) :: pos_def
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_), intent(in), optional :: scale(:)
    integer(ip_) :: lx
    integer(ip_) :: nrhs
    real (rp_), intent(inout) :: x(lx,nrhs)
    call MA77_unavailable( info, control, 'ma77_factor_solve' )
  end subroutine MA77_factor_solve_real

   subroutine MA77_resid_real(nrhs,lx,x,lresid,resid,keep,control,info,anorm)
    integer(ip_) :: nrhs
    integer(ip_) :: lx
    integer(ip_) :: lresid
    real(rp_), intent(in) :: x(lx,nrhs)
    real(rp_), intent(inout) :: resid(lresid,nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_),optional :: anorm
    call MA77_unavailable( info, control, 'ma77_resid' )
  end subroutine MA77_resid_real

  subroutine MA77_solve_real(nrhs,lx,x,keep,control,info,scale,job)
    integer(ip_), intent (in) :: nrhs
    integer(ip_), intent (in) :: lx
    real (rp_), intent (inout) :: x(lx,nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_), intent(in), optional :: scale(:)
    integer(ip_), optional, intent (in) :: job
    call MA77_unavailable( info, control, 'ma77_solve' )
  end subroutine MA77_solve_real

  subroutine MA77_solve_fredholm_real( nrhs, flag_out, lx, x,                &
                                         keep, control, info, scale )
    integer(ip_), intent (in) :: nrhs
    logical(lp_), intent(out) :: flag_out(nrhs)
    integer(ip_), intent (in) :: lx
    real (rp_), intent (inout) :: x(lx,2*nrhs)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_solve_fredholm' )
  end subroutine MA77_solve_fredholm_real

  subroutine MA77_lmultiply_real(trans,k,lx,x,ly,y,keep,control,info,scale)
    logical(lp_), intent (in) :: trans
    integer(ip_), intent (in) :: k
    integer(ip_), intent (in) :: lx, ly
    real (rp_), intent (inout) :: x(lx,k) ! On entry, x must
    real (rp_), intent (out) :: y(ly,k) ! On exit,
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_), intent(in), optional :: scale(:)
    call MA77_unavailable( info, control, 'ma77_lmultiply' )
  end subroutine MA77_lmultiply_real

  subroutine MA77_enquire_posdef_real(d,keep,control,info)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_) :: d(:)
    call MA77_unavailable( info, control, 'ma77_enquire_posdef' )
  end subroutine ma77_enquire_posdef_real

  subroutine MA77_enquire_indef_real(piv_order,d,keep,control,info)
    real(rp_) :: d(:,:)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    integer(ip_) :: piv_order(:)
    call MA77_unavailable( info, control, 'ma77_enquire_indef' )
  end subroutine ma77_enquire_indef_real

  subroutine MA77_alter_real(d,keep,control,info)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (inout) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_), intent (inout) :: d(:,:)
    call MA77_unavailable( info, control, 'ma77_alter' )
  end subroutine ma77_alter_real

  subroutine MA77_scale_real(scale,keep,control,info,anorm)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    real(rp_),optional :: anorm
    real (rp_) :: scale(:)
    call MA77_unavailable( info, control, 'ma77_scale' )
  end subroutine MA77_scale_real

  subroutine MA77_print_iflag(keep,nout,iflag,ie,st,ios)
    integer(ip_), intent (in) :: iflag, nout
    integer(ip_), intent (in), optional :: ie, st, ios
    type (MA77_keep), intent (in) :: keep
  end subroutine MA77_print_iflag

  subroutine MA77_read_integer(ifile,keep_array,loc,length,read_array, &
             flag,data,lp)
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
   integer(ip_), intent(in) :: rfile
   real(rp_), allocatable :: keep_array(:)
   integer(long_), intent(in) :: loc
   integer(ip_), intent(in) :: length
   real(rp_), dimension(:), intent(inout) :: read_array
   integer(ip_) :: flag
   type (of01_rdata), intent(inout) :: data
   integer(ip_), intent(in) :: lp
   integer(ip_), optional, intent (in) :: map(length)
   flag = GALAHAD_unavailable_option
  end subroutine MA77_read_real

  subroutine MA77_read_discard_real(rfile,keep_array,loc,length,read_array, &
             flag,data,lp,map)
   integer(ip_), intent(in) :: rfile
   real(rp_), allocatable :: keep_array(:)
   integer(long_), intent(in) :: loc
   integer(ip_), intent(in) :: length
   real(rp_), dimension(:), intent(inout) :: read_array
   integer(ip_) :: flag
   type (of01_rdata), intent(inout) :: data
   integer(ip_), intent(in) :: lp
   integer(ip_), optional, intent (in) :: map(length)
   flag = GALAHAD_unavailable_option
  end subroutine MA77_read_discard_real

  subroutine MA77_write_real(rfile,size_array,keep_array,loc,length, &
             write_array,flag,data,lp,maxstore,used,inactive)
   integer(ip_), intent(inout) :: rfile
   real(rp_), allocatable :: keep_array(:)
   integer(long_), intent(inout) :: size_array
   integer(long_), intent(in) :: loc
   integer(ip_), intent(in) :: length
   real(rp_), intent(in) :: write_array(:)
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

  subroutine MA77_finalise_real(keep,control,info,restart_file)
    type (MA77_keep), intent (inout) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info) :: info
    character (len=*), optional, intent (in) :: restart_file
    call MA77_unavailable( info, control, 'ma77_finalize' )
  end subroutine MA77_finalise_real

  subroutine MA77_restart_real(restart_file,filename,keep,control,info,path)
    type (MA77_keep) :: keep
    type (MA77_control), intent (in) :: control
    type (MA77_info), intent (inout) :: info
    character (len=*), intent (in) :: restart_file
    character (len=*), optional, intent (in) :: path(:)
    character (len=*), intent (in) :: filename(4)
    call MA77_unavailable( info, control, 'ma77_restart' )
  end subroutine MA77_restart_real

  subroutine MA77_unavailable( info, control, name )
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

end module hsl_ma77_real
