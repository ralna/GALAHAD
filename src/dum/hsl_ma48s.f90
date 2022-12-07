! THIS VERSION: 20/05/2010 AT 16:00:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 4 8   M O D U L E  -*-*-*-

module hsl_ma48_single
   use hsl_zd11_single
   USE GALAHAD_SYMBOLS
   implicit none
   integer, parameter, private :: wp = kind(0.0e0)
   integer, parameter, private :: long = selected_int_kind(18) ! Long integer

   type ma48_factors
     private
      integer, allocatable :: keep(:)
      integer, allocatable :: irn(:)
      integer, allocatable :: jcn(:)
      real(wp), allocatable :: val(:)
      integer :: m       ! Number of rows in matrix
      integer :: n       ! Number of columns in matrix
      integer :: lareq   ! Size for further factorization
      integer :: partial ! Number of columns kept to end for partial
                         ! factorization
      integer :: ndrop   ! Number of entries dropped from data structure
      integer :: first   ! Flag to indicate whether it is first call to
                         ! factorize after analyse (set to 1 in analyse).
   end type ma48_factors

   type ma48_control
      real(wp) :: multiplier ! Factor by which arrays sizes are to be
                        ! increased if they are too small
      real(wp) :: u     ! Pivot threshold
      real(wp) :: switch ! Density for switch to full code
      real(wp) :: drop   ! Drop tolerance
      real(wp) :: tolerance ! anything less than this is considered zero
      real(wp) :: cgce  ! Ratio for required reduction using IR
      integer :: lp     ! Unit for error messages
      integer :: wp     ! Unit for warning messages
      integer :: mp     ! Unit for monitor output
      integer :: ldiag  ! Controls level of diagnostic output
      integer :: btf    ! Minimum block size for BTF ... >=N to avoid
      logical :: struct ! Control to abort if structurally singular
      integer :: maxit ! Maximum number of iterations
      integer :: factor_blocking ! Level 3 blocking in factorize
      integer :: solve_blas ! Switch for using Level 1 or 2 BLAS in solve.
      integer :: pivoting  ! Controls pivoting:
!                 Number of columns searched.  Zero for Markowitz
      logical :: diagonal_pivoting  ! Set to 0 for diagonal pivoting
      integer :: fill_in ! Initially fill_in * ne space allocated for factors
      logical :: switch_mode ! Whether to switch to slow when fast mode
!               given unsuitable pivot sequence.
   end type ma48_control

   type ma48_ainfo
      real(wp) :: ops = 0.0   ! Number of operations in elimination
      integer :: flag =  GALAHAD_unavailable_option ! Flags success or failure
      integer :: more = 0   ! More information on failure
      integer(long) :: lena_analyse  = 0! Size for analysis (main arrays)
      integer(long) :: lenj_analyse  = 0! Size for analysis (integer aux array)
!! For the moment leni_factorize = lena_factorize because of BTF structure
      integer(long) :: lena_factorize  = 0 ! Size for factorize (real array)
      integer(long) :: leni_factorize  = 0 ! Size for factorize (integer array)
      integer :: ncmpa   = 0 ! Number of compresses in analyse
      integer :: rank   = 0  ! Estimated rank
      integer(long) :: drop   = 0  ! Number of entries dropped
      integer :: struc_rank  = 0! Structural rank of matrix
      integer(long) :: oor   = 0   ! Number of indices out-of-range
      integer(long) :: dup   = 0   ! Number of duplicates
      integer :: stat   = 0  ! STAT value after allocate failure
      integer :: lblock  = 0 ! Size largest non-triangular block
      integer :: sblock  = 0 ! Sum of orders of non-triangular blocks
      integer(long) :: tblock  = 0 ! Total entries in all non-triangular blocks
   end type ma48_ainfo

   type ma48_finfo
      real(wp) :: ops  = 0.0  ! Number of operations in elimination
      integer :: flag =  GALAHAD_unavailable_option ! Flags success or failure
      integer :: more   = 0  ! More information on failure
      integer(long) :: size_factor   = 0! Number of words to hold factors
!! For the moment leni_factorize = lena_factorize because of BTF structure
      integer(long) :: lena_factorize  = 0 ! Size for factorize (real array)
      integer(long) :: leni_factorize  = 0 ! Size for factorize (integer array)
      integer(long) :: drop   = 0 ! Number of entries dropped
      integer :: rank   = 0  ! Estimated rank
      integer :: stat   = 0  ! STAT value after allocate failure
   end type ma48_finfo

   type ma48_sinfo
      integer :: flag = GALAHAD_unavailable_option  ! Flags success or failure
      integer :: more = 0   ! More information on failure
      integer :: stat = 0   ! STAT value after allocate failure
   end type ma48_sinfo

contains

   subroutine ma48_initialize(factors,control)
      USE GALAHAD_SYMBOLS
      type(ma48_factors), optional :: factors
      type(ma48_control), optional :: control

!  Dummy subroutine available with GALAHAD

      IF ( present( control ) ) control%lp=-1
    end subroutine ma48_initialize

   subroutine ma48_analyse(matrix,factors,control,ainfo,finfo,perm,lastcol)
      USE GALAHAD_SYMBOLS
      type(zd11_type), Intent(in) :: matrix
      type(ma48_factors), intent(inout) :: factors
      type(ma48_control), intent(in) :: control
      type(ma48_ainfo) :: ainfo
      type(ma48_finfo), optional :: finfo
      integer, intent(in), optional :: perm(matrix%m+matrix%n) ! Input perm
      integer, intent(in), optional :: lastcol(matrix%n) ! Define last cols

      integer, allocatable :: iwork(:)
      integer :: i,job,k,la,lkeep,m,n,ne,stat,icntl(20),info(20)
      real(wp):: rinfo(10),cntl(10)

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
           "( ' We regret that the solution options that you have ', /,        &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA48_analyse with its HSL namesake ', /,               &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )

      ainfo%flag = GALAHAD_unavailable_option

   end subroutine ma48_analyse

   subroutine ma48_factorize(matrix,factors,control,finfo,fast,partial)
      USE GALAHAD_SYMBOLS
      type(zd11_type), intent(in) :: matrix
      type(ma48_factors), intent(inout) :: factors
      type(ma48_control), intent(in) :: control
      type(ma48_finfo) :: finfo
      integer, optional, intent(in) :: fast,partial

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
           "( ' We regret that the solution options that you have ', /,        &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA48_factorize with its HSL namesake ', /,             &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )

      finfo%flag = GALAHAD_unavailable_option

   end subroutine ma48_factorize

   subroutine ma48_solve(matrix,factors,rhs,x,control,sinfo,trans, &
                         resid,error)
      USE GALAHAD_SYMBOLS
      type(zd11_type), intent(in) :: matrix
      type(ma48_factors), intent(in) :: factors
      real(wp), intent(in) :: rhs(:)
      real(wp) :: x(:)
      type(ma48_control), intent(in) :: control
      type(ma48_sinfo) :: sinfo
      integer, optional, intent(in) :: trans
      real(wp), optional :: resid(2)
      real(wp), optional :: error
      integer icntl(20),info(20),job,m,n,stat
      real(wp) cntl(10),err(3)
      logical trans48

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
           "( ' We regret that the solution options that you have ', /,        &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA48_solve with its HSL namesake ', /,               &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )

      sinfo%flag = GALAHAD_unavailable_option

   end subroutine ma48_solve

   subroutine ma48_finalize(factors,control,info)
      USE GALAHAD_SYMBOLS
      type(ma48_factors), intent(inout) :: factors
      type(ma48_control), intent(in) :: control
      integer :: info
      info = GALAHAD_unavailable_option
    end subroutine ma48_finalize

  subroutine ma48_special_rows_and_cols(factors,rank,rows,cols,control,info)
      type(ma48_factors), intent(in) :: factors
      integer:: rank,info
      integer,dimension(factors%m) :: rows
      integer,dimension(factors%n) :: cols
      type(ma48_control), intent(in) :: control
  end subroutine ma48_special_rows_and_cols

  subroutine ma48_get_perm(factors,perm)
      type(ma48_factors), intent(in), optional :: factors
      integer, intent(out) :: perm(:)
  end subroutine ma48_get_perm

  subroutine ma48_determinant(factors,sgndet,logdet,control,info)
      type(ma48_factors), intent(in) :: factors
      integer,intent(out) :: sgndet,info
      real(wp),intent(out) :: logdet
      type(ma48_control), intent(in) :: control
  end subroutine ma48_determinant

end module hsl_ma48_single
