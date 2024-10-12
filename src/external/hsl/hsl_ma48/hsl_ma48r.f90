! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 14:30 GMT.

#include "hsl_subset.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 4 8   M O D U L E  -*-*-*-

module hsl_ma48_real
   use hsl_kinds_real, only: ip_, long_, lp_, rp_
   use hsl_zd11_real
#ifdef INTEGER_64
     USE GALAHAD_SYMBOLS_64, ONLY: GALAHAD_unavailable_option
#else
     USE GALAHAD_SYMBOLS, ONLY: GALAHAD_unavailable_option
#endif
   implicit none

   private
   public :: ma48_factors,ma48_control,ma48_ainfo,ma48_finfo,ma48_sinfo,       &
             ma48_initialize,ma48_analyse,ma48_factorize,ma48_solve,           &
             ma48_finalize, ma48_get_perm,ma48_special_rows_and_cols,          &
             ma48_determinant
   LOGICAL, PUBLIC, PARAMETER :: ma48_available = .FALSE.

   interface ma48_initialize
      module procedure ma48_initialize_real
   end interface

   interface ma48_analyse
      module procedure ma48_analyse_real
   end interface

   interface ma48_factorize
      module procedure ma48_factorize_real
   end interface

   interface ma48_solve
      module procedure ma48_solve_real
   end interface

   interface  ma48_finalize
      module procedure  ma48_finalize_real
   end interface

   interface ma48_get_perm
      module procedure ma48_get_perm_real
   end interface

   interface ma48_special_rows_and_cols
      module procedure ma48_special_rows_and_cols_real
   end interface

   interface ma48_determinant
      module procedure ma48_determinant_real
   end interface

   type ma48_factors
     private
      integer(ip_),  allocatable :: keep(:)
      integer(ip_),  allocatable :: irn(:)
      integer(ip_),  allocatable :: jcn(:)
      real(rp_), allocatable :: val(:)
      integer(ip_) :: m       ! Number of rows in matrix
      integer(ip_) :: n       ! Number of columns in matrix
      integer(ip_) :: lareq   ! Size for further factorization
      integer(ip_) :: partial ! Number of columns kept to end for partial
                         ! factorization
      integer(ip_) :: ndrop   ! Number of entries dropped from data structure
      integer(ip_) :: first   ! Flag to indicate whether it is first call to
                         ! factorize after analyse (set to 1 in analyse).
   end type ma48_factors

   type ma48_control
      real(rp_) :: multiplier ! Factor by which arrays sizes are to be
                        ! increased if they are too small
      real(rp_) :: u     ! Pivot threshold
      real(rp_) :: switch ! Density for switch to full code
      real(rp_) :: drop   ! Drop tolerance
      real(rp_) :: tolerance ! anything less than this is considered zero
      real(rp_) :: cgce  ! Ratio for required reduction using IR
      integer(ip_) :: lp     ! Unit for error messages
      integer(ip_) :: wp     ! Unit for warning messages
      integer(ip_) :: mp     ! Unit for monitor output
      integer(ip_) :: ldiag  ! Controls level of diagnostic output
      integer(ip_) :: btf    ! Minimum block size for BTF ... >=N to avoid
      logical(lp_) :: struct ! Control to abort if structurally singular
      integer(ip_) :: maxit ! Maximum number of iterations
      integer(ip_) :: factor_blocking ! Level 3 blocking in factorize
      integer(ip_) :: solve_blas ! Switch for using Level 1 or 2 BLAS in solve.
      integer(ip_) :: pivoting  ! Controls pivoting:
!                 Number of columns searched.  Zero for Markowitz
      logical(lp_) :: diagonal_pivoting  ! Set to 0 for diagonal pivoting
      integer(ip_) :: fill_in ! Initially fill_in * ne space for factors
      logical(lp_) :: switch_mode ! Whether to switch to slow when fast mode
!               given unsuitable pivot sequence.
   end type ma48_control

   type ma48_ainfo
      real(rp_) :: ops = 0.0   ! Number of operations in elimination
      integer(ip_) :: flag =  GALAHAD_unavailable_option ! Flags success/failure
      integer(ip_) :: more = 0   ! More information on failure
      integer(long_) :: lena_analyse  = 0! Size for analysis (main arrays)
      integer(long_) :: lenj_analyse  = 0! Size for analysis (integer aux array)
!! For the moment leni_factorize = lena_factorize because of BTF structure
      integer(long_) :: lena_factorize  = 0 ! Size for factorize (real array)
      integer(long_) :: leni_factorize  = 0 ! Size for factorize (integer array)
      integer(ip_) :: ncmpa   = 0 ! Number of compresses in analyse
      integer(ip_) :: rank   = 0  ! Estimated rank
      integer(long_) :: drop   = 0  ! Number of entries dropped
      integer(ip_) :: struc_rank  = 0! Structural rank of matrix
      integer(long_) :: oor   = 0   ! Number of indices out-of-range
      integer(long_) :: dup   = 0   ! Number of duplicates
      integer(ip_) :: stat   = 0  ! STAT value after allocate failure
      integer(ip_) :: lblock  = 0 ! Size largest non-triangular block
      integer(ip_) :: sblock  = 0 ! Sum of orders of non-triangular blocks
      integer(long_) :: tblock  = 0 ! Total entries in all non-triangular blocks
   end type ma48_ainfo

   type ma48_finfo
      real(rp_) :: ops  = 0.0  ! Number of operations in elimination
      integer(ip_) :: flag =  GALAHAD_unavailable_option ! Flags success/failure
      integer(ip_) :: more   = 0  ! More information on failure
      integer(long_) :: size_factor   = 0! Number of words to hold factors
!! For the moment leni_factorize = lena_factorize because of BTF structure
      integer(long_) :: lena_factorize  = 0 ! Size for factorize (real array)
      integer(long_) :: leni_factorize  = 0 ! Size for factorize (integer array)
      integer(long_) :: drop   = 0 ! Number of entries dropped
      integer(ip_) :: rank   = 0  ! Estimated rank
      integer(ip_) :: stat   = 0  ! STAT value after allocate failure
   end type ma48_finfo

   type ma48_sinfo
      integer(ip_) :: flag = GALAHAD_unavailable_option  ! Flags success/failure
      integer(ip_) :: more = 0   ! More information on failure
      integer(ip_) :: stat = 0   ! STAT value after allocate failure
   end type ma48_sinfo

contains

   subroutine ma48_initialize_real(factors,control)
      type(ma48_factors), optional :: factors
      type(ma48_control), optional :: control

!  Dummy subroutine available with GALAHAD

      IF ( present( control ) ) control%lp=-1
    end subroutine ma48_initialize_real

   subroutine ma48_analyse_real(matrix,factors,control,ainfo,finfo,          &
                                  perm,lastcol)
      type(zd11_type), Intent(in) :: matrix
      type(ma48_factors), intent(inout) :: factors
      type(ma48_control), intent(in) :: control
      type(ma48_ainfo) :: ainfo
      type(ma48_finfo), optional :: finfo
      integer(ip_),  intent(in), optional :: perm(matrix%m+matrix%n) ! Init perm
      integer(ip_),  intent(in), optional :: lastcol(matrix%n) ! last cols

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
           "( ' We regret that the solution options that you have ', /,        &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA48_analyse with its HSL namesake ', /,               &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )

      ainfo%flag = GALAHAD_unavailable_option

   end subroutine ma48_analyse_real

   subroutine ma48_factorize_real(matrix,factors,control,finfo,fast,partial)
      type(zd11_type), intent(in) :: matrix
      type(ma48_factors), intent(inout) :: factors
      type(ma48_control), intent(in) :: control
      type(ma48_finfo) :: finfo
      integer(ip_),  optional, intent(in) :: fast,partial

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
           "( ' We regret that the solution options that you have ', /,        &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA48_factorize with its HSL namesake ', /,             &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )

      finfo%flag = GALAHAD_unavailable_option

   end subroutine ma48_factorize_real

   subroutine ma48_solve_real(matrix,factors,rhs,x,control,sinfo,trans, &
                         resid,error)
      type(zd11_type), intent(in) :: matrix
      type(ma48_factors), intent(in) :: factors
      real(rp_), intent(in) :: rhs(:)
      real(rp_) :: x(:)
      type(ma48_control), intent(in) :: control
      type(ma48_sinfo) :: sinfo
      integer(ip_),  optional, intent(in) :: trans
      real(rp_), optional :: resid(2)
      real(rp_), optional :: error

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
           "( ' We regret that the solution options that you have ', /,        &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA48_solve with its HSL namesake ', /,                 &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )

      sinfo%flag = GALAHAD_unavailable_option

   end subroutine ma48_solve_real

   subroutine ma48_finalize_real(factors,control,info)
      type(ma48_factors), intent(inout) :: factors
      type(ma48_control), intent(in) :: control
      integer(ip_) :: info
      info = GALAHAD_unavailable_option
    end subroutine ma48_finalize_real

  subroutine ma48_special_rows_and_cols_real(factors,rank,rows,cols,         &
                                               control,info)
      type(ma48_factors), intent(in) :: factors
      integer(ip_) :: rank,info
      integer(ip_), dimension(factors%m) :: rows
      integer(ip_), dimension(factors%n) :: cols
      type(ma48_control), intent(in) :: control
  end subroutine ma48_special_rows_and_cols_real

  subroutine ma48_get_perm_real(factors,perm)
      type(ma48_factors), intent(in), optional :: factors
      integer(ip_),  intent(out) :: perm(:)
  end subroutine ma48_get_perm_real

  subroutine ma48_determinant_real(factors,sgndet,logdet,control,info)
      type(ma48_factors), intent(in) :: factors
      integer(ip_), intent(out) :: sgndet,info
      real(rp_),intent(out) :: logdet
      type(ma48_control), intent(in) :: control
  end subroutine ma48_determinant_real

end module hsl_ma48_real

