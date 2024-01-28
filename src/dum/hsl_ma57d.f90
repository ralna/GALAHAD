! THIS VERSION: GALAHAD 4.3 - 2024-01-13 AT 15:45 GMT.

#include "galahad_hsl.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 5 7   M O D U L E  -*-*-*-

module hsl_ma57_double
   use hsl_zd11_double
   use GALAHAD_KINDS
   USE GALAHAD_SYMBOLS
   implicit none

   type ma57_factors
     private
      integer(ip_),  allocatable :: keep(:)
      integer(ip_),  allocatable :: iw(:)
      real(dp_), allocatable :: val(:)
      integer(ip_) :: n        ! Matrix order
      integer(ip_) :: nrltot   ! Size for val without compression
      integer(ip_) :: nirtot   ! Size for iw without compression
      integer(ip_) :: nrlnec   ! Size for val with compression
      integer(ip_) :: nirnec   ! Size for iw with compression
      integer(ip_) :: pivoting ! Set to pivoting option used in factorize
      integer(ip_) :: scaling  ! Set to scaling option used in factorize
      integer(ip_) :: static   ! Set to indicate if static pivots chosen
      integer(ip_) :: rank     ! Set to indicate the rank of the factorization
      integer(ip_) :: nirbdu   ! Set to number of integers in factors
      integer(ip_) :: nebdu    ! Set to number of entries in factors
   end type ma57_factors

   type ma57_control
      real(dp_) :: multiplier ! Factor by which arrays sizes are to be
                        ! increased if they are too small
      real(dp_) :: reduce ! if previously allocated internal workspace arrays
                        !  are greater than reduce times the currently
                        !  required sizes, they are reset to current requirments
      real(dp_) :: u     ! Pivot threshold
      real(dp_) :: static_tolerance ! used for setting static pivot level
      real(dp_) :: static_level ! used for switch to static
      real(dp_) :: tolerance ! anything less than this is considered zero
      real(dp_) :: convergence ! used to monitor convergence in iterative
                              ! refinement
      real(dp_) :: consist ! used in test for consistency when using
                          ! fredholm alternative
      integer(ip_) :: lp     ! Unit for error messages
      integer(ip_) :: wp     ! Unit for warning messages
      integer(ip_) :: mp     ! Unit for monitor output
      integer(ip_) :: sp     ! Unit for statistical output
      integer(ip_) :: ldiag  ! Controls level of diagnostic output
      integer(ip_) :: nemin  ! Minimum number of eliminations in a step
      integer(ip_) :: factorblocking ! Level 3 blocking in factorize
      integer(ip_) :: solveblocking ! Level 2 and 3 blocking in solve
      integer(ip_) :: la     ! Initial size for real array for the factors.
                        ! If less than nrlnec, default size used.
      integer(ip_) :: liw    ! Initial size for integer array for the factors.
                        ! If less than nirnec, default size used.
      integer(ip_) :: maxla  ! Max. size for real array for the factors.
      integer(ip_) :: maxliw ! Max. size for integer array for the factors.
      integer(ip_) :: pivoting  ! Controls pivoting:
                 !  1  Numerical pivoting will be performed.
                 !  2  No pivoting will be performed and an error exit will
                 !     occur immediately a pivot sign change is detected.
                 !  3  No pivoting will be performed and an error exit will
                 !     occur if a zero pivot is detected.
                 !  4  No pivoting is performed but pivots are changed to
                 !     all be positive.
      integer(ip_) :: thresh ! threshold for detecting full rows in analyse
                 !     Registered as percentage of N
                 ! 100 Only fully dense rows detected (default)
      integer(ip_) :: ordering  ! Controls ordering:
                 !  Note that this is overridden by using optional parameter
                 !  perm in analyse call with equivalent action to 1.
                 !  0  AMD using MC47
                 !  1  User defined
                 !  2  AMD using MC50
                 !  3  Min deg as in MA57
                 !  4  Metis_nodend ordering
                 ! >4  Presently equivalent to 0 but may chnage
      integer(ip_) :: scaling  ! Controls scaling:
                 !  0  No scaling
                 ! >0  Scaling using MC64 but may change for > 1
      integer(ip_) :: rank_deficient  ! Controls handling rank deficiency:
                 !  0  No control
                 ! >0  Small entries removed during factorization

   end type ma57_control

   type ma57_ainfo
      real(dp_) :: opsa = -1.0_dp_ ! Anticipated # operations in assembly
      real(dp_) :: opse = -1.0_dp_ ! Anticipated # operations in elimination
      integer(ip_) :: flag = 0      ! Flags success or failure case
      integer(ip_) :: more = 0      ! More information on failure
      integer(ip_) :: nsteps = -1   ! Number of elimination steps
      integer(ip_) :: nrltot = -1   ! Size for a without compression
      integer(ip_) :: nirtot = -1   ! Size for iw without compression
      integer(ip_) :: nrlnec = -1   ! Size for a with compression
      integer(ip_) :: nirnec = -1   ! Size for iw with compression
      integer(ip_) :: nrladu = -1   ! Number of reals to hold factors
      integer(ip_) :: niradu = -1   ! Number of integers to hold factors
      integer(ip_) :: ncmpa = -1    ! Number of compresses
      integer(ip_) :: ordering = -1 ! Indicates the ordering actually used
      integer(ip_) :: oor = 0       ! Number of indices out-of-range
      integer(ip_) :: dup = 0       ! Number of duplicates
      integer(ip_) :: maxfrt = -1   ! Forecast maximum front size
      integer(ip_) :: stat = 0      ! STAT value after allocate failure
   end type ma57_ainfo

   type ma57_finfo
      real(dp_) :: opsa = -1.0_dp_   ! Number of operations in assembly
      real(dp_) :: opse = -1.0_dp_  ! Number of operations in elimination
      real(dp_) :: opsb = -1.0_dp_  ! Additional number of operations for BLAS
      real(dp_) :: maxchange = -1.0_dp_  ! Largest pivot mod when pivoting=4
      real(dp_) :: smin = -1.0_dp_   ! Minimum scaling factor
      real(dp_) :: smax = -1.0_dp_   ! Maximum scaling factor
      integer(ip_) :: flag = 0     ! Flags success or failure case
      integer(ip_) :: more = 0     ! More information on failure
      integer(ip_) :: maxfrt = -1  ! Largest front size
      integer(ip_) :: nebdu = -1   ! Number of entries in factors
      integer(ip_) :: nrlbdu = -1  ! Number of reals that hold factors
      integer(ip_) :: nirbdu = -1  ! Number of integers that hold factors
      integer(ip_) :: nrltot = -1  ! Size for a without compression
      integer(ip_) :: nirtot = -1  ! Size for iw without compression
      integer(ip_) :: nrlnec = -1  ! Size for a with compression
      integer(ip_) :: nirnec = -1  ! Size for iw with compression
      integer(ip_) :: ncmpbr = -1  ! Number of compresses of real data
      integer(ip_) :: ncmpbi = -1  ! Number of compresses of integer data
      integer(ip_) :: ntwo = -1    ! Number of 2x2 pivots
      integer(ip_) :: neig = -1    ! Number of negative eigenvalues
      integer(ip_) :: delay = -1   ! Number of delayed pivots (total)
      integer(ip_) :: signc = -1   ! Number of pivot sign changes (pivoting=3)
      integer(ip_) :: static = -1  ! Number of static pivots chosen
      integer(ip_) :: modstep = -1 ! First pivot modification when pivoting=4
      integer(ip_) :: rank = -1    ! Rank of original factorization
      integer(ip_) :: stat  = 0    ! STAT value after allocate failure
   end type ma57_finfo

   type ma57_sinfo
      real(dp_) :: cond = -1.0_dp_  ! Condition # of matrix (cat 1 equations)
      real(dp_) :: cond2 = -1.0_dp_ ! Condition # of matrix (cat 2 equations)
      real(dp_) :: berr = -1.0_dp_  ! Condition # of matrix (cat 1 equations)
      real(dp_) :: berr2 = -1.0_dp_ ! Condition # of matrix (cat 2 equations)
      real(dp_) :: error = -1.0_dp_ ! Estimate of forward error using above data
      integer(ip_) :: flag = 0  ! Flags success or failure case
      integer(ip_) :: stat = 0  ! STAT value after allocate failure
   end type ma57_sinfo
   interface ma57_solve
! ma57_solve1 for 1 rhs  and ma57_solve2 for more than 1.
      module procedure ma57_solve1,ma57_solve2
   end interface

   interface ma57_part_solve
! ma57_part_solve1 for 1 rhs  and ma57_part_solve2 for more than 1.
      module procedure ma57_part_solve1,ma57_part_solve2
   end interface

!  interface ma57_get_n__
!     module procedure ma57_get_n_double
!  end interface ma57_get_n__

contains

   subroutine ma57_initialize(factors,control)
      type(ma57_factors), intent(out), optional :: factors
      type(ma57_control), intent(out), optional :: control

!  Dummy subroutine available with GALAHAD

      IF ( present( factors ) ) factors%n = -1
      IF ( present( control ) ) control%lp = -1
    end subroutine ma57_initialize

   subroutine ma57_analyse(matrix,factors,control,ainfo,perm)
      type(zd11_type), intent(in) :: matrix
      type(ma57_factors), intent(inout) :: factors
      type(ma57_control), intent(in) :: control
      type(ma57_ainfo), intent(out) :: ainfo
      integer(ip_),  intent(in), optional :: perm(matrix%n) ! Pivotal sequence

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
           "( ' We regret that the solution options that you have ', /,        &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_analyse with its HSL namesake ', /,               &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      ainfo%flag = GALAHAD_unavailable_option
      ainfo%opsa = -1.0_dp_
      ainfo%opse = -1.0_dp_
      ainfo%more = 0
      ainfo%nsteps = -1
      ainfo%nrltot = -1
      ainfo%nirtot = -1
      ainfo%nrlnec = -1
      ainfo%nirnec = -1
      ainfo%nrladu = -1
      ainfo%niradu = -1
      ainfo%ncmpa = -1
      ainfo%oor = 0
      ainfo%dup = 0
      ainfo%maxfrt = -1
      ainfo%stat = 0

   end subroutine ma57_analyse

   subroutine ma57_factorize(matrix,factors,control,finfo)
      type(zd11_type), intent(in) :: matrix
      type(ma57_factors), intent(inout) :: factors
      type(ma57_control), intent(in) :: control
      type(ma57_finfo), intent(out) :: finfo
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
          "( ' We regret that the solution options that you have ', /,         &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_factorze with its HSL namesake ', /,              &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      finfo%flag = GALAHAD_unavailable_option
      finfo%opsa = -1.0_dp_
      finfo%opse = -1.0_dp_
      finfo%opsb = -1.0_dp_
      finfo%maxchange = -1.0_dp_
      finfo%smin = -1.0_dp_
      finfo%smax = -1.0_dp_
      finfo%more = 0
      finfo%maxfrt = -1
      finfo%nebdu = -1
      finfo%nrlbdu = -1
      finfo%nirbdu = -1
      finfo%nrltot = -1
      finfo%nirtot = -1
      finfo%nrlnec = -1
      finfo%nirnec = -1
      finfo%ncmpbr = -1
      finfo%ncmpbi = -1
      finfo%ntwo = -1
      finfo%neig = -1
      finfo%delay = -1
      finfo%signc = -1
      finfo%static = -1
      finfo%modstep = -1
      finfo%rank = -1
      finfo%stat  = 0

   end subroutine ma57_factorize

   subroutine ma57_solve2(matrix,factors,x,control,sinfo,rhs,iter,cond)
      type(zd11_type), intent(in) :: matrix
      type(ma57_factors), intent(in) :: factors
      real(dp_), intent(inout) :: x(:,:)
      type(ma57_control), intent(in) :: control
      type(ma57_sinfo), intent(out) :: sinfo
      real(dp_), optional, intent(in) :: rhs(:,:)
      integer(ip_),  optional, intent(in) :: iter
      integer(ip_),  optional, intent(in) :: cond
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_solve with its HSL namesake ', /,                 &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      sinfo%flag = GALAHAD_unavailable_option
      sinfo%stat = 0
      sinfo%cond = -1.0_dp_
      sinfo%cond2 = -1.0_dp_
      sinfo%berr = -1.0_dp_
      sinfo%berr2 = -1.0_dp_
      sinfo%error = -1.0_dp_
   end subroutine ma57_solve2

   subroutine ma57_solve1(matrix,factors,x,control,sinfo,rhs,iter,cond)
      type(zd11_type), intent(in) :: matrix
      type(ma57_factors), intent(in) :: factors
      real(dp_), intent(inout) :: x(:)
      type(ma57_control), intent(in) :: control
      type(ma57_sinfo), intent(out) :: sinfo
      real(dp_), optional, intent(in) :: rhs(:)
      integer(ip_),  optional, intent(in) :: iter
      integer(ip_),  optional, intent(in) :: cond
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_solve with its HSL namesake ', /,                 &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      sinfo%flag = GALAHAD_unavailable_option
      sinfo%stat = 0
      sinfo%cond = -1.0_dp_
      sinfo%cond2 = -1.0_dp_
      sinfo%berr = -1.0_dp_
      sinfo%berr2 = -1.0_dp_
      sinfo%error = -1.0_dp_
   end subroutine ma57_solve1

   subroutine ma57_fredholm_alternative(factors,control,x,fredx,sinfo)
      type(ma57_factors), intent(in) :: factors
      type(ma57_control), intent(in) :: control
      real(dp_), intent(inout) :: x(factors%n)
      real(dp_), intent(out) :: fredx(factors%n)
      type(ma57_sinfo), intent(out) :: sinfo

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy', /,  &
     &     ' subroutine MA57_fredholm_alternative with its HSL namesake ', /,  &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      sinfo%flag = GALAHAD_unavailable_option
   end subroutine ma57_fredholm_alternative

   subroutine ma57_finalize(factors,control,info)
      type(ma57_factors), intent(inout) :: factors
      type(ma57_control), intent(in) :: control
      integer(ip_),  intent(out) :: info
      info = GALAHAD_unavailable_option
   end subroutine ma57_finalize

   subroutine ma57_enquire(factors,perm,pivots,d,perturbation,scaling)
      type(ma57_factors), intent(in) :: factors
      integer(ip_),  intent(out), optional :: perm(factors%n),pivots(factors%n)
      real(dp_), intent(out), optional :: d(2,factors%n)
      real(dp_), intent(out), optional :: perturbation(factors%n)
      real(dp_), intent(out), optional :: scaling(factors%n)
   end subroutine ma57_enquire

   subroutine ma57_alter_d(factors,d,info)
      type(ma57_factors), intent(inout) :: factors
      real(dp_), intent(in) :: d(2,factors%n)
      integer(ip_),  intent(out) :: info
      info = GALAHAD_unavailable_option
   end subroutine ma57_alter_d

   subroutine ma57_part_solve2(factors,control,part,x,info)
      type(ma57_factors), intent(in) :: factors
      type(ma57_control), intent(in) :: control
      character, intent(in) :: part
      real(dp_), intent(inout) :: x(:,:)
      integer(ip_),  intent(out) :: info
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_solve with its HSL namesake ', /,                 &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      info = GALAHAD_unavailable_option
   end subroutine ma57_part_solve2

   subroutine ma57_part_solve1(factors,control,part,x,info)
      type(ma57_factors), intent(in) :: factors
      type(ma57_control), intent(in) :: control
      character, intent(in) :: part
      real(dp_), intent(inout) :: x(:)
      integer(ip_),  intent(out) :: info
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_solve with its HSL namesake ', /,                 &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      info = GALAHAD_unavailable_option
   end subroutine ma57_part_solve1

   subroutine ma57_sparse_lsolve(factors,control,nzrhs,irhs,nzsoln,isoln,      &
                                 rhs,sinfo)
      type(ma57_factors), intent(in) :: factors
      type(ma57_control), intent(in) :: control
      integer(ip_),  intent(in) :: nzrhs
      integer(ip_),  intent(in) :: irhs(nzrhs)
      integer(ip_),  intent(out) :: nzsoln,isoln(*)
      real(dp_), intent(inout) :: rhs(factors%n)
      type(ma57_sinfo), intent(out) :: sinfo
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_sparse_lsolve with its HSL namesake ', /,         &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      sinfo%flag = GALAHAD_unavailable_option
   end subroutine ma57_sparse_lsolve

   subroutine ma57_lmultiply(factors,control,trans,x,y,sinfo)
      type(ma57_factors), intent(in) :: factors
      type(ma57_control), intent(in) :: control
      character, intent(in) :: trans
      real(dp_), intent(in) :: x(:)
      real(dp_), intent(out) :: y(:)
      type(ma57_sinfo), intent(out) :: sinfo
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_lmultiply with its HSL namesake ', /,             &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      sinfo%flag = GALAHAD_unavailable_option
   end subroutine ma57_lmultiply

   subroutine ma57_get_factors(factors,control,nzl,iptrl,lrows,lvals,          &
                               nzd,iptrd,drows,dvals,perm,invperm,scale,sinfo)
      type(ma57_factors), intent(in) :: factors
      type(ma57_control), intent(in) :: control
      real(dp_), intent(out) :: lvals(factors%nebdu),dvals(2*factors%n),       &
                                scale(factors%n)
      integer(ip_),  intent(out) :: nzl,nzd,iptrl(factors%n+1),                &
                                    lrows(factors%nebdu),                      &
                                    iptrd(factors%n+1),drows(2*factors%n),     &
                                    perm(factors%n),invperm(factors%n)
      type(ma57_sinfo), intent(out) :: sinfo
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_get_factors with its HSL namesake ', /,           &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      sinfo%flag = GALAHAD_unavailable_option
   end subroutine ma57_get_factors

   pure integer function ma57_get_n__(factors)
      type(ma57_factors), intent(in) :: factors
      ma57_get_n__ = factors%n
   end function ma57_get_n__

end module hsl_ma57_double
