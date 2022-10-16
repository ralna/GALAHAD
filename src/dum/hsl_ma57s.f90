! THIS VERSION: GALAHAD 4.0 - 2022-01-07 AT 12:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 5 7   M O D U L E  -*-*-*-

module hsl_ma57_single
   use hsl_zd11_single
   USE GALAHAD_SYMBOLS
   implicit none
   integer, parameter, private :: wp = kind(0.0)

   type ma57_factors
     private
      integer, allocatable :: keep(:)
      integer, allocatable :: iw(:)
      real(wp), allocatable :: val(:)
      integer :: n        ! Matrix order
      integer :: nrltot   ! Size for val without compression
      integer :: nirtot   ! Size for iw without compression
      integer :: nrlnec   ! Size for val with compression
      integer :: nirnec   ! Size for iw with compression
      integer :: pivoting ! Set to pivoting option used in factorize
      integer :: scaling  ! Set to scaling option used in factorize
      integer :: static   ! Set to indicate if static pivots chosen
      integer :: rank     ! Set to indicate the rank of the factorization
      integer :: nirbdu   ! Set to number of integers in factors
      integer :: nebdu    ! Set to number of entries in factors
   end type ma57_factors

   type ma57_control
      real(wp) :: multiplier ! Factor by which arrays sizes are to be
                        ! increased if they are too small
      real(wp) :: reduce ! if previously allocated internal workspace arrays
                        !  are greater than reduce times the currently
                        !  required sizes, they are reset to current requirments
      real(wp) :: u     ! Pivot threshold
      real(wp) :: static_tolerance ! used for setting static pivot level
      real(wp) :: static_level ! used for switch to static
      real(wp) :: tolerance ! anything less than this is considered zero
      real(wp) :: convergence ! used to monitor convergence in iterative
                              ! refinement
      real(wp) :: consist ! used in test for consistency when using
                          ! fredholm alternative
      integer :: lp     ! Unit for error messages
      integer :: wp     ! Unit for warning messages
      integer :: mp     ! Unit for monitor output
      integer :: sp     ! Unit for statistical output
      integer :: ldiag  ! Controls level of diagnostic output
      integer :: nemin  ! Minimum number of eliminations in a step
      integer :: factorblocking ! Level 3 blocking in factorize
      integer :: solveblocking ! Level 2 and 3 blocking in solve
      integer :: la     ! Initial size for real array for the factors.
                        ! If less than nrlnec, default size used.
      integer :: liw    ! Initial size for integer array for the factors.
                        ! If less than nirnec, default size used.
      integer :: maxla  ! Max. size for real array for the factors.
      integer :: maxliw ! Max. size for integer array for the factors.
      integer :: pivoting  ! Controls pivoting:
                 !  1  Numerical pivoting will be performed.
                 !  2  No pivoting will be performed and an error exit will
                 !     occur immediately a pivot sign change is detected.
                 !  3  No pivoting will be performed and an error exit will
                 !     occur if a zero pivot is detected.
                 !  4  No pivoting is performed but pivots are changed to
                 !     all be positive.
      integer :: thresh ! Controls threshold for detecting full rows in analyse
                 !     Registered as percentage of N
                 ! 100 Only fully dense rows detected (default)
      integer :: ordering  ! Controls ordering:
                 !  Note that this is overridden by using optional parameter
                 !  perm in analyse call with equivalent action to 1.
                 !  0  AMD using MC47
                 !  1  User defined
                 !  2  AMD using MC50
                 !  3  Min deg as in MA57
                 !  4  Metis_nodend ordering
                 ! >4  Presently equivalent to 0 but may chnage
      integer :: scaling  ! Controls scaling:
                 !  0  No scaling
                 ! >0  Scaling using MC64 but may change for > 1
      integer :: rank_deficient  ! Controls handling rank deficiency:
                 !  0  No control
                 ! >0  Small entries removed during factorization

   end type ma57_control

   type ma57_ainfo
      real(wp) :: opsa = -1.0_wp ! Anticipated # operations in assembly
      real(wp) :: opse = -1.0_wp ! Anticipated # operations in elimination
      integer :: flag = 0      ! Flags success or failure case
      integer :: more = 0      ! More information on failure
      integer :: nsteps = -1   ! Number of elimination steps
      integer :: nrltot = -1   ! Size for a without compression
      integer :: nirtot = -1   ! Size for iw without compression
      integer :: nrlnec = -1   ! Size for a with compression
      integer :: nirnec = -1   ! Size for iw with compression
      integer :: nrladu = -1   ! Number of reals to hold factors
      integer :: niradu = -1   ! Number of integers to hold factors
      integer :: ncmpa = -1    ! Number of compresses
      integer :: ordering = -1 ! Indicates the ordering actually used
      integer :: oor = 0       ! Number of indices out-of-range
      integer :: dup = 0       ! Number of duplicates
      integer :: maxfrt = -1   ! Forecast maximum front size
      integer :: stat = 0      ! STAT value after allocate failure
   end type ma57_ainfo

   type ma57_finfo
      real(wp) :: opsa = -1.0_wp   ! Number of operations in assembly
      real(wp) :: opse = -1.0_wp  ! Number of operations in elimination
      real(wp) :: opsb = -1.0_wp  ! Additional number of operations for BLAS
      real(wp) :: maxchange = -1.0_wp! bigest pivot modification when pivoting=4
      real(wp) :: smin = -1.0_wp   ! Minimum scaling factor
      real(wp) :: smax = -1.0_wp   ! Maximum scaling factor
      integer :: flag = 0     ! Flags success or failure case
      integer :: more = 0     ! More information on failure
      integer :: maxfrt = -1  ! Largest front size
      integer :: nebdu = -1   ! Number of entries in factors
      integer :: nrlbdu = -1  ! Number of reals that hold factors
      integer :: nirbdu = -1  ! Number of integers that hold factors
      integer :: nrltot = -1  ! Size for a without compression
      integer :: nirtot = -1  ! Size for iw without compression
      integer :: nrlnec = -1  ! Size for a with compression
      integer :: nirnec = -1  ! Size for iw with compression
      integer :: ncmpbr = -1  ! Number of compresses of real data
      integer :: ncmpbi = -1  ! Number of compresses of integer data
      integer :: ntwo = -1    ! Number of 2x2 pivots
      integer :: neig = -1    ! Number of negative eigenvalues
      integer :: delay = -1   ! Number of delayed pivots (total)
      integer :: signc = -1   ! Number of pivot sign changes (pivoting=3)
      integer :: static = -1  ! Number of static pivots chosen
      integer :: modstep = -1 ! First pivot modification when pivoting=4
      integer :: rank = -1    ! Rank of original factorization
      integer :: stat  = 0    ! STAT value after allocate failure
   end type ma57_finfo

   type ma57_sinfo
      real(wp) :: cond = -1.0_wp  ! Condition # of matrix (category 1 equations)
      real(wp) :: cond2 = -1.0_wp ! Condition # of matrix (category 2 equations)
      real(wp) :: berr = -1.0_wp  ! Condition # of matrix (category 1 equations)
      real(wp) :: berr2 = -1.0_wp ! Condition # of matrix (category 2 equations)
      real(wp) :: error = -1.0_wp ! Estimate of forward error using above data
      integer :: flag = 0  ! Flags success or failure case
      integer :: stat = 0  ! STAT value after allocate failure
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
!     module procedure ma57_get_n_single
!  end interface ma57_get_n__

contains

   subroutine ma57_initialize(factors,control)
      type(ma57_factors), intent(out), optional :: factors
      type(ma57_control), intent(out), optional :: control
      integer icntl(20),stat
      real(wp) cntl(5)

!  Dummy subroutine available with GALAHAD

      IF ( present( factors ) ) factors%n = -1
      IF ( present( control ) ) control%lp = -1
    end subroutine ma57_initialize

   subroutine ma57_analyse(matrix,factors,control,ainfo,perm)
      type(zd11_type), intent(in) :: matrix
      type(ma57_factors), intent(inout) :: factors
      type(ma57_control), intent(in) :: control
      type(ma57_ainfo), intent(out) :: ainfo
      integer, intent(in), optional :: perm(matrix%n) ! Pivotal sequence

      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
           "( ' We regret that the solution options that you have ', /,        &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy ', /, &
     &     ' subroutine MA57_analyse with its HSL namesake ', /,               &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      ainfo%flag = GALAHAD_unavailable_option
      ainfo%opsa = -1.0_wp
      ainfo%opse = -1.0_wp
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
      finfo%opsa = -1.0_wp
      finfo%opse = -1.0_wp
      finfo%opsb = -1.0_wp
      finfo%maxchange = -1.0_wp
      finfo%smin = -1.0_wp
      finfo%smax = -1.0_wp
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
      real(wp), intent(inout) :: x(:,:)
      type(ma57_control), intent(in) :: control
      type(ma57_sinfo), intent(out) :: sinfo
      real(wp), optional, intent(in) :: rhs(:,:)
      integer, optional, intent(in) :: iter
      integer, optional, intent(in) :: cond
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
      sinfo%cond = -1.0_wp
      sinfo%cond2 = -1.0_wp
      sinfo%berr = -1.0_wp
      sinfo%berr2 = -1.0_wp
      sinfo%error = -1.0_wp
   end subroutine ma57_solve2

   subroutine ma57_solve1(matrix,factors,x,control,sinfo,rhs,iter,cond)
      type(zd11_type), intent(in) :: matrix
      type(ma57_factors), intent(in) :: factors
      real(wp), intent(inout) :: x(:)
      type(ma57_control), intent(in) :: control
      type(ma57_sinfo), intent(out) :: sinfo
      real(wp), optional, intent(in) :: rhs(:)
      integer, optional, intent(in) :: iter
      integer, optional, intent(in) :: cond
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
        "( ' We regret that the solution options that you have ', /,           &
     &     ' chosen are not all freely available with GALAHAD.', /,            &
     &     ' If you have HSL (formerly the Harwell Subroutine', /,             &
     &     ' Library), this option may be enabled by replacing the dummy', /,  &
     &     ' subroutine MA57_solve with its HSL namesake ', /,                 &
     &     ' and dependencies. See ', /,                                       &
     &     '   $GALAHAD/src/makedefs/packages for details.' )" )
      sinfo%flag = GALAHAD_unavailable_option
      sinfo%stat = 0
      sinfo%cond = -1.0_wp
      sinfo%cond2 = -1.0_wp
      sinfo%berr = -1.0_wp
      sinfo%berr2 = -1.0_wp
      sinfo%error = -1.0_wp
   end subroutine ma57_solve1

   subroutine ma57_fredholm_alternative(factors,control,x,fredx,sinfo)
      type(ma57_factors), intent(in) :: factors
      type(ma57_control), intent(in) :: control
      real(wp), intent(inout) :: x(factors%n)
      real(wp), intent(out) :: fredx(factors%n)
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
      integer, intent(out) :: info
      info = GALAHAD_unavailable_option
   end subroutine ma57_finalize

   subroutine ma57_enquire(factors,perm,pivots,d,perturbation,scaling)
      type(ma57_factors), intent(in) :: factors
      integer, intent(out), optional :: perm(factors%n),pivots(factors%n)
      real(wp), intent(out), optional :: d(2,factors%n)
      real(wp), intent(out), optional :: perturbation(factors%n)
      real(wp), intent(out), optional :: scaling(factors%n)
   end subroutine ma57_enquire

   subroutine ma57_alter_d(factors,d,info)
      type(ma57_factors), intent(inout) :: factors
      real(wp), intent(in) :: d(2,factors%n)
      integer, intent(out) :: info
      info = GALAHAD_unavailable_option
   end subroutine ma57_alter_d

   subroutine ma57_part_solve2(factors,control,part,x,info)
      type(ma57_factors), intent(in) :: factors
      type(ma57_control), intent(in) :: control
      character, intent(in) :: part
      real(wp), intent(inout) :: x(:,:)
      integer, intent(out) :: info
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
      real(wp), intent(inout) :: x(:)
      integer, intent(out) :: info
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
      integer, intent(in) :: nzrhs
      integer, intent(in) :: irhs(nzrhs)
      integer, intent(out) :: nzsoln,isoln(*)
      real(wp), intent(inout) :: rhs(factors%n)
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
      real(wp), intent(in) :: x(:)
      real(wp), intent(out) :: y(:)
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
      real(wp), intent(out) :: lvals(factors%nebdu),dvals(2*factors%n),        &
                               scale(factors%n)
      integer, intent(out) :: nzl,nzd,iptrl(factors%n+1),lrows(factors%nebdu), &
                              iptrd(factors%n+1),drows(2*factors%n),           &
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

end module hsl_ma57_single
