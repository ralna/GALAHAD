! THIS VERSION: GALAHAD 2.5 - 03/11/2011 AT 12:00 GMT.

!-*-*-*-*-*-*-*-*- G A L A H A D _ S I L S    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released pre GALAHAD Version 1.0. October 3rd 2000
!   update released with GALAHAD Version 2.0. March 28th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SILS_double

!     ---------------------------------------------
!     |                                           |
!     |  Provide a MA57-style interface for MA27  |
!     |  to allow the solution of                 |
!     |                                           |
!     |    Symmetric Indefinite Linear Systems    |
!     |                                           |
!     |      * Version for threadsafe MA27 *      |
!     |                                           |
!     ---------------------------------------------

     USE GALAHAD_SMT_double

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SILS_initialize, SILS_analyse, SILS_factorize, SILS_solve,      &
               SILS_finalize, SILS_enquire, SILS_alter_d, SILS_part_solve,     &
               SMT_type

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PRIVATE, PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PRIVATE, PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PRIVATE, PARAMETER :: one = 1.0_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

     TYPE, PUBLIC :: SILS_factors
       PRIVATE
!      INTEGER, ALLOCATABLE, DIMENSION( :, : )  :: keep
       INTEGER, ALLOCATABLE, DIMENSION( : )  :: keep
       INTEGER, ALLOCATABLE, DIMENSION( : )  :: iw
       INTEGER, ALLOCATABLE, DIMENSION( : )  :: iw1
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : )  :: val
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : )  :: w ! len maxfrt
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : )  :: r ! length n
       INTEGER :: n = - 1      ! Matrix order
       INTEGER :: nrltot = - 1 ! Size for val without compression
       INTEGER :: nirtot = - 1 ! Size for iw without compression
       INTEGER :: nrlnec = - 1 ! Size for val with compression
       INTEGER :: nirnec = - 1 ! Size for iw with compression
       INTEGER :: nsteps = - 1 ! Number of elimination steps
       INTEGER :: maxfrt = - 1 ! Largest front size
       INTEGER :: latop = - 1  ! Position of final entry of val
       INTEGER :: dim_iw1 = - 1 ! Size of iw1 for solves
       INTEGER :: pivoting = - 1 ! type of pivoting used
       REAL ( KIND = wp ) :: ops = - one
     END TYPE SILS_factors

     TYPE, PUBLIC :: SILS_control
       REAL ( KIND = wp ) :: CNTL( 5 ) ! MA27 internal real controls
       REAL ( KIND = wp ) :: multiplier ! Factor by which arrays sizes are to
                         ! be increased if they are too small
       REAL ( KIND = wp ) :: reduce ! If previously allocated internal
                         ! workspace arrays are greater than reduce times
                         ! the currently required sizes, they are reset to
                         ! current requirments
       REAL ( KIND = wp ) :: u     ! Pivot threshold
       REAL ( KIND = wp ) :: static_tolerance ! used for setting static
                           ! pivot level                                    NEW
       REAL ( KIND = wp ) :: static_level ! used for switch to static       NEW
       REAL ( KIND = wp ) :: tolerance ! Anything less than this is
                         ! considered zero
       REAL ( KIND = wp ) :: convergence = 0.5_wp ! used to monitor convergence
                                                  ! in iterative refinement
       INTEGER :: ICNTL( 30 ) ! MA27 internal integer controls
       INTEGER :: lp     ! Unit for error messages
       INTEGER :: wp     ! Unit for warning messages
       INTEGER :: mp     ! Unit for monitor output                          NEW
       INTEGER :: sp     ! Unit for statistical output                      NEW
       INTEGER :: ldiag  ! Controls level of diagnostic output
       INTEGER :: la     ! Initial size for real array for the factors.
                         ! If less than nrlnec, default size used.
       INTEGER :: liw    ! Initial size for integer array for the factors.
                         ! If less than nirnec, default size used.
       INTEGER :: maxla  ! Max. size for real array for the factors.
       INTEGER :: maxliw ! Max. size for integer array for the factors.
       INTEGER :: pivoting  ! Controls pivoting:
                  !  1  Numerical pivoting will be performed.
                  !  2  No pivoting will be performed and an error exit will
                  !     occur immediately a pivot sign change is detected.
                  !  3  No pivoting will be performed and an error exit will
                  !     occur if a zero pivot is detected.
                  !  4  No pivoting is performed but pivots are changed to
                  !     all be positive.
       INTEGER :: nemin = 1 ! Minimum number of eliminations in a step    UNUSED
       INTEGER :: factorblocking = 16 ! Level 3 blocking in factorize     UNUSED
       INTEGER :: solveblocking = 16 ! Level 2 and 3 blocking in solve    UNUSED
       INTEGER :: thresh ! Controls threshold for detecting full rows in
                  !     analyse, registered as percentage of N
                  ! 100 Only fully dense rows detected (default)            NEW
       INTEGER :: ordering  ! Controls ordering:                            NEW
                 !  0  AMD using MC47
                 !  1  User defined
                 !  2  AMD using MC50
                 !  3  Min deg as in MA57
                 !  4  Metis_nodend ordering
                 !  5  Ordering chosen depending on matrix characteristics.
                 !     At the moment choices are MC50 or Metis_nodend
                 ! >5  Presently equivalent to 5 but may chnage
       INTEGER :: scaling  ! Controls scaling:                              NEW
                 !  0  No scaling
                 ! >0  Scaling using MC64 but may change for > 1
     END TYPE SILS_control

     TYPE, PUBLIC :: SILS_ainfo
       REAL ( KIND = wp ) :: opsa = - 1.0_wp! Anticipated # ops. in assembly NEW
       REAL ( KIND = wp ) :: opse = - 1.0_wp ! Anticipated # ops. in elimin. NEW
       INTEGER :: flag = 0   ! Flags success or failure case
       INTEGER :: more = 0   ! More information on failure                  NEW
       INTEGER :: nsteps = 0 ! Number of elimination steps
       INTEGER :: nrltot = - 1 ! Size for a without compression
       INTEGER :: nirtot = - 1 ! Size for iw without compression
       INTEGER :: nrlnec = - 1 ! Size for a with compression
       INTEGER :: nirnec = - 1 ! Size for iw with compression
       INTEGER :: nrladu = - 1 ! Number of reals to hold factors
       INTEGER :: niradu = - 1 ! Number of integers to hold factors
       INTEGER :: ncmpa  = 0 ! Number of compresses
       INTEGER :: oor = 0    ! Number of indices out-of-range               NEW
       INTEGER :: dup = 0    ! Number of duplicates                         NEW
       INTEGER :: maxfrt = - 1 ! Forecast maximum front size                NEW
       INTEGER :: stat = 0   ! STAT value after allocate failure            NEW
       INTEGER :: faulty = 0 ! OLD
     END TYPE SILS_ainfo

     TYPE, PUBLIC :: SILS_finfo
       REAL ( KIND = wp ) :: opsa = - 1.0_wp ! # operations in assembly     NEW
       REAL ( KIND = wp ) :: opse = - 1.0_wp ! # operations in elimination  NEW
       REAL ( KIND = wp ) :: opsb = - 1.0_wp ! Additional # ops. for BLAS   NEW
       REAL ( KIND = wp ) :: maxchange = - 1.0_wp! Largest pivoting=4 mod.  NEW
       REAL ( KIND = wp ) :: smin = 0.0_wp ! Minimum scaling factor
       REAL ( KIND = wp ) :: smax = 0.0_wp ! Maximum scaling factor
       INTEGER :: flag = 0   ! Flags success or failure case
       INTEGER :: more = 0   ! More information on failure                  NEW
       INTEGER :: maxfrt = - 1 ! Largest front size
       INTEGER :: nebdu  = - 1 ! Number of entries in factors               NEW
       INTEGER :: nrlbdu = - 1 ! Number of reals that hold factors
       INTEGER :: nirbdu = - 1 ! Number of integers that hold factors
       INTEGER :: nrltot = - 1 ! Size for a without compression
       INTEGER :: nirtot = - 1 ! Size for iw without compression
       INTEGER :: nrlnec = - 1 ! Size for a with compression
       INTEGER :: nirnec = - 1 ! Size for iw with compression
       INTEGER :: ncmpbr = - 1 ! Number of compresses of real data
       INTEGER :: ncmpbi = - 1 ! Number of compresses of integer data
       INTEGER :: ntwo = - 1   ! Number of 2x2 pivots
       INTEGER :: neig = - 1   ! Number of negative eigenvalues
       INTEGER :: delay = - 1  ! Number of delayed pivots (total)           NEW
       INTEGER :: signc = - 1  ! Number of pivot sign changes (pivoting=3 ) NEW
       INTEGER :: static = - 1 ! Number of static pivots chosen
       INTEGER :: modstep = - 1 ! First pivot modification when pivoting=4  NEW
       INTEGER :: rank = - 1   ! Rank of original factorization
       INTEGER :: stat = - 1   ! STAT value after allocate failure
       INTEGER :: faulty = - 1 ! OLD
       INTEGER :: step = - 1   ! OLD
     END TYPE SILS_finfo

     TYPE, PUBLIC :: SILS_sinfo
       REAL ( KIND = wp ) :: cond = - 1.0_wp  ! Cond # of matrix (cat 1 eqs)
       REAL ( KIND = wp ) :: cond2 = - 1.0_wp ! Cond # of matrix (cat 2 eqs)
       REAL ( KIND = wp ) :: berr = - 1.0_wp  ! Cond # of matrix (cat 1 eqs)
       REAL ( KIND = wp ) :: berr2 = - 1.0_wp ! Cond # of matrix (cat 2 eqs)
       REAL ( KIND = wp ) :: error = - 1.0_wp  ! Estimate of forward error
       INTEGER :: flag = - 1 ! Flags success or failure case
       INTEGER :: stat = - 1 ! STAT value after allocate failure
     END TYPE SILS_sinfo

!--------------------------------
!   I n t e r f a c e  B l o c k
!--------------------------------

     INTERFACE SILS_solve
       MODULE PROCEDURE SILS_solve, SILS_solve_multiple,                       &
                        SILS_solve_refine, SILS_solve_refine_multiple
     END INTERFACE

     INTERFACE

       SUBROUTINE MA27ID( ICNTL, CNTL )
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: ICNTL( 30 )
       REAL( KIND = wp ), INTENT( OUT ) :: CNTL( 5 )
       END SUBROUTINE MA27ID

       SUBROUTINE MA27AD( n, nz, IRN, ICN, IW, liw, IKEEP, IW1, nsteps,        &
                          iflag, ICNTL, CNTL, INFO, ops )
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( IN ) :: n, nz, liw
       INTEGER, INTENT( OUT ) :: nsteps
       INTEGER, INTENT( INOUT ) :: iflag
       INTEGER, INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( INOUT ), DIMENSION( n, 3 ) :: IKEEP
       INTEGER, INTENT( OUT ), DIMENSION( n, 2 ) :: IW1
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL( KIND = wp ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       REAL( KIND = wp ), INTENT( OUT ) :: ops
       END SUBROUTINE MA27AD

       SUBROUTINE MA27BD( n, nz, IRN, ICN, A, la, IW, liw, IKEEP, nsteps,      &
                          maxfrt, IW1, ICNTL, CNTL, INFO )
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( IN ) :: n, nz, la, liw, nsteps
       INTEGER, INTENT( OUT ) :: maxfrt
       INTEGER, INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( IN ), DIMENSION( n, 3 ) :: IKEEP
       INTEGER, INTENT( OUT ), DIMENSION( n ) :: IW1
       REAL( KIND = wp ), INTENT( INOUT ), DIMENSION( la ) :: A
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL( KIND = wp ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27BD

       SUBROUTINE MA27CD( n, A, la, IW, liw, W, maxfrt, RHS, IW1, nsteps,      &
                          ICNTL, INFO )
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( IN ) :: n, la, liw, maxfrt, nsteps
       INTEGER, INTENT( IN ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( OUT ), DIMENSION( nsteps ) :: IW1
       REAL( KIND = wp ), INTENT( IN ), DIMENSION( la ) :: A
       REAL( KIND = wp ), INTENT( OUT ), DIMENSION( maxfrt ) :: W
       REAL( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: RHS
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       END SUBROUTINE MA27CD

     END INTERFACE

   CONTAINS

!-*-*-*-*-*-   S I L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE SILS_initialize( FACTORS, CONTROL )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for SILS. This routine should be called before
!  first call to SILS_analyse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

     TYPE( SILS_factors ), INTENT( OUT ), OPTIONAL :: FACTORS
     TYPE( SILS_control ), INTENT( OUT ), OPTIONAL :: CONTROL

     IF ( PRESENT( FACTORS ) ) THEN
       FACTORS%n = 0
     END IF

     IF ( present( CONTROL ) ) THEN
       CALL MA27ID( CONTROL%ICNTL, CONTROL%CNTL )
       CONTROL%multiplier = 2.0_wp
       CONTROL%reduce = 2.0_wp
       CONTROL%u = 0.1_wp ; CONTROL%CNTL( 1 ) = CONTROL%u
       CONTROL%static_tolerance = 0.0_wp
       CONTROL%static_level = 0.0_wp
       CONTROL%tolerance = 0.0_wp ; CONTROL%CNTL( 3 ) = CONTROL%tolerance
       CONTROL%lp = 6 ; CONTROL%ICNTL( 1 ) = CONTROL%lp
       CONTROL%wp = 6
       CONTROL%mp = 6 ; CONTROL%ICNTL( 2 ) = CONTROL%mp
       CONTROL%sp = - 1
       CONTROL%ldiag = 0 ; CONTROL%ICNTL( 3 ) = CONTROL%ldiag
       CONTROL%factorblocking = 16
       CONTROL%solveblocking = 16
       CONTROL%la = 0
       CONTROL%liw = 0
       CONTROL%maxla = huge( 0 )
       CONTROL%maxliw = huge( 0 )
       CONTROL%pivoting = 1
       CONTROL%thresh = 50 ; CONTROL%CNTL( 2 ) = CONTROL%thresh / 100.0_wp
       CONTROL%ordering = 3
       CONTROL%scaling = 0
     END IF

     RETURN

!  End of SILS_initialize

     END SUBROUTINE SILS_initialize

!-*-*-*-*-*-*-*-   S I L S _ A N A L Y S E  S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SILS_analyse( MATRIX, FACTORS, CONTROL, AINFO, PERM )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Analyse the sparsity pattern to obtain a good potential ordering
!  for any subsequent factorization
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SMT_type ), INTENT( IN ) :: MATRIX
     TYPE( SILS_factors ), INTENT( INOUT ) :: FACTORS
     TYPE( SILS_control ), INTENT( IN ) :: CONTROL
     TYPE( SILS_ainfo ), INTENT( INOUT ) :: AINFO
     INTEGER, INTENT( IN ), OPTIONAL :: PERM( MATRIX%n ) ! Pivot sequence

!  Local variables

     INTEGER :: i, j, liw, n, ne, stat
     INTEGER :: ICNTL( 30 ), INFO( 20 )
     REAL ( KIND = wp ) :: CNTL( 5 )
     LOGICAL :: not_perm

!  Transfer CONTROL parameters

     CNTL( 1 ) = CONTROL%u
     CNTL( 2 ) = CONTROL%thresh / 100.0_wp
     CNTL( 3 ) = CONTROL%tolerance
     CNTL( 4 : 5 ) = CONTROL%CNTL( 4 : 5 )
     ICNTL( 1 ) = CONTROL%lp
     ICNTL( 2 ) = CONTROL%mp
     ICNTL( 3 ) = CONTROL%ldiag
     ICNTL( 4 : 30 ) = CONTROL%ICNTL( 4 : 30 )

     n = MATRIX%n ; ne = MATRIX%ne ; stat = 0

!  Allocate workspace

     IF ( ALLOCATED( FACTORS%keep ) ) THEN
       IF ( SIZE( FACTORS%keep ) /= 3 * n ) THEN
         DEALLOCATE( FACTORS%keep, STAT = stat )
         IF ( stat /= 0 ) GO TO 100
         ALLOCATE( FACTORS%keep( 3 * n ), STAT = stat )
         IF ( stat /= 0 ) GO TO 100
       END IF
     ELSE
       ALLOCATE( FACTORS%keep( 3 * n ), STAT = stat )
       IF ( stat /= 0 ) GO TO 100
     END IF

     IF ( ALLOCATED( FACTORS%iw1 ) ) THEN
       IF ( SIZE( FACTORS%iw1 ) /= 2 * n ) THEN
         DEALLOCATE( FACTORS%iw1, STAT = stat )
         IF ( stat /= 0 ) GO TO 100
         ALLOCATE( FACTORS%iw1( 2 * n ), STAT = stat )
         IF ( stat /= 0 ) GO TO 100
       END IF
     ELSE
       ALLOCATE( FACTORS%iw1( 2 * n ), STAT = stat )
       IF ( stat /= 0 ) GO TO 100
     END IF

     IF ( present( PERM ) ) THEN

! check that the input perm is indeed a permutation

       not_perm = .FALSE.
       FACTORS%keep( 1 : n ) = 0
       DO i = 1, n
         j = PERM( i )
         IF ( j < 1 .OR. j > n ) THEN
           not_perm = .TRUE.
         ELSE IF ( FACTORS%keep( j ) == 1 ) THEN
           not_perm = .TRUE.
         END IF
         IF ( not_perm ) THEN
           IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                       &
             WRITE( CONTROL%lp, '( A, /, A, I0 )')                             &
              ' Error return from SILS_analyze:',                              &
              ' allocate or deallocate failed with STAT = ', stat
           AINFO%flag = - 9
           AINFO%more = 0
           AINFO%nsteps = - 1
           AINFO%opsa   = 0.0
           AINFO%opse   = 0.0
           AINFO%nrltot = - 1
           AINFO%nirtot = - 1
           AINFO%nrlnec = - 1
           AINFO%nirnec = - 1
           AINFO%nrladu = - 1
           AINFO%niradu = - 1
           AINFO%ncmpa  = - 1
           AINFO%oor = - 1
           AINFO%dup = - 1
           AINFO%maxfrt = - 1
           AINFO%stat = stat
           RETURN
         END IF
         FACTORS%keep( j ) = 1
       END DO
       FACTORS%keep( 1 : n ) = PERM( 1 : n )
       liw = INT( 1.2_wp * REAL( ne + 3 * n + 1, KIND = wp ) )
       AINFO%flag = 1
     ELSE
       liw = INT( 1.2_wp * REAL( 2 * ne + 3 * n + 1, KIND = wp ) )
       AINFO%flag = 0
     END IF

     IF ( ALLOCATED( FACTORS%iw ) ) THEN
       IF ( SIZE( FACTORS%iw ) /= liw ) THEN
         DEALLOCATE( FACTORS%iw, STAT = stat )
         IF ( stat /= 0 ) GO TO 100
         ALLOCATE( FACTORS%iw( liw ), STAT = stat )
         IF ( stat /= 0 ) GO TO 100
       END IF
     ELSE
       ALLOCATE( FACTORS%iw( liw ), STAT = stat )
       IF ( stat /= 0 ) GO TO 100
     END IF

!  Analyse the matrix

     FACTORS%ops = - one
     CALL MA27AD( n, ne, MATRIX%row, MATRIX%col, FACTORS%iw, liw,              &
                  FACTORS%keep, FACTORS%iw1, FACTORS%nsteps, AINFO%flag,       &
                  ICNTL, CNTL, INFO, FACTORS%ops )

!  Record return information

     FACTORS%nrltot = INFO( 3 )
     FACTORS%nirtot = INFO( 4 )
     FACTORS%nrlnec = INFO( 5 )
     FACTORS%nirnec = INFO( 6 )
     FACTORS%n = n

     AINFO%flag = INFO( 1 )
     AINFO%more = INFO( 2 )
     AINFO%nsteps = FACTORS%nsteps
     IF ( AINFO%flag == - 1 .OR. AINFO%flag == - 2 ) THEN
       AINFO%opsa = zero
       AINFO%opse = zero
     ELSE
       AINFO%opsa = FACTORS%ops / 2.0
       AINFO%opse = FACTORS%ops / 2.0
     END IF
     AINFO%nrltot = INFO( 3 )
     AINFO%nirtot = INFO( 4 )
     AINFO%nrlnec = INFO( 5 )
     AINFO%nirnec = INFO( 6 )
     AINFO%nrladu = INFO( 7 )
     AINFO%niradu = INFO( 8 )
     AINFO%ncmpa  = INFO( 11 )
     IF ( AINFO%flag == 1 ) THEN
       AINFO%oor =  AINFO%more
     ELSE
       AINFO%oor = 0
     END IF
     AINFO%dup = 0
     AINFO%maxfrt = - 1
     AINFO%stat = 0

     RETURN

 100 CONTINUE
     IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                             &
       WRITE( CONTROL%lp, '( A, /, A, I0 )')                                   &
        ' Error return from SILS_analyze:',                                    &
        ' allocate or deallocate failed with STAT = ', stat
     AINFO%flag = - 3
     AINFO%more = 0
     AINFO%nsteps = - 1
     AINFO%opsa   = 0.0
     AINFO%opse   = 0.0
     AINFO%nrltot = - 1
     AINFO%nirtot = - 1
     AINFO%nrlnec = - 1
     AINFO%nirnec = - 1
     AINFO%nrladu = - 1
     AINFO%niradu = - 1
     AINFO%ncmpa  = - 1
     AINFO%oor = - 1
     AINFO%dup = - 1
     AINFO%maxfrt = - 1
     AINFO%stat = stat

     RETURN

!  End of SILS_analyse

     END SUBROUTINE SILS_analyse

!-*-*-*-*-*-*-   S I L S _ F A C T O R I Z E  S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE SILS_factorize( MATRIX, FACTORS, CONTROL, FINFO )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Factorize the matrix using the ordering suggested from the analysis
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SMT_type ), INTENT( IN ) :: MATRIX
     TYPE( SILS_factors ), INTENT( INOUT ) :: FACTORS
     TYPE( SILS_control ), INTENT( IN ) :: CONTROL
     TYPE( SILS_finfo ), INTENT( OUT ) :: FINFO

!  Local variables

     INTEGER :: i, kw, nblks, ncols, nrows, stat, la, liw, block, la_extra
     INTEGER :: ICNTL( 30 ), INFO( 20 )
     REAL ( KIND = wp ) :: CNTL( 5 )
     INTEGER, ALLOCATABLE :: flag( : ) ! Workarray for completing the
                                       ! permutation in the rank-deficient case
     INTEGER :: FACTORS_iw1( MATRIX%n )

     IF ( FACTORS%n /= MATRIX%n ) THEN
       IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                           &
         WRITE( CONTROL%lp, '( A, I7, A, I7 )')                                &
           ' Error return from SILS_FACTORIZE: MATRIX%n and the value',        &
           MATRIX%n, ' instead of', FACTORS%n
       FINFO%flag = - 1
       FINFO%more = FACTORS%n
     END IF

!  Transfer CONTROL parameters

     SELECT CASE ( CONTROL%pivoting )
       CASE default ; CNTL( 1 ) = CONTROL%u
       CASE ( 2 ) ; CNTL( 1 ) = - CONTROL%u
       CASE ( 3 ) ; CNTL( 1 ) = 0.0_wp
     END SELECT
     CNTL( 2 ) = CONTROL%thresh / 100.0_wp
     CNTL( 3 ) = CONTROL%tolerance
     CNTL( 4 : 5 ) = CONTROL%CNTL( 4 : 5 )
     ICNTL( 1 ) = CONTROL%lp
     ICNTL( 2 ) = CONTROL%mp
     ICNTL( 3 ) = CONTROL%ldiag
     ICNTL( 4 : 30 ) = CONTROL%ICNTL( 4 : 30 )
     FACTORS%pivoting = CONTROL%pivoting

     stat = 0

!  Allocate workspace

     IF ( CONTROL%pivoting == 4 ) THEN
       la_extra = MATRIX%n
     ELSE
       la_extra = 0
     END IF
     la = CONTROL%la
     IF ( la < FACTORS%nrlnec ) THEN
       la = INT( CONTROL%reduce * REAL( FACTORS%nrltot, KIND = wp ) )
       IF ( ALLOCATED( FACTORS%val ) )                                         &
         la = MIN( SIZE( FACTORS%val ) - la_extra, la )
       IF ( la < FACTORS%nrlnec ) la = FACTORS%nrltot
     END IF

     IF ( ALLOCATED( FACTORS%val ) ) THEN
       IF ( la + la_extra /= SIZE( FACTORS%val ) ) THEN
         DEALLOCATE( FACTORS%val, STAT = stat )
         IF ( stat /= 0 ) GO TO 100
         ALLOCATE( FACTORS%val( la + la_extra ), STAT = stat )
         IF ( stat /= 0 ) GO TO 100
       END IF
     ELSE
       ALLOCATE( FACTORS%val( la + la_extra ), STAT = stat )
       IF ( stat /= 0 ) GO TO 100
     END IF

     liw = CONTROL%liw
     IF ( liw < FACTORS%nirnec ) THEN
       liw = INT( CONTROL%reduce * REAL( FACTORS%nirtot, KIND = wp ) )
       IF ( ALLOCATED( FACTORS%iw ) )                                          &
         liw = MIN( SIZE( FACTORS%iw ), liw )
       IF ( liw < FACTORS%nirnec ) liw = FACTORS%nirtot
     END IF

     IF ( ALLOCATED( FACTORS%iw ) ) THEN
       IF ( liw /= SIZE( FACTORS%iw ) ) THEN
         DEALLOCATE( FACTORS%iw, STAT = stat )
         IF ( stat /= 0 ) GO TO 100
         ALLOCATE( FACTORS%iw( liw ), STAT = stat )
         IF ( stat /= 0 ) GO TO 100
       END IF
     ELSE
       ALLOCATE( FACTORS%iw( liw ), STAT = stat )
       IF ( stat /= 0 ) GO TO 100
     END IF

!  Factorize the matrix

     DO
       FACTORS%val( 1 : MATRIX%ne ) = MATRIX%val( 1 : MATRIX%ne )

!  Schnabel-Eskow modified Cholesky factorization

       FINFO%modstep = 0
       IF ( CONTROL%pivoting == 4 ) THEN
         CALL SILS_schnabel_eskow( MATRIX%n, MATRIX%ne, MATRIX%row,            &
           MATRIX%col, FACTORS%val( : la ), la, FACTORS%iw, liw,               &
           FACTORS%keep, FACTORS%nsteps, FINFO%maxfrt, FACTORS_iw1,            &
           ICNTL, CNTL, INFO, FACTORS%val( la + 1 : ), FINFO%maxchange,        &
           FINFO%modstep )

!  Multifrontal factorization

       ELSE
         CALL MA27BD( MATRIX%n, MATRIX%ne, MATRIX%row, MATRIX%col,             &
                      FACTORS%val, la, FACTORS%iw, liw, FACTORS%keep,          &
                      FACTORS%nsteps, FINFO%maxfrt, FACTORS_iw1,               &
                      ICNTL, CNTL, INFO )
         FINFO%maxchange = zero
         FINFO%modstep = MATRIX%n + 1
       END IF

       FINFO%flag = INFO( 1 )

!  Check to see if there was sufficient workspace. If not, allocate
!  more and retry

       IF ( FINFO%flag == - 3 ) THEN
         IF ( ALLOCATED( FACTORS%iw ) ) THEN
           DEALLOCATE( FACTORS%iw, STAT = stat )
           IF ( stat /= 0 ) GO TO 100
         END IF
         liw = INT( CONTROL%multiplier * REAL( liw, KIND = wp ) )
         IF ( liw > CONTROL%maxliw ) THEN
           IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                       &
             WRITE( CONTROL%lp, 2000 ) 'integer', CONTROL%maxliw
           FINFO%flag = - 8
           return
         END IF
         ALLOCATE( FACTORS%iw( liw ), STAT = stat )
         IF ( stat /= 0 ) GO TO 100
       ELSE IF ( FINFO%flag == - 4 ) THEN
         IF ( ALLOCATED( FACTORS%iw ) ) THEN
           DEALLOCATE( FACTORS%val, STAT = stat )
           IF ( stat /= 0 ) GO TO 100
         END IF
         la = INT( CONTROL%multiplier * REAL( la, KIND = wp ) )
         IF ( la > CONTROL%maxla ) THEN
           IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                       &
             WRITE( CONTROL%lp, 2000 ) 'real', CONTROL%maxla
           FINFO%flag = - 7
           RETURN
         END IF
         ALLOCATE( FACTORS%val( la + la_extra ), STAT = stat )
         IF ( stat /= 0 ) GO TO 100
       ELSE
         EXIT
       END IF
     END DO

     IF ( FINFO%flag >= 0 ) THEN
       kw = 2
       nblks = ABS( FACTORS%iw( 1 ) )
       DO block = 1, nblks
         FACTORS%iw1( block ) = kw - 1
         ncols = FACTORS%iw( kw )
         IF ( ncols > 0 ) THEN
           kw = kw + 1
         ELSE
           ncols = - ncols
         END IF
         kw = kw + ncols + 1
       END DO

       FACTORS%latop = INFO( 9 )
       FACTORS%maxfrt = FINFO%maxfrt

       IF ( FINFO%flag == 3 ) THEN

! Supplement the arrays in the singular case

         kw = 2
         ALLOCATE( flag( FACTORS%n ), STAT = stat )
         IF ( stat /= 0 ) GO TO 100
         flag = 0
         DO block = 1, nblks
           ncols = FACTORS%iw( kw )
           IF ( ncols > 0 ) THEN
             kw = kw + 1
             nrows = FACTORS%iw( kw )
             flag ( ABS( FACTORS%iw( kw + 1 : kw + nrows ) ) ) = 1
           ELSE
             ncols = - ncols
             flag( FACTORS%iw( kw + 1 ) ) = 1
           END IF
           kw = kw + ncols + 1
         END DO
         DO i = 1, FACTORS%n
           IF ( flag( i ) == 0 ) THEN
             nblks = nblks + 1
             FACTORS%iw1( nblks ) = INFO( 10 )
             INFO( 9 ) = INFO( 9 ) + 1
             FACTORS%val( INFO( 9 ) ) = 1.0_wp
             FACTORS%iw( INFO( 10 ) + 1 ) = - 1
             FACTORS%iw( INFO( 10 ) + 2 ) = i
             INFO( 10 ) = INFO( 10 ) + 2
           END IF
         END DO
         DEALLOCATE( flag, STAT = stat )
         IF ( stat /= 0 ) GO TO 100
         FACTORS%iw( 1 ) = sign ( nblks, FACTORS%iw( 1 ) )
       END IF
     END IF
     FACTORS%dim_iw1 = ABS( FACTORS%iw( 1 ) )

!  Record return information

     IF ( FINFO%flag <= - 5 ) THEN
       FINFO%step = INFO( 2 )
     ELSE
       FINFO%step = MATRIX%n + 1
     END IF
     FINFO%more = INFO( 2 )
     FINFO%nrlbdu = INFO( 9 )
     FINFO%nebdu  = INFO( 9 )
     FINFO%nirbdu = INFO( 10 )
     FINFO%ncmpbr = INFO( 12 )
     FINFO%ncmpbi = INFO( 13 )
     FINFO%ntwo   = INFO( 14 )
     FINFO%neig   = INFO( 15 )
     FINFO%static = 0
     FINFO%maxfrt = - 1
     FINFO%nrltot = FACTORS%nrltot
     FINFO%nirtot = FACTORS%nirtot
     FINFO%nrlnec = FACTORS%nrlnec
     FINFO%nirnec = FACTORS%nirnec
     FINFO%delay = 0
     FINFO%signc = 0
     FINFO%opsa = FACTORS%ops / 2.0
     FINFO%opse = FACTORS%ops / 2.0
     FINFO%opsb = 0.0
     FINFO%smin = 1.0
     FINFO%smax = 1.0

!  Reset FINFO%flag to MA57-style return values

     IF ( FINFO%flag == 3 ) THEN
        FINFO%flag = 4
        FINFO%rank = INFO( 2 )
     ELSE
        FINFO%rank = MATRIX%n
     END IF
     IF ( FINFO%flag == 2 ) FINFO%flag = 5
     FINFO%stat = 0

     IF ( ALLOCATED( FACTORS%w ) ) THEN
       IF ( SIZE( FACTORS%w ) < FACTORS%maxfrt ) THEN
         DEALLOCATE( FACTORS%w, STAT = i )
         IF ( i /= 0 ) THEN
           IF ( FINFO%flag >= 0 ) FINFO%flag = i
           IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                       &
             WRITE( CONTROL%lp, 2010 ) 'deallocate', FINFO%flag
           RETURN
         END IF
         ALLOCATE( FACTORS%w( FACTORS%maxfrt ), STAT = i )
         IF ( i /= 0 ) THEN
           IF ( FINFO%flag >= 0 ) FINFO%flag = i
           IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                       &
             WRITE( CONTROL%lp, 2010 ) 'allocate', FINFO%flag
           RETURN
         END IF
       END IF
     ELSE
       ALLOCATE( FACTORS%w( FACTORS%maxfrt ), STAT = i )
       IF ( i /= 0 ) THEN
         IF ( FINFO%flag >= 0 ) FINFO%flag = i
         IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                         &
           WRITE( CONTROL%lp, 2010 ) 'allocate', FINFO%flag
         RETURN
       END IF
     END IF

     RETURN

 100 CONTINUE
     IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                             &
        WRITE( CONTROL%lp, '( A / A, I0 )')                                    &
        ' Error return from SILS_FACTORIZE:',                                  &
        ' allocate or DEALLOCATE failed with STAT=', stat
     FINFO%flag = - 3
     FINFO%more = 0
     FINFO%nrlbdu = - 1
     FINFO%nebdu  = - 1
     FINFO%nirbdu = - 1
     FINFO%ncmpbr = - 1
     FINFO%ncmpbi = - 1
     FINFO%ntwo   = - 1
     FINFO%neig   = - 1
     FINFO%static = 0
     FINFO%maxfrt = - 1
     FINFO%nrltot = - 1
     FINFO%nirtot = - 1
     FINFO%nrlnec = - 1
     FINFO%nirnec = - 1
     FINFO%delay = - 1
     FINFO%signc = 0
     FINFO%modstep = - 1
     FINFO%opsa = 0.0
     FINFO%opse = 0.0
     FINFO%opsb = 0.0
     FINFO%smin = 1.0
     FINFO%smax = 1.0
     FINFO%maxchange = 0.0
     FINFO%rank = - 1
     FINFO%stat = stat

     RETURN

!  Non-executable statement

 2000 FORMAT( ' Error return from SILS_FACTORIZE: ', ' main ', A,              &
              ' array needs to be bigger than', I0 )
 2010 FORMAT( ' Error return from SILS_FACTORIZE: ', A, ' failed',             &
              ' with STAT = ', I0 )


     CONTAINS

!-*-*-*-   S I L S _ S C H N A B E L _ E S K O W   S U B R O U T I N E  -*-*-*-

       SUBROUTINE SILS_schnabel_eskow( n, nz, IRN, ICN, A, la, IW, liw, IKEEP, &
                                       nsteps, maxfrt, IW1, ICNTL, CNTL, INFO, &
                                       DIAG, maxchange, modstep )

!  Given an elimination ordering, factorize a symmetric matrix A
!  (modified version of MA27B from LANCELOT A)

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER n, nz, la, liw, nsteps, maxfrt, modstep
       REAL ( KIND = wp ) :: maxchange
       INTEGER, DIMENSION( * ) :: IRN, ICN
       INTEGER, DIMENSION( liw ) :: IW
       INTEGER, DIMENSION( n, 3 ) :: IKEEP
       INTEGER, DIMENSION( n ) :: IW1
       INTEGER, DIMENSION( 30 ) :: ICNTL
       INTEGER, DIMENSION( 20 ) :: INFO
       REAL ( KIND = wp ), DIMENSION( 5 ) :: CNTL
       REAL ( KIND = wp ), DIMENSION( la ) :: A
       REAL ( KIND = wp ), DIMENSION( n ) :: DIAG

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

       INTEGER :: k, kz, nz1, iphase, j2, j1, irows
       INTEGER :: len, nrows, ipos, kblk, iapos, ncols, iblk
       REAL ( KIND = wp ) :: addon

       INFO( 1 ) = 0
       maxchange = zero
       IF ( ICNTL( 3 ) > 0 .AND. ICNTL( 2 ) > 0 ) THEN

!  Print input parameters

         WRITE( ICNTL( 2 ), 2010 ) n, nz, la, liw, nsteps, CNTL( 1 )
         kz = MIN( 6, nz ) ; IF ( ICNTL( 3 ) > 1 ) kz = nz
         IF ( nz > 0 ) WRITE( ICNTL( 2 ), 2020 )                               &
            ( A( k ), IRN( k ), ICN( k ), k = 1, kz )
         k = MIN( 9, n ) ; IF ( ICNTL( 3 ) > 1 ) k = n
         IF ( k > 0 ) WRITE( ICNTL( 2 ), 2030 ) IKEEP( : k, 1 )
         k = MIN( k, nsteps )
         IF ( k > 0 ) THEN
           WRITE( ICNTL( 2 ), 2040 ) IKEEP( : k, 2 )
           WRITE( ICNTL( 2 ), 2050 ) IKEEP( : k, 3 )
         END IF
       END IF
       IF ( n >= 1 .AND. n <= ICNTL( 4 ) ) THEN
         IF ( nz < 0 ) THEN
           INFO( 1 ) = - 2
           IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
           IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2110 ) nz
           GO TO 130
         END IF
         IF ( liw < nz ) THEN
           INFO( 1 ) = - 3 ; INFO( 2 ) = nz
           GO TO 130
         END IF
         IF ( la < nz + n ) THEN
           INFO( 1 ) = - 4 ; INFO( 2 ) = nz + n
           GO TO 130
         END IF

!  Set phase of Cholesky modification

         iphase = 1

!  Sort

         CALL SILS_sort_entries( n, nz, nz1, A, la, IRN, ICN, IW, liw, IKEEP,  &
                                 IW1, ICNTL, INFO, DIAG, addon )
         IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                           &
           WRITE( ICNTL( 2 ), 2000 ) addon
         IF ( INFO( 1 ) == - 3 .OR. INFO( 1 ) == - 4 ) GO TO 130

!  Factorize

         CALL SILS_schnabel_eskow_main(                                        &
           n, nz1, A, la, IW, liw, IKEEP, IKEEP( 1, 3 ), nsteps, maxfrt,       &
           IKEEP( 1, 2 ), IW1, ICNTL, CNTL, INFO, DIAG, addon, iphase,         &
           maxchange, modstep )

         IF ( INFO( 1 ) == - 3 .OR. INFO( 1 ) == - 4 ) GO TO 130
         IF ( INFO( 1 ) == - 5 ) THEN
           IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
           IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2190 ) INFO( 2 )
         END IF
         IF ( INFO( 1 ) == - 6 ) THEN
           IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
           IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2210 )
         END IF

! **** Warning message ****

         IF ( INFO( 1 ) == 3 .AND. ICNTL( 2 ) > 0 )                            &
           WRITE( ICNTL( 2 ), 2060 ) INFO( 1 ), INFO( 2 )

! **** Error returns ****

       ELSE
         INFO( 1 ) = - 1
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2090 ) n
       END IF

  130  CONTINUE
       IF ( INFO( 1 ) == - 3 ) THEN
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2140 ) liw, INFO( 2 )
       ELSE IF ( INFO( 1 ) == - 4 ) THEN
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2170 ) la, INFO( 2 )
       END IF
       IF ( ICNTL( 3 ) > 0 .AND. ICNTL( 2 ) > 0 ) THEN

!  Print output parameters

         WRITE( ICNTL( 2 ), 2230 ) maxfrt, INFO( 1 ),INFO( 9 ), INFO( 10 ),    &
           INFO( 12 ), INFO( 13 ),INFO( 14 ), INFO( 2 )
         IF ( INFO( 1 ) >= 0 ) THEN

!  Print out matrix factors from SILS_schnabel_eskow

           kblk = ABS( IW( 1 ) + 0 )
           IF ( kblk /= 0 ) THEN
             IF ( ICNTL( 3 ) == 1 ) kblk = 1
             ipos = 2 ; iapos = 1
             DO iblk = 1, kblk
               ncols = IW( ipos ) ; nrows = IW( ipos + 1 )
               j1 = ipos + 2
               IF ( ncols <= 0 ) THEN
                 ncols = - ncols ; nrows = 1
                 j1 = j1 - 1
               END IF
               WRITE( ICNTL( 2 ), 2250 ) iblk, nrows, ncols
               j2 = j1 + ncols - 1
               ipos = j2 + 1
               WRITE( ICNTL( 2 ), 2260 ) IW( j1 : j2 )
               WRITE( ICNTL( 2 ), 2270 )
               len = ncols
               DO irows = 1, nrows
                 j1 = iapos ; j2 = iapos + len - 1
                 WRITE( ICNTL( 2 ), 2280 ) A( j1 : j2 )
                 len = len - 1
                 iapos = j2 + 1
               END DO
             END DO
           END IF
         END IF
       END IF

       RETURN

!  Non executable statements

 2000  FORMAT( ' addon = ', ES12.4 )
 2010  FORMAT( //, ' entering SILS_schnabel_eskow with      n     nz     la ', &
                   '   liw  nsteps      u', / , 38X, 5I7, F7.2 )
 2020  FORMAT( ' Matrix non-zeros', 2( ES16.4, 2I6 ), /,                       &
               ( 17X, ES16.4, 2I6, ES16.4, 2I6 ) )
 2030  FORMAT( ' IKEEP( ., 1 )=', 10I6, /, ( 12X, 10I6 ) )
 2040  FORMAT( ' IKEEP( ., 2 )=', 10I6, /, ( 12X, 10I6 ) )
 2050  FORMAT( ' IKEEP( ., 3 )=', 10I6, /, ( 12X, 10I6 ) )
 2060  FORMAT( ' *** Warning message from subroutine SILS_schnabel_eskow ***', &
               ' info(1) = ', I0, /, 5X, 'matrix is singular. Rank=', I0 )
 2080  FORMAT( ' **** Error return from SILS_schnabel_eskow **** info(1) =', I0)
 2090  FORMAT( ' Value of n out of range ... =', I0 )
 2110  FORMAT( ' Value of nz out of range .. =', I0 )
 2140  FORMAT( ' liw too small, must be increased from', I0,                   &
               ' to at least', I0 )
 2170  FORMAT( ' la too small, must be increased from ', I0,                   &
               ' to at least', I0 )
 2190  FORMAT( ' Zero pivot at stage', I0,                                     &
               ' when input matrix declared definite' )
 2210  FORMAT( ' Change in sign of pivot encountered ',                        &
               ' when factoring allegedly definite matrix' )
 2230  FORMAT( /' Leaving SILS_schnabel_eskow with maxfrt  info(1) nrlbdu',    &
                ' nirbdu ncmpbr ncmpbi   ntwo ierror', /, 37X, 8I7 )
 2250  FORMAT( ' Block pivot =', I8,' nrows =', I8,' ncols =', I0 )
 2260  FORMAT( ' Column indices =', 10I6, /, ( 17X, 10I6 ) )
 2270  FORMAT( ' Real entries .. each row starts on a new line' )
 2280  FORMAT( 5ES16.8 )

!  End of subroutine SILS_schnabel_eskow

       END SUBROUTINE SILS_schnabel_eskow

!-*-*-*-*-   S I L S _ S O R T _ E N T R I E S    S U B R O U T I N E  -*-*-*-*-

       SUBROUTINE SILS_sort_entries( n, nz, nz1, A, la, IRN, ICN, IW, liw,     &
                                     PERM, IW2, ICNTL, INFO, DIAG, addon )

!  Sort the entries of A prior to the factorization
!  (modified version of MA27N from LANCELOT A)

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER n, nz, nz1, la, liw
       REAL ( KIND = wp ) addon
       INTEGER, DIMENSION( * ) :: IRN, ICN
       INTEGER, DIMENSION( liw ) :: IW
       INTEGER, DIMENSION( n ) :: PERM, IW2
       INTEGER, DIMENSION( 30 ) :: ICNTL
       INTEGER, DIMENSION( 20 ) :: INFO
       REAL ( KIND = wp ), DIMENSION( la ) :: A
       REAL ( KIND = wp ), DIMENSION( n ) :: DIAG

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

       INTEGER :: k, iold, inew, jold, ia, jnew, j2, j1, iiw, jj, ii
       INTEGER :: ich, i, ipos, jpos
       REAL ( KIND = wp ) :: anext, anow, machep, maxdag

! ** Obtain machep

       machep = EPSILON( one )
       INFO( 1 ) = 0

!  Initialize work array (IW2) in preparation for counting numbers of
!  non-zeros in the rows and initialize last n entries in A which will
!  hold the diagonal entries

       ia = la
       IW2 = 1
       A( ia - n + 1 : ia ) = zero

!  Scan input copying row indices from IRN to the first nz positions
!  in IW. The negative of the index is held to flag entries for
!  the in-place sort. Entries in IW corresponding to diagonals and
!  entries with out-of-range indices are set to zero. For diagonal entries,
!  reals are accumulated in the last n locations OF A.
!  The number of entries in each row of the permuted matrix is
!  accumulated in IW2. Indices out of range are ignored after being counted
!  and after appropriate messages have been printed.

       INFO( 2 ) = 0

!  nz1 is the number of non-zeros held after indices out of range have
!  been ignored and diagonal entries accumulated.

       nz1 = n
       IF ( nz /= 0 ) THEN
         DO k = 1, nz
           iold = IRN( k )
           IF ( iold <= n .AND. iold > 0 ) THEN
             jold = ICN( k )
             IF ( jold <= n .AND. jold > 0 ) THEN
               inew = PERM( iold )
               jnew = PERM( jold )
               IF ( inew == jnew ) THEN
                 ia = la - n + iold
                 A( ia ) = A( ia ) + A( k )
                 IW( k ) = 0
                 CYCLE
               END IF
               inew = MIN( inew, jnew )

!  Increment number of entries in row inew.

               IW2( inew ) = IW2( inew ) + 1
               IW( k ) = - iold
               nz1 = nz1 + 1
               CYCLE

!  Entry out of range. It will be ignored and a flag set.

             END IF
           END IF
           INFO( 1 ) = 1
           INFO( 2 ) = INFO( 2 ) + 1
           IF ( INFO( 2 ) <= 1 .AND. ICNTL( 2 ) > 0 )                          &
             WRITE( ICNTL( 2 ), 2040 ) INFO( 1 )
           IF ( INFO( 2 ) <= 10 .AND. ICNTL( 2 ) > 0 )                         &
             WRITE( ICNTL( 2 ), 2050 ) k, IRN( k ), ICN( k )
           IW( k ) = 0
         END DO

!  Calculate pointers (in IW2) to the position immediately after the end
!  of each row.

       END IF

!  Room is included for the diagonals.

       IF ( nz >= nz1 .OR. nz1 == n ) THEN
         k = 1
         DO i = 1, n
           k = k + IW2( i )
           IW2( i ) = k
         END DO
       ELSE

!  Room is not included for the diagonals.

         k = 1
         DO i = 1, n
           k = k + IW2( i ) - 1
           IW2( i ) = k
         END DO

!  Fail if insufficient space in arrays A or IW.

       END IF

!  **** Error return ****

       IF ( nz1 > liw ) THEN
          INFO( 1 ) = - 3 ; INFO( 2 ) = nz1
          RETURN
       END IF

       IF ( nz1 + n > la ) THEN
         INFO( 1 ) = - 4 ; INFO( 2 ) = nz1 + n
         RETURN
       END IF

!  Now run through non-zeros in order placing them in their new
!  position and decrementing appropriate IW2 entry. If we are
!  about to overwrite an entry not yet moved, we must deal with
!  this at this time.

       IF ( nz1 /= n ) THEN
 L140:   DO k = 1, nz
           iold = - IW( k )
           IF ( iold <= 0 ) CYCLE  L140
           jold = ICN( k )
           anow = A( k )
           IW( k ) = 0
           DO ich = 1, nz
             inew = PERM( iold ) ; jnew = PERM( jold )
             inew = MIN( inew, jnew )
             IF ( inew == PERM( jold ) ) jold = iold
             jpos = IW2( inew ) - 1
             iold = -IW( jpos )
             anext = A( jpos )
             A( jpos ) = anow
             IW( jpos ) = jold
             IW2( inew ) = jpos
             IF ( iold == 0 ) CYCLE  L140
             anow = anext
             jold = ICN( jpos )
           END DO
         END DO L140
         IF ( nz < nz1 ) THEN

!  Move up entries to allow for diagonals.

           ipos = nz1 ; jpos = nz1 - n
           DO ii = 1, n
             i = n - ii + 1
             j1 = IW2( i ) ; j2 = jpos
             IF ( j1 <= jpos ) THEN
                DO jj = j1, j2
                   IW( ipos ) = IW( jpos )
                   A( ipos ) = A( jpos )
                   ipos = ipos - 1
                   jpos = jpos - 1
                END DO
             END IF
             IW2( i ) = ipos + 1
             ipos = ipos - 1
           END DO

!  Run through rows inserting diagonal entries and flagging beginning
!  of each row by negating first column index.

         END IF
       END IF
       maxdag = machep
       DO iold = 1, n
         inew = PERM( iold )
         jpos = IW2( inew ) - 1
         ia = la - n + iold
         A( jpos ) = A( ia )

!  Set diag to value of diagonal entry (original numbering)

         DIAG( iold ) = A( ia )
         maxdag = MAX( maxdag, ABS( A( ia ) ) )
         IW( jpos ) = - iold
       END DO

!  Compute addition to off-diagonal 1-norm

       addon = maxdag * machep ** 0.75
!      addon = maxdag * machep ** 0.33333

!  Move sorted matrix to the end of the arrays

       ipos = nz1
       ia = la
       iiw = liw
       DO i = 1, nz1
         A( ia ) = A( ipos )
         IW( iiw ) = IW( ipos )
         ipos = ipos - 1
         ia = ia - 1
         iiw = iiw - 1
       END DO

       RETURN

!  Non executable statements

 2040  FORMAT( ' *** Warning message from subroutine SILS_sort_entries ***',   &
               ' iflag =', I2 )
 2050  FORMAT( I6, 'th non-zero (in row', I6, ' and column ', I6, ') ignored' )

!  End of subroutine SILS_sort_entries

       END SUBROUTINE SILS_sort_entries

!-*-  S I L S _ S C H N A B E L _ E S K O W _ M A I N   S U B R O U T I N E  -*-

       SUBROUTINE SILS_schnabel_eskow_main(                                    &
           n, nz, A, la, IW, liw, PERM, NSTK, nsteps, maxfrt, NELIM, IW2,      &
           ICNTL, CNTL, INFO, DIAG, addon, iphase, maxchange, modstep )

!  Perform the multifrontal factorization
!  (modified version of MA27O from LANCELOT A)

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER n, nz, la, liw, nsteps, maxfrt, iphase, modstep
       REAL ( KIND = wp ) addon, maxchange
       INTEGER, DIMENSION( liw ) :: IW
       INTEGER, DIMENSION( n ) :: PERM
       INTEGER, DIMENSION( nsteps ) :: NSTK, NELIM
       INTEGER, DIMENSION( n ) :: IW2
       INTEGER, DIMENSION( 30 ) :: ICNTL
       INTEGER, DIMENSION( 20 ) :: INFO
       REAL ( KIND = wp ), DIMENSION( 5 ) :: CNTL
       REAL ( KIND = wp ), DIMENSION( la ) :: A
       REAL ( KIND = wp ), DIMENSION( n ) :: DIAG

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

       INTEGER :: idummy, numorg, jnew, jj, j, laell, lapos2, ifr, iorg
       INTEGER :: jdummy, j2, iell, jcol, npiv, newel, istk, i, azero
       INTEGER :: ltopst, lnass, numstk, jfirst, nfront, jlast, j1, jnext
       INTEGER :: iswap, ibeg, iexch, krow, ipos, liell, kmax, ioldps, iend
       INTEGER :: kdummy, lnpiv, irow, jjj, jay, kk, ipiv, npivp1, jpiv
       INTEGER :: istk2, iwpos, k, nblk, iass, nass, numass, iinput, ntotpv
       INTEGER :: posfac, astk, astk2, apos, apos1, apos2, ainput, pivsiz
       INTEGER :: ntwo, neig, ncmpbi, ncmpbr, nrlbdu, nirbdu
       REAL ( KIND = wp ) :: amax, rmax, swap, amult, w1, onenrm, uu

       IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 ) THEN
         WRITE( ICNTL( 2 ), 2000 ) DIAG( : MIN( n, 4 ) )
         WRITE( ICNTL( 2 ), 2100 ) iphase
       END IF

!  Initialization.
!  nblk is the number of block pivots used.

       nblk = 0 ; ntwo = 0 ; neig = 0
       ncmpbi = 0 ; ncmpbr = 0 ; maxfrt = 0
       nrlbdu = 0 ; nirbdu = 0

!  A private variable uu is set to u, so that u will remain unaltered.

       uu = MIN( CNTL( 1 ), half )
       uu = MAX( uu, - half )

       IW2 = 0

!  iwpos is pointer to first free position for factors in IW.
!  posfac is pointer for factors in A. At each pass through the
!      major loop posfac initially points to the first free location
!      in A and then is set to the position of the current pivot in A.
!  istk is pointer to top of stack in IW.
!  istk2 is pointer to bottom of stack in IW (needed by compress).
!  astk is pointer to top of stack in A.
!  astk2 is pointer to bottom of stack in A (needed by compress).
!  iinput is pointer to current position in original rows in IW.
!  ainput is pointer to current position in original rows in A.
!  azero is pointer to last position zeroed in A.
!  ntotpv is the total number of pivots selected. This is used
!      to determine whether the matrix is singular.

       iwpos = 2
       posfac = 1
       istk = liw - nz + 1 ; istk2 = istk - 1
       astk = la - nz + 1 ; astk2 = astk - 1
       iinput = istk ; ainput = astk
       azero = 0
       ntotpv = 0

!  numass is the accumulated number of rows assembled so far.

       numass = 0

!  Each pass through this main loop performs all the operations
!      associated with one set of assembly/eliminations.

       DO iass = 1, nsteps

!  nass will be set to the number of fully assembled variables in
!      current newly created element.

         nass = NELIM( iass )

!  newel is a pointer into IW to control output of integer information
!      for newly created element.

         newel = iwpos + 1

!  Symbolically assemble incoming rows and generated stack elements
!  ordering the resultant element according to permutation PERM.  We
!  assemble the stack elements first because these will already be ordered.

!  Set header pointer for merge of index lists.

         jfirst = n + 1

!  Initialize number of variables in current front.

         nfront = 0
         numstk = NSTK( iass )
         ltopst = 1
         lnass = 0

!  Jump if no stack elements are being assembled at this stage.

         IF ( numstk /= 0 ) THEN
           j2 = istk - 1
           lnass = nass
           ltopst = ( ( IW( istk ) + 1 ) * IW( istk ) ) / 2
           DO iell = 1, numstk

!  Assemble element iell placing the indices into a linked list in IW2
!  ordered according to PERM.

             jnext = jfirst
             jlast = n + 1
             j1 = j2 + 2
             j2 = j1 - 1 + IW( j1 - 1 )

!  Run through index list of stack element iell.

             DO jj = j1, j2
               j = IW( jj )
               IF ( IW2( j ) > 0 ) CYCLE
               jnew = PERM( j )

!  If variable was previously fully summed but was not pivoted on earlier
!  because of numerical test, increment number of fully summed rows/columns
!  in front.

               IF ( jnew <= numass ) nass = nass + 1

!  Find position in linked list for new variable.  Note that we start
!  from where we left off after assembly of previous variable.

               DO idummy = 1, n
                 IF ( jnext == n + 1 ) EXIT
                 IF ( PERM( jnext ) > jnew ) EXIT
                 jlast = jnext
                 jnext = IW2( jlast )
               END DO

               IF ( jlast == n + 1 ) THEN
                 jfirst = j
               ELSE
                 IW2( jlast ) = j
               END IF

               IW2( j ) = jnext
               jlast = j

!  Increment number of variables in the front.

               nfront = nfront + 1
             END DO
           END DO
           lnass = nass - lnass
         END IF

!  Now incorporate original rows.  Note that the columns in these rows need not
!  be in order. We also perform a swap so that the diagonal entry is the first
!  in its row. This allows us to avoid storing the inverse of array PERM.

         numorg = NELIM( iass )
         j1 = iinput
 L150:   DO iorg = 1, numorg
           j = - IW( j1 )
           DO idummy = 1, liw
             jnew = PERM( j )

!  Jump if variable already included.

             IF ( IW2( j ) <= 0 ) THEN

!  Here we must always start our search at the beginning.

               jlast = n + 1
               jnext = jfirst
               DO jdummy = 1, n
                 IF ( jnext == n + 1 ) EXIT
                 IF ( PERM( jnext ) > jnew ) EXIT
                 jlast = jnext
                 jnext = IW2( jlast )
               END DO
               IF ( jlast == n + 1 ) THEN
                 jfirst = j
               ELSE
                 IW2( jlast ) = j
               END IF
               IW2( j ) = jnext

!  Increment number of variables in front.

               nfront = nfront + 1
             END IF

             j1 = j1 + 1
             IF ( j1 > liw ) CYCLE L150
             j = IW( j1 )
             IF ( j < 0 ) CYCLE L150
           END DO
         END DO L150

!  Now run through linked list IW2 putting indices of variables in new
!  element into IW and setting IW2 entry to point to the relative
!  position of the variable in the new element.

         IF ( newel + nfront >= istk ) THEN

!  Compress IW.

           CALL SILS_compress( A, IW, istk, istk2, iinput, 2, ncmpbr, ncmpbi )
           IF ( newel + nfront >= istk ) THEN
             INFO( 2 ) = liw + 1 + newel + nfront - istk
             INFO( 1 ) = - 3
             RETURN
           END IF
         END IF

         j = jfirst
         DO ifr = 1, nfront
           newel = newel + 1
           IW( newel ) = j
           jnext = IW2( j )
           IW2( j ) = newel - iwpos - 1
           j = jnext
         END DO

!  Assemble reals into frontal matrix.

         maxfrt = MAX( maxfrt, nfront )
         IW( iwpos ) = nfront

!  First zero out frontal matrix as appropriate first checking to see
!  if there is sufficient space.

         laell = ( ( nfront + 1 ) * nfront )/2
         apos2 = posfac + laell - 1
         IF ( numstk /= 0 ) lnass = lnass * ( 2 * nfront - lnass + 1 ) / 2
         IF ( posfac + lnass - 1 < astk ) THEN
           IF ( apos2 < astk + ltopst - 1 ) GO TO 190
         END IF

!  Compress A.

         CALL SILS_compress( A, IW, astk, astk2, ainput, 1, ncmpbr, ncmpbi )

!  Error returns

         IF ( posfac + lnass - 1 >= astk ) THEN
           INFO( 1 ) = - 4
           INFO( 2 ) = la + MAX( posfac + lnass, apos2 - ltopst + 2 ) - astk
           RETURN
         END IF
         IF ( apos2 >= astk + ltopst - 1 ) THEN
           INFO( 1 ) = - 4
           INFO( 2 ) = la + MAX( posfac + lnass, apos2 - ltopst + 2 ) - astk
           RETURN
         END IF

  190    CONTINUE
         IF ( apos2 > azero ) THEN
           apos = azero + 1
           lapos2 = MIN( apos2, astk - 1 )
           IF ( lapos2 >= apos ) THEN
             A( apos : lapos2 ) = zero
           END IF
           azero = apos2
         END IF

!  Jump if there are no stack elements to assemble.

         IF ( numstk /= 0 ) THEN

!  Place reals corresponding to stack elements in correct positions in A.

           DO iell = 1, numstk
             j1 = istk + 1 ; j2 = istk + IW( istk )
             DO jj = j1, j2
               irow = IW2( IW( jj ) )
               apos = posfac + SILS_idiag( nfront, irow )
               DO jjj = jj, j2
                 j = IW( jjj )
                 apos2 = apos + IW2( j ) - irow
                 A( apos2 ) = A( apos2 ) + A( astk )
                 A( astk ) = zero
                 astk = astk + 1
               END DO
             END DO

!  Increment stack pointer.

             istk = j2 + 1
           END DO
         END IF

!  Incorporate reals from original rows.

 L280:   DO iorg = 1, numorg
           j = - IW( iinput )

!  We can do this because the diagonal is now the first entry.

           irow = IW2( j )
           apos = posfac + SILS_idiag( nfront, irow )

!  The following loop goes from 1 to nz because there may be duplicates.

           DO idummy = 1, nz
             apos2 = apos + IW2( j ) - irow
             A( apos2 ) = A( apos2 ) + A( ainput )
             ainput = ainput + 1 ; iinput = iinput + 1
             IF ( iinput > liw ) CYCLE L280
             j = IW( iinput )
             IF ( j < 0 ) CYCLE L280
           END DO
         END DO L280

!  Reset IW2 and numass.

         numass = numass + numorg
         j1 = iwpos + 2 ; j2 = iwpos + nfront + 1
         IW2( IW( j1: j2 ) ) = 0

!  Perform pivoting on assembled element.
!  npiv is the number of pivots so far selected.
!  lnpiv is the number of pivots selected after the last pass through
!      the the following loop.

         lnpiv = - 1 ; npiv = 0

         DO kdummy = 1, nass
           IF ( npiv == nass ) EXIT
           IF ( npiv == lnpiv ) EXIT
           lnpiv = npiv ; npivp1 = npiv + 1

!  jpiv is used as a flag to indicate when 2 by 2 pivoting has occurred
!      so that ipiv is incremented correctly.

           jpiv = 1

!  nass is maximum possible number of pivots. We either take the diagonal
!  entry or the 2 by 2 pivot with the largest off-diagonal at each stage.
!  Each pass through this loop tries to choose one pivot.

           DO ipiv = npivp1, nass
             jpiv = jpiv - 1
             IF ( jpiv == 1 ) CYCLE
             apos = posfac + SILS_idiag( nfront - npiv, ipiv - npiv )

!  If the user has indicated that the matrix is definite, we do not need to
!  test for stability but we do check to see if the pivot is non-zero or has
!  changed sign. If it is zero, we exit with an error. If it has changed sign
!  and u was set negative, then we again exit immediately. If the pivot changes
!  sign and u was zero, we continue with the factorization but print a warning
!  message

!  First check if pivot is positive.

             IF ( A( apos ) <= zero ) THEN
               IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                     &
                 WRITE( ICNTL( 2 ), 2050 ) npiv
               iphase = 2
             END IF
             amax = zero
             rmax = amax

! i is pivot row.

             i = IW( iwpos + ipiv + 1 )
             IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 ) THEN
               WRITE( ICNTL( 2 ), 2060 ) npiv, i
               WRITE( ICNTL( 2 ), 2010 ) A( apos )
             END IF

!  Find largest entry to right of diagonal in row of prospective pivot
!  in the fully-summed part. Also record column of this largest entry.
!  onenrm is set to 1-norm of off-diagonals in row.

             onenrm = 0.0
             j1 = apos + 1 ; j2 = apos + nfront - ipiv
             DO jj = j1, j2
               jay = jj - j1 + 1
               IF ( iphase == 1 ) THEN
                 j = IW( iwpos + ipiv + jay + 1 )
                 w1 = DIAG( j ) - A( jj ) * A( jj ) / A( apos )
                 IF ( w1 <= addon ) THEN
                   IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                 &
                     WRITE( ICNTL( 2 ), 2020 ) j, w1
                   iphase = 2
                 END IF
               END IF
               onenrm = onenrm + ABS( A( jj ) )
               rmax = MAX( ABS( A( jj ) ),rmax )
               IF ( jay <= nass - ipiv ) THEN
                 IF ( ABS( A( jj ) ) > amax ) amax = ABS( A( jj ) )
               END IF
            END DO

!  Now calculate largest entry in other part of row.

            apos1 = apos
            kk = nfront - ipiv
            IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                        &
              WRITE( ICNTL( 2 ), 2070 ) npiv, i, onenrm

! Jump if still in phase 1.

            IF ( iphase /= 1 ) THEN

!  Check to see if pivot must be increased

              IF ( A( apos ) < addon + onenrm ) THEN

                IF ( modstep == 0 ) modstep = ipiv

!  Adjust diagonal entry and record change in DIAG

                DIAG( i ) = addon + onenrm - A( apos )
                maxchange = MAX( maxchange, ABS( DIAG( i ) ) )
                IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                    &
                  WRITE( ICNTL( 3 ), 2080 ) i, DIAG( i )
                A( apos ) = onenrm + addon
              ELSE
                DIAG( i ) = zero
              END IF
            ELSE
              DIAG( i ) = zero
            END IF
            pivsiz = 1 ; irow = ipiv - npiv

!  Pivot has been chosen. If block pivot of order 2, pivsiz is equal to 2,
!  otherwise pivsiz is 1. The following loop moves the pivot block to the top
!  left hand corner of the frontal matrix.

            DO krow = 1, pivsiz
              IF ( irow == 1 ) CYCLE
              j1 = posfac + irow
              j2 = posfac + nfront - npiv - 1
              IF ( j2 >= j1 ) THEN
                apos2 = apos + 1

!  Swap portion of rows whose column indices are greater than later row.

                DO jj = j1, j2
                  swap = A( apos2 )
                  A( apos2 ) = A( jj )
                  A( jj ) = swap
                  apos2 = apos2 + 1
                END DO
              END IF
              j1 = posfac + 1 ; j2 = posfac + irow - 2
              apos2 = apos
              kk = nfront - irow - npiv

!  Swap portion of rows/columns whose indices lie between the two rows.

              DO jj = j2, j1, - 1
                kk = kk + 1
                apos2 = apos2 - kk
                swap = A( apos2 )
                A( apos2 ) = A( jj )
                A( jj ) = swap
              END DO
              IF ( npiv /= 0 ) THEN
                apos1 = posfac
                kk = kk + 1
                apos2 = apos2 - kk

!  Swap portion of columns whose indices are less than earlier row.

                DO jj = 1, npiv
                  kk = kk + 1
                  apos1 = apos1 - kk ; apos2 = apos2 - kk
                  swap = A( apos2 )
                  A( apos2 ) = A( apos1 )
                  A( apos1 ) = swap
                END DO
              END IF

!  Swap diagonals and integer indexing information

              swap = A( apos )
              A( apos ) = A( posfac )
              A( posfac ) = swap
              ipos = iwpos + npiv + 2
              iexch = iwpos + irow + npiv + 1
              iswap = IW( ipos )
              IW( ipos ) = IW( iexch )
              IW( iexch ) = iswap
            END DO

!  Perform the elimination using entry (ipiv,ipiv) as pivot.
!  We store u and D(inverse).

            A( posfac ) = one / A( posfac )
            IF ( A( posfac ) < zero ) neig = neig + 1
            j1 = posfac + 1 ; j2 = posfac + nfront - npiv - 1
            IF ( j2 >= j1 ) THEN
              ibeg = j2 + 1
              DO jj = j1, j2
                amult = - A( jj ) * A( posfac )

!  Update diag array.

                j = IW( iwpos + npiv + jj - j1 + 3 )
                DIAG( j ) = DIAG( j ) + amult * A( jj )
                IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                    &
                  WRITE( ICNTL( 2 ), 2090 ) j, DIAG( j )
                iend = ibeg + nfront - ( npiv + jj - j1 + 2 )

!DIR$ IVDEP
                DO irow = ibeg, iend
                  jcol = jj + irow - ibeg
                  A( irow ) = A( irow ) + amult * A( jcol )
                END DO
                ibeg = iend + 1
                A( jj ) = amult
              END DO
            END IF
            npiv = npiv + 1
            ntotpv = ntotpv + 1
            jpiv = 1
            posfac = posfac + nfront - npiv + 1
          END DO
        END DO
        IF ( npiv /= 0 ) nblk = nblk + 1
        ioldps = iwpos ; iwpos = iwpos + nfront + 2
        IF ( npiv /= 0 ) THEN
          IF ( npiv <= 1 ) THEN
            IW( ioldps ) = - IW( ioldps )
            DO k = 1, nfront
              j1 = ioldps + k
              IW( j1 ) = IW( j1 + 1 )
            END DO
            iwpos = iwpos - 1
          ELSE
            IW( ioldps + 1 ) = npiv

!  Copy remainder of element to top of stack

          END IF
        END IF
        liell = nfront - npiv
        IF ( liell /= 0 .AND. iass /= nsteps ) THEN

          IF ( iwpos + liell >= istk )                                         &
            CALL SILS_compress( A, IW, istk, istk2, iinput, 2, ncmpbr, ncmpbi )
          istk = istk - liell - 1
          IW( istk ) = liell
          j1 = istk ; kk = iwpos - liell - 1

! DIR$ IVDEP
          DO k = 1, liell
            j1 = j1 + 1
            kk = kk + 1
            IW( j1 ) = IW( kk )
          END DO

!  We copy in reverse direction to avoid overwrite problems.

           laell = ( ( liell + 1 ) * liell ) / 2
           kk = posfac + laell
           IF ( kk == astk ) THEN
             astk = astk - laell
           ELSE

!  The move and zeroing of array A is performed with two loops so
!  that they may be vectorized

             kmax = kk - 1

!DIR$ IVDEP
              DO k = 1, laell
                kk = kk - 1
                astk = astk - 1
                A( astk ) = A( kk )
              END DO
              kmax = MIN( kmax, astk - 1 )
              A( kk : kmax ) = zero
            END IF
            azero = MIN( azero, astk - 1 )
         END IF
         IF ( npiv == 0 ) iwpos = ioldps
       END DO

!  End of loop on tree nodes.

       IW( 1 ) = nblk
       IF ( ntwo > 0 ) IW( 1 ) = - nblk
       nrlbdu = posfac - 1 ; nirbdu = iwpos - 1

       IF ( ntotpv /= n ) THEN
         INFO( 1 ) = 3
         INFO( 2 ) = ntotpv
       END IF
       INFO( 9 ) = nrlbdu
       INFO( 10 ) = nirbdu
       INFO( 12 ) = ncmpbr
       INFO( 13 ) = ncmpbi
       INFO( 14 ) = ntwo
       INFO( 15 ) = neig

       RETURN

 2000  FORMAT( ' Diag ', 2ES24.16, /, '      ', 2ES24.16 )
 2010  FORMAT( ' Pivot has value ', ES24.16 )
 2020  FORMAT( ' Phase 2, j, w1 ', I6, ES24.16 )
 2050  FORMAT( ' Negative pivot encountered at stage', I8 )
 2060  FORMAT( ' Pivot, pivot row', 2I6 )
 2070  FORMAT( ' npiv, i, onenrm ', 2I6, ES11.4 )
 2080  FORMAT( ' i, Perturbation ', I6, ES12.4 )
 2090  FORMAT( ' j, DIAG  ', I6, ES12.4 )
 2100  FORMAT( ' Phase = ', I1 )

!  End of subroutine SILS_schnabel_eskow_main

       END SUBROUTINE SILS_schnabel_eskow_main

!-*-*-*-*-*-*-   S I L S _ C O M P R E S S   S U B R O U T I N E  -*-*-*-*-*-*-

       SUBROUTINE SILS_compress( A, IW, j1, j2, itop, ireal, ncmpbr, ncmpbi )

!  Compress the data structures (modified version of MA27P from LANCELOT A)

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER j1, j2, itop, ireal, ncmpbr, ncmpbi
       INTEGER, DIMENSION( * ) :: IW
       REAL ( KIND = wp ), DIMENSION( * ) :: A

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

       INTEGER :: jj, ipos

       ipos = itop - 1
       IF ( j2 /= ipos ) THEN
         IF ( ireal /= 2 ) THEN
           ncmpbr = ncmpbr + 1
           DO jj = j2, j1, - 1
             A( ipos ) = A( jj )
             ipos = ipos - 1
           END DO
         ELSE
           ncmpbi = ncmpbi + 1
           DO jj = j2, j1, - 1
             IW( ipos ) = IW( jj )
             ipos = ipos - 1
           END DO
         END IF
         j2 = itop - 1 ; j1 = ipos + 1
       END IF
       RETURN

!  End of subroutine SILS_compress

       END SUBROUTINE SILS_compress

!-*-*-*-*-*-*-*-   S I L S _ C O M P R E S S   F U N C T I O N  -*-*-*-*-*-*-*-

       FUNCTION SILS_idiag( ix, iy )

!  Obtain the displacement from the start of the assembled matrix (of order IX)
!  of the diagonal entry in its row IY

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER :: SILS_idiag
       INTEGER, INTENT( in ) :: ix, iy

       SILS_idiag = ( ( iy - 1 ) * ( 2 * ix - iy + 2 ) ) / 2
       RETURN

!  End of function SILS_idiag

       END FUNCTION SILS_idiag

!  End of SILS_factorize

     END SUBROUTINE SILS_factorize

!-*-*-*-*-*-*-*-*-*-   S I L S _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

     SUBROUTINE SILS_solve( MATRIX, FACTORS, X, CONTROL, SINFO )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Solve the linear system using the factors obtained in the factorization
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SMT_type ), INTENT( IN ) :: MATRIX
     TYPE( SILS_factors ), INTENT( IN ) :: FACTORS
     REAL ( KIND = wp ), INTENT( INOUT ) :: X( FACTORS%n )
     TYPE( SILS_control ), INTENT( IN ) :: CONTROL
     TYPE( SILS_sinfo ), INTENT( OUT ) :: SINFO

!  Local variables

     INTEGER :: la, size_factors_iw
     INTEGER :: ICNTL( 30 ), INFO( 20 )

!  Transfer CONTROL parameters

     INTEGER :: FACTORS_iw1( FACTORS%dim_iw1 )
     REAL ( KIND = wp ) :: FACTORS_w( FACTORS%maxfrt )

     ICNTL( 1 ) = CONTROL%lp
     ICNTL( 2 ) = CONTROL%mp
     ICNTL( 3 ) = CONTROL%ldiag
     ICNTL( 4 : 30 ) = CONTROL%ICNTL( 4 : 30 )

     IF ( CONTROL%pivoting == 4 ) THEN
       la = SIZE( FACTORS%val ) - MATRIX%n
     ELSE
       la = SIZE( FACTORS%val )
     END IF

     size_factors_iw = SIZE( FACTORS%iw )
     CALL MA27CD( FACTORS%n, FACTORS%val( : la ), la, FACTORS%iw,              &
                  size_factors_iw, FACTORS_w, FACTORS%maxfrt, X,               &
                  FACTORS_iw1, FACTORS%dim_iw1, ICNTL, INFO )
     SINFO%flag = INFO( 1 )
     SINFO%stat = 0
     SINFO%cond = - 1.0 ; SINFO%cond2 = - 1.0
     SINFO%berr = - 1.0 ; SINFO%berr2 = - 1.0
     SINFO%error = - 1.0

     RETURN

!  End of SILS_solve

     END SUBROUTINE SILS_solve

!-*-**-*-   S I L S _ S O L V E _ M U L T I P L E   S U B R O U T I N E   -*-*-*-

     SUBROUTINE SILS_solve_multiple( MATRIX, FACTORS, X, CONTROL, SINFO )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Solve the linear system using the factors obtained in the factorization
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SMT_type ), INTENT( IN ) :: MATRIX
     TYPE( SILS_factors ), INTENT( IN ) :: FACTORS
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : , : ) :: X
     TYPE( SILS_control ), INTENT( IN ) :: CONTROL
     TYPE( SILS_sinfo ), INTENT( OUT ) :: SINFO

!  Local variables

     INTEGER :: la, i, size_factors_iw

     INTEGER :: ICNTL( 30 ), INFO( 20 )

!  Transfer CONTROL parameters

     INTEGER :: FACTORS_iw1( FACTORS%dim_iw1 )
     REAL ( KIND = wp ) :: FACTORS_w( FACTORS%maxfrt )

     ICNTL( 1 ) = CONTROL%lp
     ICNTL( 2 ) = CONTROL%mp
     ICNTL( 3 ) = CONTROL%ldiag
     ICNTL( 4 : 30 ) = CONTROL%ICNTL( 4 : 30 )

     IF ( CONTROL%pivoting == 4 ) THEN
       la = SIZE( FACTORS%val ) - MATRIX%n
     ELSE
       la = SIZE( FACTORS%val )
     END IF

     DO i = 1, SIZE( X, 2 )
       size_factors_iw = SIZE( FACTORS%iw )
       CALL MA27CD( FACTORS%n, FACTORS%val( : la ), la, FACTORS%iw,            &
                    size_factors_iw, FACTORS_w, FACTORS%maxfrt, X( : , i ),    &
                    FACTORS_iw1, FACTORS%dim_iw1, ICNTL, INFO )
     END DO
     SINFO%flag = INFO( 1 )
     SINFO%stat = 0
     SINFO%cond = - 1.0 ; SINFO%cond2 = - 1.0
     SINFO%berr = - 1.0 ; SINFO%berr2 = - 1.0
     SINFO%error = - 1.0

     RETURN

!  End of SILS_solve_multiple

     END SUBROUTINE SILS_solve_multiple

!-*-*-*-*-  S I L S _ S O L V E  _ R E F I N E   S U B R O U T I N E   -*-*-*-

     SUBROUTINE SILS_solve_refine( MATRIX, FACTORS, X, CONTROL, SINFO, RHS )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Solve the linear system using the factors obtained in the factorization
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SMT_type ), INTENT( IN ) :: MATRIX
     TYPE( SILS_factors ), INTENT( INOUT ) :: FACTORS
     REAL ( KIND = wp ), INTENT( INOUT ) :: X( FACTORS%n )
     TYPE( SILS_control ), INTENT( IN ) :: CONTROL
     TYPE( SILS_sinfo ), INTENT( OUT ) :: SINFO
     REAL ( KIND = wp ), INTENT( IN ) :: RHS( FACTORS%n )

!  Local variables

     INTEGER :: la, size_factors_iw
     INTEGER :: ICNTL( 30 ), INFO( 20 )
!    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: FACTORS_w
!    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: FACTORS_r

!    INTEGER :: FACTORS_iw1( FACTORS%dim_iw1 )
!    REAL ( KIND = wp ) :: FACTORS_w( FACTORS%maxfrt )
     REAL ( KIND = wp ) :: FACTORS_r( FACTORS%n )

!  Transfer CONTROL parameters

     ICNTL( 1 ) = CONTROL%lp
     ICNTL( 2 ) = CONTROL%mp
     ICNTL( 3 ) = CONTROL%ldiag
     ICNTL( 4 : 30 ) = CONTROL%ICNTL( 4 : 30 )

     size_factors_iw = SIZE( FACTORS%iw )
     IF ( CONTROL%pivoting == 4 ) THEN
       la = SIZE( FACTORS%val ) - MATRIX%n
       CALL SILS_residual( MATRIX%val, MATRIX%row, MATRIX%col, FACTORS_r,      &
                           DIAG = FACTORS%val( la + 1 : ) )
       CALL MA27CD( FACTORS%n, FACTORS%val( : la ), la, FACTORS%iw,            &
                    size_factors_iw, FACTORS%w, FACTORS%maxfrt, FACTORS_r,     &
                    FACTORS%iw1, FACTORS%dim_iw1, ICNTL, INFO )
     ELSE
       la = SIZE( FACTORS%val )
       CALL SILS_residual( MATRIX%val, MATRIX%row, MATRIX%col, FACTORS_r )
       CALL MA27CD( FACTORS%n, FACTORS%val, la, FACTORS%iw,                    &
                    size_factors_iw, FACTORS%w, FACTORS%maxfrt, FACTORS_r,     &
                    FACTORS%iw1, FACTORS%dim_iw1, ICNTL, INFO )
     END IF

     SINFO%flag = INFO( 1 )
     SINFO%stat = 0
     SINFO%cond = - 1.0 ; SINFO%cond2 = - 1.0
     SINFO%berr = - 1.0 ; SINFO%berr2 = - 1.0
     SINFO%error = - 1.0
     X = X - FACTORS_r

     RETURN

     CONTAINS

       SUBROUTINE SILS_residual( A, IRN, ICN, R, DIAG )

!  ====================
!  Calculate r = Ax - b
!  ====================

!  Dummy arguments

       INTEGER, DIMENSION( : ) :: IRN, ICN
       REAL ( KIND = wp ), DIMENSION( : ) :: A, R
       REAL ( KIND = wp ), OPTIONAL, DIMENSION( : ) :: DIAG

!  Local variables

       INTEGER :: i, j, k

       R( : MATRIX%n ) = - RHS( : MATRIX%n )
       DO k = 1, MATRIX%ne
         i = IRN( k )
         j = ICN( k )
         IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= MATRIX%n ) THEN
           R( i ) = R( i ) + A( k ) * X( j )
           IF ( i /= j ) R( j ) = R( j ) + A( k ) * X( i )
         END IF
       END DO
       IF ( PRESENT( DIAG ) ) THEN
         R = R + DIAG * X
       END IF

       RETURN

!  End of SILS_residual

       END SUBROUTINE SILS_residual

     END SUBROUTINE SILS_solve_refine

!-  S I L S _ S O L V E  _ R E F I N E _ M U T I P L E    S U B R O U T I N E  -

     SUBROUTINE SILS_solve_refine_multiple( MATRIX, FACTORS, X, CONTROL,       &
                                            SINFO, RHS )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Solve the linear system using the factors obtained in the factorization
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SMT_type ), INTENT( IN ) :: MATRIX
     TYPE( SILS_factors ), INTENT( INOUT ) :: FACTORS
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : , : ) :: X
     TYPE( SILS_control ), INTENT( IN ) :: CONTROL
     TYPE( SILS_sinfo ), INTENT( OUT ) :: SINFO
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : , : ) :: RHS

!  Local variables

     INTEGER :: i

     DO i = 1, SIZE( X, 2 )
       CALL SILS_solve_refine( MATRIX, FACTORS, X( : , i ), CONTROL, SINFO,   &
                               RHS( : , i ) )
     END DO

     RETURN

     END SUBROUTINE SILS_solve_refine_multiple

!-*-*-*-*-*-*-*-   S I L S _ F I N A L I Z E  S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE SILS_finalize( FACTORS, CONTROL, info )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Deallocate all currently allocated arrays
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SILS_factors ), INTENT( INOUT ) :: FACTORS
     TYPE( SILS_control ), INTENT( IN ) :: CONTROL
     INTEGER, INTENT( OUT ) :: info

     INTEGER :: dealloc_stat

     info = 0

     IF ( ALLOCATED( FACTORS%keep ) ) THEN
       DEALLOCATE( FACTORS%keep, STAT = dealloc_stat )
       IF ( dealloc_stat /= 0 ) info = dealloc_stat
     END IF

     IF ( ALLOCATED( FACTORS%iw ) ) THEN
       DEALLOCATE( FACTORS%iw, STAT = dealloc_stat )
       IF ( dealloc_stat /= 0 ) info = dealloc_stat
     END IF

     IF ( ALLOCATED( FACTORS%val ) ) THEN
       DEALLOCATE( FACTORS%val, STAT = dealloc_stat )
       IF ( dealloc_stat /= 0 ) info = dealloc_stat
     END IF

     IF ( ALLOCATED( FACTORS%w ) ) THEN
       DEALLOCATE( FACTORS%w, STAT = dealloc_stat )
       IF ( dealloc_stat /= 0 ) info = dealloc_stat
     END IF

     IF ( ALLOCATED( FACTORS%r ) ) THEN
       DEALLOCATE( FACTORS%r, STAT = dealloc_stat )
       IF ( dealloc_stat /= 0 ) info = dealloc_stat
     END IF

     IF ( ALLOCATED( FACTORS%iw1 ) ) THEN
       DEALLOCATE( FACTORS%iw1, STAT = dealloc_stat )
       IF ( dealloc_stat /= 0 ) info = dealloc_stat
     END IF

     IF ( info /= 0 ) THEN
       IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                           &
         WRITE( CONTROL%lp, '( A, I0 )')                                       &
       ' Error return from SILS_finalize: deallocate failed with STAT=', info
     END IF

     RETURN

!  End of SILS_finalize

     END SUBROUTINE SILS_finalize

!-*-*-*-*-*-*-*-   S I L S _ E N Q U I R E  S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SILS_enquire( FACTORS, PERM, PIVOTS, D, PERTURBATION )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Interogate the factorization to obtain additional information
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SILS_factors ), INTENT( IN ) :: FACTORS
     INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( FACTORS%n ) :: PIVOTS
     INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( FACTORS%n ) :: PERM
     REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( 2, FACTORS%n ) :: D
     REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL,                              &
       DIMENSION( FACTORS%n ) :: PERTURBATION

!  Local variables

     INTEGER :: block, i, ka, kd, kp, kw, ncols, nrows

     IF ( present( PERM ) ) THEN
       PERM = FACTORS%keep( 1 : FACTORS%n )
     ENDIF

     IF ( present( PERTURBATION ) ) THEN
       IF ( FACTORS%pivoting == 4 ) THEN
         PERTURBATION = FACTORS%val( SIZE( FACTORS%val ) -  FACTORS%n + 1 : )
       ELSE
         PERTURBATION = zero
       END IF
     ENDIF

     ka = 1 ; kd = 0 ; kp = 0 ; kw = 2
     IF ( PRESENT( D ) ) D = zero
     DO block = 1, ABS( FACTORS%iw( 1 ) )
       IF ( FACTORS%pivoting == 1 ) THEN
         ncols = FACTORS%iw( kw )
         IF ( ncols > 0 ) THEN
           kw = kw + 1
           nrows = FACTORS%iw( kw )
         ELSE
           ncols = - ncols
           nrows = 1
         END IF
       ELSE
         ncols = 1
         nrows = 1
       END IF
       IF ( PRESENT( PIVOTS ) ) THEN
          PIVOTS( kp + 1 : kp + nrows ) = FACTORS%iw( kw + 1 : kw + nrows )
          kp = kp + nrows
       END IF
       IF ( PRESENT( D ) ) THEN
         DO i = 1, nrows
           kd = kd + 1
           D( 1, kd ) = FACTORS%val( ka )
           IF ( FACTORS%iw( kw + i ) < 0 ) D( 2, kd ) = FACTORS%val( ka + 1 )
           ka = ka + ncols + 1 - i
         END DO
       END IF
       kw = kw + ncols + 1
     END DO

     RETURN

!  End of SILS_enquire

     END SUBROUTINE SILS_enquire

!-*-*-*-*-*-*-*-   S I L S _ A L T E R _ D   S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SILS_alter_d( FACTORS, D, info )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Alter the diagonal blocks
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SILS_factors ), INTENT( INOUT ) :: FACTORS
     REAL ( KIND = wp ), INTENT( IN ) :: D( 2, FACTORS%n )
     INTEGER, INTENT( OUT ) :: info

!  Local variables

     INTEGER :: block, i, ka, kd, kw, ncols, nrows

     info = 0
     ka = 1 ; kd = 0 ; kw = 2
     DO block = 1, ABS( FACTORS%iw( 1 ) )
       ncols = FACTORS%iw( kw )
       IF ( ncols > 0 ) THEN
         kw = kw + 1
         nrows = FACTORS%iw( kw )
       ELSE
         ncols = - ncols
         nrows = 1
       END IF
       DO i = 1, nrows
         kd = kd + 1
         FACTORS%val( ka ) = D( 1, kd )
         IF ( FACTORS%iw( kw + i ) < 0 ) THEN
           FACTORS%val( ka + 1 ) = D( 2, kd )
         ELSE
           IF ( D( 2, kd ) /= zero ) info = kd
         END IF
         ka = ka + ncols + 1 - i
       END DO
       kw = kw + ncols + 1
     END DO
     RETURN

!  End of SILS_alter_d

     END SUBROUTINE SILS_alter_d

!-*-*-*-*-*-*-   S I L S _ P A R T _ S O L V E   S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE SILS_part_solve( FACTORS, CONTROL, part, X, info )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Solve a system involving individual factors
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE( SILS_factors ), INTENT( INOUT ) :: FACTORS
     TYPE( SILS_control ), INTENT( IN ) :: CONTROL
     CHARACTER, INTENT( IN ) :: part
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( FACTORS%n ) :: X
     INTEGER, INTENT( OUT ) :: info

!  Local variables

     INTEGER :: size_factors_val, size_factors_iw
     INTEGER :: ICNTL( 30 )

!  Transfer CONTROL parameters

     ICNTL( 1 ) = CONTROL%lp
     ICNTL( 2 ) = CONTROL%mp
     ICNTL( 3 ) = CONTROL%ldiag
     ICNTL( 4 : 30 ) = CONTROL%ICNTL( 4 : 30 )

     info = 0

     IF ( ALLOCATED( FACTORS%w ) ) THEN
       IF ( SIZE( FACTORS%w )<FACTORS%maxfrt ) THEN
         DEALLOCATE( FACTORS%w, STAT = info )
         IF ( info /= 0 ) THEN
           IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                       &
             WRITE( CONTROL%lp, 2000 ) 'deallocate', info
           RETURN
         END IF
         ALLOCATE( FACTORS%w( FACTORS%maxfrt ), STAT = info )
         IF ( info /= 0 ) THEN
           IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                       &
             WRITE( CONTROL%lp, 2000 ) 'allocate', info
           RETURN
         END IF
       END IF
     ELSE
       ALLOCATE( FACTORS%w( FACTORS%maxfrt ), STAT = info )
       IF ( info /= 0 ) THEN
         IF ( CONTROL%ldiag > 0 .AND. CONTROL%lp > 0 )                         &
           WRITE( CONTROL%lp, 2000 ) 'allocate', info
         RETURN
       END IF
     END IF

!  Solution involving the lower-triangular factor

     size_factors_val = SIZE( FACTORS%val )
     size_factors_iw =  SIZE( FACTORS%iw )
     IF ( part=='L') THEN
       CALL MA27QD( FACTORS%n, FACTORS%val, size_factors_val,                  &
                    FACTORS%iw( 2 : ), size_factors_iw - 1,                    &
                    FACTORS%w, FACTORS%maxfrt, X, FACTORS%iw1,                 &
                    ABS( FACTORS%iw( 1 ) ), FACTORS%latop, ICNTL )

!  Solution involving the block-diagonal factor

     ELSE IF ( part=='D') THEN
       CALL SILS_solve_d( FACTORS%val,size_factors_val,                        &
                          FACTORS%iw, size_factors_iw )

!  Solution involving the upper-triangular factor

     ELSE IF ( part=='U') THEN
       CALL SILS_solve_u( FACTORS%n, FACTORS%val, size_factors_val,            &
                          FACTORS%iw, size_factors_iw, FACTORS%iw1,            &
                          FACTORS%w, FACTORS%maxfrt, FACTORS%latop,            &
                          ICNTL )
     END IF

     RETURN

!  Non-executable statement

 2000 FORMAT( ' Error return from SILS_part_solve: ', A, ' failed',           &
              ' with STAT = ', I0 )

     CONTAINS

!-*-*-*-*-*-*-*-   S I L S _ S O L V E _ d   S U B R O U T I N E   -*-*-*-*-*-*-

       SUBROUTINE SILS_solve_d( a, la, iw, liw )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Solve a system involving the block-diagonal factors
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

       INTEGER :: la, liw
       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( la ) :: A
       INTEGER, INTENT( IN ), DIMENSION( liw ) :: IW

!  Local variables

       INTEGER :: block, i, j, j1, ka, kw, ncols, nrows
       REAL ( KIND = wp ) :: xj

       ka = 1 ; kw = 2
       DO block = 1, ABS( IW( 1 ) )
         ncols = IW( kw )
         IF ( ncols > 0 ) THEN
           kw = kw + 1
           nrows = IW( kw )
         ELSE
           ncols = - ncols
           nrows = 1
         END IF
         i = 1
         DO
           IF ( i > nrows ) EXIT
           j = IW( kw + i )
           IF ( j > 0 ) THEN
              X( j ) = A( ka ) * X( j )
           ELSE
             j = - j
             j1 = IW( kw + i + 1 )
             xj = X( j )
             X( j ) = A( ka ) * xj + A( ka + 1 ) * X( j1 )
             X( j1 ) = A( ka + 1 ) * xj + A( ka + ncols + 1 - i ) * X( j1 )
             ka = ka + ncols + 1 - i
             i = i + 1
           END IF
           ka = ka + ncols + 1 - i
           i = i + 1
         END DO
         kw = kw + ncols + 1
       END DO

       RETURN

!  End of SILS_solve_d

       END SUBROUTINE SILS_solve_d

!-*-*-*-*-*-*-*-   S I L S _ S O L V E _ u   S U B R O U T I N E   -*-*-*-*-*-*-

       SUBROUTINE SILS_solve_u( n, A, la, IW, liw, IW2, W, lw, latop, ICNTL )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  Solve a system involving the upper triangular factors
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

       INTEGER, INTENT( IN )  :: n, la, liw, lw, latop
       INTEGER, DIMENSION( liw ), INTENT( IN )  :: IW
       INTEGER, DIMENSION( ABS( IW( 1 ) ) ), INTENT( IN ) :: IW2
       REAL ( KIND = wp ), DIMENSION( la ), INTENT( IN ) :: A
       REAL ( KIND = wp ), DIMENSION( lw ), INTENT( OUT )  :: W
       INTEGER, INTENT( IN ) :: ICNTL( 30 )

!  Local variables

       INTEGER :: apos, apos2, i1x, i2x, iblk, ifr, iipiv, iix, ilvl, ipiv
       INTEGER :: ipos, ix, ist, j, j1, j2, jj, jj1, jj2, jpiv, jpos, k
       INTEGER :: liell, loop, npiv
       REAL ( KIND = wp ) :: x1, x2
       INTRINSIC IABS, MIN0
       INTEGER, PARAMETER :: ifrlvl = 5

       apos = latop + 1
       npiv = 0
       iblk = ABS( IW( 1 ) ) + 1

! Run through block pivot rows in the reverse order

       DO loop = 1, n
         IF ( npiv > 0 ) GO TO 10
         iblk = iblk - 1
         IF ( iblk < 1 ) EXIT
         ipos = IW2( iblk )
         liell = - IW( ipos + 1 )
         npiv = 1
         IF ( liell <= 0 ) THEN
           liell = - liell
           ipos = ipos + 1
           npiv = IW( ipos + 1 )
         END IF
         jpos = ipos + npiv
         j2 = ipos + liell
         ilvl = MIN0( 10, npiv ) + 10
         IF ( liell < ICNTL( ifrlvl + ilvl ) ) GO TO 10

! Perform operations using direct addressing

         j1 = ipos + 1

! Load appropriate components of X into W

         ifr = 0
         DO jj = j1, j2
           j = IABS( IW( jj + 1 ) + 0 )
           ifr = ifr + 1
           W( ifr ) = X( j )
         END DO

! Perform eliminations

         jpiv = 1
         DO iipiv = 1, npiv
           jpiv = jpiv - 1
           IF ( jpiv == 1 ) CYCLE
           ipiv = npiv - iipiv + 1

           IF ( ipiv /= 1 ) THEN

! Perform back-substitution operations with 2 by 2 pivot

             IF ( IW( jpos ) < 0 ) THEN
               jpiv = 2
               apos2 = apos - ( liell + 1 - ipiv )
               apos = apos2 - ( liell + 2 - ipiv )
               ist = ipiv + 1
               x1 = W( ipiv - 1 )
               x2 = W( ipiv )
               jj1 = apos + 2
               jj2 = apos2 + 1
               DO j = ist, liell
                 x1 = x1 + W( j ) * A( jj1 )
                 x2 = x2 + W( j ) * A( jj2 )
                 jj1 = jj1 + 1
                 jj2 = jj2 + 1
               END DO
               W( ipiv - 1 ) = x1
               W( ipiv ) = x2
               jpos = jpos - 2
               CYCLE
             END IF
           END IF

! Perform back-substitution using 1 by 1 pivot

           jpiv = 1
           apos = apos - ( liell + 1 - ipiv )
           ist = ipiv + 1
           x1 = W( ipiv )
           jj1 = apos + 1
           DO j = ist, liell
             x1 = x1 + A( jj1 ) * W( j )
             jj1 = jj1 + 1
           END DO
           W( ipiv ) = x1
           jpos = jpos - 1
         END DO

! Reload working vector into solution vector

         ifr = 0
         DO jj = j1, j2
           j = IABS( IW( jj + 1 ) + 0 )
           ifr = ifr + 1
           X( j ) = W( ifr )
         END DO
         npiv = 0
         CYCLE

! Perform operations using indirect addressing

   10    CONTINUE
         IF ( npiv /= 1 ) THEN

! Perform operations with 2 by 2 pivot

           IF ( IW( jpos ) < 0 ) THEN
             npiv = npiv - 2
             apos2 = apos - ( j2 - jpos + 1 )
             apos = apos2 - ( j2 - jpos + 2 )
             i1x = - IW( jpos )
             i2x = IW( jpos + 1 )
             x1 = X( i1x )
             x2 = X( i2x )
             j1 = jpos + 1
             jj1 = apos + 2
             jj2 = apos2 + 1
             DO j = j1, j2
               ix = IABS( IW( j + 1 ) + 0 )
               x1 = x1 + X( ix ) * A( jj1 )
               x2 = x2 + X( ix ) * A( jj2 )
               jj1 = jj1 + 1
               jj2 = jj2 + 1
             END DO
             X( i1x ) = x1
             X( i2x ) = x2
             jpos = jpos - 2
             CYCLE
           END IF
         END IF

! Perform back-substitution using 1 by 1 pivot

         npiv = npiv - 1
         apos = apos - ( j2 - jpos + 1 )
         iix = IW( jpos + 1 )
         x1 = X( iix )
         j1 = jpos + 1
         k = apos + 1
         DO j = j1, j2
           ix = IABS( IW( j + 1 ) + 0 )
           x1 = x1 + A( k ) * X( ix )
           k = k + 1
         END DO
         X( iix ) = x1
         jpos = jpos - 1
       END DO

       RETURN

!  End of SILS_solve_u

       END SUBROUTINE SILS_solve_u

!  End of SILS_part_solve

     END SUBROUTINE SILS_part_solve

   END MODULE GALAHAD_SILS_double
