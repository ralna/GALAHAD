! THIS VERSION: GALAHAD 3.1 - 16/06/2018 AT 13:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N Q P B _ D A T A  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.0. January 19th 2006

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNQPB_DATA

!    -----------------------------------------------------------
!    | Main program for the problem-data-file interface to QPB,|
!    | an interior-point algorithm for quadratic programming   |
!    -----------------------------------------------------------

!NOT95USE GALAHAD_CPU_time
   USE GALAHAD_CLOCK
   USE GALAHAD_QPT_double
   USE GALAHAD_RPD_double
   USE GALAHAD_LSQP_double
   USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
   USE GALAHAD_SMT_double, only: SMT_put
   USE GALAHAD_QPB_double
   USE GALAHAD_PRESOLVE_double
   USE GALAHAD_SPECFILE_double
   USE GALAHAD_STRING_double, ONLY: STRING_upper_word
   USE GALAHAD_COPYRIGHT
   USE SCALING
   USE GALAHAD_SYMBOLS,                                                        &
       ACTIVE                => GALAHAD_ACTIVE,                                &
       TRACE                 => GALAHAD_TRACE,                                 &
       DEBUG                 => GALAHAD_DEBUG,                                 &
       GENERAL               => GALAHAD_GENERAL,                               &
       ALL_ZEROS             => GALAHAD_ALL_ZEROS
   USE GALAHAD_SCALE_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 5

!  --------------------------------------------
!
!  Solve the quadratic program
!
!     minimize     1/2 x(T) H x + g(T) x + f
!
!     subject to     c_l <= A x <= c_u
!                    x_l <=  x <= x_u
!
!  using the GALAHAD package GALAHAD_QPB
!
!  --------------------------------------------

!  ****************************************************************************

!  The data should be input in a file on unit 5. The data is in
!  free format (blanks separate values),but must occur in the order
!  given here. Any blank lines, or lines starting with any of the
!  characters "!", "%" or "#" are ignored. Each term in "quotes"
!  denotes a required value. Any strings beyond those required on a
!  given lines will be regarded as comments and ignored.

!  "problem name"
!  "problem type" one of LP (an LP), BQP (a bound-constrained QP) or QP (a QP)
!  "number variables"
!  "number general linear constraints"
!  "number of nonzeros in lower traingle of H"
!  "row" "column" "value" for each entry of H (if any), one triple on each line
!  "default value for entries in g"
!  "number of non-default entries in g"
!  "index" "value" for each non-default term in g (if any), one pair per line
!  "value of f"
!  "number of nonzeros in A"
!  "row" "column" "value" for each entry of A (if any), one triple on each line
!  "default value for entries in c_l"
!  "value for infinity" for bounds - any bound greater than or equal to this
!     in absolute value is infinite
!  "number of non-default entries in c_l"
!  "index" "value" for each non-default term in c_l (if any), one pair per line
!  "default value for entries in c_u"
!  "number of non-default entries in c_u"
!  "index" "value" for each non-default term in c_u (if any), one pair per line
!  "default value for entries in x_l"
!  "number of non-default entries in x_l"
!  "index" "value" for each non-default term in x_l (if any), one pair per line
!  "default value for entries in x_u"
!  "number of non-default entries in x_u"
!  "index" "value" for each non-default term in x_u (if any), one pair per line
!  "default value for starting value for variables x"
!  "number of non-default starting entries in x"
!  "index" "value" for each non-default term in x (if any), one pair per line
!  "default value for starting value for Lagrange multipliers y for constraints"
!  "number of non-default starting entries in y"
!  "index" "value" for each non-default term in y (if any), one pair per line
!  "default value for starting value for dual varibales z for simple bounds"
!  "number of non-default starting entries in z"
!  "index" "value" for each non-default term in z (if any), one pair per line
!  "number of non-default names of variables" -default for variable i is "i"
!  "index" "name" for each non-default name for variable x_i with index i
!     (if any)
!  "number of non-default names of constraints" -default for constraint i is "i"
!  "index" "name" for each non-default name for constraint with index i
!     (if any)

!  For full details, see README.data-file

!  *****************************************************************************

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: i, j, l, n, m, A_ne, H_ne, ir, ic, liw, iores
!     INTEGER :: npm, np1
      INTEGER :: status, mfixed, mdegen, iter, nfacts, nfixed, ndegen, mequal
      INTEGER :: alloc_stat, newton, nmods, mredun, smt_stat
      REAL :: time, timeo, times, timet, timep1, timep2, timep3, timep4
      REAL ( KIND = wp ) :: clock, clocko, clocks, clockt
      REAL ( KIND = wp ) :: qfval, stopr, dummy
      REAL ( KIND = wp ) :: res_c, res_k, max_cs
      LOGICAL :: filexx, phase1, printo, printe
      LOGICAL :: warn_h_not_diagonal, warn_h_indefinite
      LOGICAL :: center = .FALSE.
!     LOGICAL :: ldummy

!  Functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 24
      CHARACTER ( LEN = 16 ) :: specname = 'RUNQPB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNQPB.SPC'

!  The default values for QPB could have been set as:

! BEGIN RUNQPB SPECIFICATIONS (DEFAULT)
!  write-problem-data                        NO
!  problem-data-file-name                    QPA.data
!  problem-data-file-device                  26
!  write-initial-sif                         NO
!  initial-sif-file-name                     INITIAL.SIF
!  initial-sif-file-device                   51
!  least-squares-qp                          NO
!  scale-problem                             0
!  pre-solve-problem                         YES
!  write-presolved-sif                       NO
!  presolved-sif-file-name                   PRESOLVE.SIF
!  presolved-sif-file-device                 53
!  write-scaled-sif                          NO
!  scaled-sif-file-name                      SCALED.SIF
!  scaled-sif-file-device                    58
!  solve-problem                             YES
!  print-full-solution                       NO
!  write-solution                            NO
!  solution-file-name                        QPASOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  QPARES.d
!  result-summary-file-device                47
!  perturb-bounds-by                         0.0
! END RUNQPB SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: scale = 0
      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
      INTEGER :: pfiledevice = 53
      INTEGER :: qfiledevice = 58
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_presolved_sif  = .FALSE.
      LOGICAL :: write_scaled_sif     = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'QPB.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: pfilename = 'PRESOLVE.SIF'
      CHARACTER ( LEN = 30 ) :: qfilename = 'SCALED.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'QPBRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'QPBSOL.d'
      LOGICAL :: islsqp = .FALSE.
      LOGICAL :: do_presolve = .TRUE.
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = wp ) :: pert_bnd = zero

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 10 ) ::  pname = '          '
      CHARACTER ( LEN = 30 ) :: sls_solv

!  Arrays

      TYPE ( RPD_inform_type ) :: RPD_inform
      TYPE ( LSQP_control_type ) :: LSQP_control
      TYPE ( LSQP_inform_type ) :: LSQP_inform
      TYPE ( SCALING_control_type ) :: control
      TYPE ( QPB_data_type ) :: data
      TYPE ( QPB_control_type ) :: QPB_control
      TYPE ( QPB_inform_type ) :: QPB_inform
      TYPE ( QPT_problem_type ) :: prob
      TYPE ( PRESOLVE_control_type ) :: PRE_control
      TYPE ( PRESOLVE_inform_type )  :: PRE_inform
      TYPE ( PRESOLVE_data_type )    :: PRE_data
      TYPE ( SCALE_trans_type ) :: SCALE_trans
      TYPE ( SCALE_data_type ) :: SCALE_data
      TYPE ( SCALE_control_type ) :: SCALE_control
      TYPE ( SCALE_inform_type ) :: SCALE_inform

!  Allocatable arrays

      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: AY, HX
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW

!  Open the data input file

      OPEN( input, FORM = 'FORMATTED', STATUS = 'OLD'  )
      REWIND input

      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

      CALL RPD_read_problem_data( input, prob, RPD_inform )
      IF ( RPD_inform%status < 0 ) THEN
        SELECT CASE( RPD_inform%status )
        CASE ( - 2 )
          WRITE( out, 2150 ) RPD_inform%bad_alloc, RPD_inform%alloc_status
        CASE ( - 3 )
          WRITE( out, "( ' ** premature end of problem-input file',            &
         &  ' encountered on line ', I0, ' io_status = ', I0 )" )              &
            RPD_inform%line, RPD_inform%io_status
        CASE ( - 4 )
          WRITE( out, "( ' ** read error of problem-input file occured',       &
         &  ' on line ', I0, ' io_status = ', I0 )" )                          &
            RPD_inform%line, RPD_inform%io_status
        CASE DEFAULT
          WRITE( out, "( ' ** error reported when reading qplib file by',      &
         &     ' RPD, status = ', I0 )" )  RPD_inform%status
        END SELECT
        STOP
      END IF
      pname = TRANSFER( prob%name, pname )

!  check that the problem variables are continuous

      SELECT CASE ( RPD_inform%p_type( 2 : 2 ) )
      CASE ( 'C' )
      CASE DEFAULT
        WRITE( out, "( /, ' ** Problem ', A, ', some variables are not',       &
       & ' continuous. Stopping' )" ) TRIM( pname )
        STOP
      END SELECT

!  check that the problem is a QP

      SELECT CASE ( RPD_inform%p_type( 3 : 3 ) )
      CASE ( 'N', 'B', 'L' )
      CASE DEFAULT
        WRITE( out, "( /, ' ** Problem ', A, ', constraints are not',          &
       &  ' linear. Stopping' )" ) TRIM( pname )
        STOP
      END SELECT

      n = prob%n
      m = prob%m
      H_ne = prob%H%ne
      A_ne = prob%A%ne

!  Allocate derived types

      ALLOCATE( prob%X0( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X0', alloc_stat
        STOP
      END IF
      prob%X0 = prob%X

!  Set up the initial estimate of the solution and
!  right-hand-side of the Kuhn-Tucker system.

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ),                                    &
                           MAX( prob%X_l( : n ), prob%X( : n ) ) )

!  Allocate and initialize dual variables.

      liw = MAX( m, n ) + 1
      ALLOCATE( prob%A%ptr( m + 1 ), prob%H%ptr( n + 1 ) )
      ALLOCATE( IW( liw ) )

!     WRITE( 27, "( ( 3( 2I6, ES12.4 ) ) )" )                                  &
!        ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, H_ne )
!     WRITE( 26, "( ' H_row ', /, ( 10I6 ) )" ) prob%H%row( : H_ne )
!     WRITE( 26, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
!     WRITE( 26, "( ' H_val ', /, ( 5ES12.4 ) )" ) prob%H%val( : H_ne )

!  Transform A to row storage format

      IF ( A_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( m, n, A_ne, prob%A%row, prob%A%col, A_ne,   &
                                   prob%A%val, prob%A%ptr, m + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%A%ptr = 0
      END IF

!  Same for H

      IF ( H_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( n, n, H_ne, prob%H%row, prob%H%col, H_ne,   &
                                   prob%H%val, prob%H%ptr, n + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%H%ptr = 0
      END IF

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( prob%A%row, prob%H%row )
      DEALLOCATE( IW )
      ALLOCATE( prob%A%row( 0 ), prob%H%row( 0 ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, "( ' whoa there - allocate error ', i6 )" ) alloc_stat ; STOP
      END IF

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )

!     WRITE( out, "( ' maximum element of A = ', ES12.4,                       &
!    &                ' maximum element of H = ', ES12.4 )" )                  &
!      MAXVAL( ABS( prob%A%val( : A_ne ) ) ),                                  &
!      MAXVAL( ABS( prob%H%val( : H_ne ) ) )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times ) ;  CALL CLOCK_time( clocks )

!  ------------------ Open the specfile for runqpb ----------------

      OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD' )

!   Define the keywords

      spec( 1 )%keyword = 'write-problem-data'
      spec( 2 )%keyword = 'problem-data-file-name'
      spec( 3 )%keyword = 'problem-data-file-device'
      spec( 4 )%keyword = 'write-initial-sif'
      spec( 5 )%keyword = 'initial-sif-file-name'
      spec( 6 )%keyword = 'initial-sif-file-device'
      spec( 7 )%keyword = 'least-squares-qp'
      spec( 8 )%keyword = 'scale-problem'
      spec( 9 )%keyword = 'pre-solve-problem'
      spec( 10 )%keyword = 'write-presolved-sif'
      spec( 11 )%keyword = 'presolved-sif-file-name'
      spec( 12 )%keyword = 'presolved-sif-file-device'
      spec( 13 )%keyword = 'solve-problem'
      spec( 14 )%keyword = 'print-full-solution'
      spec( 15 )%keyword = 'write-solution'
      spec( 16 )%keyword = 'solution-file-name'
      spec( 17 )%keyword = 'solution-file-device'
      spec( 18 )%keyword = 'write-result-summary'
      spec( 19 )%keyword = 'result-summary-file-name'
      spec( 20 )%keyword = 'result-summary-file-device'
      spec( 21 )%keyword = 'perturb-bounds-by'
      spec( 22 )%keyword = 'write-scaled-sif'
      spec( 23 )%keyword = 'scaled-sif-file-name'
      spec( 24 )%keyword = 'scaled-sif-file-device'

!   Read the specfile

      CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

      CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
      CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
      CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
      CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
      CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 7 ), islsqp, errout )
      CALL SPECFILE_assign_integer( spec( 8 ), scale, errout )
      CALL SPECFILE_assign_logical( spec( 9 ), do_presolve, errout )
      CALL SPECFILE_assign_logical( spec( 10 ), write_presolved_sif, errout )
      CALL SPECFILE_assign_string ( spec( 11 ), pfilename, errout )
      CALL SPECFILE_assign_integer( spec( 12 ), pfiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 13 ), do_solve, errout )
      CALL SPECFILE_assign_logical( spec( 14 ), fulsol, errout )
      CALL SPECFILE_assign_logical( spec( 15 ), write_solution, errout )
      CALL SPECFILE_assign_string ( spec( 16 ), sfilename, errout )
      CALL SPECFILE_assign_integer( spec( 17 ), sfiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 18 ), write_result_summary, errout )
      CALL SPECFILE_assign_string ( spec( 19 ), rfilename, errout )
      CALL SPECFILE_assign_integer( spec( 20 ), rfiledevice, errout )
      CALL SPECFILE_assign_real( spec( 21 ), pert_bnd, errout )
      CALL SPECFILE_assign_logical( spec( 22 ), write_scaled_sif, errout )
      CALL SPECFILE_assign_string ( spec( 23 ), qfilename, errout )
      CALL SPECFILE_assign_integer( spec( 24 ), qfiledevice, errout )

!  Perturb bounds if required

      IF ( pert_bnd /= zero ) THEN
        DO i = 1, n
          IF (  prob%X_l( i ) /= prob%X_u( i ) ) THEN
            IF ( prob%X_l( i ) > - infinity )                                  &
              prob%X_l( i ) = prob%X_l( i ) - pert_bnd
            IF ( prob%X_u( i ) < infinity )                                    &
              prob%X_u( i ) = prob%X_u( i ) + pert_bnd
          END IF
        END DO

        DO i = 1, m
          IF (  prob%C_l( i ) /= prob%C_u( i ) ) THEN
            IF ( prob%C_l( i ) > - infinity )                                  &
              prob%C_l( i ) = prob%C_l( i ) - pert_bnd
            IF ( prob%C_u( i ) < infinity )                                    &
              prob%C_u( i ) = prob%C_u( i ) + pert_bnd
          END IF
        END DO
      END IF

!  If required, print out the (raw) problem data

      IF ( write_problem_data ) THEN
        INQUIRE( FILE = dfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( dfiledevice, FILE = dfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', IOSTAT = iores )
        ELSE
           OPEN( dfiledevice, FILE = dfilename, FORM = 'FORMATTED',            &
                  STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2160 ) iores, dfilename
          STOP
        END IF

        WRITE( dfiledevice, "( 'n, m = ', 2I6, ' obj = ', ES12.4 )" )          &
          n, m, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
        WRITE( dfiledevice, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : m )
        WRITE( dfiledevice, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : m )
        WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" ) prob%A%ptr( : m + 1 )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) prob%A%col( : A_ne )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                  &
          prob%A%val( : A_ne )
        WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" ) prob%H%ptr( : n + 1 )
        WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
        WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                  &
          prob%H%val( : H_ne )

        CLOSE( dfiledevice )
      END IF

!  If required, append results to a file

      IF ( write_result_summary ) THEN
        INQUIRE( FILE = rfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
        ELSE
           OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',            &
                 STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2160 ) iores, rfilename
          STOP
        END IF
        WRITE( rfiledevice, 2180 ) pname
      END IF

!  Set all default values, and override defaults if requested

      CALL SCALING_initialize( control )
      IF ( islsqp ) THEN
        CALL LSQP_initialize( data, LSQP_control, LSQP_inform )
        CALL LSQP_read_specfile( LSQP_control, input_specfile )
        phase1 = LSQP_control%just_feasible
        LSQP_control%restore_problem = 2
        control%print_level = LSQP_control%print_level
        control%out         = LSQP_control%out
        control%out_error   = LSQP_control%error
      ELSE
        CALL QPB_initialize( data, QPB_control, QPB_inform )
        CALL QPB_read_specfile( QPB_control, input_specfile )
        phase1 = QPB_control%LSQP_control%just_feasible
        IF ( phase1 ) THEN
          LSQP_control = QPB_control%LSQP_control
          control%print_level = LSQP_control%print_level
          control%out         = LSQP_control%out
          control%out_error   = LSQP_control%error
        ELSE
          control%print_level = QPB_control%print_level
          control%out         = QPB_control%out
          control%out_error   = QPB_control%error
        END IF
        center = QPB_control%center
        QPB_control%LSQP_control%pivot_tol = QPB_control%pivot_tol
        QPB_control%LSQP_control%pivot_tol_for_dependencies =                  &
          QPB_control%pivot_tol_for_dependencies
!       QPB_control%LSQP_control%maxit = 1
!       QPB_control%LSQP_control%print_level = 3
        QPB_control%restore_problem = 2
      END IF

      printo = out > 0 .AND. control%print_level > 0
      printe = out > 0 .AND. control%print_level >= 0

      WRITE( out, 2020 ) pname
      WRITE( out, 2200 ) n, m, A_ne, H_ne

      IF ( printo ) CALL COPYRIGHT( out, '2006' )

      IF ( phase1 ) THEN
        qfval = zero
        prob%G( : n ) = zero
        H_ne = 0
      END IF

!     IF ( .NOT. islsqp ) THEN
      IF ( islsqp ) THEN
        qfval = zero
        H_ne = 0
      END IF

!  If the preprocessor is to be used, or the problem to be output,
!  allocate sufficient space

      IF ( write_initial_sif .OR. do_presolve ) THEN

        ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%X_status = ACTIVE

        ALLOCATE( prob%C_status( m ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%C_status = ACTIVE

        ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'Z_lu', alloc_stat
          STOP
        END IF
        prob%Z_l( : n ) = - infinity
        prob%Z_u( : n ) =   infinity

        ALLOCATE( prob%Y_l( m ), prob%Y_u( m ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'C_lu', alloc_stat
          STOP
        END IF
        prob%Y_l( : m ) = - infinity
        prob%Y_u( : m ) =   infinity

!  Writes the initial SIF file, if needed

        IF ( write_initial_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,          &
                                 .FALSE., .FALSE., infinity )
          IF ( .NOT. ( do_presolve .OR. do_solve ) ) STOP
        END IF
      END IF

!  If required, scale the problem

      IF ( scale < 0 ) THEN
        CALL SCALE_get( prob, - scale, SCALE_trans, SCALE_data,                &
                        SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
        CALL SCALE_apply( prob, SCALE_trans, SCALE_data,                       &
                          SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
        IF ( write_scaled_sif )                                                &
          CALL QPT_write_to_sif( prob, pname, qfilename, qfiledevice,          &
                                 .FALSE., .FALSE., infinity )
      END IF

!  Presolve

      IF ( do_presolve ) THEN

        CALL CPU_TIME( timep1 )

!       set the control variables

        CALL PRESOLVE_initialize( PRE_control, PRE_inform, PRE_data )
        CALL PRESOLVE_read_specfile( input_specfile, PRE_control, PRE_inform )
        CLOSE( input_specfile )

        IF ( PRE_inform%status /= 0 ) STOP

!       Overide some defaults

        IF ( islsqp ) THEN
          PRE_control%infinity   = LSQP_control%infinity
          PRE_control%c_accuracy = ten * LSQP_control%stop_p
          PRE_control%z_accuracy = ten * LSQP_control%stop_d
        ELSE
          PRE_control%infinity   = QPB_control%infinity
          PRE_control%c_accuracy = ten * QPB_control%stop_p
          PRE_control%z_accuracy = ten * QPB_control%stop_d
        END IF

!  Call the presolver

        CALL PRESOLVE_apply( prob, PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from PRESOLVE (exitc =', I6, ')' )" ) &
            PRE_inform%status
          STOP
        END IF

        CALL CPU_TIME( timep2 )

        A_ne = MAX( 0, prob%A%ptr( prob%m + 1 ) - 1 )
        H_ne = MAX( 0, prob%H%ptr( prob%n + 1 ) - 1 )
        IF ( printo ) WRITE( out, 2300 ) prob%n, prob%m, A_ne, H_ne,           &
           timep2 - timep1, PRE_inform%nbr_transforms

!  If required, write a SIF file containing the presolved problem

        IF ( write_presolved_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, pfilename, pfiledevice,          &
                                 .FALSE., .FALSE., PRE_control%infinity )
        END IF
      END IF

!  Call the optimizer

      qfval = prob%f

      IF ( do_solve .AND. prob%n > 0 ) THEN

!  If required, scale the problem

        IF ( scale > 0 ) THEN
          CALL SCALE_get( prob, - scale, SCALE_trans, SCALE_data,              &
                          SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF
          CALL SCALE_apply( prob, SCALE_trans, SCALE_data,                     &
                            SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF

          IF ( write_scaled_sif )                                              &
            CALL QPT_write_to_sif( prob, pname, qfilename, qfiledevice,        &
                                   .FALSE., .FALSE., infinity )
        END IF

        CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )

!       IF ( .NOT. do_presolve ) THEN
!          prob%m = m
!          prob%n = n
!       END IF

        IF ( phase1 ) THEN
          IF ( printo ) WRITE( out, "( /, ' *** Solving the phase-1 problem' )")
          solv = ' LSQP'
          prob%gradient_kind = 0
          IF ( center ) THEN
            prob%Hessian_kind = 0
            DEALLOCATE( prob%X0 )
          ELSE IF ( scale > 0 ) THEN
            ALLOCATE( prob%WEIGHT( n ), STAT = alloc_stat )
            IF ( alloc_stat /= 0 ) THEN
              IF ( printe ) WRITE( out, 2150 ) 'WEIGHT', alloc_stat ; STOP
            END IF
            prob%Hessian_kind = 2
            prob%WEIGHT = SCALE_trans%X_scale
            prob%X0 = prob%X
          ELSE
            prob%Hessian_kind = 1
            prob%X0 = prob%X
          END IF
          CALL LSQP_solve( prob, data, LSQP_control, LSQP_inform )
          qfval = LSQP_inform%obj ; newton = LSQP_inform%nfacts ; nmods = 0
        ELSE IF ( islsqp ) THEN
          solv = ' LSQP'
          prob%gradient_kind = 2
          IF ( prob%H%ptr( n + 1 ) > 1 ) THEN
            IF ( printo ) WRITE( out, "( /, ' *** Solving as a separable QP' )")
            prob%Hessian_kind = 2
            ALLOCATE( prob%WEIGHT( n ), STAT = alloc_stat )
            IF ( alloc_stat /= 0 ) THEN
              IF ( printe ) WRITE( out, 2150 ) 'WEIGHT', alloc_stat ; STOP
            END IF
!  Set the Hessian; warn if is not diagonal or indefinite,
!  and modify appropriately

            prob%WEIGHT = zero
            prob%X0 = zero
            warn_h_not_diagonal = .FALSE. ; warn_h_indefinite = .FALSE.
            DO i = 1, n
              DO l = prob%H%ptr( i ) , prob%H%ptr( i + 1 ) - 1
                IF ( prob%H%col( l ) == i ) THEN
                  IF ( prob%H%val( l ) >= zero ) THEN
                    prob%WEIGHT( i ) = prob%H%val( l )
                  ELSE
                    warn_h_indefinite = .TRUE.
                    prob%WEIGHT( i ) = - prob%H%val( l )
                  END IF
                ELSE
                  warn_h_not_diagonal = .FALSE.
                END IF
              END DO
            END DO
            IF ( printe .AND. warn_h_not_diagonal )                            &
              WRITE( out, "( ' *** off-diagonal Hessian terms ignored ' )" )
            IF ( printe .AND. warn_h_indefinite )                              &
              WRITE( out, "( ' *** negative diagonal Hessian terms modified' )")
            prob%WEIGHT = SQRT( prob%WEIGHT )
          ELSE
            IF ( printo ) WRITE( out, "( /, ' *** Solving as an LP ' )" )
            prob%Hessian_kind = 0
          END IF
          CALL LSQP_solve( prob, data, LSQP_control, LSQP_inform )
          qfval = LSQP_inform%obj ; newton = LSQP_inform%nfacts ; nmods = 0
        ELSE
          IF ( printo ) WRITE( out, "( ' ** Solving as a general QP ' )" )
          DEALLOCATE( prob%X0 )
!         WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!         WRITE( 6, "( ' y ', /, (5ES12.4) )" ) prob%Y
!         WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z
          solv = ' QPB'
          IF ( printo ) WRITE( out, " ( /, ' ** QPB solver used ** ' ) " )
          CALL QPB_solve( prob, data, QPB_control, QPB_inform )

          IF ( printo ) WRITE( out, " ( /, ' ** QPB solver used ** ' ) " )
          qfval = QPB_inform%obj ; nmods = QPB_inform%nmods ; newton = 0
        END IF
        CALL CPU_TIME( timet ) ; ; CALL CLOCK_time( clockt )

!  Deallocate arrays from the minimization

        IF ( phase1 .OR. islsqp ) THEN
          status = LSQP_inform%status ; iter = LSQP_inform%iter
          nfacts = LSQP_inform%nfacts ; stopr = LSQP_control%stop_d
          CALL LSQP_terminate( data, LSQP_control, LSQP_inform )

        ELSE
          status = QPB_inform%status ; iter = QPB_inform%iter
          nfacts = QPB_inform%nfacts ; stopr = QPB_control%stop_d
          CALL QPB_terminate( data, QPB_control, QPB_inform )
        END IF

!  If the problem was scaled, unscale it

        IF ( scale > 0 ) THEN
          CALL SCALE_recover( prob, SCALE_trans, SCALE_data,                   &
                              SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF
        END IF
      ELSE
        timeo  = 0.0
        timet  = 0.0
        iter   = 0
        solv   = ' NONE'
        status = 0
        IF ( islsqp ) THEN
          stopr  = LSQP_control%stop_d
        ELSE
          stopr  = QPB_control%stop_d
        END IF
        nfacts = 0
        newton = 0
        nmods  = 0
        qfval  = prob%f
      END IF

!  Restore from presolve

      IF ( do_presolve ) THEN
        IF ( PRE_control%print_level >= DEBUG )                                &
          CALL QPT_write_problem( out, prob )
        CALL CPU_TIME( timep3 )
        CALL PRESOLVE_restore( prob, PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status /= 0 .AND. printo )                             &
          WRITE( out, " ( /, ' Warning: info%status following',                &
       &  ' PRESOLVE_restore is ', I5, / ) " ) PRE_inform%status
!       IF ( PRE_inform%status /= 0 ) STOP
        qfval = prob%q
        CALL PRESOLVE_terminate( PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status /= 0 .AND. printo )                             &
          WRITE( out, " ( /, ' Warning: info%status following',                &
       &    ' PRESOLVE_terminate is ', I5, / ) " ) PRE_inform%status
!       IF ( PRE_inform%status /= 0 ) STOP
        IF ( .NOT. do_solve ) STOP
        CALL CPU_TIME( timep4 )
        IF ( printo ) WRITE( out, 2210 )                                       &
          timep4 - timep3, timep2 - timep1 + timep4 - timep3
      ELSE
        PRE_control%print_level = TRACE
      END IF

!  If the problem was scaled, unscale it

      IF ( scale < 0 ) THEN
        CALL SCALE_recover( prob, SCALE_trans, SCALE_data,                     &
                            SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
      END IF

!  Compute maximum contraint residual and complementary slackness

      res_c = zero ; max_cs = zero
      DO i = 1, prob%m
        dummy = zero
        DO j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          dummy = dummy +  prob%A%val( j ) * prob%X( prob%A%col( j ) )
        END DO
        res_c = MAX( res_c, MAX( zero, prob%C_l( i ) - dummy,                  &
                                       dummy - prob%C_u( i ) ) )
        IF ( prob%C_l( i ) > - infinity ) THEN
          IF ( prob%C_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
                 MIN( ABS( ( prob%C_l( i ) - dummy ) * prob%Y( i ) ),          &
                      ABS( ( prob%C_u( i ) - dummy ) * prob%Y( i ) ) ) )
          ELSE
            max_cs = MAX( max_cs,                                              &
                          ABS( ( prob%C_l( i ) - dummy ) * prob%Y( i ) ) )
          END IF
        ELSE IF ( prob%C_u( i ) < infinity ) THEN
          max_cs = MAX( max_cs, ABS( ( prob%C_u( i ) - dummy ) * prob%Y( i ) ) )
        END IF
      END DO

      DO i = 1, prob%n
        dummy = prob%X( i )
        IF ( prob%X_l( i ) > - infinity ) THEN
          IF ( prob%X_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
                 MIN( ABS( ( prob%X_l( i ) - dummy ) * prob%Z( i ) ),          &
                      ABS( ( prob%X_u( i ) - dummy ) * prob%Z( i ) ) ) )
          ELSE
            max_cs = MAX( max_cs,                                              &
                          ABS( ( prob%X_l( i ) - dummy ) * prob%Z( i ) ) )
          END IF
        ELSE IF ( prob%X_u( i ) < infinity ) THEN
          max_cs = MAX( max_cs, ABS( ( prob%X_u( i ) - dummy ) * prob%Z( i ) ) )
        END IF
      END DO

!  Compute maximum KKT residual

      ALLOCATE( AY( n ), HX( n ), STAT = alloc_stat )
      AY = zero ; HX = prob%G( : n )
!     prob%G( : n ) = prob%G( : n ) - prob%Z( : n )
      DO i = 1, m
        DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          j = prob%A%col( l )
!         prob%G( j ) = prob%G( j ) - prob%A%val( l ) * prob%Y( i )
          AY( j ) = AY( j ) - prob%A%val( l ) * prob%Y( i )
        END DO
      END DO
      DO i = 1, n
        DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
          j = prob%H%col( l )
!         prob%G( i ) = prob%G( i ) + prob%H%val( l ) * prob%X( j )
!         IF ( j /= i )                                                        &
!           prob%G( j ) = prob%G( j ) + prob%H%val( l ) * prob%X( i )
          HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
          IF ( j /= i )                                                        &
            HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
        END DO
      END DO
!     DO i = 1, n
!       WRITE(6,"( i6, 4ES12.4 )" ) i, HX( i ), prob%Z( i ), AY( i ),          &
!                                   HX( i ) - prob%Z( i ) + AY( i )
!     END DO
!     WRITE(6,"( ( 5ES12.4 ) ) " ) MAXVAL( ABS( prob%Z ) )
!     WRITE(6,"( ' G ', /, ( 5ES12.4 ) )" ) prob%G( : n )
      res_k = MAXVAL( ABS( HX( : n ) - prob%Z( : n ) + AY( : n ) ) )

!  Print details of the solution obtained

      WRITE( out, 2010 ) status
      IF ( status == 0 .OR. status == - 8 .OR. status == - 9 .OR.              &
           status == - 10 ) THEN
        l = 4
        IF ( fulsol ) l = n
        IF ( do_presolve ) THEN
          IF ( PRE_control%print_level >= DEBUG ) l = n
        END IF

!  Print details of the primal and dual variables

        WRITE( out, 2090 )
        DO j = 1, 2
          IF ( j == 1 ) THEN
            ir = 1 ; ic = MIN( l, n )
          ELSE
            IF ( ic < n - l ) WRITE( out, 2000 )
            ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
          END IF
          DO i = ir, ic
            state = ' FREE'
            IF ( ABS( prob%X  ( i ) - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X  ( i ) - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) <     1.0D-10 )          &
              state = 'FIXED'
            WRITE( out, 2050 ) i, prob%X_names( i ), state, prob%X( i ),       &
                               prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO
        END DO

!  Compute the number of fixed and degenerate variables.

        nfixed = 0 ; ndegen = 0
        DO i = 1, n
          IF ( ABS( prob%X_u( i ) - prob%X_l( i ) ) < stopr ) THEN
            nfixed = nfixed + 1
            IF ( ABS( prob%Z( i ) ) < ten * stopr ) ndegen = ndegen + 1
          ELSE IF ( MIN( ABS( prob%X( i ) - prob%X_l( i ) ),                   &
                    ABS( prob%X( i ) - prob%X_u( i ) ) ) <=                    &
                    MAX( ten * stopr, ABS( prob%Z( i ) ) ) ) THEN
            nfixed = nfixed + 1
            IF ( ABS( prob%Z( i ) ) < ten * stopr ) ndegen = ndegen + 1
          END IF
        END DO

!  Print details of the constraints.

        IF ( m > 0 ) THEN

          WRITE( out, 2040 )
          l = 2  ; IF ( fulsol ) l = m
          IF ( do_presolve ) THEN
            IF ( PRE_control%print_level >= DEBUG ) l = m
          END IF
          DO j = 1, 2
            IF ( j == 1 ) THEN
              ir = 1 ; ic = MIN( l, m )
            ELSE
              IF ( ic < m - l ) WRITE( out, 2000 )
              ir = MAX( ic + 1, m - ic + 1 ) ; ic = m
            END IF
            DO i = ir, ic
              state = ' FREE'
              IF ( ABS( prob%C( I )   - prob%C_l( i ) ) < ten * stopr )        &
                state = 'LOWER'
              IF ( ABS( prob%C( I )   - prob%C_u( i ) ) < ten * stopr )        &
                state = 'UPPER'
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) <       stopr )        &
                state = 'EQUAL'
              WRITE( out, 2130 ) i, prob%C_names( i ), STATE, prob%C( i ),     &
                                 prob%C_l( i ), prob%C_u( i ), prob%Y( i )
            END DO
          END DO

!  Compute the number of equality, fixed inequality and degenerate constraints

          mequal = 0 ; mfixed = 0 ; mdegen = 0  ; mredun = 0
          DO i = 1, m
           IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr ) THEN
              mequal = mequal + 1
              IF ( ABS( prob%Y( i ) ) < stopr ) mredun = mredun + 1
            ELSE IF ( MIN( ABS( prob%C( i ) - prob%C_l( i ) ),                 &
                      ABS( prob%C( i ) - prob%C_u( i ) ) ) <=                  &
                 MAX( ten * stopr, ABS( prob%Y( i ) ) ) ) THEN
              mfixed = mfixed + 1
              IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1
            END IF
!           IF ( ABS( prob%C( i ) - prob%C_l( i ) ) < ten * stopr .OR.         &
!                ABS( prob%C( i ) - prob%C_u( i ) ) < ten * stopr ) THEN
!             IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < ten * stopr ) THEN
!                mequal = mequal + 1
!             ELSE
!                mfixed = mfixed + 1
!             END IF
!             IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1
!           END IF
          END DO
        END IF
        WRITE( out, 2100 ) n, nfixed, ndegen
        IF ( m > 0 ) THEN
           WRITE( out, 2110 ) m, mequal, mredun
           IF ( m /= mequal ) WRITE( out, 2120 ) m - mequal, mfixed, mdegen
        END IF
        WRITE( out, 2030 ) qfval, res_c, res_k, max_cs, iter

!  If required, write the solution to a file

        IF ( write_solution ) THEN
          INQUIRE( FILE = sfilename, EXIST = filexx )
          IF ( filexx ) THEN
             OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',          &
                 STATUS = 'OLD', IOSTAT = iores )
          ELSE
             OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',          &
                  STATUS = 'NEW', IOSTAT = iores )
          END IF
          IF ( iores /= 0 ) THEN
            write( out, 2160 ) iores, sfilename
            STOP
          END IF

          WRITE( sfiledevice, 2250 ) pname, solv, qfval
          WRITE( sfiledevice, 2090 )

          DO i = 1, n
            state = ' FREE'
            IF ( ABS( prob%X( i )   - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X( i )   - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( I ) - prob%X_u( I ) ) < stopr )                &
              state = 'FIXED'
            WRITE( sfiledevice, 2050 ) i, prob%X_names( i ), STATE,            &
              prob%X( i ), prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO

          IF ( m > 0 ) THEN
            WRITE( sfiledevice, 2040 )
            DO i = 1, m
              state = ' FREE'
              IF ( ABS( prob%C( I ) - prob%C_l( i ) ) < ten * stopr )          &
                state = 'LOWER'
              IF ( ABS( prob%C( I ) - prob%C_u( i ) ) < ten * stopr )          &
                state = 'UPPER'
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr )              &
                state = 'EQUAL'
              WRITE( sfiledevice, 2130 ) i, prob%C_names( i ), STATE,          &
                prob%C( i ), prob%C_l( i ), prob%C_u( i ), prob%Y( i )
            END DO
          END IF

          WRITE( sfiledevice, 2030 ) qfval, res_c, res_k, max_cs, iter
          CLOSE( sfiledevice )
        END IF
      END IF

      IF ( phase1 .OR. islsqp ) THEN
        sls_solv = LSQP_control%SBLS_control%symmetric_linear_solver
        CALL STRING_upper_word( sls_solv )
        WRITE( out, "( /, 1X, A, ' symmetric equation solver used' )" )        &
          TRIM( sls_solv )
        WRITE( out, "( ' Typically ', I0, ', ', I0,                            &
    &                  ' entries in matrix, factors' )" )                      &
          LSQP_inform%SBLS_inform%SLS_inform%entries,                          &
          LSQP_inform%SBLS_inform%SLS_inform%entries_in_factors
        WRITE( out, "( ' Analyse, factorize & solve CPU   times =',            &
    &     3( 1X, F8.3 ), /, ' Analyse, factorize & solve clock times =',       &
    &     3( 1X, F8.3))") LSQP_inform%time%analyse, LSQP_inform%time%factorize,&
          LSQP_inform%time%solve, LSQP_inform%time%clock_analyse,              &
          LSQP_inform%time%clock_factorize, LSQP_inform%time%clock_solve
      ELSE
        sls_solv = QPB_control%SBLS_control%symmetric_linear_solver
        CALL STRING_upper_word( sls_solv )
        WRITE( out, "( /, 1X, A, ' symmetric equation solver used' )" )        &
          TRIM( sls_solv )
        WRITE( out, "( ' Typically ', I0, ', ', I0,                            &
    &                  ' entries in matrix, factors' )" )                      &
          QPB_inform%SBLS_inform%SLS_inform%entries,                           &
          QPB_inform%SBLS_inform%SLS_inform%entries_in_factors
        WRITE( out, "( ' Analyse, factorize & solve CPU   times =',            &
    &     3( 1X, F8.3 ), /, ' Analyse, factorize & solve clock times =',       &
    &     3( 1X, F8.3 ))") QPB_inform%time%analyse, QPB_inform%time%factorize, &
          QPB_inform%time%solve, QPB_inform%time%clock_analyse,                &
          QPB_inform%time%clock_factorize, QPB_inform%time%clock_solve
      END IF

      times = times - time ; timet = timet - timeo
      clocks = clocks - clock ; clockt = clockt - clocko
      WRITE( out, "( /, ' Total CPU, clock times = ', F8.3, ', ', F8.3 )" )    &
        times + timet, clocks + clockt
!$    WRITE( out, "( ' number of threads = ', I0 )" ) OMP_GET_MAX_THREADS( )
      WRITE( out, 2070 ) pname

!  Compare the variants used so far

      WRITE( out, 2080 ) solv, iter, nfacts, qfval, status, clocks, clockt,    &
                         clocks + clockt

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        WRITE( rfiledevice, 2190 )                                             &
           pname, n, m, iter, newton, nmods, qfval, status, timet
      END IF

!  Print the first and last few components of the solution.

!     WRITE( out, 2070 )
!     WRITE( out, 2120 )
!     j = MIN( npm, 12 )
!     RES( : n ) = prob%X( : n ) ; RES( np1 : npm ) = prob%Y( : m )
!     WRITE( out, 2100 ) ( RES( i ), i = 1, j )
!     IF ( j < npm ) THEN
!        IF ( j + 1 < npm - 11 ) WRITE( out, 2110 )
!        WRITE( out, 2100 ) ( RES( i ), i = MAX( j + 1, npm - 11 ), npm )
!     END IF
!     DEALLOCATE( RES )

!  Close the data input file

   CLOSE( input  )
   STOP

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' )
 2010 FORMAT( /,' Stopping with inform%status = ', I0 )
 2020 FORMAT( ' Problem: ', A )
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, /,              &
                 ' Total number of iterations = ', I0 )
 2040 FORMAT( /, ' Constraints : ', /, '                             ',        &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper     Multiplier ' )
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 )
 2070 FORMAT( /, ' Problem: ', A, //,                                          &
                 '                                 objective',                 &
                 '          < ------ time ----- > ', /,                        &
                 ' Method  iterations   factors      value  ',                 &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -------    ---------',                 &
                 '   ------ -----    ----   -----  ' )
 2080 FORMAT( A5, 2I10, 5X, ES12.4, 1X,  I6, 0P, 3F8.2 )
 2090 FORMAT( /, ' Solution : ', /, '                              ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' )
 2100 FORMAT( /, ' Of the ', I0, ' variables, ', I0,                           &
              ' are on bounds, & ', I0, ' are dual degenerate' )
 2110 FORMAT( ' Of the ', I0, ' constraints, ', I0,' are equations, & ',       &
              I0, ' are redundant' )
 2120 FORMAT( ' Of the ', I0, ' inequalities, ', I0, ' are on bounds, & ',     &
              I0, ' are degenerate' )
 2130 FORMAT( I7, 1X, A10, A6, 4ES12.4 )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I0, ' when opening file ', A, '. Stopping ' )
 2180 FORMAT( A10 )
 2190 FORMAT( A10, 2I7, 3I6, ES13.4, I6, 0P, F8.2 )
 2200 FORMAT( /, ' problem dimensions:  n = ', I7, ' m = ', I7,                &
              ' a_ne = ', I9, ' h_ne = ', I9 )
 2300 FORMAT( /, ' after presolve - ', /,                                      &
              ' updated dimensions:  n = ', I7, ' m = ', I7,                   &
              ' a_ne = ', I9, ' h_ne = ', I9, /,                               &
              ' preprocessing time     =', F9.2,                               &
              '        number of transformations =', I10 )
 2210 FORMAT( /, ' postprocessing complete: time = ', F9.2,                    &
              ' total processing time =', F9.2 )
 2250 FORMAT( /, ' Problem:    ', A, /, ' Solver :   ', A,                     &
              /, ' Objective:', ES24.16 )

!  End of RUNQPB_DATA

   END PROGRAM RUNQPB_DATA
