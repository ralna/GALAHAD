! THIS VERSION: GALAHAD 3.1 - 16/06/2018 AT 13:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N B Q P B _ D A T A  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. January 18th 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNBQPB_DATA

!    ----------------------------------------------------
!    | Main program for the problem-data-file interface |
!    | to BQPB, a preconditioned conjugate-gradient     |
!    | interior-point algorithm for bound-constrained   |
!    | convex quadratic programming                     |
!    ----------------------------------------------------

!NOT95USE GALAHAD_CPU_time
   USE GALAHAD_QPT_double
   USE GALAHAD_RPD_double
   USE GALAHAD_SMT_double, only: SMT_put
   USE GALAHAD_BQPB_double
   USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
   USE GALAHAD_PRESOLVE_double
   USE GALAHAD_SPECFILE_double
   USE GALAHAD_COPYRIGHT
   USE SCALING
   USE GALAHAD_SYMBOLS,                                                        &
       ACTIVE                => GALAHAD_ACTIVE,                                &
       TRACE                 => GALAHAD_TRACE,                                 &
       DEBUG                 => GALAHAD_DEBUG,                                 &
       GENERAL               => GALAHAD_GENERAL,                               &
       ALL_ZEROS             => GALAHAD_ALL_ZEROS

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 5

!  --------------------------------------------
!
!  Solve the quadratic program
!
!     minimize     1/2 x(T) H x + g(T) x + f
!
!     subject to       x_l <= x <= x_u
!
!  using the GALAHAD package GALAHAD_BQPB20
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
!  "problem type" must be BQP (a bound-constrained QP)
!  "number variables"
!  "number of nonzeros in lower traingle of H"
!  "row" "column" "value" for each entry of H (if any), one triple on each line
!  "default value for entries in g"
!  "number of non-default entries in g"
!  "index" "value" for each non-default term in g (if any), one pair per line
!  "value of f"
!  "value for infinity" for bounds - any bound greater than or equal to this
!     in absolute value is infinite
!  "default value for entries in x_l"
!  "number of non-default entries in x_l"
!  "index" "value" for each non-default term in x_l (if any), one pair per line
!  "default value for entries in x_u"
!  "number of non-default entries in x_u"
!  "index" "value" for each non-default term in x_u (if any), one pair per line
!  "default value for starting value for variables x"
!  "number of non-default starting entries in x"
!  "index" "value" for each non-default term in x (if any), one pair per line
!  "default value for starting value for dual varibales z for simple bounds"
!  "number of non-default starting entries in z"
!  "index" "value" for each non-default term in z (if any), one pair per line
!  "number of non-default names of variables" -default for variable i is "i"
!  "index" "name" for each non-default name for variable x_i with index i
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

      INTEGER :: n, ir, ic, ifail, liw, iores, smt_stat
!     INTEGER :: np1, npm, factorization_integer, factorization_real
      INTEGER :: i, j, l, nfixed, ndegen, status, alloc_stat, H_ne, iter
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: qfval, stopr, dummy
      REAL ( KIND = wp ) :: res_k, max_cs
      LOGICAL :: filexx, printo, printe, is_specfile
!     LOGICAL :: ldummy

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 21
      CHARACTER ( LEN = 16 ) :: specname = 'RUNBQPB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNBQPB.SPC'

!  The default values for BQPB could have been set as:

! BEGIN RUNBQPB SPECIFICATIONS (DEFAULT)
!  write-problem-data                        NO
!  problem-data-file-name                    BQPB.data
!  problem-data-file-device                  26
!  write-initial-sif                         NO
!  initial-sif-file-name                     INITIAL.SIF
!  initial-sif-file-device                   51
!  least-squares-qp                          NO
!  scale-problem                             0
!  solve-problem                             YES
!  print-full-solution                       NO
!  write-solution                            NO
!  solution-file-name                        BQPBSOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  BQPBRES.d
!  result-summary-file-device                47
!  perturb-bounds-by                         0.0
! END RUNBQPB SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: scale = 0
      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'BQPB.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'BQPBRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'BQPBSOL.d'
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = wp ) :: pert_bnd = zero

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( RPD_inform_type ) :: RPD_inform
      TYPE ( SCALING_control_type ) :: control
      TYPE ( BQPB_data_type ) :: data
      TYPE ( BQPB_control_type ) :: BQPB_control
      TYPE ( BQPB_inform_type ) :: BQPB_inform
      TYPE ( QPT_problem_type ) :: prob
      TYPE ( NLPT_userdata_type ) :: userdata

!  Allocatable arrays

      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SH, SA
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HX
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, B_stat

     CALL CPU_TIME( time )

!  Open the data input file

      OPEN( input, FORM = 'FORMATTED', STATUS = 'OLD'  )
      REWIND input

      CALL CPU_TIME( time )

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

!  check that the problem is a BQP

      SELECT CASE ( RPD_inform%p_type( 1 : 1 ) )
      CASE ( 'L', 'D', 'C', 'Q' )
      CASE DEFAULT
        WRITE( out, "( /, ' ** Problem ', A, ', objective function',           &
       &  ' is not convex. Stopping' )" ) TRIM( pname )
        STOP
      END SELECT

      SELECT CASE ( RPD_inform%p_type( 3 : 3 ) )
      CASE ( 'N', 'B' )
      CASE DEFAULT
        WRITE( out, "( /, ' ** Problem ', A, ', constraints are not',          &
       &  ' simple bounds. Stopping' )" ) TRIM( pname )
        STOP
      END SELECT

      n = prob%n
      H_ne = prob%H%ne

!  Allocate derived types

      ALLOCATE( prob%X0( n ), B_stat( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X0', alloc_stat
        STOP
      END IF
      prob%X0 = prob%X

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ),                                    &
                           MAX( prob%X_l( : n ), prob%X( : n ) ) )

!  Allocate and initialize dual variables.

      liw = n + 1
      ALLOCATE( prob%A%ptr( 0 ), prob%H%ptr( n + 1 ) )
      ALLOCATE( IW( liw ) )

!     WRITE( 27, "( ( 3( 2I6, ES12.4 ) ) )" )                                  &
!        ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, H_ne )
!     WRITE( 26, "( ' H_row ', /, ( 10I6 ) )" ) prob%H%row( : H_ne )
!     WRITE( 26, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
!     WRITE( 26, "( ' H_val ', /, ( 5ES12.4 ) )" ) prob%H%val( : H_ne )

!  Transform H to row storage format

      IF ( H_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( n, n, H_ne, prob%H%row, prob%H%col, H_ne,   &
                                   prob%H%val, prob%H%ptr, n + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%H%ptr = 0
      END IF

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( prob%H%row )
      DEALLOCATE( IW )
      ALLOCATE( prob%H%row( 0 ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, "( ' whoa there - allocate error ', I0 )" ) alloc_stat; STOP
      END IF

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )

!     WRITE( out, "( ' maximum element of H = ', ES12.4 )" )                   &
!       MAXVAL( ABS( prob%H%val( : H_ne ) ) )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times )

!  ------------------ Open the specfile for runbqpb ----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!   Define the keywords

        spec( 1 )%keyword = 'write-problem-data'
        spec( 2 )%keyword = 'problem-data-file-name'
        spec( 3 )%keyword = 'problem-data-file-device'
        spec( 4 )%keyword = 'write-initial-sif'
        spec( 5 )%keyword = 'initial-sif-file-name'
        spec( 6 )%keyword = 'initial-sif-file-device'
        spec( 8 )%keyword = 'scale-problem'
        spec( 13 )%keyword = 'solve-problem'
        spec( 14 )%keyword = 'print-full-solution'
        spec( 15 )%keyword = 'write-solution'
        spec( 16 )%keyword = 'solution-file-name'
        spec( 17 )%keyword = 'solution-file-device'
        spec( 18 )%keyword = 'write-result-summary'
        spec( 19 )%keyword = 'result-summary-file-name'
        spec( 20 )%keyword = 'result-summary-file-device'
        spec( 21 )%keyword = 'perturb-bounds-by'

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
        CALL SPECFILE_assign_integer( spec( 8 ), scale, errout )
        CALL SPECFILE_assign_logical( spec( 13 ), do_solve, errout )
        CALL SPECFILE_assign_logical( spec( 14 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 15 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 16 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 17 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 18 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 19 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 20 ), rfiledevice, errout )
        CALL SPECFILE_assign_real( spec( 21 ), pert_bnd, errout )
      END IF

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

        WRITE( dfiledevice, "( 'n = ', 2I6, ' obj = ', ES12.4 )" ) n, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
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

      CALL BQPB_initialize( data, BQPB_control, BQPB_inform )
      CALL BQPB_read_specfile( BQPB_control, input_specfile )

      control%print_level = BQPB_control%print_level
      control%out         = BQPB_control%out
      control%out_error   = BQPB_control%error

      printo = out > 0 .AND. control%print_level > 0
      printe = out > 0 .AND. control%print_level >= 0

      IF ( printo ) CALL COPYRIGHT( out, '2006' )

      WRITE( out, 2020 ) pname
      WRITE( out, 2200 ) n, H_ne

      B_stat = 0

!  If required, scale the problem

      IF ( scale > 0 ) THEN
        ALLOCATE( SH( n ), SA( 0 ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'SH/SA', alloc_stat ; STOP
        END IF

!  Scale using K

        IF ( scale == 1 .OR. scale == 4 ) THEN
          IF ( printo ) WRITE( out, 2140 ) 'K'
          CALL SCALING_get_factors_from_K( n, 0, prob%H%val, prob%H%col,       &
                                           prob%H%ptr, prob%A%val, prob%A%col, &
                                           prob%A%ptr, SH, SA, control, ifail )
!  Scale using A

        ELSE IF ( scale == 2 .OR. scale == 5 ) THEN
          IF ( printo ) WRITE( out, 2140 ) 'A'
          CALL SCALING_get_factors_from_A( n, 0, prob%A%val, prob%A%col,       &
                                           prob%A%ptr, SH, SA, control, ifail )
        ELSE IF ( scale == 3 ) THEN
          SH = one ; SA = one
        END IF

!  Rescale A

        IF ( scale >= 3 ) THEN
          IF ( printo ) WRITE( out, 2170 )
          CALL SCALING_normalize_rows_of_A( n, 0, prob%A%val, prob%A%col,      &
                                            prob%A%ptr, SH, SA )
        END IF

!  Apply the scaling factors

        CALL SCALING_apply_factors( n, 0, prob%H%val, prob%H%col, prob%H%ptr,  &
                                    prob%A%val, prob%A%col, prob%A%ptr,        &
                                    prob%G, prob%X, prob%X_l, prob%X_u,        &
                                    prob%C_l, prob%C_u, prob%Y, prob%Z,        &
                                    infinity, SH, SA, .TRUE. )
      END IF

!  If the preprocessor is to be used, or the problem to be output,
!  allocate sufficient space

      IF ( write_initial_sif ) THEN

        ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%X_status = ACTIVE

        ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'Z_lu', alloc_stat
          STOP
        END IF
        prob%Z_l( : n ) = - infinity
        prob%Z_u( : n ) =   infinity

!  Writes the initial SIF file, if needed

        IF ( write_initial_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,          &
                                 .FALSE., .FALSE., infinity )
          IF ( .NOT. do_solve ) STOP
        END IF
      END IF

!  Call the optimizer

      qfval = prob%f

      IF ( do_solve .AND. prob%n > 0 ) THEN

        CALL CPU_TIME( timeo )

        prob%n = n
        DEALLOCATE( prob%X0 )

        solv = ' BQPB'
        IF ( printo ) WRITE( out, " ( ' ** BQPB solver used ** ' ) " )
        CALL BQPB_solve( prob, B_stat, data, BQPB_control, BQPB_inform, userdata )

        IF ( printo ) WRITE( out, " ( /, ' ** BQPB solver used ** ' ) " )
        qfval = BQPB_inform%obj

        CALL CPU_TIME( timet )

!  Deallocate arrays from the minimization

        status = BQPB_inform%status
        iter = BQPB_inform%iter
        stopr = BQPB_control%stop_d
        CALL BQPB_terminate( data, BQPB_control, BQPB_inform )
      ELSE
        timeo  = 0.0
        timet  = 0.0
        iter  = 0
        solv   = ' NONE'
        status = 0
        stopr = BQPB_control%stop_d
        qfval  = prob%f
      END IF

!  If the problem was scaled, unscale it.

      IF ( scale > 0 ) THEN
        CALL SCALING_apply_factors( n, 0, prob%H%val, prob%H%col, prob%H%ptr,  &
                                    prob%A%val, prob%A%col, prob%A%ptr,        &
                                    prob%G, prob%X, prob%X_l, prob%X_u,        &
                                    prob%C_l, prob%C_u, prob%Y, prob%Z,        &
                                    infinity, SH, SA, .FALSE., C = prob%C )
        DEALLOCATE( SH, SA )
      END IF

!  Compute maximum complementary slackness

      max_cs = zero
      DO i = 1, n
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

      ALLOCATE( HX( n ), STAT = alloc_stat )
      HX = prob%G( : n )
      DO i = 1, n
        DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
          j = prob%H%col( l )
          HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
          IF ( j /= i )                                                        &
            HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
        END DO
      END DO
      res_k = MAXVAL( ABS( HX( : n ) - prob%Z( : n ) ) )


!  Print details of the solution obtained

      WRITE( out, 2010 ) status
      IF ( status == 0 .OR. status == - 8 .OR. status == - 9 .OR.              &
           status == - 10 ) THEN
        l = 4
        IF ( fulsol ) l = n

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
            WRITE( out, 2050 ) i, state, prob%X( i ),                          &
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

        WRITE( out, 2100 ) n, nfixed, ndegen
        WRITE( out, 2030 ) qfval, res_k, max_cs, iter
!                          factorization_integer, factorization_real

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
            WRITE( sfiledevice, 2050 ) i, STATE, prob%X( i ),                  &
              prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO

          WRITE( sfiledevice, 2030 ) qfval, res_k, max_cs, iter
!           factorization_integer, factorization_real
          CLOSE( sfiledevice )
        END IF
      END IF

      times = times - time ; timet = timet - timeo
      WRITE( out, 2060 ) times + timet
      WRITE( out, 2070 ) pname

!  Compare the variants used so far

      WRITE( out, 2080 ) solv, iter, qfval, status, times,                     &
                         timet, times + timet

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        WRITE( rfiledevice, 2190 )                                             &
           pname, n, iter, qfval, status, timet
      END IF

      IF ( is_specfile ) CLOSE( input_specfile )

      STOP

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' )
 2010 FORMAT( /,' Stopping with inform%status = ', I3 )
 2020 FORMAT( /, ' Problem: ', A10 )
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, //,             &
                 ' Number of iterations = ', I0 )
!           //, I0, ' integer and ', I0, ' real words required',               &
!                ' for the factorization' )
 2050 FORMAT( I7, 1X, A6, 4ES12.4 )
 2060 FORMAT( /, ' Total time = ', 0P, F12.2 )
 2070 FORMAT( /, ' Problem: ', A10, //,                                        &
                 '         iterations               objective',                &
                 '          < ------ time ----- > ', /,                        &
                 ' Method  QPB   factors       value  ',                &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -------     ---------',                &
                 '   ------ -----    ----   -----  ' )
 2080 FORMAT( A5, I7, 6X, ES12.4, I6, 0P, 3F8.2 )
 2090 FORMAT( /, ' Solution : ', /, '                              ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' )
 2100 FORMAT( /, ' Of the ', I7, ' variables, ', 2X, I7,                       &
              ' are on bounds &', I7, ' are dual degenerate' )
! 2110 FORMAT( ' Of the ', I7, ' constraints, ', I7,' are equations &', I7,    &
!              ' are degenerate' )
 2140 FORMAT( /, ' *** Problem will be scaled based on ', A1, ' *** ' )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2170 FORMAT( /, ' *** Further scaling applied to A *** ' )
 2180 FORMAT( A10 )
 2190 FORMAT( A10, I7, I6, ES13.4, I6, 0P, F8.2 )
 2200 FORMAT( /, ' problem dimensions:  n = ', I7, ' h_ne = ', I9 )
 2250 FORMAT( /, ' Problem:    ', A10, /, ' Solver :   ', A5,                  &
              /, ' Objective:', ES24.16 )

!  End of RUNBQPB_DATA

   END PROGRAM RUNBQPB_DATA
