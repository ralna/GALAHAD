  PROGRAM RUNRB_sls
    USE spral_rutherford_boeing
    USE GALAHAD_CLOCK
    USE GALAHAD_SPECFILE_double
    USE GALAHAD_SLS_double

    IMPLICIT none

    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
    INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
    INTEGER, PARAMETER :: out = 6
    INTEGER :: errout = 6

!  local variables

    CHARACTER( LEN = 80 ) :: filename
    INTEGER :: info                    ! return code
    INTEGER :: m                       ! # rows
    INTEGER :: n                       ! # columns
    INTEGER ( KIND = long ) :: nelt    ! # elements (0 if asm)
    INTEGER ( KIND = long ) :: nvar    ! # indices in file
    INTEGER ( KIND = long ) :: a_ne    ! # values in file
    INTEGER :: matrix_type             ! SPRAL matrix type
    CHARACTER ( LEN = 3 ) :: type_code ! eg "rsa"
    CHARACTER ( LEN = 72 ) :: title    ! title field of file
    CHARACTER ( LEN = 8 ) :: identifier
    TYPE ( SMT_type ) :: A
    REAL( wp ), DIMENSION( : ), ALLOCATABLE :: X, B
    TYPE( rb_read_options ) :: options ! control variables
    TYPE ( SLS_data_type ) :: data
    TYPE ( SLS_control_type ) :: SLS_control
    TYPE ( SLS_inform_type ) :: SLS_inform

    INTEGER :: ir, ic, la, lh, liw, iores, smt_stat
    INTEGER :: i, j, l
    INTEGER :: status, alloc_stat
    REAL :: time, timeo, times, timet
    REAL ( KIND = wp ) :: clock, clocko, clocks, clockt, res
    LOGICAL :: filexx, printo, is_specfile

!  Specfile characteristics

    INTEGER, PARAMETER :: input_specfile = 34
    INTEGER, PARAMETER :: lspec = 14
    CHARACTER ( LEN = 16 ) :: specname = 'RUNSLS'
    TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
    CHARACTER ( LEN = 16 ) :: runspec = 'RUNSLS.SPC'
    CHARACTER ( LEN = 30 ) :: solver = "sils" // REPEAT( ' ', 26 )

!  Default values for specfile-defined parameters

    INTEGER :: dfiledevice = 26
    INTEGER :: sfiledevice = 62
    INTEGER :: rfiledevice = 47
    LOGICAL :: write_problem_data   = .FALSE.
    LOGICAL :: write_solution       = .FALSE.
    LOGICAL :: write_result_summary = .FALSE.
    LOGICAL :: kkt_system = .TRUE.
    LOGICAL :: solve = .TRUE.
    CHARACTER ( LEN = 30 ) :: dfilename = 'SLS.data'
    CHARACTER ( LEN = 30 ) :: sfilename = 'SLSSOL.d'
    CHARACTER ( LEN = 30 ) :: rfilename = 'SLSRES.d'
    LOGICAL :: fulsol = .FALSE.
    REAL ( KIND = wp ) :: barrier_pert = 1.0_wp

    CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

!  ----------------------- data set-up --------------------------

!  read the name of the RB file

    READ( 5, "( A80 )" ) filename

!  read header information from the Rutheford-Boeing file

    CALL rb_peek( TRIM( filename ), info, m, n, nelt, nvar, a_ne,              &
                  matrix_type, type_code, title, identifier )

!  check that the file exists and is not faulty

    IF ( info < 0 ) THEN
      WRITE( out, "( ' input filename faulty, info = ', I0 )" ) info
      STOP
     END IF

!  print details of matrix

    WRITE( out, "( ' Matrix identifier = ', A, ' ( ', A, ')', /                &
                   ' m = ', I0, ', n = ', I0, ', nnz = ', I0 )" )              &
      TRIM( identifier ), TRIM( type_code ), m, n, a_ne

    IF ( m /= n .OR. type_code /= 'rsa' ) THEN
      WRITE( out, "( ' matrix does not seem to be real, symmetric' )" )
      STOP
    END IF

!  read the matrix from the Rutheford-Boeing file and translate as
!  uper-triangluar CSC = lower triangular CSR

    options%lwr_upr_full = 2
    CALL rb_read( TRIM( filename ), m, n, A%ptr, A%col, A%val, options, info )
    CALL SMT_put( A%type, 'COORDINATE', info )

!  pick solution vector of ones

    ALLOCATE( X( n ), B( n ), STAT = alloc_stat )
    X = 1.0_wp

!  generate RHS

    B = 0.0+wp
    DO i = 1, n
      DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
        j = A%col( l )
        B( i ) = B( i ) + A%val( i ) * X( j )
        IF ( i /= j ) B( j ) = B( j ) + A%val( i ) * X( i )
      END DO
    END DO

!  ------------------- data set-up complete ----------------------

    CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )
    times = times - time ; clocks = clocks - clock


!  The default values for SLS could have been set as:

! BEGIN RUNSLS SPECIFICATIONS (DEFAULT)
!  write-problem-data                                NO
!  problem-data-file-name                            SLS.data
!  problem-data-file-device                          26
!  kkt-system                                        YES
!  symmetric-linear-equation-solver                  sils
!  print-full-solution                               NO
!  write-solution                                    NO
!  solution-file-name                                SLSSOL.d
!  solution-file-device                              62
!  write-result-summary                              NO
!  result-summary-file-name                          SLSRES.d
!  result-summary-file-device                        47
!  barrier-perturbation                              1.0
!  solve                                             YES
! END RUNSLS SPECIFICATIONS

!  ------------------ Open the specfile for runsls ----------------

    INQUIRE( FILE = runspec, EXIST = is_specfile )
    IF ( is_specfile ) THEN
      OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',                &
            STATUS = 'OLD' )

!   Define the keywords (9 and 10 are irrelevant for the RB interface)

      spec( 1 )%keyword = 'write-problem-data'
      spec( 2 )%keyword = 'problem-data-file-name'
      spec( 3 )%keyword = 'problem-data-file-device'
      spec( 4 )%keyword = 'print-full-solution'
      spec( 5 )%keyword = 'write-solution'
      spec( 6 )%keyword = 'solution-file-name'
      spec( 7 )%keyword = 'solution-file-device'
      spec( 8 )%keyword = 'symmetric-linear-equation-solver'
      spec( 11 )%keyword = 'solve'
      spec( 12 )%keyword = 'write-result-summary'
      spec( 13 )%keyword = 'result-summary-file-name'
      spec( 14 )%keyword = 'result-summary-file-device'
      spec( 9 )%keyword = 'barrier-perturbation'
      spec( 10 )%keyword = 'kkt-system'

!   Read the specfile

      CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

      CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
      CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
      CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
      CALL SPECFILE_assign_logical( spec( 4 ), fulsol, errout )
      CALL SPECFILE_assign_logical( spec( 5 ), write_solution, errout )
      CALL SPECFILE_assign_string ( spec( 6 ), sfilename, errout )
      CALL SPECFILE_assign_integer( spec( 7 ), sfiledevice, errout )
      CALL SPECFILE_assign_value( spec( 8 ), solver, errout )
      CALL SPECFILE_assign_logical( spec( 11 ), solve, errout )
      CALL SPECFILE_assign_logical( spec( 12 ), write_result_summary, errout )
      CALL SPECFILE_assign_string ( spec( 13 ), rfilename, errout )
      CALL SPECFILE_assign_integer( spec( 14 ), rfiledevice, errout )
      CALL SPECFILE_assign_real( spec( 9 ), barrier_pert, errout )
      CALL SPECFILE_assign_logical( spec( 10 ), kkt_system, errout )
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
        write( out, 2000 ) iores, dfilename
        STOP
      END IF

      a_ne = A%ptr( m + 1 )
      WRITE( dfiledevice, "( 'n, m = ', 2I6 )" ) n, m
      WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" ) A%ptr( : m + 1 )
      WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) A%col( : a_ne )
      WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" ) A%val( : a_ne )

      CLOSE( dfiledevice )
    END IF

!  If required, append results to a file

    IF ( write_result_summary ) THEN
      INQUIRE( FILE = rfilename, EXIST = filexx )
      IF ( filexx ) THEN
         OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',              &
               STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
      ELSE
         OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',              &
               STATUS = 'NEW', IOSTAT = iores )
      END IF
      IF ( iores /= 0 ) THEN
        write( out, 2000 ) iores, rfilename
        STOP
      END IF
      WRITE( rfiledevice, "( A )" ) TRIM( identifier )
    END IF

!  analyse

    CALL SLS_analyse( A, data, SLS_control, SLS_inform )
    WRITE( 6, "( /, ' analyse   time = ', F8.3, ' clock = ', F8.3,             &
   &  ' ststus = ', I0 )" ) SLS_inform%time%analyse,                           &
      SLS_inform%time%clock_analyse, SLS_inform%status
    WRITE( 6, "( ' external  time = ', F8.3, ' clock = ', F8.3 )" )            &
      SLS_inform%time%analyse_external,                                        &
      SLS_inform%time%clock_analyse_external
    WRITE( 6, "( ' A n = ', I0,                                                &
     &  ', nnz(prec,predicted factors) = ', I0, ', ', I0 )" )                  &
           A%n, A%ne, SLS_inform%entries_in_factors

!  factorize

    IF ( SLS_inform%status >= 0 ) THEN
      CALL SLS_factorize( A, data, SLS_control, SLS_inform )
      WRITE( 6, "( ' factorize time = ', F8.3, ' clock = ', F8.3,              &
   &    ' ststus = ', I0 )" ) SLS_inform%time%factorize,                       &
        SLS_inform%time%clock_factorize, SLS_inform%status
      WRITE( 6, "( ' external  time = ', F8.3, ' clock = ', F8.3 )" )          &
        SLS_inform%time%factorize_external,                                    &
        SLS_inform%time%clock_factorize_external

!  solve

      IF ( SLS_inform%status >= 0 .AND. solve ) THEN
        X = B
        CALL SLS_solve( A, X, data, SLS_control, SLS_inform )
        WRITE( 6, "( ' solve status = ', I0 )" ) SLS_inform%status
      END IF
    END IF
    IF ( printo ) WRITE( out, " ( /, ' ** SLS solver used ** ' ) " )

    WRITE( 6, "( ' nullity, # -ve eigenvalues = ', I0, ', ', I0 )" )           &
          A%n - SLS_inform%rank, SLS_inform%negative_eigenvalues

!  Deallocate arrays from the minimization

    status = SLS_inform%status
    CALL SLS_terminate( data, SLS_control, SLS_inform )
    CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )
    timet = timet - timeo ; clockt = clockt - clocko

    WRITE( out, "( /, ' Solver: ', A, ' with ordering = ', I0 )" )             &
      TRIM( solver ), SLS_control%ordering
    WRITE( out, "(  ' Stopping with inform%status = ', I0 )" ) status

    IF ( write_result_summary ) THEN
      BACKSPACE( rfiledevice )
      IF ( SLS_inform%status >= 0 ) THEN
        WRITE( rfiledevice, "( A10, I8, A10, 2I8, F10.2, I4 )" )               &
          TRIM( identifier ), A%n, TRIM( solver ),                             &
          A%n - SLS_inform%rank, SLS_inform%negative_eigenvalues,              &
          clockt, SLS_inform%status
      ELSE
        WRITE( rfiledevice, "( A10, I8, A10, 2I8, F10.2, I4 )" )               &
          TRIM( identifier ), A%n, TRIM( solver ),                             &
          A%n - SLS_inform%rank, SLS_inform%negative_eigenvalues,              &
          - clockt, SLS_inform%status
      END IF
    END IF
    IF ( .NOT. solve ) RETURN

!  Compute maximum contraint residual

    IF ( status >= 0 ) THEN
      DO i = 1, n
        DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
          j = A%col( l )
          B( i ) = B( i ) - A%val( i ) * X( j )
          IF ( i /= j ) B( j ) = B( j ) - A%val( i ) * X( i )
        END DO
      END DO
      res = MAXVAL( ABS( B ) )
      WRITE( out, "( /, ' Maximum error =', ES21.14 )" ) res

!  If required, write the solution to a file

      IF ( write_solution ) THEN
        INQUIRE( FILE = sfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',        &
               STATUS = 'OLD', IOSTAT = iores )
        ELSE
           OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',        &
                STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2000 ) iores, sfilename
          STOP
        END IF
        WRITE( sfiledevice, "( /, ' Problem:    ', A )" ) TRIM( identifier )
        WRITE( sfiledevice, "( /, ' Solution : ', /, '     value   ' )" )
        DO i = 1, n
          WRITE( sfiledevice, "( I8, ES22.14 )" ) i, X( i )
        END DO
        WRITE( sfiledevice, "( /, ' Maximum error =', ES21.14 )" ) res
        CLOSE( sfiledevice )
      END IF
    END IF

    WRITE( out, "( /, ' Analyse time, clock = ', F0.2, ', ', F0.2 )" )         &
      SLS_inform%time%analyse_external,                                        &
      SLS_inform%time%clock_analyse_external
    WRITE( out, "( /, ' Factorize time, clock = ', F0.2, ', ', F0.2 )" )       &
      SLS_inform%time%factorize_external,                                      &
      SLS_inform%time%clock_factorize_external
      WRITE( out, "( /, ' Solve time, clock = ', F0.2, ', ', F0.2 )" )         &
      SLS_inform%time%solve_external, SLS_inform%time%clock_solve_external

    WRITE( out, "( /, ' Total time, clock = ', F0.2, ', ', F0.2 )" )           &
      times + timet, clocks + clockt
    WRITE( out, "( /, ' Solver: ', A, ' with ordering = ', I0 )" )             &
      TRIM( solver ), SLS_control%ordering
    WRITE( out, "(  ' Stopping with inform%status = ', I0 )" ) status
    WRITE( out, "( /, ' Problem: ', A10, //,                                   &
   &                  '          < ------ time ----- > ',                      &
   &                  '  < ----- clock ---- > ', /,                            &
   &                  '   status setup   solve   total',                       &
   &                  '   setup   solve   total', /,                           &
   &                  '   ------ -----    ----   -----',                       &
   &                  '   -----   -----   -----  ' )" ) TRIM( identifier )

    DEALLOCATE( A%ptr, A%col, A%val, X, B )
      CALL SLS_initialize( solver, data, SLS_control, SLS_inform )

!  Non-executable statements

 2000 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A, '. Stopping ' )

  END PROGRAM RUNRB_sls
