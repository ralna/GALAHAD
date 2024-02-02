! THIS VERSION: GALAHAD 4.3 - 2024-02-02 AT 08:00 GMT.

#include "galahad_modules.h"

  PROGRAM LSRB_main

!  read a Rutherford-Boeing matrix A and store it in csc scheme
!  and then solve the least-squares problem
!    min 1/2 || A x - b ||^W_2 + 1/2 sigma ||x||^2^2
!  possibly subject to constraints using an iterative method

!!$  USE omp_lib
!   USE SMT_precision
    USE GALAHAD_KINDS_precision
    USE GALAHAD_SPECFILE_precision
    USE GALAHAD_SYMBOLS
    USE GALAHAD_CONVERT_precision
    USE GALAHAD_BLLS_precision
    USE GALAHAD_SLLS_precision
    USE hsl_mi35_precision, ONLY: mi35_control, mi35_info, mi35_check_matrix
    USE SPRAL_RAL_BOEING_precision

    IMPLICIT NONE

!  parameters

    REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
    REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
    REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
    REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
!   REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 19
    REAL ( KIND = rp_ ) :: feas_tol = ten ** ( - 5 )
    REAL ( KIND = rp_ ) :: same_tol = ten ** ( - 10 )
    INTEGER ( KIND = ip_ ), PARAMETER :: out = 6

!  scalars

    INTEGER ( KIND = ip_ ) :: errout = 6
    INTEGER ( KIND = ip_ ) :: alloc_stat, status, smt_stat, iter
    INTEGER ( KIND = ip_ ) :: i, ir, ic, j, l, m, n, ne
    INTEGER ( KIND = ip_ ) :: iores, nfixed, ndegen
!   INTEGER ( KIND = ip_ ) :: matrix_type
    REAL ( KIND = rp_ ) :: rjm1, rnm1, xj, obj
    REAL :: time, timeo, times, timet
    LOGICAL :: filexx, is_file, is_specfile, transpose_a
    CHARACTER ( LEN = 5 ) :: state
    CHARACTER ( LEN = 59 ) :: filea
    CHARACTER ( LEN = 72 ) :: title
    CHARACTER ( LEN = 80 ) :: name = REPEAT( ' ', 80 )

!  allocatable arrays

    INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: AT_row, AT_ptr
    REAL( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: AT_val
    INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_stat

!  structures

    TYPE ( SMT_type ) :: A
    TYPE ( rb_read_options ) :: options_rb
    INTEGER ( KIND = ip_ ) :: info_rb
    TYPE ( mi35_control ) :: control_mi35
    TYPE ( mi35_info ) :: info_mi35
    TYPE ( QPT_problem_type ) :: p
    TYPE ( GALAHAD_userdata_type ) :: userdata
    TYPE ( BLLS_data_type ) :: BLLS_data
    TYPE ( BLLS_control_type ) :: BLLS_control
    TYPE ( BLLS_inform_type ) :: BLLS_inform
    TYPE ( SLLS_data_type ) :: SLLS_data
    TYPE ( SLLS_control_type ) :: SLLS_control
    TYPE ( SLLS_inform_type ) :: SLLS_inform

!$  INTEGER ( KIND = ip_ ) :: OMP_GET_MAX_THREADS

!  read the specfile

!  Specfile characteristics

      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 18
      INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
      CHARACTER ( LEN = 16 ) :: specname = 'RUNLSRB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNLSRB.SPC'

!  Default values for specfile-defined parameters

      INTEGER ( KIND = ip_ ) :: dfiledevice = 26
      INTEGER ( KIND = ip_ ) :: ifiledevice = 51
      INTEGER ( KIND = ip_ ) :: rfiledevice = 47
      INTEGER ( KIND = ip_ ) :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: solver = 'blls'
      CHARACTER ( LEN = 30 ) :: dfilename = 'LSRB.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'LSRBRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'LSRBSOL.d'
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = rp_ ) :: res_rel = ten ** ( - 8 )
      REAL ( KIND = rp_ ) :: res_abs = 0.0_rp_
      REAL ( KIND = rp_ ) :: time_limit = - 1.0_rp_

      CALL CPU_TIME( time )

!  The default values for LSRB could have been set as:

! BEGIN RUNLSRB SPECIFICATIONS (DEFAULT)
!  write-problem-data                            NO
!  problem-data-file-name                        LSRB.data
!  problem-data-file-device                      26
!  write-initial-sif                             NO
!  initial-sif-file-name                         INITIAL.SIF
!  initial-sif-file-device                       51
!  solve                                         blls
!  solve-problem                                 YES
!  relative-residual-accuracy-sought             1.0E-8
!  absolute-residual-accuracy-sought             0.0
!  time-limit                                    -1.0
!  print-full-solution                           NO
!  write-solution                                NO
!  solution-file-name                            LSRBSOL.d
!  solution-file-device                          62
!  write-result-summary                          NO
!  result-summary-file-name                      LSRBRES.d
!  result-summary-file-device                    47
! END RUNLSRB SPECIFICATIONS

!  ----------------- Open and interpret the specfile for LSRB ---------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )

      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!   Define the keywords

        spec(  1 )%keyword = 'write-problem-data'
        spec(  2 )%keyword = 'problem-data-file-name'
        spec(  3 )%keyword = 'problem-data-file-device'
        spec(  4 )%keyword = 'write-initial-sif'
        spec(  5 )%keyword = 'initial-sif-file-name'
        spec(  6 )%keyword = 'initial-sif-file-device'
        spec(  7 )%keyword = 'solve-problem'
        spec(  8 )%keyword = 'solver'
        spec(  9 )%keyword = 'print-full-solution'
        spec( 10 )%keyword = 'write-solution'
        spec( 11 )%keyword = 'solution-file-name'
        spec( 12 )%keyword = 'solution-file-device'
        spec( 13 )%keyword = 'write-result-summary'
        spec( 14 )%keyword = 'result-summary-file-name'
        spec( 15 )%keyword = 'result-summary-file-device'
        spec( 16 )%keyword = 'time-limit'
        spec( 17 )%keyword = 'relative-residual-accuracy-sought'
        spec( 18 )%keyword = 'absolute-residual-accuracy-sought'

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 7 ), do_solve, errout )
        CALL SPECFILE_assign_string ( spec( 8 ), solver, errout )
        CALL SPECFILE_assign_logical( spec( 9 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 10 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 11 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 12 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 13 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 14 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 15 ), rfiledevice, errout )
        CALL SPECFILE_assign_real( spec( 16 ), time_limit, errout )
        CALL SPECFILE_assign_real( spec( 17 ), res_rel, errout )
        CALL SPECFILE_assign_real( spec( 18 ), res_abs, errout )
      END IF

!  ========================= assemble the problem ========================

!  read the problem name from standard input

    READ( 5, "( A )" ) name
    INQUIRE( FILE = TRIM( name ) // ".rb", EXIST = is_file )
    IF ( .NOT. is_file ) THEN
      WRITE( out, "( ' file ', A, '.rb does not exist' )" ) TRIM( name )
      STOP
    END IF

    WRITE( out, "( 15( ' -' ), 1X, A, 15( ' -' ) )" ) TRIM( name )

!   matrix_type = 1 !  real rectangular
!   write(*, "(a,i0)") "matrix type = ", matrix_type

!  read in an mxn matrix in CSC format from a file matrix.rb

!   WRITE( out, "( ' reading ', A, '.rb' )" ) TRIM( name )

!  generate values automatically if not supplied

    options_rb%values = 2
    CALL rb_read( TRIM( name ) // ".rb", p%A%m, p%A%n, p%A%ptr, p%A%row,       &
                  p%A%val, options_rb, info_rb, title = title )
    IF ( info_rb /= 0 ) THEN
      WRITE( out, "( ' rb_read failed with error ', I0 )" ) info_rb
      STOP
    END IF

    DO i = 1, LEN( title )
      IF ( title( i : i ) == ';') EXIT
    END DO
    i = MIN( i - 1 , 59 )
    filea = title( 1 : i )
    WRITE( out, "( ' matrix is formally ', A59 )" ) filea

!  removing null rows and columns

    WRITE( out, "( ' before check m, n, = ', I0, ', ', I0 )" ) p%A%m, p%A%n

!   control_mi35%limit_rowA = MAX( 100, p%A%m / 1000 )
    control_mi35%limit_rowA = - 1
    CALL mi35_check_matrix( p%A%m, p%A%n, p%A%ptr, p%A%row, p%A%val,           &
                            control_mi35, info_mi35 )

    WRITE( out, "(' after check  m, n, = ', I0, ', ', I0,                      &
   &  ' check flag = ', I0 )" ) p%A%m, p%A%n, info_mi35%flag
    WRITE( out, "(' max, min a = ', 2ES12.4 )" )                               &
     MAXVAL( ABS( p%A%val ) ), MINVAL( ABS( p%A%val ) )

    IF ( info_mi35%flag < 0 ) THEN
      IF ( info_mi35%flag == - 4 .AND. p%A%n > 1 ) THEN
      ELSE
        status = GALAHAD_error_mi35
        WRITE( out, "( ' on exit from mi35_check_matrix, status = ', I0 )" )   &
          info_mi35%flag
        STOP
      END IF
    END IF

!  transpose the matrix if m < n

    p%A%ne = p%A%ptr( p%A%n + 1 ) - 1 ; ne = p%A%ne
    transpose_a = p%A%m < p%A%n
    IF ( transpose_a ) THEN
      WRITE( out, "( ' ** transposing matrix as input m < n' )" )
      m = p%A%n ; n = p%A%m
      ALLOCATE( AT_ptr( n + 1 ), AT_row( ne ), AT_val( ne ) )
      CALL CONVERT_transpose( p%A%m, p%A%n, p%A%ne, p%A%ptr, p%A%row, p%A%val, &
                              AT_ptr, AT_row, AT_val )
      DEALLOCATE( p%A%ptr )
      ALLOCATE( p%A%ptr( n + 1 ) )
      p%A%ptr( : n + 1 ) = AT_ptr( : n + 1 )
      p%A%row = AT_row ; p%A%val = AT_val
      p%A%m = m ; p%A%n = n
      DEALLOCATE( AT_ptr, AT_row, AT_val )
!     DO i = 1, p%A%n
!       WRITE( out, "( ' column ', I0, ' rows ', /, ( 10I8 ) )" ) i,           &
!        p%A%row( p%A%ptr( i ) : p%A%ptr( i + 1 ) - 1 )
!     END DO

    ELSE
      m = p%A%m ; n = p%A%n
    END IF
    p%m = m ; p%n = n
    CALL SMT_put( p%A%type, 'SPARSE_BY_COLUMNS', smt_stat )

    WRITE ( out, "( ' m, n, ne = ', I0, ', ', I0, ', ', I0 )" ) m, n, ne

!  obtain the right-hand side; pick a "non" solution that is excluded by
!  the constraints x >= 0

    ALLOCATE( p%B( p%A%m ), p%X( p%A%n ), STAT = alloc_stat )
    rnm1 = REAL( n - 1, KIND = rp_ )
    DO j = 1, n
      rjm1 = REAL( j - 1, KIND = rp_ )
      p%X( j ) = - one + two * rjm1 / rnm1
    END DO

!  generate the right-hand side b = A x

    p%B = zero
    DO j = 1, n
      xj = p%X( j )
      DO l = p%A%ptr( j ), p%A%ptr( j + 1 ) - 1
        i = p%A%row( l )
        p%B( i ) = p%B( i ) + p%A%val( l ) * xj
      END DO
    END DO

    WRITE ( out, "( ' problem data input complete' )" )

!  ------------------- problem set-up complete ----------------------

    CALL CPU_TIME( times )

!  solve the relevant problem

    ALLOCATE( p%C( p%A%m ), p%G( p%A%n ), p%Z( p%A%n ), STAT = alloc_stat )
    ALLOCATE( X_stat( p%A%n ), STAT = alloc_stat )
    SELECT CASE( TRIM( solver ) )
    CASE ( 'blls' )
      ALLOCATE( p%X_l( p%A%n ), p%X_u( p%A%n ), STAT = alloc_stat )
      p%X_l = 0.0_rp_ ; p%X_u = 1.0_rp_
      CALL BLLS_initialize( BLLS_data, BLLS_control, BLLS_inform )
      IF ( is_specfile )                                                       &
        CALL BLLS_read_specfile( BLLS_control, input_specfile )
!     BLLS_control%print_level = 1
      BLLS_inform%status = 1
      CALL BLLS_solve( p, X_stat, BLLS_data, BLLS_control, BLLS_inform,        &
                       userdata )
      status = BLLS_inform%status
      iter = BLLS_inform%iter ; obj = BLLS_inform%obj
      CALL BLLS_terminate( BLLS_data, BLLS_control, BLLS_inform )
    CASE ( 'slls' )
      CALL SLLS_initialize( SLLS_data, SLLS_control, SLLS_inform )
      IF ( is_specfile )                                                       &
        CALL SLLS_read_specfile( SLLS_control, input_specfile )
      SLLS_inform%status = 1
      CALL SLLS_solve( p, X_stat, SLLS_data, SLLS_control, SLLS_inform,        &
                       userdata )
      status = SLLS_inform%status
      iter = SLLS_inform%iter ; obj = SLLS_inform%obj
      CALL SLLS_terminate( SLLS_data, SLLS_control, SLLS_inform )
    END SELECT

    CALL CPU_TIME( timet )
    timet = timet - times ; times = times - time
    WRITE( out, "( /, ' Total time = ', 0P, F12.2 )" ) times + timet
!   WRITE( out, 2000 ) TRIM( name ), TRIM( solver )
!   WRITE( out, "( A5, I7, 6X, ES22.14, I6, 0P, 3F8.2 )" ) TRIM( solver ),     &
!                  iter, obj, status, times, timet, times + timet

    CALL SYMBOLS_status( status, out, ' ', 'solve' )
    CALL CPU_TIME( timet )

!  Print details of the solution obtained

    WRITE( out, "( /,' Stopping with inform%status = ', I0 )" ) status

    IF ( status == 0 .OR. status == - 8 .OR. status == - 9 .OR.                &
        status == - 10 ) THEN
      l = 4
      IF ( fulsol ) l = p%n
      ir = 1 ; ic = MIN( l, p%n )

!  Print details of the primal and dual variables

      SELECT CASE( TRIM( solver ) )
      CASE ( 'blls' )
        WRITE( out, 2010 )
        DO j = 1, 2
          IF ( j == 2 ) THEN
            IF ( ic < p%n - l ) WRITE( out, 2030 )
            ir = MAX( ic + 1, p%n - ic + 1 ) ; ic = p%n
          END IF
          DO i = ir, ic
            state = ' FREE'
            IF ( ABS( p%X  ( i ) - p%X_l( i ) ) < feas_tol ) state = 'LOWER'
            IF ( ABS( p%X  ( i ) - p%X_u( i ) ) < feas_tol ) state = 'UPPER'
            IF ( ABS( p%X_l( i ) - p%X_u( i ) ) < same_tol ) state = 'FIXED'
            WRITE( out, 2020 ) i, state, p%X( i ), p%X_l( i ), p%X_u( i ),     &
                               p%Z( i )
          END DO
        END DO
      CASE ( 'slls' )
        WRITE( out, 2040 )
        DO j = 1, 2
          IF ( j == 2 ) THEN
            IF ( ic < p%n - l ) WRITE( out, 2060 )
            ir = MAX( ic + 1, p%n - ic + 1 ) ; ic = p%n
          END IF
          DO i = ir, ic
            state = ' FREE'
            IF ( p%X ( i ) < feas_tol ) state = 'LOWER'
            WRITE( out, 2050 ) i, state, p%X( i ), p%Z( i )
          END DO
        END DO
      END SELECT

!  Compute the number of fixed and degenerate variables.

      nfixed = 0 ; ndegen = 0
      SELECT CASE( TRIM( solver ) )
      CASE ( 'blls' )
        DO i = 1, n
          IF ( ABS( p%X_u( i ) - p%X_l( i ) ) < same_tol ) THEN
            nfixed = nfixed + 1
            IF ( ABS( p%Z( i ) ) < feas_tol ) ndegen = ndegen + 1
          ELSE IF ( MIN( ABS( p%X( i ) - p%X_l( i ) ),                         &
                    ABS( p%X( i ) - p%X_u( i ) ) ) <=                          &
                    MAX( feas_tol, ABS( p%Z( i ) ) ) ) THEN
            nfixed = nfixed + 1
            IF ( ABS( p%Z( i ) ) < feas_tol ) ndegen = ndegen + 1
          END IF
        END DO
      CASE ( 'slls' )
        DO i = 1, n
          IF ( ABS( p%Z( i ) ) < feas_tol ) ndegen = ndegen + 1
          IF ( p%X( i ) < feas_tol ) nfixed = nfixed + 1
        END DO
      END SELECT

      WRITE( out, "( /, ' Of the ', I0, ' variables, ', I0,                    &
     &                  ' are on bounds & ', I0, ' are dual degenerate' )" )   &
             n, nfixed, ndegen

!  If required, write the solution to a file

      IF ( write_solution ) THEN
        INQUIRE( FILE = sfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',            &
               STATUS = 'OLD', IOSTAT = iores )
        ELSE
           OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',            &
                STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2060 ) iores, sfilename
          STOP
        END IF
        WRITE( sfiledevice, "( /, ' Problem: ', A, /, ' Solver : ',  A, /,     &
       &       ' Objective:', ES24.16 )" ) TRIM( name ), TRIM( solver ), obj
        SELECT CASE( TRIM( solver ) )
        CASE ( 'blls' )
          WRITE( sfiledevice, 2010 )
          DO i = 1, n
            state = ' FREE'
            IF ( ABS( p%X( i )   - p%X_l( i ) ) < feas_tol ) state = 'LOWER'
            IF ( ABS( p%X( i )   - p%X_u( i ) ) < feas_tol ) state = 'UPPER'
            IF ( ABS( p%X_l( I ) - p%X_u( I ) ) < same_tol ) state = 'FIXED'
            WRITE( sfiledevice, 2020 ) i, state, p%X( i ), p%X_l( i ),         &
                                       p%X_u( i ), p%Z( i )
          END DO
        CASE ( 'slls' )
          WRITE( sfiledevice, 2040 )
          DO i = 1, n
            state = ' FREE'
            IF ( p%X( i ) < feas_tol ) state = 'LOWER'
            WRITE( sfiledevice, 2050 ) i, state, p%X( i ), p%Z( i )
          END DO
        END SELECT
        CLOSE( sfiledevice )
      END IF
    END IF

!  Compare the variants used so far

    WRITE( out, 2000 ) TRIM( name ), TRIM( solver )
    WRITE( out, "( A5, I7, 6X, ES22.14, I6, 0P, 3F8.2 )" ) TRIM( solver ),     &
                   iter, obj, status, times, timet, times + timet

    IF ( write_result_summary ) THEN
      BACKSPACE( rfiledevice )
      WRITE( rfiledevice, "( 2A10, 2I7, I6, ES13.4, I6, 0P, F8.2 )" )          &
         TRIM( solver ), TRIM( name ), m, n, iter, obj, status, timet
    END IF

    IF ( is_specfile ) CLOSE( input_specfile )

!  deallocate arrays

    DEALLOCATE( p%X, p%C, p%G, p%Z, X_stat, STAT = alloc_stat )
    DEALLOCATE( p%B, p%A%ptr, p%A%row, p%A%val, STAT = alloc_stat )
    IF ( TRIM( solver ) == 'blls' ) DEALLOCATE( p%X_l, p%X_u, STAT = alloc_stat)

!  Non-executable statements

 2000 FORMAT( /, ' Problem: ', A, ', solver: ', A, //,                         &
                 '                          objective',                        &
                 '              < ------ time ----- > ', /,                    &
                 ' Method  iterations         value      ',                    &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -----------------  ',                  &
                 ' ------ -----    ----   -----  ' )
 2010 FORMAT( /, ' Solution :', /, '                    ',                     &
                 '        <------ Bounds ------> ', /                          &
                 '      #  state    value   ',                                 &
                 '    Lower       Upper       Dual ' )
 2020 FORMAT( I7, 1X, A6, 4ES12.4 )
 2030 FORMAT( '      .  .         ..... ',                                     &
              '   ..........  ..........  .......... ' )
 2040 FORMAT( /, ' Solution :', /, '      #  state    value        Dual ' )
 2050 FORMAT( I7, 1X, A6, 2ES12.4 )
 2060 FORMAT( '      .  .      ..........  ..........' )
 2070 FORMAT( ' IOSTAT = ', I0, ' when opening file ', A, '. Stopping ' )

!  END of program LSRB_main

  END PROGRAM LSRB_main



