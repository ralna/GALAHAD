! THIS VERSION: GALAHAD 2.5 - 10/04/2013 AT 16:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ S H A  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  April 10th 2013

   MODULE GALAHAD_USESHA_double

!  This is the driver program for running SHA for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE GALAHAD_SHA_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_double
     USE GALAHAD_RAND_double
     USE GALAHAD_CLOCK
     USE GALAHAD_CUTEST_FUNCTIONS_double
     USE CUTEst_interface_double

     USE GALAHAD_NORMS_double, ONLY : TWO_NORM
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_SHA

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ S H A   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_SHA( input )

!  Dummy argument

     INTEGER, INTENT( IN ) :: input

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( SHA_control_type ) :: control
     TYPE ( SHA_inform_type ) :: inform
     TYPE ( SHA_data_type ) :: data

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

!  problem input characteristics

     INTEGER :: iores, difs, difs_max, i, k, lh, m, n, nnzh, alloc_stat, status
     REAL ( KIND = wp ) :: pert, clocka, clocke, clocks, level, reduce
     REAL ( KIND = wp ) :: clock( 2 ), error( 2 )
     LOGICAL :: filexx, is_specfile
     CHARACTER ( LEN =  1 ) :: ptype
     CHARACTER ( LEN =  10 ) :: pname
     TYPE ( RAND_seed ) :: seed

     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROW, COL, RD, ROW_COUNT
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VAL, VAL_est
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, X_l, X_u
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: LAMBDA, C_l, C_u
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: S, Y
     LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR

!  input/output file characteristics

     INTEGER :: out  = 6
     INTEGER :: errout = 6
     INTEGER, PARAMETER :: io_buffer = 11

!  functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  specfile characteristics

     INTEGER, PARAMETER :: input_specfile = 34
     INTEGER, PARAMETER :: lspec = 8
     CHARACTER ( LEN = 16 ) :: specname = 'RUNSHA'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNSHA.SPC'

!  default values for specfile-defined parameters

     INTEGER :: print_level = 1
     INTEGER :: max_sy = 100
     INTEGER :: rfiledevice = 47
     INTEGER :: sfiledevice = 62
     LOGICAL :: write_solution       = .FALSE.
     LOGICAL :: write_result_summary = .FALSE.
     CHARACTER ( LEN = 30 ) :: rfilename = 'SHARES.d'
     CHARACTER ( LEN = 30 ) :: sfilename = 'SHASOL.d'

!  ------------------ Open the specfile for SHA ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword = 'write-solution'
       spec( 2 )%keyword = 'solution-file-name'
       spec( 3 )%keyword = 'solution-file-device'
       spec( 4 )%keyword = 'write-result-summary'
       spec( 5 )%keyword = 'result-summary-file-name'
       spec( 6 )%keyword = 'result-summary-file-device'
       spec( 7 )%keyword = 'maximum-differences-allowed'
       spec( 8 )%keyword = 'print-level'

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 2 ), sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 3 ), sfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 4 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 5 ), rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 6 ), rfiledevice, errout )
       CALL SPECFILE_assign_integer( spec( 7 ), max_sy, errout )
       CALL SPECFILE_assign_integer( spec( 8 ), print_level, errout )
     END IF

!  If required, open a file for the results

     IF ( write_result_summary ) THEN
       INQUIRE( FILE = rfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',             &
                STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
       ELSE
          OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',             &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         WRITE( errout,                                                        &
        &  "( ' IOSTAT = ', I0, ' when opening file ', A, '. Stopping ' )" )   &
           iores, rfilename
         STOP
       END IF
       READ( INPUT, "( /, I2, A8  )" ) iores, pname
       REWIND( input )
       WRITE( rfiledevice, "( A10 )" ) pname
     END IF

!  If required, open a file for the solution

     IF ( write_solution ) THEN
       INQUIRE( FILE = sfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',             &
                STATUS = 'OLD', IOSTAT = iores )
       ELSE
          OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',             &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         WRITE( errout,                                                        &
        &  "( ' IOSTAT = ', I0, ' when opening file ', A, '. Stopping ' )" )   &
           iores, sfilename
         STOP
       END IF
     END IF

!  set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2013' )

!  set up control parameters prior to the next solution

     CALL SHA_initialize( data, control, inform )
     IF ( is_specfile ) CALL SHA_read_specfile( control, input_specfile )

!  determine the number of variables

     CALL CUTEST_cdimen( status, input, n, m )
     IF ( status /= 0 ) GO TO 910

!  allocate suitable arrays

     ALLOCATE( X( n ), X_l( n ), X_u( n ), ROW_COUNT( n ), STAT = alloc_stat )
     IF ( alloc_stat /= 0 ) THEN
       WRITE( out, 2000 ) 'X', alloc_stat ; STOP
     END IF

     IF ( m > 0 ) THEN
       ALLOCATE( LAMBDA( m ), C_l( m ), C_u( m ), EQUATN( m ), LINEAR( m ),    &
                 STAT = alloc_stat )
       IF ( alloc_stat /= 0 ) THEN
         WRITE( out, 2000 ) 'X', alloc_stat ; STOP
       END IF
     END IF

!  set up the data structures necessary to hold the group partially
!  separable objective function

     IF ( m > 0 ) THEN
       CALL CUTEST_csetup( status, input, out, io_buffer, n, m, X, X_l, X_u,   &
                           LAMBDA, C_l, C_u, EQUATN, LINEAR, 0, 0, 0 )
       ptype = 'C'
     ELSE
       CALL CUTEST_usetup( status, input, out, io_buffer, n, X, X_l, X_u )
       ptype = 'U'
     END IF
     IF ( status /= 0 ) GO TO 910

!  perturb X

     DO i = 1, n
       CALL RAND_random_real( seed, .TRUE., pert )
       IF ( X_l( i ) == X_u( i ) ) THEN
         X( i ) = X_l( i )
       ELSE IF ( X( i ) <= X_l( i ) ) THEN
         X( i ) = X_l( i ) + pert * MIN( 1.0_wp, X_u( i ) - X_l( i ) )
       ELSE IF ( X( i ) >= X_u( i ) ) THEN
         X( i ) = X_u( i ) - pert * MIN( 1.0_wp, X_u( i ) - X_l( i ) )
       ELSE
         X( i ) = X( i ) + pert * MIN( 1.0_wp, X_u( i ) - X( i ) )
       END IF
     END DO

     IF ( m > 0 ) THEN
       DO i = 1, m
         CALL RAND_random_real( seed, .FALSE., LAMBDA( i ) )
       END DO
     END IF

!  discover the problem name

     CALL CUTEST_probname( status, pname )
     IF ( status /= 0 ) GO TO 910

!  compute the number of nonzeros in the Hessian

     IF ( m > 0 ) THEN
       CALL CUTEST_cdimsh( status, nnzh )
     ELSE
       CALL CUTEST_udimsh( status, nnzh )
     END IF
     IF ( status /= 0 ) GO TO 910

     WRITE( 6, "( ' Problem ', A, ': n = ', I0, ', nnzh = ', I0 )" )           &
       TRIM( pname ), n, nnzh

!  allocate space to hold the Hessian

     lh = nnzh
     ALLOCATE( ROW( lh ), COL( lh ), VAL( lh ), VAL_est( lh ),                 &
               STAT = alloc_stat )
     IF ( alloc_stat /= 0 ) THEN
       WRITE( out, 2000 ) 'VAL', alloc_stat ; STOP
     END IF

!  compute the Hessian sparsity pattern

     IF ( m > 0 ) THEN
!      CALL CUTEST_cshp( status, n, nnzh, lh, ROW, COL )
     ELSE
!      CALL CUTEST_ushp( status, n, nnzh, lh, ROW, COL )
     END IF
     IF ( status /= 0 ) GO TO 910

!  compute the exact Hessian for comparison purposes

     IF ( m > 0 ) THEN
       CALL CUTEST_csh( status, n, m, X, LAMBDA, nnzh, lh, VAL, ROW, COL )
     ELSE
       CALL CUTEST_ush( status, n, X, nnzh, lh, VAL, ROW, COL )
      END IF

     IF ( status /= 0 ) GO TO 910

!  for debugging ... report on row counts

!  ---------------------------------- IGNORED ----------------------------------
!  |                                                                           |
!  v                                                                           v
     IF ( .FALSE. ) THEN
       CALL SHA_count( n, nnzh, ROW, COL, ROW_COUNT )

       WRITE( out, "( A10, 8I7 )" )                                            &
         pname, n, MAXVAL( ROW_COUNT( 1 : n ) ),                               &
         COUNT( ROW_COUNT( 1 : n ) >= 5 ),                                     &
         COUNT( ROW_COUNT( 1 : n ) >= 10 ),                                    &
         COUNT( ROW_COUNT( 1 : n ) >= 20 ),                                    &
         COUNT( ROW_COUNT( 1 : n ) >= 50 ),                                    &
         COUNT( ROW_COUNT( 1 : n ) >= 100 ),                                   &
         COUNT( ROW_COUNT( 1 : n ) >= 500 )

       IF ( write_result_summary ) THEN
         BACKSPACE( rfiledevice )
         WRITE( rfiledevice, "( A10, 8I7 )" )                                  &
           pname, n, MAXVAL( ROW_COUNT( 1 : n ) ),                             &
           COUNT( ROW_COUNT( 1 : n ) >= 5 ),                                   &
           COUNT( ROW_COUNT( 1 : n ) >= 10 ),                                  &
           COUNT( ROW_COUNT( 1 : n ) >= 20 ),                                  &
           COUNT( ROW_COUNT( 1 : n ) >= 50 ),                                  &
           COUNT( ROW_COUNT( 1 : n ) >= 100 ),                                 &
           COUNT( ROW_COUNT( 1 : n ) >= 500 )
       END IF
       RETURN
     END IF
!  ^                                                                           ^
!  |                                                                           |
!  --------------------------- END OF IGNORED ----------------------------------

 ! analyse the sparsity

     CALL CLOCK_time( clocks )
     CALL SHA_analyse( n, nnzh, ROW, COL, data, control, inform )
     CALL CLOCK_time( clocke )
     clocka = clocke - clocks
     IF ( inform%status /= 0 ) THEN
       WRITE( 6, "( ' return with error status ', I0, ' from SHA_analyse' )" ) &
         inform%status ; STOP
     END IF

     WRITE( 6, "( ' maximum degree in the connectivity graph = ', I0 )" )      &
       inform%max_degree
     WRITE( 6, "( 1X, I0, ' differences required, partial clock time = ',      &
    &             F0.3 )" ) inform%differences_needed, clocka

!  allocate space for the differences

     difs_max = MIN( inform%differences_needed, max_sy )
     ALLOCATE( S( n, difs_max ), Y( n, difs_max ), RD( difs_max ),             &
               STAT = alloc_stat )
     IF ( alloc_stat /= 0 ) THEN
       WRITE( out, 2000 ) 'S, Y, RD', alloc_stat ; STOP
     END IF

! choose random S and compute the product Y = H * S

     level = 1.0_wp
!    reduce = ( 0.00000001_wp ) ** ( 1.0_wp / REAL( difs_max, wp ) )
     reduce = 1.0_wp
     CALL RAND_initialize( seed )
     DO k = 1, difs_max
       RD( k ) = difs_max - k + 1
       DO i = 1, n
         CALL RAND_random_real( seed, .FALSE., S( i, k ) )
         S( i, k ) = S( i, k ) * level
       END DO
!write(6, "( ' ||s(', I0, ')|| = ', ES12.4 )" ) k, TWO_NORM( S( : n, k ) )
       level = level * reduce
       IF ( m > 0 ) THEN
         CALL CUTEST_chprod( status, n, m, .FALSE., X, LAMBDA, S( : , k ),     &
                             Y( : , k ) )
       ELSE
         CALL CUTEST_uhprod( status, n, .FALSE., X, S( : , k ), Y( : , k ) )
       END IF
       IF ( status /= 0 ) GO TO 910
!      Y( : , k ) = Y( : , k ) + 0.0001_wp
     END DO

!  ---------------------------------- IGNORED ----------------------------------
!  |                                                                           |
!  v                                                                           v
     IF ( .FALSE. ) THEN

! approximate the Hessian ignoring symmetry

       IF ( inform%differences_needed <= max_sy ) THEN
         difs = inform%differences_needed
         control%approximation_algorithm = 0

         CALL CLOCK_time( clocks )
         CALL SHA_estimate( n, nnzh, ROW, COL, difs, difs, RD, n, difs, S,     &
                            n, difs, Y, VAL_est, data, control, inform )
         CALL CLOCK_time( clocke )
         IF ( inform%status /= 0 ) THEN
           WRITE( 6, "( ' return with error status ',I0,' from SHA_estimate')")&
             inform%status ; STOP
         END IF

!  compute the error

         IF ( nnzh > 0 ) THEN
           error( 1 ) = MAXVAL( ABS( VAL( : nnzh ) - VAL_est( : nnzh ) ) /    &
                                MAX( 1.0_wp, ABS( VAL( : nnzh ) ) ) )
         ELSE
           error( 1 ) = 0.0_wp
         END IF
         clock( 1 ) = clocke - clocks + clocka
         WRITE( 6, "( ' maximum error (nosym) = ', ES11.4, ' clock time = ',   &
        &              F0.3 )" ) error( 1 ), clock( 1 )
!      WRITE( 6, "( ' maximum values (true, estimate) = ', 2ES11.4 )" )        &
!        MAXVAL( ABS( VAL( : nnzh ) ) ), MAXVAL( ABS( VAL_est( : nnzh ) ) )

!DO i = 1, nnzh
! write(6,"( 3ES12.4 )" ) ABS( VAL( i ) - VAL_est( i ) ), ABS( VAL( i ) ), ABS( VAL_est( i ) )
!END DO
       ELSE
         error( 1 ) = - 1.0_wp ; clock( 1 ) = - 1.0_wp
       END IF

! approximate the Hessian accounting for symmetry

       IF ( inform%differences_needed <= max_sy ) THEN
         difs = inform%differences_needed
         control%approximation_algorithm = 1

         CALL CLOCK_time( clocks )
         CALL SHA_estimate( n, nnzh, ROW, COL, difs, difs, RD, n, difs, S,     &
                            n, difs, Y, VAL_est, data, control, inform )
         CALL CLOCK_time( clocke )
         IF ( inform%status /= 0 ) THEN
           WRITE( 6, "( ' return with error status ',I0,' from SHA_estimate')")&
             inform%status ; STOP
         END IF

!  compute the error

         IF ( nnzh > 0 ) THEN
           error( 2 ) = MAXVAL( ABS( VAL( : nnzh ) - VAL_est( : nnzh ) ) /     &
                                MAX( 1.0_wp, ABS( VAL( : nnzh ) ) ) )
         ELSE
           error( 2 ) = 0.0_wp
         END IF
         clock( 2 ) = clocke - clocks + clocka
         WRITE( 6, "( ' maximum error (sym)   = ', ES11.4, ' clock time = ',   &
        &              F0.3 )" ) error( 2 ), clock( 2 )
!      WRITE( 6, "( ' maximum values (true, estimate) = ', 2ES11.4 )" )        &
!        MAXVAL( ABS( VAL( : nnzh ) ) ), MAXVAL( ABS( VAL_est( : nnzh ) ) )

!DO i = 1, nnzh
! write(6,"( 3ES12.4 )" ) ABS( VAL( i ) - VAL_est( i ) ), ABS( VAL( i ) ), ABS( VAL_est( i ) )
!END DO
       ELSE
         error( 2 ) = - 1.0_wp ; clock( 2 ) = - 1.0_wp
       END IF

       IF ( write_result_summary ) THEN
         BACKSPACE( rfiledevice )
         WRITE( rfiledevice, "( A10, 3I7, 2( ES12.4, F12.3 ) )" ) pname, n,    &
           inform%max_degree, inform%differences_needed, error( 1 ),           &
           clock( 1 ), error( 2 ), clock( 2 )
       END IF
     END IF
!  ^                                                                           ^
!  |                                                                           |
!  --------------------------- END OF IGNORED ----------------------------------

!  approximate the Hessian

!    IF ( inform%differences_needed <= max_sy ) THEN
!      control%approximation_algorithm = 0
!      difs = inform%differences_needed
       difs = difs_max

       CALL CLOCK_time( clocks )
       CALL SHA_estimate( n, nnzh, ROW, COL, difs_max, difs, RD, n, difs, S,   &
                          n, difs, Y, VAL_est, data, control, inform )
       CALL CLOCK_time( clocke )
       IF ( inform%status /= 0 ) THEN
         WRITE( 6, "( ' return with error status ', I0, ' from SHA_estimate')")&
           inform%status ; STOP
       END IF

!  compute the error

       IF ( nnzh > 0 ) THEN
         error( 1 ) = MAXVAL( ABS( VAL( : nnzh ) - VAL_est( : nnzh ) ) /       &
                              MAX( 1.0_wp, ABS( VAL( : nnzh ) ) ) )
       ELSE
         error( 1 ) = 0.0_wp
       END IF
       clock( 1 ) = clocke - clocks + clocka
       WRITE( 6, "( ' maximum error (alg=', I0, ',lin=', I0, ') = ', ES10.4,   &
     &  ', clock time = ', F0.3 )" ) control%approximation_algorithm,          &
          control%dense_linear_solver, error( 1 ), clock( 1 )

!    WRITE( 6, "( ' maximum values (true, estimate) = ', 2ES11.4 )" )          &
!      MAXVAL( ABS( VAL( : nnzh ) ) ), MAXVAL( ABS( VAL_est( : nnzh ) ) )
     IF ( print_level > 1 ) THEN
       WRITE( 6, "( '    row    col     true         est       error' )" )
       DO i = 1, nnzh
        WRITE( 6, "( 2I7, 3ES12.4 )" ) ROW( i ), COL( i ),                     &
         ABS( VAL( i ) ), ABS( VAL_est( i ) ), ABS( VAL( i ) - VAL_est( i ) )
       END DO
     END IF

     IF ( write_result_summary ) THEN
       BACKSPACE( rfiledevice )
       WRITE( rfiledevice, "( A10, 1X, A1, 3I7, 2( ES12.4, F12.3, I4 ) )" )    &
         pname, ptype, n, inform%max_degree, inform%differences_needed,        &
         error( 1 ), clock( 1 ), control%approximation_algorithm
       CLOSE( rfiledevice )
     END IF

     IF ( write_solution ) THEN
       WRITE( sfiledevice, "( A, ': dimension = ', I0, ', nonzeros = ', I0,    &
      &  ', row, col, val, true val =', /, ( 2I10, 2ES24.16, ES11.4 ) )" )     &
          TRIM( pname ), n, nnzh, ( ROW( i ), COL( i ), VAL_est( i ),          &
          VAL( i ), ABS( VAL( i ) - VAL_est( i ) ) /                           &
          MAX( 1.0_wp, ABS( VAL( i ) ) ), i = 1, nnzh )
       CLOSE( sfiledevice )
     END IF

!  close any opened files and deallocate arrays

     IF ( is_specfile ) CLOSE( input_specfile )
     IF ( m > 0 ) THEN
       CALL CUTEST_cterminate( status )
     ELSE
       CALL CUTEST_uterminate( status )
     END IF
     RETURN

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', I0, ', stopping' )" ) status
     RETURN

!  Non-executable statements

 2000 FORMAT( ' Allocation error, variable ', A, ' status = ', I0 )

!  End of subroutine USE_SHA

     END SUBROUTINE USE_SHA

!  End of module USESHA_double

   END MODULE GALAHAD_USESHA_double
