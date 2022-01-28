   PROGRAM PSLS_TEST_PROGRAM  !  GALAHAD 4.0 - 2022-01-24 AT 09:30 GMT.
   USE GALAHAD_PSLS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( SMT_type ) :: matrix
   TYPE ( PSLS_data_type ) :: data
   TYPE ( PSLS_control_type ) control
   TYPE ( PSLS_inform_type ) :: inform
   INTEGER, PARAMETER :: n = 5
   INTEGER, PARAMETER :: ne = 7
   REAL ( KIND = wp ) :: X( n )
   INTEGER :: SUB( 3 ) = (/ 1, 3, 5 /)
   INTEGER :: i, s, stat_faf, stat_sol, prec
   CHARACTER ( LEN = 1 ) :: st

!  test basic storage and preconditioner options

   WRITE( 6, "( ' storage and preconditioner tests', / )" )
! run through storage types
   DO i = 1, 3
! allocate and set lower triangle of matrix
    IF ( i == 1 ) THEN ! co-ordinate form
       st = 'C'
       CALL SMT_put( matrix%type, 'COORDINATE', s )
       matrix%n = n ; matrix%ne = ne
       ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
       matrix%row = (/ 1, 2, 3, 3, 4, 5, 5 /)
       matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
       matrix%val = (/ 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp,               &
                               6.0_wp, 1.0_wp /)
     ELSE IF ( i == 2 ) THEN ! sparse-row form
       st = 'R'
       CALL SMT_put( matrix%type, 'SPARSE_BY_ROWS', s )
       matrix%n = n
       ALLOCATE( matrix%val( ne ), matrix%col( ne ), matrix%ptr( n + 1 ) )
       matrix%ptr = (/ 1, 2, 3, 5, 6, 8 /)
       matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
       matrix%val = (/ 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp,               &
                               6.0_wp, 1.0_wp /)
     ELSE IF ( i == 3 ) THEN ! dense form
       st = 'D'
       CALL SMT_put( matrix%type, 'DENSE', s )
       matrix%n = n
       ALLOCATE( matrix%val( n * ( n + 1 ) / 2 ) )
       matrix%val = (/ 2.0_wp, 3.0_wp, 0.0_wp, 0.0_wp, 4.0_wp, 1.0_wp,       &
                       0.0_wp, 0.0_wp, 5.0_wp, 0.0_wp, 0.0_wp, 6.0_wp,       &
                       0.0_wp, 0.0_wp, 1.0_wp /)
     END IF
! run through preconditioners
     DO prec = - 1, 7
! specify the solver used by SLS (in this case sils)
       CALL PSLS_initialize( data, control, inform )
       control%preconditioner = prec
       control%semi_bandwidth = 1  ! semi-bandwidth of one
       control%definite_linear_solver = 'sils'
! problem setup complete. form and factorize the preconditioner, P
       CALL PSLS_form_and_factorize( matrix, data, control, inform )
       stat_faf = inform%status
       IF ( stat_faf == 0 ) THEN
! use the factors to solve P x = b, with b input in x
         X( : n ) = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
         CALL PSLS_apply( X, data, control, inform )
         stat_sol = inform%status
!        IF ( inform%status == 0 ) THEN
!          WRITE( 6, "( ' PSLS - Preconditioned solution is ', 5F6.2 )" ) X
!        ELSE 
!          WRITE( 6, "( ' PSLS - exit status = ', I0 )" ) inform%status
!        END IF
       ELSE
         stat_sol = - 1
       END IF
       WRITE( 6, "( I2, A, ' prec:storage: status form & fact = ', I3,         &
      &           ' solve = ', I3 )" ) prec, st, stat_faf, stat_sol
  ! clean up
       CALL PSLS_terminate( data, control, inform )
     END DO
     IF ( i == 1 ) THEN ! co-ordinate form
       DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
     ELSE IF ( i == 2 ) THEN ! spares-row form
       DEALLOCATE( matrix%type, matrix%val, matrix%col, matrix%ptr )
     ELSE IF ( i == 3 ) THEN ! dense form
       DEALLOCATE( matrix%type, matrix%val )
     END IF
   END DO

!  test basic storage and preconditioner options for submatrices

   WRITE( 6, "( /, ' storage and preconditioner tests with submatrices', / )" )
! run through storage types
   DO i = 1, 3
! allocate and set lower triangle of matrix
    IF ( i == 1 ) THEN ! co-ordinate form
       st = 'C'
       CALL SMT_put( matrix%type, 'COORDINATE', s )
       matrix%n = n ; matrix%ne = ne
       ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
       matrix%row = (/ 1, 2, 3, 3, 4, 5, 5 /)
       matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
       matrix%val = (/ 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp,               &
                               6.0_wp, 1.0_wp /)
     ELSE IF ( i == 2 ) THEN ! sparse-row form
       st = 'R'
       CALL SMT_put( matrix%type, 'SPARSE_BY_ROWS', s )
       matrix%n = n
       ALLOCATE( matrix%val( ne ), matrix%col( ne ), matrix%ptr( n + 1 ) )
       matrix%ptr = (/ 1, 2, 3, 5, 6, 8 /)
       matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
       matrix%val = (/ 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp,               &
                               6.0_wp, 1.0_wp /)
     ELSE IF ( i == 3 ) THEN ! dense form
       st = 'D'
       CALL SMT_put( matrix%type, 'DENSE', s )
       matrix%n = n
       ALLOCATE( matrix%val( n * ( n + 1 ) / 2 ) )
       matrix%val = (/ 2.0_wp, 3.0_wp, 0.0_wp, 0.0_wp, 4.0_wp, 1.0_wp,       &
                       0.0_wp, 0.0_wp, 5.0_wp, 0.0_wp, 0.0_wp, 6.0_wp,       &
                       0.0_wp, 0.0_wp, 1.0_wp /)
     END IF
! run through preconditioners
     DO prec = - 1, 7
! specify the solver used by SLS (in this case sils)
       CALL PSLS_initialize( data, control, inform )
       control%preconditioner = prec
       control%semi_bandwidth = 1  ! semi-bandwidth of one
       control%definite_linear_solver = 'sils'
! problem setup complete. form and factorize the preconditioner, P
       CALL PSLS_form_and_factorize( matrix, data, control, inform, SUB = SUB )
       stat_faf = inform%status
       IF ( stat_faf == 0 ) THEN
! use the factors to solve P x = b, with b input in x
         X( : n ) = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
         CALL PSLS_apply( X, data, control, inform )
         stat_sol = inform%status
!        IF ( inform%status == 0 ) THEN
!          WRITE( 6, "( ' PSLS - Preconditioned solution is ', 5F6.2 )" ) X
!        ELSE 
!          WRITE( 6, "( ' PSLS - exit status = ', I0 )" ) inform%status
!        END IF
       ELSE
         stat_sol = - 1
       END IF
       WRITE( 6, "( I2, A, ' prec:storage: status form & fact = ', I3,         &
      &           ' solve = ', I3 )" ) prec, st, stat_faf, stat_sol
  ! clean up
       CALL PSLS_terminate( data, control, inform )
     END DO
     IF ( i == 1 ) THEN ! co-ordinate form
       DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
     ELSE IF ( i == 2 ) THEN ! spares-row form
       DEALLOCATE( matrix%type, matrix%val, matrix%col, matrix%ptr )
     ELSE IF ( i == 3 ) THEN ! dense form
       DEALLOCATE( matrix%type, matrix%val )
     END IF
   END DO

! Test error returns

   WRITE( 6, "( /, ' error tests', / )" )
! specify the generic problem
   CALL SMT_put( matrix%type, 'COORDINATE', s )
   matrix%n = n ; matrix%ne = ne
   ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
   matrix%row = (/ 1, 2, 3, 3, 4, 5, 5 /)
   matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
   matrix%val = (/ 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp, 6.0_wp, 1.0_wp /)
   DO i = 1, 4
! specify the solver used by SLS (in this case sils)
     CALL PSLS_initialize( data, control, inform )
     control%preconditioner = 2  ! band preconditioner
     control%semi_bandwidth = 1  ! semi-bandwidth of one
     control%definite_linear_solver = 'sils'
! provoke error condition
     SELECT CASE ( i )
     CASE( 1 )
       matrix%n = - 1 ; matrix%ne = ne
     CASE( 2 )
       CALL SMT_put( matrix%type, 'BAD', s )
       matrix%n = n ; matrix%ne = ne
     CASE( 3 )
       CALL SMT_put( matrix%type, 'COORDINATE', s )
       control%preconditioner = 5
       control%definite_linear_solver = 'bad'
     CASE( 4 )
       control%definite_linear_solver = 'sils'
       control%preconditioner = 10
     CASE( 5 )
       control%preconditioner = 7
       control%mi28_lsize = - 1
       control%mi28_rsize = - 1
     CASE DEFAULT
       control%preconditioner = 2
     END SELECT

! problem setup complete. form and factorize the preconditioner, P
     CALL PSLS_form_and_factorize( matrix, data, control, inform )
     stat_faf = inform%status
     WRITE( 6, "( ' status form  = ', I3 )" ) stat_faf
   END DO
   DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
   STOP
   END PROGRAM PSLS_TEST_PROGRAM
