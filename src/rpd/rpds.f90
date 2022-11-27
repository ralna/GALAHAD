! THIS VERSION: GALAHAD 4.1 - 2022-11-27 AT 14:00 GMT.
   PROGRAM GALAHAD_RPD_example
   USE GALAHAD_RPD_double                       ! double precision version
   IMPLICIT NONE
   TYPE ( RPD_control_type ) :: control
   TYPE ( RPD_inform_type ) :: inform
   TYPE ( QPT_problem_type ) :: prob
   INTEGER :: i, length
   INTEGER :: qplib_unit = 21
   CHARACTER ( LEN = 8 ) :: galahad_var = 'GALAHAD'
   CHARACTER( LEN = : ), ALLOCATABLE :: galahad
!  open the QPLIB file ALLINIT.qplib for reading on unit 21
   CALL GET_ENVIRONMENT_VARIABLE( galahad_var, length = length )
   ALLOCATE( CHARACTER( LEN = length ):: galahad )
   CALL GET_ENVIRONMENT_VARIABLE( galahad_var, value = galahad )
   OPEN( qplib_unit, file = galahad // "/examples/ALLINIT.qplib",              &
         FORM = 'FORMATTED', STATUS = 'OLD' )
   CALL RPD_initialize( control, inform )
   control%qplib = qplib_unit
!  collect the problem statistics
   CALL RPD_read_problem_data( prob, control, inform )
   WRITE( 6, "( ' read status = ', I0 )" ) inform%status
   WRITE( 6, "( ' qplib example ALLINIT type = ', A )" ) inform%p_type
   WRITE( 6, "( ' n, m, h_ne, a_ne, h_c_ne =', 5I3 )" )                        &
     prob%n, prob%m, prob%H%ne, prob%A%ne, prob%H_c%ne
!  close the QPLIB file after reading
   CLOSE( qplib_unit )
   WRITE( 6, "( ' G =', 5F5.1 )" ) prob%G
   WRITE( 6, "( ' f =', F5.1 )" ) prob%f
   WRITE( 6, "( ' X_l =', 5F4.1 )" ) prob%X_l
   WRITE( 6, "( ' X_u =', 5F4.1 )" ) prob%X_u
   WRITE( 6, "( ' C_l =', 2F4.1 )" ) prob%C_l
   WRITE( 6, "( ' C_u =', 2ES8.1 )" ) prob%C_u
   IF ( ALLOCATED( prob%H%row ) .AND. ALLOCATED( prob%H%col ) .AND.            &
        ALLOCATED( prob%H%val ) ) THEN
     DO i = 1, prob%H%ne
       WRITE( 6, "( ' H(row, col, val) =', 2I3, F5.1 )" )                      &
         prob%H%row( i ), prob%H%col( i ), prob%H%val( i )
     END DO
   END IF
   IF ( ALLOCATED( prob%A%row ) .AND. ALLOCATED( prob%A%col ) .AND.            &
        ALLOCATED( prob%A%val ) ) THEN
     DO i = 1, prob%A%ne
       WRITE( 6, "( ' A(row, col, val) =', 2I3, F5.1 )" )                      &
         prob%A%row( i ), prob%A%col( i ), prob%A%val( i )
     END DO
   END IF
   IF ( ALLOCATED( prob%H_c%ptr ) .AND. ALLOCATED( prob%H_c%row ) .AND.        &
        ALLOCATED( prob%H_c%col ) .AND. ALLOCATED( prob%H_c%val ) ) THEN
     DO i = 1, prob%H_c%ne
       WRITE( 6, "( ' H_c(ptr, row, col, val) =', 3I3, F5.1 )" )               &
         prob%H_c%ptr( i ), prob%H_c%row( i ), prob%H_c%col( i ),              &
         prob%H_c%val( i )
     END DO
   END IF
   WRITE( 6, "( ' X_type =', 5I2 )") prob%X_type
   WRITE( 6, "( ' X =', 5F4.1 )" ) prob%X
   WRITE( 6, "( ' Y =', 2F4.1 )" ) prob%Y
   WRITE( 6, "( ' Z =', 5F4.1 )" ) prob%Z
!  deallocate internal array space
   CALL RPD_terminate( prob, control, inform )
   END PROGRAM GALAHAD_RPD_example
