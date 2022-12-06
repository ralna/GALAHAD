! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_SBLS_EXAMPLE
   USE GALAHAD_SBLS_double                    ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SMT_type ) :: H, A, C
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
   TYPE ( SBLS_data_type ) :: data
   TYPE ( SBLS_control_type ) :: control        
   TYPE ( SBLS_inform_type ) :: inform
   INTEGER :: s
   INTEGER :: n = 3, m = 2, h_ne = 4, a_ne = 4, c_ne = 1
   INTEGER :: data_storage_type = 0
! start problem data
   ALLOCATE( SOL( n + m ) )
   SOL( 1 : n ) = (/ 7.0_wp, 4.0_wp, 8.0_wp /)  ! RHS a
   SOL( n + 1 : n + m ) = (/ 2.0_wp, 1.0_wp /)  ! RHS b
! sparse co-ordinate storage format
   IF ( data_storage_type == 0 ) THEN
   CALL SMT_put( H%type, 'COORDINATE', s )  ! Specify co-ordinate 
   CALL SMT_put( A%type, 'COORDINATE', s )  ! storage for H, A and C
   CALL SMT_put( C%type, 'COORDINATE', s )  
   ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
   ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
   ALLOCATE( C%val( c_ne ), C%row( c_ne ), C%col( c_ne ) )
   H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /) ! matrix H
   H%row = (/ 1, 2, 3, 3 /)                     ! NB lower triangle
   H%col = (/ 1, 2, 3, 1 /) ; H%ne = h_ne
   A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! matrix A
   A%row = (/ 1, 1, 2, 2 /)
   A%col = (/ 1, 2, 2, 3 /) ; A%ne = a_ne
   C%val = (/ 1.0_wp /) ! matrix C
   C%row = (/ 2 /) ! NB lower triangle
   C%col = (/ 1 /) ; C%ne = c_ne
! sparse row-wise storage format
   ELSE IF ( data_storage_type == - 1 ) THEN
   CALL SMT_put( H%type, 'SPARSE_BY_ROWS', s )  ! Specify sparse-by-rows
   CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )  ! storage for H, A and C
   CALL SMT_put( C%type, 'SPARSE_BY_ROWS', s )
   ALLOCATE( H%val( h_ne ), H%col( h_ne ), H%ptr( n + 1 ) )
   ALLOCATE( A%val( a_ne ), A%col( a_ne ), A%ptr( m + 1 ) )
   ALLOCATE( C%val( c_ne ), C%col( c_ne ), C%ptr( m + 1 ) )
   H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /) ! matrix H
   H%col = (/ 1, 2, 3, 1 /)                     ! NB lower triangular
   H%ptr = (/ 1, 2, 3, 5 /)                     ! Set row pointers
   A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! matrix A
   A%col = (/ 1, 2, 2, 3 /)
   A%ptr = (/ 1, 3, 5 /)                        ! Set row pointers  
   C%val = (/ 1.0_wp /)                         ! matrix C
   C%col = (/ 1 /)                              ! NB lower triangular
   C%ptr = (/ 1, 1, 2 /)                        ! Set row pointers
! dense storage format
   ELSE
   CALL SMT_put( H%type, 'DENSE', s )  ! Specify dense
   CALL SMT_put( A%type, 'DENSE', s )  ! storage for H, A and C
   CALL SMT_put( C%type, 'DENSE', s )
   ALLOCATE( H%val( n * ( n + 1 ) / 2 ) )
   ALLOCATE( A%val( n * m ) )
   ALLOCATE( C%val( m * ( m + 1 ) / 2 ) )
   H%val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 4.0_wp, 0.0_wp, 3.0_wp /) ! H
   A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 1.0_wp /) ! A
   C%val = (/ 0.0_wp, 1.0_wp, 0.0_wp /)                         ! C
! problem data complete   
   END IF
   CALL SBLS_initialize( data, control )        ! Initialize control parameters
!  control%print_level = 1
   control%preconditioner = 2
! factorize matrix
   CALL SBLS_form_and_factorize( n, m, H, A, C, data, control, inform )
   IF ( inform%status < 0 ) THEN                !  Unsuccessful call
     WRITE( 6, "( ' SBLS_form_and_factorize exit status = ', I6 ) " )          &
       inform%status
     STOP
   END IF
! solve system
   CALL SBLS_solve( n, m, A, C, data, control, inform, SOL )
   IF ( inform%status == 0 ) THEN                !  Successful return
     WRITE( 6, "( ' SBLS: Solution = ', /, ( 5ES12.4 ) )" ) SOL
   ELSE                                          !  Error returns
     WRITE( 6, "( ' SBLS_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL SBLS_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_SBLS_EXAMPLE

