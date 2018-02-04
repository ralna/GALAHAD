! THIS VERSION: GALAHAD 2.6 - 13/05/2014 AT 15:00 GMT.
   PROGRAM GALAHAD_MIQR_TEST
   USE GALAHAD_MIQR_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SMT_type ) :: mat
   TYPE ( MIQR_data_type ) :: data
   TYPE ( MIQR_control_type ) :: control        
   TYPE ( MIQR_inform_type ) :: inform
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
   INTEGER :: s
   INTEGER, PARAMETER :: prob_number = 1
   INTEGER, PARAMETER :: m_1 = 4, n_1 = 3, a_ne_1 = 5
   INTEGER, PARAMETER :: m_2 = 3, n_2 = 3, a_ne_2 = 4
   INTEGER, PARAMETER :: m_3 = 3, n_3 = 3, a_ne_3 = 4
! set problem data
   SELECT CASE ( prob_number )
   CASE ( 2 ) 
     mat%m = 3 ; mat%n = 3 ; mat%ne = 4
   CASE ( 3 ) 
     mat%m = 3 ; mat%n = 3 ; mat%ne = 4
   CASE DEFAULT
     mat%m = 4 ; mat%n = 3 ; mat%ne = 5
   END SELECT
   ALLOCATE( mat%ptr( mat%m + 1 ), mat%col( mat%ne ), mat%val( mat%ne ) ) 
   SELECT CASE ( prob_number )
   CASE ( 2 ) 
     mat%ptr = (/ 1, 3, 4, 5 /)
     mat%col = (/ 1, 2, 1, 3 /)
     mat%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
   CASE ( 3 ) 
     mat%ptr = (/ 1, 3, 4, 5 /)
     mat%col = (/ 1, 2, 2, 3 /)
     mat%val = (/ 2.0_wp, 1.0_wp, 3.0_wp, 4.0_wp /)
   CASE DEFAULT
     mat%ptr = (/ 1, 3, 4, 5, 6 /)
     mat%col = (/ 1, 2, 1, 2, 3 /)
     mat%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp /)
   END SELECT
   CALL SMT_put( mat%type, 'SPARSE_BY_ROWS', s )
! problem data complete
   CALL MIQR_initialize( data, control, inform ) ! Initialize control parameters
   control%print_level = 1
!  control%multi_level = .FALSE.
!  data%control%deallocate_after_factorization = .TRUE.
   CALL MIQR_form( mat, data, control, inform ) ! form factors
   ALLOCATE( SOL( mat%n ) )
   SELECT CASE ( prob_number )
   CASE ( 2 ) 
     SOL = (/ 14.0, 10.0, 48.0 /)
   CASE ( 3 ) 
     SOL = (/ 8.0, 22.0, 48.0 /)
   CASE DEFAULT
     SOL = (/ 14.0, 42.0, 75.0 /)
   END SELECT
   CALL MIQR_apply( SOL, .TRUE., data, inform )
   WRITE( 6, "( ' sol(transpose) ', /, ( 5ES12.4 ) )" ) SOL
   CALL MIQR_apply( SOL, .FALSE., data, inform )
   WRITE( 6, "( ' sol ', /, ( 5ES12.4 ) )" ) SOL
   CALL MIQR_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_MIQR_TEST

