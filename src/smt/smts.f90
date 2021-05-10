! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_SMT_example
   USE GALAHAD_SMT_double 
   INTEGER :: i, s
   TYPE ( SMT_type ) :: A 
   A%n = 3 ; A%ne = 2 
   ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) ) 
   CALL SMT_put( A%id, 'Sparse', s )   ! Put name into A%id
   A%row( 1 ) = 1 ; A%col( 1 ) = 1 ; A%val( 1 ) = 1.0 
   A%row( 2 ) = 2 ; A%col( 2 ) = 3 ; A%val( 2 ) = 1.0 
   WRITE( 6, "( 3A, I2, //, A )" ) ' Matrix ', SMT_get( A%id ), & 
          ' dimension', A%n, ' row col  value ' 
   DO i = 1, A%ne 
      WRITE( 6, "( I3, 1X, I3, ES9.1 )" ) A%row( i ), A%col( i ), A%val( i ) 
   END DO 
   DEALLOCATE( A%id, A%row, A%col, A%val )
   END PROGRAM GALAHAD_SMT_example

