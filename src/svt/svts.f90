! THIS VERSION: GALAHAD 4.2 - 2023-10-25 AT 16:30 GMT.
   PROGRAM GALAHAD_SVT_example
   USE GALAHAD_SVT_double 
   INTEGER :: i
   TYPE ( SVT_type ) :: V
   V%n = 5 ; V%ne = 2 
   ALLOCATE( V%ind( V%ne ), V%val( V%ne ) ) 
   V%row( 1 ) = 1 ; V%val( 1 ) = 1.0 
   V%row( 2 ) = 3 ; V%val( 2 ) = 3.0 
   WRITE( 6, "( ' Vector of dimension ', I0, //, ' ind  value' )" ) V%n
   DO i = 1, V%ne 
      WRITE( 6, "( I3, ES9.1 )" ) V%ind( i ),V%val( i ) 
   END DO 
   DEALLOCATE( A%ind, A%val )
   END PROGRAM GALAHAD_SVT_example

