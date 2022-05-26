! THIS VERSION: GALAHAD 4.1 - 2022-05-26 AT 07:15 GMT
! Used to test components of package
   PROGRAM GALAHAD_SLLS_EXAMPLE2
   USE GALAHAD_SLLS_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_proj
   INTEGER, PARAMETER :: n = 10
   INTEGER :: j, status
   REAL ( KIND = wp ) :: scale, flipflop
   ALLOCATE( p%X( n ), X_proj( n ) )
   scale = REAL( n, KIND = wp )
   DO j = 1, n
     p%X( j ) = 1.0_wp / scale
   END DO
   CALL SLLS_project_onto_simplex( n, p%X, X_proj, status )
   WRITE( 6, "( ' status = ', I0 )" ) status
   WRITE( 6, "( 8X, '        x          p(x)' )" )
   DO j = 1, n
     WRITE(6, "( I8, 2ES12.4 )" ) j, p%X( j ), X_proj( j )
   END DO
   scale = REAL( n * ( n + 1 ) / 2, KIND = wp )
   DO j = 1, n
     p%X( j ) = REAL( j, KIND = wp ) / scale
   END DO
   CALL SLLS_project_onto_simplex( n, p%X, X_proj, status )
   WRITE( 6, "( ' status = ', I0 )" ) status
   WRITE( 6, "( 8X, '        x          p(x)' )" )
   DO j = 1, n
     WRITE(6, "( I8, 2ES12.4 )" ) j, p%X( j ), X_proj( j )
   END DO
   scale = REAL( n * ( n + 1 ) / 2, KIND = wp )
   DO j = 1, n
     p%X( j ) = - REAL( j, KIND = wp ) / scale
   END DO
   CALL SLLS_project_onto_simplex( n, p%X, X_proj, status )
   WRITE( 6, "( ' status = ', I0 )" ) status
   WRITE( 6, "( 8X, '        x          p(x)' )" )
   DO j = 1, n
     WRITE(6, "( I8, 2ES12.4 )" ) j, p%X( j ), X_proj( j )
   END DO
   scale = REAL( n * ( n + 1 ) / 2, KIND = wp )
   flipflop = 1.0_wp
   DO j = 1, n
     p%X( j ) = flipflop * REAL( j, KIND = wp ) / scale
     flipflop = - 1.0_wp * flipflop
   END DO
   CALL SLLS_project_onto_simplex( n, p%X, X_proj, status )
   WRITE( 6, "( ' status = ', I0 )" ) status
   WRITE( 6, "( 8X, '        x          p(x)' )" )
   DO j = 1, n
     WRITE(6, "( I8, 2ES12.4 )" ) j, p%X( j ), X_proj( j )
   END DO
   DEALLOCATE( p%X, X_proj )
   END PROGRAM GALAHAD_SLLS_EXAMPLE2
