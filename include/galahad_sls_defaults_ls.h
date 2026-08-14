!  Symmetric (indefinite) and definite linear solvers used by the GALAHAD test
!  programs. They are read at run time from the environment variables
!  GALAHAD_SYMMETRIC_LINEAR_SOLVER / GALAHAD_DEFINITE_LINEAR_SOLVER, so the same
!  test binaries can exercise any available solver without recompiling; if a
!  variable is unset (or empty) it falls back to the always-available LAPACK
!  dense solvers sytr / potr. This header is self-contained (it does not include
!  galahad_sls_defaults_{sls,dls}.h) because Fortran forbids the declarations of
!  the second header from following the run-time reads of the first.

     CHARACTER ( LEN = 30 ) :: symmetric_linear_solver, definite_linear_solver
     INTEGER :: ls_default_status
     CALL GET_ENVIRONMENT_VARIABLE( 'GALAHAD_SYMMETRIC_LINEAR_SOLVER',         &
                                    symmetric_linear_solver,                   &
                                    STATUS = ls_default_status )
     IF ( ls_default_status /= 0 .OR.                                          &
          LEN_TRIM( symmetric_linear_solver ) == 0 )                          &
       symmetric_linear_solver = 'sytr'
     CALL GET_ENVIRONMENT_VARIABLE( 'GALAHAD_DEFINITE_LINEAR_SOLVER',          &
                                    definite_linear_solver,                    &
                                    STATUS = ls_default_status )
     IF ( ls_default_status /= 0 .OR.                                          &
          LEN_TRIM( definite_linear_solver ) == 0 )                           &
       definite_linear_solver = 'potr'
