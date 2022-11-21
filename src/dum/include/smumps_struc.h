! THIS VERSION: GALAHAD 4.1 - 2022-11-02 AT 14:50 GMT.

      TYPE SMUMPS_STRUC

!  This dummy structure contains a subset of the parameters for the 
!  interface to the user, plus internal information from the MUMPS solver.

!  Extracted from package-provided smumps_struc.h

        INTEGER :: COMM
        INTEGER :: SYM, PAR
        INTEGER :: JOB 
        INTEGER :: MYID
        INTEGER :: N
        INTEGER :: NZ
        INTEGER( 8 ) :: NNZ
        REAL, DIMENSION( : ), POINTER :: A
        INTEGER, DIMENSION( : ), POINTER :: IRN, JCN
        REAL, DIMENSION( : ), POINTER :: RHS
        INTEGER, DIMENSION( : ), POINTER :: PERM_IN
        REAL, DIMENSION( : ), POINTER :: COLSCA, ROWSCA
        INTEGER :: INFOG( 80 )
      END TYPE SMUMPS_STRUC
