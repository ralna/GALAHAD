! THIS VERSION: GALAHAD 4.1 - 2022-11-24 AT 09:45 GMT.

      TYPE SMUMPS_STRUC( long )
      INTEGER, KIND :: long

!  This dummy structure contains a subset of the parameters for the
!  interface to the user, plus internal information from the MUMPS solver.

!  Extracted from package-provided smumps_struc.h

        INTEGER :: COMM
        INTEGER :: SYM, PAR
        INTEGER :: JOB
        INTEGER :: MYID
        INTEGER :: N
        INTEGER :: NZ
        INTEGER( KIND = C_INT64_T ) :: NNZ
        REAL, DIMENSION( : ), POINTER :: A
        INTEGER, DIMENSION( : ), POINTER :: IRN, JCN
        REAL, DIMENSION( : ), POINTER :: RHS
        INTEGER, DIMENSION( : ), POINTER :: PERM_IN
        REAL, DIMENSION( : ), POINTER :: COLSCA, ROWSCA
        INTEGER :: INFOG( 80 )
      END TYPE SMUMPS_STRUC
