! THIS VERSION: GALAHAD 4.1 - 2022-11-24 AT 09:45 GMT.

      TYPE DMUMPS_STRUC( long )
      INTEGER, KIND :: long

!  This dummy structure contains a subset of the parameters for the
!  interface to the user, plus internal information from the MUMPS solver.

!  Extracted from package-provided dmumps_struc.h

        INTEGER :: COMM
        INTEGER :: SYM, PAR
        INTEGER :: JOB
        INTEGER :: MYID
        INTEGER :: N
        INTEGER :: NZ
        INTEGER( KIND = long ) :: NNZ
        DOUBLE PRECISION, DIMENSION( : ), POINTER :: A
        INTEGER, DIMENSION( : ), POINTER :: IRN, JCN
        DOUBLE PRECISION, DIMENSION( : ), POINTER :: RHS
        INTEGER, DIMENSION( : ), POINTER :: PERM_IN
        DOUBLE PRECISION, DIMENSION( : ), POINTER :: COLSCA, ROWSCA
        INTEGER :: INFOG( 80 )
      END TYPE DMUMPS_STRUC
