! THIS VERSION: GALAHAD 4.1 - 2022-12-01 AT 13:00 GMT.

!     TYPE SMUMPS_STRUC( long )
!     INTEGER, KIND :: long

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
        INTEGER :: NRHS
        INTEGER :: LRHS
!       INTEGER( KIND = long ) :: NNZ
        INTEGER( KIND = SELECTED_INT_KIND( 18 ) ) :: NNZ
        REAL, DIMENSION( : ), POINTER :: A
        INTEGER, DIMENSION( : ), POINTER :: IRN, JCN
        REAL, DIMENSION( : ), POINTER :: RHS
        INTEGER, DIMENSION( : ), POINTER :: PERM_IN
        REAL, DIMENSION( : ), POINTER :: COLSCA, ROWSCA
        INTEGER, DIMENSION( : ), POINTER :: SYM_PERM, UNS_PERM
        INTEGER, DIMENSION( 60 ) ::  ICNTL
        REAL, DIMENSION( 15 ) :: CNTL
        INTEGER, DIMENSION( 80 ) :: INFOG
        REAL, DIMENSION( 40 ) ::  RINFOG
      END TYPE SMUMPS_STRUC
