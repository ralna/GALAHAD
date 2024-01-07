! THIS VERSION: GALAHAD 4.1 - 2022-12-03 AT 14:20 GMT.

      TYPE SMUMPS_STRUC

!  This dummy structure contains a subset of the parameters for the
!  interface to the user, plus internal information from the MUMPS solver.

!  Extracted from package-provided smumps_struc.h

        INTEGER ( KIND = ip_ ) :: COMM
        INTEGER ( KIND = ip_ ) :: SYM, PAR
        INTEGER ( KIND = ip_ ) :: JOB
        INTEGER ( KIND = ip_ ) :: MYID
        INTEGER ( KIND = ip_ ) :: N
        INTEGER ( KIND = ip_ ) :: NZ
        INTEGER ( KIND = ip_ ) :: NRHS
        INTEGER ( KIND = ip_ ) :: LRHS
        INTEGER ( KIND = long_ ) :: NNZ
        REAL, DIMENSION( : ), POINTER :: A
        INTEGER ( KIND = ip_ ), DIMENSION( : ), POINTER :: IRN, JCN
        REAL, DIMENSION( : ), POINTER :: RHS
        INTEGER ( KIND = ip_ ), DIMENSION( : ), POINTER :: PERM_IN
        REAL, DIMENSION( : ), POINTER :: COLSCA, ROWSCA
        INTEGER ( KIND = ip_ ), DIMENSION( : ), POINTER :: SYM_PERM, UNS_PERM
        INTEGER ( KIND = ip_ ), DIMENSION( 60 ) ::  ICNTL
        REAL, DIMENSION( 15 ) :: CNTL
        INTEGER ( KIND = ip_ ), DIMENSION( 80 ) :: INFOG
        REAL, DIMENSION( 40 ) ::  RINFOG
      END TYPE SMUMPS_STRUC
