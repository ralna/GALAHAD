! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM SILS_EXAMPLE
   USE GALAHAD_SMT_double
   USE GALAHAD_SILS_DOUBLE
   IMPLICIT NONE
   TYPE(SMT_type) MATRIX
   TYPE(SILS_control) CONTROL
   TYPE(SILS_ainfo) AINFO
   TYPE(SILS_finfo) FINFO
   TYPE(SILS_sinfo) SINFO
   TYPE(SILS_factors) FACTORS

   DOUBLE PRECISION, ALLOCATABLE :: B(:),X(:)
   INTEGER I,N,NE

! Read matrix order and number of entries
      READ (5,*) N,NE
      MATRIX%N = N
      MATRIX%NE = NE

! Allocate arrays of appropriate sizes
      ALLOCATE(MATRIX%VAL(NE), MATRIX%ROW(NE), MATRIX%COL(NE))
      ALLOCATE(B(N), X(N))

! Read matrix and right-hand side
      READ (5,*) (MATRIX%ROW(I),MATRIX%COL(I),MATRIX%VAL(I),I=1,NE)
      READ (5,*) B

! Initialize the structures
      CALL SILS_INITIALIZE(FACTORS,CONTROL)

! Analyse
      CALL SILS_ANALYSE(MATRIX,FACTORS,CONTROL,AINFO)
      IF(AINFO%FLAG<0) THEN
         WRITE(6,'(A,I2)') &
            ' Failure of SILS_ANALYSE with AINFO%FLAG=', AINFO%FLAG
         STOP
      END IF

! Factorize
      CALL SILS_FACTORIZE(MATRIX,FACTORS,CONTROL,FINFO)
      IF(FINFO%FLAG<0) THEN
         WRITE(6,'(A,I2)') &
            ' Failure of SILS_FACTORIZE with FINFO%FLAG=', FINFO%FLAG
         STOP
      END IF

! Solve without refinement
      X = B
      CALL SILS_SOLVE(MATRIX,FACTORS,X,CONTROL,SINFO)
      IF(SINFO%FLAG==0)WRITE(6,'(A,/,(3F20.16))')  &
         ' Solution without refinement is',X

! Perform one refinement
      CALL SILS_SOLVE(MATRIX,FACTORS,X,CONTROL,SINFO,B)
      IF(SINFO%FLAG==0)WRITE(6,'(A,/,(3F20.16))') &
          ' Solution after one refinement is',X

END PROGRAM SILS_EXAMPLE
