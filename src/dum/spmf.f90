! THIS VERSION: GALAHAD 4.1 - 2022-10-18 AT 15:30 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   S P M F   M O D U L E  -*-*-*-

MODULE spmf

  USE iso_c_binding, ONLY : c_double, c_int, c_ptr, c_int32_t, c_int64_t

  INTEGER, PARAMETER :: spm_int_t = c_int32_t
! INTEGER, PARAMETER :: spm_int_t = c_int64_t

  TYPE, BIND( C ) :: MPI_Comm
    INTEGER ( kind = c_int ) :: MPI_VAL = 0
  END TYPE MPI_Comm

  TYPE, BIND( C ) :: spmatrix_t
    INTEGER ( c_int ) :: mtxtype
    INTEGER ( c_int ) :: flttype
    INTEGER ( c_int ) :: fmttype
    INTEGER ( KIND = spm_int_t ) :: baseval
    INTEGER ( KIND = spm_int_t ) :: gN
    INTEGER ( KIND = spm_int_t ) :: n
    INTEGER ( KIND = spm_int_t ) :: gnnz
    INTEGER ( KIND = spm_int_t ) :: nnz
    INTEGER ( KIND = spm_int_t ) :: gNexp
    INTEGER ( KIND = spm_int_t ) :: nexp
    INTEGER ( KIND = spm_int_t ) :: gnnzexp
    INTEGER ( KIND = spm_int_t ) :: nnzexp
    INTEGER ( KIND = spm_int_t ) :: dof
    TYPE ( c_ptr) :: dofs
    INTEGER ( c_int ) :: layout
    TYPE ( c_ptr ) :: colptr
    TYPE ( c_ptr ) :: rowptr
    TYPE ( c_ptr ) :: loc2glob
    TYPE ( c_ptr ) :: values
    TYPE ( c_ptr ) :: glob2loc
    INTEGER ( KIND = c_int ) :: clustnum
    INTEGER ( KIND = c_int ) :: clustnbr
    TYPE ( MPI_Comm ) :: comm
  END type spmatrix_t

CONTAINS

   SUBROUTINE spmInit( spm )
    IMPLICIT NONE
    TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
    spm%mtxtype = - 1
   END SUBROUTINE spmInit

   SUBROUTINE spmUpdateComputedFields( spm )
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmUpdateComputedFields

   SUBROUTINE spmAlloc( spm )
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmAlloc

   SUBROUTINE spmCheckAndCorrect( spm_in, spm_out, info )
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm_in
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm_out
     INTEGER(kind=c_int), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE spmCheckAndCorrect

   SUBROUTINE spmGetArray( spm, colptr, rowptr, dvalues )
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
     INTEGER ( spm_int_t ), DIMENSION(:), INTENT( OUT ), OPTIONAL,             &
                         POINTER :: colptr
     INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,           &
                         POINTER :: rowptr
     REAL ( c_double ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,               &
                        POINTER :: dvalues
   END SUBROUTINE spmGetArray

   SUBROUTINE spmCheckAxb( eps, nrhs, spm, opt_X0, opt_ldx0, B, ldb, X,        &
                           ldx, info )
     IMPLICIT NONE
     REAL ( KIND = c_double ), INTENT( IN ) :: eps
     INTEGER ( KIND = spm_int_t ),INTENT( IN ) :: nrhs
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
     CLASS( * ), INTENT( INOUT ), TARGET, DIMENSION( :, : ), OPTIONAL :: opt_X0
     INTEGER ( KIND = spm_int_t ), INTENT( IN ), OPTIONAL :: opt_ldx0
     CLASS( * ), INTENT( INOUT ), TARGET, DIMENSION( :, : ) :: B
     INTEGER ( KIND = spm_int_t ), INTENT( IN ) :: ldb
     CLASS( * ), INTENT( IN ), TARGET, DIMENSION( :, : ) :: X
     INTEGER ( KIND = spm_int_t ), INTENT( IN ) :: ldx
     INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE spmCheckAxb

   SUBROUTINE spmExit( spm )
     IMPLICIT NONE
     TYPE( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmExit

END MODULE spmf
