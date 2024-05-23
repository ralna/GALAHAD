! THIS VERSION: GALAHAD 5.0 - 2024-05-22 AT 09:10 GMT.

#include "galahad_modules.h"

!-*-*-  G A L A H A D  -  D U M M Y   S P M F _ E N U M S   M O D U L E  -*-*-

MODULE spmf_enums

  USE iso_c_binding, ONLY : c_ptr, c_int, c_int32_t, c_int64_t

#ifdef INTEGER_64
  INTEGER, PARAMETER :: spm_int_t = c_int64_t
#else
  INTEGER, PARAMETER :: spm_int_t = c_int32_t
#endif
  TYPE, BIND( C ) :: MPI_Comm
    INTEGER ( kind = c_int ) :: MPI_VAL = 0
  END TYPE MPI_Comm

  TYPE( MPI_Comm ), PARAMETER :: MPI_COMM_WORLD = MPI_Comm( 0 )

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

  INTEGER, PARAMETER :: SpmCSC = 0
  INTEGER, PARAMETER :: SpmCSR = 1
  INTEGER, PARAMETER :: SpmIJV = 2
  INTEGER, PARAMETER :: SpmGeneral = 111
  INTEGER, PARAMETER :: SpmSymmetric = 112
  INTEGER, PARAMETER :: SpmFloat = 2
  INTEGER, PARAMETER :: SpmDouble = 3
  INTEGER, PARAMETER :: SpmComplex32 = 4
  INTEGER, PARAMETER :: SpmComplex64 = 5

END MODULE spmf_enums
