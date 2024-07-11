! THIS VERSION: GALAHAD 5.0 - 2024-05-22 AT 11:50 GMT.

#include "galahad_modules.h"

!-*-*-*-  G A L A H A D  -  D U M M Y   S P M F  S U B R O U T I N E S -*-*-*-

   SUBROUTINE spmInit_f08( spm )
    USE spmf_enums, ONLY : spmatrix_t
    IMPLICIT NONE
    TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
    spm%mtxtype = - 1
   END SUBROUTINE spmInit_f08

   SUBROUTINE spmUpdateComputedFields_f08( spm )
     USE spmf_enums, ONLY : spmatrix_t
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmUpdateComputedFields_f08

   SUBROUTINE spmAlloc_f08( spm )
     USE spmf_enums, ONLY : spmatrix_t
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmAlloc_f08

   SUBROUTINE spmCheckAndCorrect_f08( spm_in, spm_out, info )
     USE GALAHAD_KINDS, ONLY : ipc_
     USE spmf_enums, ONLY : spmatrix_t
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm_in
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm_out
     INTEGER ( KIND = ipc_ ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE spmCheckAndCorrect_f08

   SUBROUTINE spmGetArray_f08( spm, colptr, rowptr, zvalues, cvalues,          &
                               dvalues, svalues, dofs, loc2glob, glob2loc )
     USE iso_c_binding, ONLY : c_double_complex, c_float_complex, c_double,    &
                               c_float, c_f_pointer
     USE spmf_enums, ONLY : spmatrix_t, spm_int_t, SpmCSC, SpmCSR,             &
                            SpmComplex64, SpmComplex32, SpmDouble, SpmFloat
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
     INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,           &
                            POINTER :: colptr
     INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,           &
                            POINTER :: rowptr
     COMPLEX ( c_double_complex ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,    &
                                   POINTER :: zvalues
     COMPLEX ( c_float_complex ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,     &
                                  POINTER :: cvalues
     REAL ( c_double ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,               &
                        POINTER :: dvalues
     REAL ( c_float ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,                &
                       POINTER :: svalues
     INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,           &
                            POINTER :: dofs
     INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,           &
                            POINTER :: loc2glob
     INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,           &
                            POINTER :: glob2loc

     INTEGER ( spm_int_t ) :: colsize
     INTEGER ( spm_int_t ) :: rowsize

     IF ( spm%fmttype == SpmCSC ) THEN
        colsize = spm%n + 1
        rowsize = spm%nnz
     ELSE IF ( spm%fmttype == SpmCSR ) THEN
        colsize = spm%nnz
        rowsize = spm%n + 1
     ELSE
        colsize = spm%nnz
        rowsize = spm%nnz
     END IF

     IF ( PRESENT( colptr ) )                                                  &
       CALL c_f_pointer( spm%colptr, colptr, [colsize] )
     IF ( PRESENT( rowptr ) )                                                  &
       CALL c_f_pointer( spm%rowptr, rowptr, [rowsize] )
     IF ( PRESENT( dofs ) )                                                    &
       CALL c_f_pointer( spm%dofs, dofs, [ spm%gN+1 ] )
     IF ( PRESENT( loc2glob ) )                                                &
       CALL c_f_pointer( spm%loc2glob, loc2glob, [ spm%n ] )
     IF ( PRESENT( glob2loc ) )                                                &
       CALL c_f_pointer( spm%glob2loc, glob2loc, [spm%gN]   )
     IF ( PRESENT( zvalues ) .AND. ( spm%flttype == SpmComplex64 ) )           &
        CALL c_f_pointer( spm%values, zvalues, [ spm%nnzexp ] )
     IF ( PRESENT( cvalues ) .AND. ( spm%flttype == SpmComplex32 ) )           &
        CALL c_f_pointer( spm%values, cvalues, [ spm%nnzexp ] )
     IF ( PRESENT( dvalues ) .AND. ( spm%flttype == SpmDouble ) )              &
        CALL c_f_pointer( spm%values, dvalues, [ spm%nnzexp ] )
     IF ( PRESENT( svalues ) .AND. ( spm%flttype == SpmFloat ) )              &
        CALL c_f_pointer( spm%values, svalues, [ spm%nnzexp ] )

   END SUBROUTINE spmGetArray_f08

   SUBROUTINE spmCheckAxb_f08( eps, nrhs, spm, opt_X0, opt_ldx0, B, ldb, X,    &
                               ldx, info )
     USE GALAHAD_KINDS_precision, ONLY : ipc_, dpc_
     USE spmf_enums, ONLY : spmatrix_t, spm_int_t
     IMPLICIT NONE
     REAL ( KIND = dpc_ ), INTENT( IN ) :: eps
     INTEGER ( KIND = spm_int_t ),INTENT( IN ) :: nrhs
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
     CLASS( * ), INTENT( INOUT ), TARGET, DIMENSION( :, : ), OPTIONAL :: opt_X0
     INTEGER ( KIND = spm_int_t ), INTENT( IN ), OPTIONAL :: opt_ldx0
     CLASS( * ), INTENT( INOUT ), TARGET, DIMENSION( :, : ) :: B
     INTEGER ( KIND = spm_int_t ), INTENT( IN ) :: ldb
     CLASS( * ), INTENT( IN ), TARGET, DIMENSION( :, : ) :: X
     INTEGER ( KIND = spm_int_t ), INTENT( IN ) :: ldx
     INTEGER ( KIND = ipc_ ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE spmCheckAxb_f08

   SUBROUTINE spmExit_f08( spm )
     USE spmf_enums, ONLY : spmatrix_t
     IMPLICIT NONE
     TYPE( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmExit_f08
