! THIS VERSION: GALAHAD 4.1 - 2022-10-26 AT 15:30 GMT.

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
     USE iso_c_binding, ONLY : c_int
     USE spmf_enums, ONLY : spmatrix_t
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm_in
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm_out
     INTEGER( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE spmCheckAndCorrect_f08

   SUBROUTINE spmGetArray_f08( spm, colptr, rowptr, zvalues, cvalues,          &
                               dvalues, svalues, dofs, loc2glob, glob2loc )
     USE iso_c_binding, ONLY : c_double_complex, c_float_complex, c_double,    &
                               c_float
     USE spmf_enums, ONLY : spmatrix_t, spm_int_t
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
   END SUBROUTINE spmGetArray_f08

   SUBROUTINE spmCheckAxb_f08( eps, nrhs, spm, opt_X0, opt_ldx0, B, ldb, X,    &
                               ldx, info )
     USE iso_c_binding, ONLY : c_double, c_int
     USE spmf_enums, ONLY : spmatrix_t, spm_int_t
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
   END SUBROUTINE spmCheckAxb_f08

   SUBROUTINE spmExit_f08( spm )
     USE spmf_enums, ONLY : spmatrix_t
     IMPLICIT NONE
     TYPE( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmExit_f08
