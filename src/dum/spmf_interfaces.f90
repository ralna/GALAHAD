! THIS VERSION: GALAHAD 4.1 - 2022-10-25 AT 16:30 GMT.

!-  G A L A H A D  -  D U M M Y   S P M F _ I N T E R F A C E S   M O D U L E  -

 MODULE spmf_interfaces

   USE iso_c_binding, ONLY : c_float, c_double, c_ptr,                         &
                             c_int, c_int32_t, c_int64_t
   USE spmf_enums

   INTERFACE spmInit
     MODULE PROCEDURE spmInit_f08 
   END INTERFACE spmInit

   INTERFACE spmUpdateComputedFields
     MODULE PROCEDURE spmUpdateComputedFields_f08
   END INTERFACE spmUpdateComputedFields

   INTERFACE spmAlloc
     MODULE PROCEDURE spmAlloc_f08
   END INTERFACE spmAlloc

   INTERFACE spmCheckAndCorrect
     MODULE PROCEDURE spmCheckAndCorrect_f08
   END INTERFACE spmCheckAndCorrect

   INTERFACE spmGetArray
     MODULE PROCEDURE spmGetArray_f08
   END INTERFACE spmGetArray

   INTERFACE spmCheckAxb
     MODULE PROCEDURE spmCheckAxb_f08
   END INTERFACE spmCheckAxb

   INTERFACE spmExit
     MODULE PROCEDURE spmExit_f08
   END INTERFACE spmExit

 CONTAINS

   SUBROUTINE spmInit_f08( spm )
    IMPLICIT NONE
    TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
    spm%mtxtype = - 1
   END SUBROUTINE spmInit_f08

   SUBROUTINE spmUpdateComputedFields_f08( spm )
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmUpdateComputedFields_f08

   SUBROUTINE spmAlloc_f08( spm )
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmAlloc_f08

   SUBROUTINE spmCheckAndCorrect_f08( spm_in, spm_out, info )
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm_in
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm_out
     INTEGER( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE spmCheckAndCorrect_f08

   SUBROUTINE spmGetArray_f08( spm, colptr, rowptr, dvalues, svalues )
     IMPLICIT NONE
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
     INTEGER ( spm_int_t ), DIMENSION(:), INTENT( OUT ), OPTIONAL,             &
                         POINTER :: colptr
     INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,           &
                         POINTER :: rowptr
     REAL ( c_double ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,               &
                        POINTER :: dvalues
     REAL ( c_float ),  DIMENSION( : ), INTENT( OUT ), OPTIONAL,               &
                        POINTER :: svalues
   END SUBROUTINE spmGetArray_f08

   SUBROUTINE spmCheckAxb_f08( eps, nrhs, spm, opt_X0, opt_ldx0, B, ldb, X,    &
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
   END SUBROUTINE spmCheckAxb_f08

   SUBROUTINE spmExit_f08( spm )
     IMPLICIT NONE
     TYPE( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
   END SUBROUTINE spmExit_f08

 END MODULE spmf_interfaces
