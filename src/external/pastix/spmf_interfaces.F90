! THIS VERSION: GALAHAD 5.1 - 2024-07-11 AT 10:10 GMT.

#include "galahad_modules.h"

!-  G A L A H A D  -  D U M M Y   S P M F _ I N T E R F A C E S   M O D U L E  -

!  Extracted from
!> @file spmf_interfaces.f90
!>
!> SPM Fortran 90 wrapper
!>
!> @copyright 2017-2023 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
!>                      Univ. Bordeaux. All rights reserved.
!>
!> @version 1.2.2
!> @author Mathieu Faverge
!> @author Tony Delarue
!> @date 2023-12-06

#ifdef INTEGER_64
#define spmf_enums spmf_enums_64
#endif

#ifdef REAL_32
#ifdef INTEGER_64
#define spmf_interfaces_precision spmf_interfaces_single_64
#else
#define spmf_interfaces_precision spmf_interfaces_single
#endif
#elif REAL_128
#ifdef INTEGER_64
#define spmf_interfaces_precision spmf_interfaces_quadruple_64
#else
#define spmf_interfaces_precision spmf_interfaces_quadruple
#endif
#else
#ifdef INTEGER_64
#define spmf_interfaces_precision spmf_interfaces_double_64
#else
#define spmf_interfaces_precision spmf_interfaces_double
#endif
#endif

 MODULE spmf_interfaces_precision

   USE iso_c_binding, ONLY : c_float, c_double, c_ptr,                         &
                             c_int, c_int32_t, c_int64_t
   INTERFACE spmInit
     SUBROUTINE spmInit_f08( spm )
      USE spmf_enums, ONLY : spmatrix_t
      IMPLICIT NONE
      TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
     END SUBROUTINE spmInit_f08
   END INTERFACE spmInit

   INTERFACE spmUpdateComputedFields
     SUBROUTINE spmUpdateComputedFields_f08( spm )
       USE spmf_enums, ONLY : spmatrix_t
       IMPLICIT NONE
       TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
     END SUBROUTINE spmUpdateComputedFields_f08
   END INTERFACE spmUpdateComputedFields

   INTERFACE spmAlloc
     SUBROUTINE spmAlloc_f08( spm )
       USE spmf_enums, ONLY : spmatrix_t
       IMPLICIT NONE
       TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
     END SUBROUTINE spmAlloc_f08
   END INTERFACE spmAlloc

   INTERFACE spmCheckAndCorrect
     SUBROUTINE spmCheckAndCorrect_f08( spm_in, spm_out, info )
       USE GALAHAD_KINDS, ONLY : ipc_
       USE spmf_enums, ONLY : spmatrix_t
       IMPLICIT NONE
       TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm_in
       TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm_out
       INTEGER ( KIND = ipc_ ), INTENT( OUT ), OPTIONAL :: info
     END SUBROUTINE spmCheckAndCorrect_f08
   END INTERFACE spmCheckAndCorrect

   INTERFACE spmGetArray
#ifdef REAL_128
     SUBROUTINE spmGetArray_f08( spm, colptr, rowptr, zvalues, cvalues,        &
                                 dvalues, svalues, dofs, loc2glob, glob2loc,   &
                                 xvalues, qvalues )
       USE iso_c_binding, ONLY : c_double_complex, c_float_complex, c_double,  &
                                 c_float, c_float128_complex, c_float128
#else
     SUBROUTINE spmGetArray_f08( spm, colptr, rowptr, zvalues, cvalues,        &
                                 dvalues, svalues, dofs, loc2glob, glob2loc )
       USE iso_c_binding, ONLY : c_double_complex, c_float_complex, c_double,  &
                                 c_float
#endif
       USE spmf_enums, ONLY : spmatrix_t, spm_int_t
       IMPLICIT NONE
       TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
       INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,         &
                              POINTER :: colptr
       INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,         &
                              POINTER :: rowptr
       COMPLEX ( c_double_complex ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,  &
                                     POINTER :: zvalues
       COMPLEX ( c_float_complex ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,   &
                                    POINTER :: cvalues
       REAL ( c_double ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,             &
                          POINTER :: dvalues
       REAL ( c_float ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,              &
                         POINTER :: svalues
#ifdef REAL_128
       COMPLEX ( c_float128_complex ), DIMENSION( : ), INTENT( OUT ),          &
                                    OPTIONAL, POINTER :: xvalues
       REAL ( c_float128 ), DIMENSION( : ), INTENT( OUT ),                     &
                         OPTIONAL, POINTER :: qvalues
#endif
       INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,         &
                              POINTER :: dofs
       INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,         &
                              POINTER :: loc2glob
       INTEGER ( spm_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,         &
                              POINTER :: glob2loc
     END SUBROUTINE spmGetArray_f08
   END INTERFACE spmGetArray

   INTERFACE spmCheckAxb
     SUBROUTINE spmCheckAxb_f08( eps, nrhs, spm, opt_X0, opt_ldx0, B, ldb, X,  &
                                 ldx, info )
       USE GALAHAD_KINDS_precision, ONLY : ipc_, dpc_
       USE spmf_enums, ONLY : spmatrix_t, spm_int_t
       IMPLICIT NONE
       REAL ( KIND = dpc_ ), INTENT( IN ) :: eps
       INTEGER ( KIND = spm_int_t ),INTENT( IN ) :: nrhs
       TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
       CLASS( * ), INTENT( INOUT ), TARGET, DIMENSION( :, : ),                 &
                   OPTIONAL :: opt_X0
       INTEGER ( KIND = spm_int_t ), INTENT( IN ), OPTIONAL :: opt_ldx0
       CLASS( * ), INTENT( INOUT ), TARGET, DIMENSION( :, : ) :: B
       INTEGER ( KIND = spm_int_t ), INTENT( IN ) :: ldb
       CLASS( * ), INTENT( IN ), TARGET, DIMENSION( :, : ) :: X
       INTEGER ( KIND = spm_int_t ), INTENT( IN ) :: ldx
       INTEGER ( KIND = ipc_ ), INTENT( OUT ), OPTIONAL :: info
     END SUBROUTINE spmCheckAxb_f08
   END INTERFACE spmCheckAxb

   INTERFACE spmExit
     SUBROUTINE spmExit_f08( spm )
       USE spmf_enums, ONLY : spmatrix_t
       IMPLICIT NONE
       TYPE( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
     END SUBROUTINE spmExit_f08
   END INTERFACE spmExit

 END MODULE spmf_interfaces_precision
