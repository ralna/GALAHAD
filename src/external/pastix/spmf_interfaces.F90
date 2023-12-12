! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

!-  G A L A H A D  -  D U M M Y   S P M F _ I N T E R F A C E S   M O D U L E  -

!  Extracted from
!> @file spmf_interfaces.f90
!>
!> SPM Fortran 90 wrapper
!>
!> @copyright 2017-2022 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
!>                      Univ. Bordeaux. All rights reserved.
!>
!> @version 1.2.0
!> @author Mathieu Faverge
!> @author Tony Delarue
!> @date 2022-02-22

 MODULE spmf_interfaces

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
       INTEGER( KIND = ipc_ ), INTENT( OUT ), OPTIONAL :: info
     END SUBROUTINE spmCheckAndCorrect_f08
   END INTERFACE spmCheckAndCorrect

   INTERFACE spmGetArray
     SUBROUTINE spmGetArray_f08( spm, colptr, rowptr, zvalues, cvalues,        &
                                 dvalues, svalues, dofs, loc2glob, glob2loc )
       USE iso_c_binding, ONLY : c_double_complex, c_float_complex, c_double,  &
                                 c_float
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
#ifdef GALAHAD_SINGLE
       USE GALAHAD_KINDS, ONLY : ipc_, spc_
#else
       USE GALAHAD_KINDS, ONLY : ipc_, dpc_
#endif
       USE spmf_enums, ONLY : spmatrix_t, spm_int_t
       IMPLICIT NONE
#ifdef GALAHAD_SINGLE
       REAL ( KIND = spc_ ), INTENT( IN ) :: eps
#else
       REAL ( KIND = dpc_ ), INTENT( IN ) :: eps
#endif
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

 END MODULE spmf_interfaces





