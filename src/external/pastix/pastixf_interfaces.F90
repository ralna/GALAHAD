! THIS VERSION: GALAHAD 5.0 - 2024-06-11 AT 10:10 GMT.

#include "galahad_modules.h"

! G A L A H A D  -  D U M M Y   P A S T I X F _ I N T E R F A C E S  M O D U L E

!  Extracted from
!> @file pastixf_interfaces.f90
!>
!> PaStiX Fortran 90 wrapper
!>
!> @copyright 2017-2022 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
!>                      Univ. Bordeaux. All rights reserved.
!>
!> @version 6.3.2
!> @author Mathieu Faverge
!> @author Tony Delarue
!> @author Selmane Lebdaoui
!> @date 2023-07-21

#ifdef INTEGER_64
#define pastixf_enums pastixf_enums_64
#define spmf_enums spmf_enums_64
#endif

#ifdef REAL_32
#ifdef INTEGER_64
#define pastixf_interfaces_precision pastixf_interfaces_single_64
#else
#define pastixf_interfaces_precision pastixf_interfaces_single
#endif
#elif REAL_128
#ifdef INTEGER_64
#define pastix_interfaces_precision pastix_interfaces_quadruple_64
#else
#define pastix_interfaces_precision pastix_interfaces_quadruple
#endif
#else
#ifdef INTEGER_64
#define pastixf_interfaces_precision pastixf_interfaces_double_64
#else
#define pastixf_interfaces_precision pastixf_interfaces_double
#endif
#endif

 MODULE pastixf_interfaces_precision

   INTERFACE pastixInitParam
     SUBROUTINE pastixInitParam_f08( iparm, dparm )
       USE GALAHAD_KINDS_precision, ONLY : rpc_
       USE pastixf_enums, ONLY : pastix_int_t
       IMPLICIT NONE
       INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target :: iparm( : )
       REAL ( KIND = rpc_ ), INTENT( INOUT ), target :: dparm( : )
     END SUBROUTINE pastixInitParam_f08
   END INTERFACE pastixInitParam

   INTERFACE pastixInit
     SUBROUTINE pastixInit_f08( pastix_data, pastix_comm, iparm, dparm )
       USE GALAHAD_KINDS_precision, ONLY : rpc_
       USE spmf_enums, ONLY : MPI_Comm
       USE pastixf_enums, ONLY : pastix_data_t, pastix_int_t
       IMPLICIT NONE
       TYPE ( pastix_data_t ), INTENT( INOUT ), pointer :: pastix_data
       TYPE ( MPI_Comm ), INTENT( IN ) :: pastix_comm
       INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target  :: iparm( : )
       REAL ( KIND = rpc_ ), INTENT( INOUT ), target :: dparm( : )
     END SUBROUTINE pastixInit_f08
   END INTERFACE pastixInit

   INTERFACE pastix_task_analyze
     SUBROUTINE pastix_task_analyze_f08( pastix_data, spm, info )
       USE GALAHAD_KINDS, ONLY : ipc_
       USE spmf_enums, ONLY : spmatrix_t
       USE pastixf_enums, ONLY : pastix_data_t
       IMPLICIT NONE
       TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
       TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
       INTEGER ( KIND = ipc_ ), INTENT( OUT ), OPTIONAL :: info
     END SUBROUTINE pastix_task_analyze_f08
   END INTERFACE pastix_task_analyze

   INTERFACE pastix_task_numfact
     SUBROUTINE pastix_task_numfact_f08( pastix_data, spm, info )
       USE GALAHAD_KINDS, ONLY : ipc_
       USE spmf_enums, ONLY : spmatrix_t
       USE pastixf_enums, ONLY : pastix_data_t
       IMPLICIT NONE
       TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
       TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
       INTEGER ( KIND = ipc_ ), INTENT( OUT ), OPTIONAL :: info
     END SUBROUTINE pastix_task_numfact_f08
   END INTERFACE pastix_task_numfact

   INTERFACE pastix_task_solve
     SUBROUTINE pastix_task_solve_f08( pastix_data, m, nrhs, B, ldb, info )
       USE GALAHAD_KINDS, ONLY : ipc_
       USE pastixf_enums, ONLY : pastix_data_t, pastix_int_t
       IMPLICIT NONE
       TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
       INTEGER ( KIND = pastix_int_t ), INTENT (IN ) :: m
       INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: nrhs
       CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), target :: B
       INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldb
       INTEGER ( KIND = ipc_ ), INTENT( OUT ), OPTIONAL :: info
     END SUBROUTINE pastix_task_solve_f08
   END INTERFACE pastix_task_solve

   INTERFACE pastix_task_refine
     SUBROUTINE pastix_task_refine_f08( pastix_data, n, nrhs, B, ldb, X, ldx,  &
                                        info )
       USE GALAHAD_KINDS, ONLY : ipc_
       USE pastixf_enums, ONLY : pastix_data_t, pastix_int_t
       IMPLICIT NONE
       TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET   :: pastix_data
       INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: n
       INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: nrhs
       CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), TARGET :: B
       INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldb
       CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), TARGET :: X
       INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldx
       INTEGER ( KIND = ipc_ ),        INTENT( OUT ), OPTIONAL :: info
     END SUBROUTINE pastix_task_refine_f08
   END INTERFACE pastix_task_refine

   INTERFACE pastixOrderGet
     SUBROUTINE pastixOrderGet_f08( pastix_data, order )
       USE pastixf_enums, ONLY : pastix_data_t, pastix_order_t
       IMPLICIT NONE
       TYPE( pastix_data_t ), INTENT( IN ),  target  :: pastix_data
       TYPE( pastix_order_t ), INTENT( OUT ), pointer :: order
     END SUBROUTINE pastixOrderGet_f08
   END INTERFACE pastixOrderGet

   INTERFACE pastixOrderGetArray
     SUBROUTINE pastixOrderGetArray_f08( order, permtab, peritab, rangtab,     &
                                         treetab, sndetab )
       USE pastixf_enums, ONLY : pastix_int_t, pastix_order_t
       IMPLICIT NONE
       TYPE( pastix_order_t ), INTENT( IN ), target  :: order
       INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,      &
                                 POINTER :: permtab
       INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,      &
                                 POINTER :: peritab
       INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,      &
                                 POINTER :: rangtab
       INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,      &
                                 POINTER :: treetab
       INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,      &
                                 POINTER :: sndetab
     END SUBROUTINE pastixOrderGetArray_f08
   END INTERFACE pastixOrderGetArray

   INTERFACE pastixFinalize
     SUBROUTINE pastixFinalize_f08( pastix_data )
       USE pastixf_enums, ONLY : pastix_data_t
       IMPLICIT NONE
       TYPE ( pastix_data_t ), INTENT( INOUT ), POINTER :: pastix_data
     END SUBROUTINE pastixFinalize_f08
   END INTERFACE pastixFinalize

 END MODULE pastixf_interfaces_precision












