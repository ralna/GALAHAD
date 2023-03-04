! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

! G A L A H A D  -  D U M M Y   P A S T I X F _ I N T E R F A C E S  M O D U L E

!  Extracted from
!> @file pastixf_interfaces.f90
!>
!> PaStiX Fortran 90 wrapper
!>
!> @copyright 2017-2022 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
!>                      Univ. Bordeaux. All rights reserved.
!>
!> @version 6.2.0
!> @author Mathieu Faverge
!> @author Tony Delarue
!> @author Selmane Lebdaoui
!> @date 2022-01-13

 MODULE pastixf_interfaces

   INTERFACE pastixInitParam
     SUBROUTINE pastixInitParam_f08( iparm, dparm )
       USE GALAHAD_KINDS, ONLY : spc_, dpc_
       USE pastixf_enums, ONLY : pastix_int_t
       IMPLICIT NONE
       INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target :: iparm( : )
#ifdef GALAHAD_SINGLE
       REAL ( KIND = spc_ ), INTENT( INOUT ), target :: dparm( : )
#else
       REAL ( KIND = dpc_ ), INTENT( INOUT ), target :: dparm( : )
#endif
     END SUBROUTINE pastixInitParam_f08
   END INTERFACE pastixInitParam

   INTERFACE pastixInit
     SUBROUTINE pastixInit_f08( pastix_data, pastix_comm, iparm, dparm )
       USE GALAHAD_KINDS, ONLY : spc_, dpc_
       USE spmf_enums, ONLY : MPI_Comm
       USE pastixf_enums, ONLY : pastix_data_t, pastix_int_t
       IMPLICIT NONE
       TYPE ( pastix_data_t ), INTENT( INOUT ), pointer :: pastix_data
       TYPE ( MPI_Comm ), INTENT( IN ) :: pastix_comm
       INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target  :: iparm( : )
#ifdef GALAHAD_SINGLE
       REAL ( KIND = spc_ ), INTENT( INOUT ), target :: dparm( : )
#else
       REAL ( KIND = dpc_ ), INTENT( INOUT ), target :: dparm( : )
#endif
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
     SUBROUTINE pastix_task_solve_f08( pastix_data, nrhs, B, ldb, info )
       USE GALAHAD_KINDS, ONLY : ipc_
       USE pastixf_enums, ONLY : pastix_data_t, pastix_int_t
       IMPLICIT NONE
       TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
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

 END MODULE pastixf_interfaces












