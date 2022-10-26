! THIS VERSION: GALAHAD 4.1 - 2022-10-25 AT 16:25 GMT.

! G A L A H A D  -  D U M M Y   P A S T I X F _ I N T E R F A C E S  M O D U L E

 MODULE pastixf_interfaces

   USE pastixf_enums
   USE iso_c_binding, ONLY : c_double, c_int, c_ptr, c_int32_t, c_int64_t

   INTERFACE pastixInitParam
     MODULE PROCEDURE pastixInitParam_f08
   END INTERFACE pastixInitParam

   INTERFACE pastixInit
     MODULE PROCEDURE pastixInit_f08
   END INTERFACE pastixInit

   INTERFACE pastix_task_analyze
     MODULE PROCEDURE pastix_task_analyze_f08
   END INTERFACE pastix_task_analyze

   INTERFACE pastix_task_numfact
     MODULE PROCEDURE pastix_task_numfact_f08
   END INTERFACE pastix_task_numfact

   INTERFACE pastix_task_solve
     MODULE PROCEDURE pastix_task_solve_f08
   END INTERFACE pastix_task_solve

   INTERFACE pastix_task_refine
     MODULE PROCEDURE pastix_task_refine_f08
   END INTERFACE pastix_task_refine

   INTERFACE pastixOrderGet
     MODULE PROCEDURE pastixOrderGet_f08
   END INTERFACE pastixOrderGet

   INTERFACE pastixOrderGetArray
     MODULE PROCEDURE pastixOrderGetArray_f08
   END INTERFACE pastixOrderGetArray

   INTERFACE pastixFinalize
     MODULE PROCEDURE pastixFinalize_f08
   END INTERFACE pastixFinalize

 CONTAINS

   SUBROUTINE pastixInitParam_f08( iparm, dparm )
     IMPLICIT NONE
     INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target :: iparm( : )
     REAL ( KIND = c_double ), INTENT( INOUT ), target :: dparm( : )
   END SUBROUTINE pastixInitParam_f08

   SUBROUTINE pastixInit_f08( pastix_data, pastix_comm, iparm, dparm )
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), pointer :: pastix_data
     TYPE ( MPI_Comm ), INTENT( IN ) :: pastix_comm
     INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target  :: iparm( : )
     REAL ( KIND = c_double ), INTENT( INOUT ), target  :: dparm( : )
   END SUBROUTINE pastixInit_f08

   SUBROUTINE pastix_task_analyze_f08( pastix_data, spm, info )
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
     INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE pastix_task_analyze_f08

   SUBROUTINE pastix_task_numfact_f08( pastix_data, spm, info )
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
     INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE pastix_task_numfact_f08

   SUBROUTINE pastix_task_solve_f08( pastix_data, nrhs, B, ldb, info )
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
     INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: nrhs
     CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), target :: B
     INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldb
     INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE pastix_task_solve_f08

   SUBROUTINE pastix_task_refine_f08( pastix_data, n, nrhs, B, ldb, X, ldx,    &
                                      info )
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET   :: pastix_data
     INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: n
     INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: nrhs
     CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), TARGET :: B
     INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldb
     CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), TARGET :: X
     INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldx
     INTEGER ( KIND = c_int ),        INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE pastix_task_refine_f08

   SUBROUTINE pastixOrderGet_f08( pastix_data, order )
     IMPLICIT NONE
     TYPE( pastix_data_t ), INTENT( IN ),  target  :: pastix_data
     TYPE( pastix_order_t ), INTENT( OUT ), pointer :: order
   END SUBROUTINE pastixOrderGet_f08

   SUBROUTINE pastixOrderGetArray_f08( order, permtab )
     IMPLICIT NONE
     TYPE( pastix_order_t ), INTENT( IN ), target  :: order
     INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,        &
                               POINTER :: permtab
   END SUBROUTINE pastixOrderGetArray_f08

   SUBROUTINE pastixFinalize_f08( pastix_data )
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), POINTER :: pastix_data
   END SUBROUTINE pastixFinalize_f08

 END MODULE pastixf_interfaces



