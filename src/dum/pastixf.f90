! THIS VERSION: GALAHAD 4.1 - 2022-10-26 AT 16:00 GMT.

! -*-*- G A L A H A D  -  D U M M Y   P A S T I X F  S U B R O U T I N E S -*-*-

   SUBROUTINE pastixInitParam_f08( iparm, dparm )
     USE iso_c_binding, ONLY : c_double
     USE pastixf_enums, ONLY : pastix_int_t
     IMPLICIT NONE
     INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target :: iparm( : )
     REAL ( KIND = c_double ), INTENT( INOUT ), target :: dparm( : )
   END SUBROUTINE pastixInitParam_f08

   SUBROUTINE pastixInit_f08( pastix_data, pastix_comm, iparm, dparm )
     USE iso_c_binding, ONLY : c_double
     USE spmf_enums, ONLY : MPI_Comm
     USE pastixf_enums, ONLY : pastix_data_t, pastix_int_t
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), pointer :: pastix_data
     TYPE ( MPI_Comm ), INTENT( IN ) :: pastix_comm
     INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target  :: iparm( : )
     REAL ( KIND = c_double ), INTENT( INOUT ), target  :: dparm( : )
   END SUBROUTINE pastixInit_f08

   SUBROUTINE pastix_task_analyze_f08( pastix_data, spm, info )
     USE iso_c_binding, ONLY : c_int
     USE spmf_enums, ONLY : spmatrix_t
     USE pastixf_enums, ONLY : pastix_data_t
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
     TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
     INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
     IF ( PRESENT( info ) ) info = - 26
   END SUBROUTINE pastix_task_analyze_f08

   SUBROUTINE pastix_task_numfact_f08( pastix_data, spm, info )
     USE iso_c_binding, ONLY : c_int
     USE spmf_enums, ONLY : spmatrix_t
     USE pastixf_enums, ONLY : pastix_data_t
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
     TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
     INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE pastix_task_numfact_f08

   SUBROUTINE pastix_task_solve_f08( pastix_data, nrhs, B, ldb, info )
     USE iso_c_binding, ONLY : c_int
     USE pastixf_enums, ONLY : pastix_data_t, pastix_int_t
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
     INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: nrhs
     CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), target :: B
     INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldb
     INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
   END SUBROUTINE pastix_task_solve_f08

   SUBROUTINE pastix_task_refine_f08( pastix_data, n, nrhs, B, ldb, X, ldx,    &
                                      info )
     USE iso_c_binding, ONLY : c_int
     USE pastixf_enums, ONLY : pastix_data_t, pastix_int_t
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
     USE pastixf_enums, ONLY : pastix_data_t, pastix_order_t
     IMPLICIT NONE
     TYPE( pastix_data_t ), INTENT( IN ),  target  :: pastix_data
     TYPE( pastix_order_t ), INTENT( OUT ), pointer :: order
   END SUBROUTINE pastixOrderGet_f08

   SUBROUTINE pastixOrderGetArray_f08( order, permtab, peritab, rangtab,       &
                                       treetab, sndetab )
     USE pastixf_enums, ONLY : pastix_int_t, pastix_order_t
     IMPLICIT NONE
     TYPE( pastix_order_t ), INTENT( IN ), target  :: order
     INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,        &
                               POINTER :: permtab
     INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,        &
                               POINTER :: peritab
     INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,        &
                               POINTER :: rangtab
     INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,        &
                               POINTER :: treetab
     INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,        &
                               POINTER :: sndetab
   END SUBROUTINE pastixOrderGetArray_f08

   SUBROUTINE pastixFinalize_f08( pastix_data )
     USE pastixf_enums, ONLY : pastix_data_t
     IMPLICIT NONE
     TYPE ( pastix_data_t ), INTENT( INOUT ), POINTER :: pastix_data
   END SUBROUTINE pastixFinalize_f08
