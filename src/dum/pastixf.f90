! THIS VERSION: GALAHAD 4.1 - 2022-10-18 AT 16:30 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   P A S T I X F   M O D U L E  -*-*-*-

MODULE pastixf

  USE spmf
  USE iso_c_binding, ONLY : c_double, c_int, c_ptr, c_int32_t, c_int64_t

  INTEGER, PARAMETER :: pastix_int_t = c_int32_t
! INTEGER, PARAMETER :: pastix_int_t = c_int64_t

  TYPE, BIND( C ) :: pastix_data_t
    TYPE( c_ptr ) :: ptr
  END TYPE pastix_data_t

  TYPE, BIND( c ) :: pastix_order_t
    INTEGER ( KIND = pastix_int_t ) :: baseval
    INTEGER ( KIND = pastix_int_t ) :: vertnbr
    INTEGER ( KIND = pastix_int_t ) :: cblknbr
    TYPE ( c_ptr ) :: permtab
    TYPE ( c_ptr ) :: peritab
    TYPE ( c_ptr ) :: rangtab
    TYPE ( c_ptr ) :: treetab
    TYPE ( c_ptr ) :: selevtx
    INTEGER ( KIND = pastix_int_t ) :: sndenbr
    TYPE ( c_ptr ) :: sndetab
  END TYPE pastix_order_t

CONTAINS

  SUBROUTINE pastixInitParam( iparm, dparm )
    IMPLICIT NONE
    INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target :: iparm( : )
    REAL ( KIND = c_double ), INTENT( INOUT ), target :: dparm( : )
  END SUBROUTINE pastixInitParam

  SUBROUTINE pastixInit( pastix_data, pastix_comm, iparm, dparm )
    IMPLICIT NONE
    TYPE ( pastix_data_t ), INTENT( INOUT ), pointer :: pastix_data
    TYPE ( MPI_Comm ), INTENT( IN ) :: pastix_comm
    INTEGER ( KIND = pastix_int_t ), INTENT( INOUT ), target  :: iparm( : )
    REAL ( KIND = c_double ), INTENT( INOUT ), target  :: dparm( : )
  END SUBROUTINE pastixInit

  SUBROUTINE pastix_task_analyze( pastix_data, spm, info )
    IMPLICIT NONE
    TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
    TYPE ( spmatrix_t ), INTENT( IN ), TARGET :: spm
    INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
  END SUBROUTINE pastix_task_analyze

  SUBROUTINE pastix_task_numfact( pastix_data, spm, info )
    IMPLICIT NONE
    TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
    TYPE ( spmatrix_t ), INTENT( INOUT ), TARGET :: spm
    INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
  END SUBROUTINE pastix_task_numfact

  SUBROUTINE pastix_task_solve( pastix_data, nrhs, B, ldb, info )
    IMPLICIT NONE
    TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET :: pastix_data
    INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: nrhs
    CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), target :: B
    INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldb
    INTEGER ( KIND = c_int ), INTENT( OUT ), OPTIONAL :: info
  END SUBROUTINE pastix_task_solve

  SUBROUTINE pastix_task_refine( pastix_data, n, nrhs, B, ldb, X, ldx, info )
    IMPLICIT NONE
    TYPE ( pastix_data_t ), INTENT( INOUT ), TARGET   :: pastix_data
    INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: n
    INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: nrhs
    CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), TARGET :: B
    INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldb
    CLASS ( * ), DIMENSION( :, : ), INTENT( INOUT ), TARGET :: X
    INTEGER ( KIND = pastix_int_t ), INTENT( IN ) :: ldx
    INTEGER ( KIND = c_int ),        INTENT( OUT ), OPTIONAL :: info
  END SUBROUTINE pastix_task_refine

  SUBROUTINE pastixOrderGet( pastix_data, order )
    IMPLICIT NONE
    TYPE( pastix_data_t ), INTENT( IN ),  target  :: pastix_data
    TYPE( pastix_order_t ), INTENT( OUT ), pointer :: order
  END SUBROUTINE pastixOrderGet

  SUBROUTINE pastixOrderGetArray( order, permtab )
    IMPLICIT NONE
    TYPE( pastix_order_t ), INTENT( IN ), target  :: order
    INTEGER ( pastix_int_t ), DIMENSION( : ), INTENT( OUT ), OPTIONAL,         &
                              POINTER :: permtab
  END SUBROUTINE pastixOrderGetArray

END MODULE pastixf
