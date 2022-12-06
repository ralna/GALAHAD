! THIS VERSION: GALAHAD 4.1 - 2022-10-25 AT 16:25 GMT.

!-*-  G A L A H A D  -  D U M M Y   P A S T I X F _ E N U M S   M O D U L E  -*-

 MODULE pastixf_enums

   USE spmf_enums
   USE iso_c_binding, ONLY : c_double, c_int, c_ptr, c_int32_t, c_int64_t

   INTEGER, PARAMETER :: pastix_int_t = c_int32_t
 ! INTEGER, PARAMETER :: pastix_int_t = c_int64_t
   PUBLIC :: MPI_COMM_WORLD

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

   INTEGER, PARAMETER :: PASTIX_SUCCESS = 0
   INTEGER, PARAMETER :: PASTIX_ERR_OUTOFMEMORY = 4
   INTEGER, PARAMETER :: PASTIX_ERR_BADPARAMETER = 7

 END MODULE pastixf_enums



