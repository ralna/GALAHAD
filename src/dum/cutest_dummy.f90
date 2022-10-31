! THIS VERSION: GALAHAD 4.1 - 2022-10-29 AT 10:45 GMT.

!-*-*-  C U T E S T _ D U M M Y  P A C K A G E S / S U B P R O G R A M S  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Daniel Robinson and Nick Gould
!
!  History -
!   originally released GALAHAD Version 4.1. August 14th 2022

   SUBROUTINE RANGE()
   END SUBROUTINE RANGE

   SUBROUTINE ELFUN()
   END SUBROUTINE ELFUN

   SUBROUTINE ELFUN_flexible()
   END SUBROUTINE ELFUN_flexible

   SUBROUTINE GROUP()
   END SUBROUTINE GROUP

   SUBROUTINE dgqt()
   END SUBROUTINE dgqt

   MODULE GALAHAD_CUTEST_FUNCTIONS_double

     IMPLICIT NONE

!---------------------
!   P r e c i s i o n
!---------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

     PRIVATE
     PUBLIC :: CUTEst_initialize, CUTEst_eval_F, CUTEst_eval_FC,               &
               CUTEst_eval_C, CUTEst_eval_G, CUTEst_eval_GJ, CUTEst_eval_J,    &
               CUTEst_eval_H, CUTEst_eval_HPROD, CUTEst_eval_SHPROD,           &
               CUTEst_eval_JPROD, CUTEst_eval_SJPROD, CUTEst_eval_HL,          &
               CUTEst_eval_HLC, CUTEst_eval_HLPROD, CUTEst_eval_SHLPROD,       &
               CUTEst_eval_HLCPROD, CUTEst_eval_SHLCPROD, CUTEst_eval_HCPRODS, &
               CUTEst_start_timing, CUTEst_timing,                             &
               CUTEst_terminate

   CONTAINS

     SUBROUTINE CUTEst_initialize( )
     END SUBROUTINE CUTEst_initialize

     SUBROUTINE CUTEst_eval_F( )
     END SUBROUTINE CUTEst_eval_F

     SUBROUTINE CUTEst_eval_C( )
     END SUBROUTINE CUTEst_eval_C

     SUBROUTINE CUTEst_eval_FC( )
     END SUBROUTINE CUTEst_eval_FC

     SUBROUTINE CUTEst_eval_G( )
     END SUBROUTINE CUTEst_eval_G

     SUBROUTINE CUTEst_eval_J( )
     END SUBROUTINE CUTEst_eval_J

     SUBROUTINE CUTEst_eval_GJ( )
     END SUBROUTINE CUTEst_eval_GJ

     SUBROUTINE CUTEst_eval_H( )
     END SUBROUTINE CUTEst_eval_H

     SUBROUTINE CUTEst_eval_HL( )
     END SUBROUTINE CUTEst_eval_HL

     SUBROUTINE CUTEst_eval_HLC( )
     END SUBROUTINE CUTEst_eval_HLC

     SUBROUTINE CUTEst_eval_JPROD( )
     END SUBROUTINE CUTEst_eval_JPROD

     SUBROUTINE CUTEst_eval_SJPROD( )
     END SUBROUTINE CUTEst_eval_SJPROD

     SUBROUTINE CUTEst_eval_HPROD( )
     END SUBROUTINE CUTEst_eval_HPROD

     SUBROUTINE CUTEst_eval_SHPROD( )
     END SUBROUTINE CUTEst_eval_SHPROD

     SUBROUTINE CUTEst_eval_HLPROD( )
     END SUBROUTINE CUTEst_eval_HLPROD

     SUBROUTINE CUTEst_eval_SHLPROD( )
     END SUBROUTINE CUTEst_eval_SHLPROD

     SUBROUTINE CUTEst_eval_HLCPROD( )
     END SUBROUTINE CUTEst_eval_HLCPROD

     SUBROUTINE CUTEst_eval_SHLCPROD( )
     END SUBROUTINE CUTEst_eval_SHLCPROD

     SUBROUTINE CUTEst_eval_HCPRODS( )
     END SUBROUTINE CUTEst_eval_HCPRODS

     SUBROUTINE CUTEst_terminate( )
     END SUBROUTINE CUTEst_terminate

     SUBROUTINE CUTEst_start_timing( )
     END SUBROUTINE CUTEst_start_timing

     SUBROUTINE CUTEst_timing( )
     END SUBROUTINE CUTEst_timing

   END MODULE GALAHAD_CUTEST_FUNCTIONS_double

   SUBROUTINE CUTEST_probname( cutest_status, p_name )
   INTEGER, INTENT( OUT ) :: cutest_status
   CHARACTER ( LEN = 10 ), INTENT( OUT ) :: p_name
   END SUBROUTINE CUTEST_probname

   SUBROUTINE CUTEST_varnames( cutest_status, n, X_names )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   CHARACTER ( LEN = 10 ), INTENT( OUT ), DIMENSION( n ) :: X_names
   END SUBROUTINE CUTEST_varnames

   SUBROUTINE CUTEST_udimen( cutest_status, input, n )
   INTEGER, INTENT( IN ) :: input
   INTEGER, INTENT( OUT ) :: cutest_status, n
   END SUBROUTINE CUTEST_udimen

   SUBROUTINE CUTEST_usetup( cutest_status, input, out, io_buffer,         &
                             n, X, X_l, X_u )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: input, out, io_buffer
   INTEGER, INTENT( INOUT ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X, X_l, X_u
   END SUBROUTINE CUTEST_usetup

   SUBROUTINE CUTEST_unames( cutest_status, n, p_name, X_names )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   CHARACTER ( LEN = 10 ), INTENT( OUT ) :: p_name
   CHARACTER ( LEN = 10 ), INTENT( OUT ), DIMENSION( n ) :: X_names
   END SUBROUTINE CUTEST_unames

   SUBROUTINE CUTEST_uvartype( cutest_status, n, X_type )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   INTEGER, INTENT( OUT ) :: X_type( n )
   END SUBROUTINE CUTEST_uvartype

   SUBROUTINE CUTEST_ufn( cutest_status, n, X, f )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   END SUBROUTINE CUTEST_ufn

   SUBROUTINE CUTEST_ugr( cutest_status, n, X, G )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   END SUBROUTINE CUTEST_ugr

   SUBROUTINE CUTEST_uofg( cutest_status, n, X, f, G, grad )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   LOGICAL, INTENT( IN ) :: grad
   END SUBROUTINE CUTEST_uofg

   SUBROUTINE CUTEST_udh( cutest_status, n, X, lh1, H )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lh1
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_udh

   SUBROUTINE CUTEST_ugrdh( cutest_status, n, X, G, lh1, H )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lh1
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_ugrdh

   SUBROUTINE CUTEST_udimsh( cutest_status, nnzh )
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   END SUBROUTINE CUTEST_udimsh

   SUBROUTINE CUTEST_ushp( cutest_status, n, nnzh, lh, IRNH, ICNH )
   INTEGER, INTENT( IN ) :: n, lh
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   END SUBROUTINE CUTEST_ushp

   SUBROUTINE CUTEST_ush( cutest_status, n, X, nnzh, lh, H, IRNH, ICNH )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lh
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_ush

   SUBROUTINE CUTEST_ugrsh( cutest_status, n, X, G,                        &
                            nnzh, lh, H, IRNH, ICNH )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lh
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_ugrsh

   SUBROUTINE CUTEST_udimse( cutest_status, ne, nnzh, nzirnh )
   INTEGER, INTENT( OUT ) :: cutest_status, ne, nnzh, nzirnh
   END SUBROUTINE CUTEST_udimse

   SUBROUTINE CUTEST_ueh( cutest_status, n, X, ne, le, IPRNHI, IPRHI,      &
                          lirnhi, IRNHI, lhi, Hi, byrows )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, le, lirnhi, lhi
   INTEGER, INTENT( OUT ) :: cutest_status, ne
   LOGICAL, INTENT( IN ) :: byrows
   INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
   INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
   END SUBROUTINE CUTEST_ueh

   SUBROUTINE CUTEST_ugreh( cutest_status, n, X, G, ne, le, IPRNHI, IPRHI, &
                          lirnhi, IRNHI, lhi, Hi, byrows )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, le, lirnhi, lhi
   INTEGER, INTENT( OUT ) :: cutest_status, ne
   LOGICAL, INTENT( IN ) :: byrows
   INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
   INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
   END SUBROUTINE CUTEST_ugreh

   SUBROUTINE CUTEST_uhprod( cutest_status, n, goth, X, P, RESULT )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: goth
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, P
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_uhprod

   SUBROUTINE CUTEST_ushprod( status, n, goth, X,                          &
                              nnz_vector, INDEX_nz_vector, VECTOR,         &
                              nnz_result, INDEX_nz_result, RESULT )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, nnz_vector
   INTEGER, INTENT( OUT ) :: status, nnz_result
   LOGICAL, INTENT( IN ) :: goth
   INTEGER, DIMENSION( nnz_vector ), INTENT( IN ) :: INDEX_nz_vector
   INTEGER, DIMENSION( n ), INTENT( OUT ) :: INDEX_nz_result
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, VECTOR
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_ushprod

   SUBROUTINE CUTEST_ubandh( cutest_status, n, X, nsemib, BANDH, lbandh,   &
                             maxsbw )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, nsemib, lbandh
   INTEGER, INTENT( OUT ) :: cutest_status
   INTEGER, INTENT( OUT ) :: maxsbw
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) ::  X
   REAL ( KIND = wp ), INTENT( OUT ),                                      &
                            DIMENSION( 0 : lbandh, n ) ::  BANDH
   END SUBROUTINE CUTEST_ubandh

   SUBROUTINE CUTEST_ureport( cutest_status, CALLS, CPU )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 4 ) :: CALLS
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 4 ) :: CPU
   END SUBROUTINE CUTEST_ureport

   SUBROUTINE CUTEST_uterminate( cutest_status )
   INTEGER, INTENT( OUT ) :: cutest_status
   END SUBROUTINE CUTEST_uterminate

   SUBROUTINE CUTEST_cdimen( cutest_status, input, n, m )
   INTEGER, INTENT( IN ) :: input
   INTEGER, INTENT( OUT ) :: cutest_status, n, m
   END SUBROUTINE CUTEST_cdimen

   SUBROUTINE CUTEST_csetup( cutest_status, input, out, io_buffer,         &
                             n, m, X, X_l, X_u,                            &
                             Y, C_l, C_u, EQUATN, LINEAR,                  &
                             e_order, l_order, v_order )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) ::  input, out, io_buffer
   INTEGER, INTENT( IN ) ::  e_order, l_order, v_order
   INTEGER, INTENT( INOUT ) :: n, m
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X, X_l, X_u
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: Y, C_l, C_u
   LOGICAL, INTENT( OUT ), DIMENSION( m ) :: EQUATN, LINEAR
   END SUBROUTINE CUTEST_csetup

   SUBROUTINE CUTEST_cnames( cutest_status, n, m, p_name, X_names, C_names )
   INTEGER, INTENT( IN ) :: n, m
   INTEGER, INTENT( OUT ) :: cutest_status
   CHARACTER ( LEN = 10 ), INTENT( OUT ) :: p_name
   CHARACTER ( LEN = 10 ), INTENT( OUT ), DIMENSION( n ) :: X_names
   CHARACTER ( LEN = 10 ), INTENT( OUT ), DIMENSION( m ) :: C_names
   END SUBROUTINE CUTEST_cnames

   SUBROUTINE CUTEST_connames( cutest_status, m, C_names )
   INTEGER, INTENT( IN ) :: m
   INTEGER, INTENT( OUT ) :: cutest_status
   CHARACTER ( LEN = 10 ), INTENT( OUT ), DIMENSION( m ) :: C_names
   END SUBROUTINE CUTEST_connames

   SUBROUTINE CUTEST_cvartype( cutest_status, n, X_type )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   INTEGER, INTENT( OUT ) :: X_type( n )
   END SUBROUTINE CUTEST_cvartype

   SUBROUTINE CUTEST_cfn( cutest_status, n, m, X, f, C )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C
   END SUBROUTINE CUTEST_cfn

   SUBROUTINE CUTEST_cgr( cutest_status, n, m, X, Y, grlagf, G, jtrans,    &
                          lcjac1, lcjac2, CJAC  )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac1, lcjac2
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: grlagf, jtrans
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( OUT ),                                      &
                            DIMENSION( lcjac1, lcjac2 ) :: CJAC
   END SUBROUTINE CUTEST_cgr

   SUBROUTINE CUTEST_cofg( cutest_status, n, X, f, G, grad )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   LOGICAL, INTENT( IN ) :: grad
   END SUBROUTINE CUTEST_cofg

   SUBROUTINE CUTEST_cofsg( status, n, X, f, nnzg, lg, G_val, G_var, grad )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lg
   INTEGER, INTENT( OUT ) :: status, nnzg
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   LOGICAL, INTENT( IN ) :: grad
   INTEGER, INTENT( OUT ), DIMENSION( lg ) :: G_var
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lg ) :: G_val
   END SUBROUTINE CUTEST_cofsg

   SUBROUTINE CUTEST_cdimsj( cutest_status, nnzj )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: cutest_status, nnzj
   END SUBROUTINE CUTEST_cdimsj

   SUBROUTINE CUTEST_csjp( status, nnzj, lj, J_var, J_fun )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: lj
   INTEGER, INTENT( OUT ) :: nnzj, status
   INTEGER, INTENT( OUT ), DIMENSION( lj ) :: J_var, J_fun
   END SUBROUTINE CUTEST_csjp

   SUBROUTINE CUTEST_csgrp( status, n, nnzj, lj, J_var, J_fun )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lj
   INTEGER, INTENT( OUT ) :: nnzj, status
   INTEGER, INTENT( OUT ), DIMENSION( lj ) :: J_var, J_fun
   END SUBROUTINE CUTEST_csgrp

   SUBROUTINE CUTEST_csgr( cutest_status, n, m, X, Y, grlagf, nnzj,        &
                           lcjac, CJAC, INDVAR, INDFUN )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac
   INTEGER, INTENT( OUT ) :: cutest_status, nnzj
   LOGICAL, INTENT( IN ) :: grlagf
   INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
   END SUBROUTINE CUTEST_csgr

   SUBROUTINE CUTEST_ccfg( cutest_status, n, m, X, C, jtrans,              &
                           lcjac1, lcjac2, CJAC, grad )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac1, lcjac2
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C
   REAL ( KIND = wp ), INTENT( OUT ),                                      &
                            DIMENSION( lcjac1, lcjac2 ) :: CJAC
   LOGICAL, INTENT( IN ) :: jtrans, grad
   END SUBROUTINE CUTEST_ccfg

   SUBROUTINE CUTEST_ccfsg( cutest_status, n, m, X, C, nnzj, lcjac, CJAC,  &
                            INDVAR, INDFUN, grad )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac
   INTEGER, INTENT( OUT ) :: cutest_status, nnzj
   LOGICAL, INTENT( IN ) :: grad
   INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
   END SUBROUTINE CUTEST_ccfsg

   SUBROUTINE CUTEST_clfg( status, n, m, X, Y, f, G, grad )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   LOGICAL, INTENT( IN ) :: grad
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   END SUBROUTINE CUTEST_clfg

   SUBROUTINE CUTEST_ccifg( cutest_status, n, icon, X, ci, GCI, grad )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, icon
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: grad
   REAL ( KIND = wp ), INTENT( OUT ) :: ci
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GCI
   END SUBROUTINE CUTEST_ccifg

   SUBROUTINE CUTEST_ccifsg( cutest_status, n, icon, X, ci,                &
                             nnzgci, lgci, GCI, INDVAR, grad )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, icon, lgci
   INTEGER, INTENT( OUT ) :: cutest_status, nnzgci
   LOGICAL, INTENT( IN ) :: grad
   INTEGER, INTENT( OUT ), DIMENSION( lgci ) :: INDVAR
   REAL ( KIND = wp ), INTENT( OUT ) :: ci
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lgci ) :: GCI
   END SUBROUTINE CUTEST_ccifsg

   SUBROUTINE CUTEST_cdh( cutest_status, n, m, X, Y, lh1, H )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lh1
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_cdh

   SUBROUTINE CUTEST_cdhc( cutest_status, n, m, X, Y, lh1, H )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lh1
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_cdhc

   SUBROUTINE CUTEST_cidh( cutest_status, n, X, iprob, lh1, H )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, iprob, lh1
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_cidh

   SUBROUTINE CUTEST_cgrdh( cutest_status, n, m, X, Y, grlagf, G,          &
                            jtrans, lcjac1, lcjac2, CJAC, lh1, H     )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lh1, lcjac1, lcjac2
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: grlagf, jtrans
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   REAL ( KIND = wp ), INTENT( OUT ),                                      &
                            DIMENSION( lcjac1, lcjac2 ) :: CJAC
   END SUBROUTINE CUTEST_cgrdh

   SUBROUTINE CUTEST_cdimsh( cutest_status, nnzh )
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   END SUBROUTINE CUTEST_cdimsh

   SUBROUTINE CUTEST_cshp( cutest_status, n, nnzh, lh, IRNH, ICNH )
   INTEGER, INTENT( IN ) :: n, lh
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   END SUBROUTINE CUTEST_cshp

   SUBROUTINE CUTEST_csh( cutest_status, n, m, X, Y,                       &
                          nnzh, lh, H, IRNH, ICNH  )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lh
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_csh

   SUBROUTINE CUTEST_cshc( cutest_status, n, m, X, Y,                      &
                           nnzh, lh, H, IRNH, ICNH  )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lh
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_cshc

   SUBROUTINE CUTEST_cish( cutest_status, n, X, iprob,                     &
                           nnzh, lh, H, IRNH, ICNH  )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, iprob, lh
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_cish

   SUBROUTINE CUTEST_csgrshp( status, n, nnzj, lj, J_var, J_fun,           &
                              nnzh, lh, H_row, H_col )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lj, lh
   INTEGER, INTENT( OUT ) :: nnzh, nnzj, status
   INTEGER, INTENT( OUT ), DIMENSION( lj ) :: J_var, J_fun
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: H_row, H_col
   END SUBROUTINE CUTEST_csgrshp

   SUBROUTINE CUTEST_csgrsh( cutest_status, n, m, X, Y, grlagf, nnzj,      &
                             lcjac, CJAC, INDVAR, INDFUN, nnzh,            &
                             lh, H, IRNH, ICNH  )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac, lh
   INTEGER, INTENT( OUT ) :: cutest_status, nnzj, nnzh
   LOGICAL, INTENT( IN ) ::  grlagf
   INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
   END SUBROUTINE CUTEST_csgrsh

   SUBROUTINE CUTEST_cdimse( cutest_status, ne, nnzh, nzirnh )
   INTEGER, INTENT( OUT ) :: cutest_status, ne, nnzh, nzirnh
   END SUBROUTINE CUTEST_cdimse

   SUBROUTINE CUTEST_ceh( cutest_status, n, m, X, Y, ne, le, IPRNHI,       &
                          IPRHI, lirnhi, IRNHI, lhi, Hi, byrows )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, le, lirnhi, lhi
   INTEGER, INTENT( OUT ) :: cutest_status, ne
   LOGICAL, INTENT( IN ) :: byrows
   INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
   INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
   END SUBROUTINE CUTEST_ceh

   SUBROUTINE CUTEST_csgreh( cutest_status, n, m, X, Y, grlagf,            &
                             nnzj, lcjac, CJAC, INDVAR, INDFUN,            &
                             ne, le, IPRNHI, IPRHI, lirnhi, IRNHI, lhi,    &
                             Hi, byrows )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac, le, lirnhi, lhi
   INTEGER, INTENT( OUT ) :: cutest_status, ne, nnzj
   LOGICAL, INTENT( IN ) :: grlagf, byrows
   INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
   INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
   INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
   END SUBROUTINE CUTEST_csgreh

   SUBROUTINE CUTEST_chprod( cutest_status, n, m, goth, X, Y, P, RESULT )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: goth
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, P
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_chprod

   SUBROUTINE CUTEST_cshprod( status, n, m, goth, X, Y,                    &
                              nnz_vector, INDEX_nz_vector, VECTOR,         &
                              nnz_result, INDEX_nz_result, RESULT )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, nnz_vector
   INTEGER, INTENT( OUT ) :: status, nnz_result
   LOGICAL, INTENT( IN ) :: goth
   INTEGER, DIMENSION( nnz_vector ), INTENT( IN ) :: INDEX_nz_vector
   INTEGER, DIMENSION( n ), INTENT( OUT ) :: INDEX_nz_result
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, VECTOR
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_cshprod

   SUBROUTINE CUTEST_chcprod( cutest_status, n, m, goth, X, Y, P, RESULT )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: goth
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, P
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_chcprod

   SUBROUTINE CUTEST_cshcprod( status, n, m, goth, X, Y,                   &
                               nnz_vector, INDEX_nz_vector, VECTOR,        &
                               nnz_result, INDEX_nz_result, RESULT )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, nnz_vector
   INTEGER, INTENT( OUT ) :: status, nnz_result
   LOGICAL, INTENT( IN ) :: goth
   INTEGER, DIMENSION( nnz_vector ), INTENT( IN ) :: INDEX_nz_vector
   INTEGER, DIMENSION( n ), INTENT( OUT ) :: INDEX_nz_result
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, VECTOR
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_cshcprod

   SUBROUTINE CUTEST_cjprod( cutest_status, n, m, gotj, jtrans, X,         &
                             VECTOR, lvector, RESULT, lresult )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lvector, lresult
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: gotj, jtrans
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lvector ) :: VECTOR
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lresult ) :: RESULT
   END SUBROUTINE CUTEST_cjprod

   SUBROUTINE CUTEST_csjprod( cutest_status, n, m, gotj, jtrans, X,        &
                              nnz_vector, INDEX_nz_vector, VECTOR, lvector,&
                              nnz_result, INDEX_nz_result, RESULT, lresult )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, nnz_vector, lvector, lresult
   INTEGER, INTENT( OUT ) :: cutest_status, nnz_result
   LOGICAL, INTENT( IN ) :: gotj, jtrans
   INTEGER, DIMENSION( nnz_vector ), INTENT( IN ) :: INDEX_nz_vector
   INTEGER, DIMENSION( n ), INTENT( OUT ) :: INDEX_nz_result
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lvector ) :: VECTOR
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lresult ) :: RESULT
   END SUBROUTINE CUTEST_csjprod

   SUBROUTINE CUTEST_cdimchp( cutest_status, nnzchp )
   INTEGER, INTENT( OUT ) :: cutest_status, nnzchp
   END SUBROUTINE CUTEST_cdimchp

   SUBROUTINE CUTEST_cchprodsp( status, m, lchp, CHP_ind, CHP_ptr )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: m, lchp
   INTEGER, INTENT( OUT ) :: status
   INTEGER, INTENT( INOUT ), DIMENSION( m + 1 ) :: CHP_ptr
   INTEGER, INTENT( INOUT ), DIMENSION( lchp ) :: CHP_ind
   END SUBROUTINE CUTEST_cchprodsp

   SUBROUTINE CUTEST_cchprods( cutest_status, n, m, goth, X, VECTOR,       &
                               lchp, CHP_val, CHP_ind, CHP_ptr )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lchp
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: goth
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, VECTOR
   INTEGER, INTENT( INOUT ), DIMENSION( m + 1 ) :: CHP_ptr
   INTEGER, INTENT( INOUT ), DIMENSION( lchp ) :: CHP_ind
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lchp ) :: CHP_val
   END SUBROUTINE CUTEST_cchprods

   SUBROUTINE CUTEST_creport( cutest_status, CALLS, CPU )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 7 ) :: CALLS
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 4 ) :: CPU
   END SUBROUTINE CUTEST_creport

   SUBROUTINE CUTEST_cstats( cutest_status, nonlinear_variables_objective, &
                             nonlinear_variables_constraints,              &
                             equality_constraints, linear_constraint )
   INTEGER, INTENT( OUT ) :: cutest_status, nonlinear_variables_objective
   INTEGER, INTENT( OUT ) :: nonlinear_variables_constraints
   INTEGER, INTENT( OUT ) :: equality_constraints, linear_constraint
   END SUBROUTINE CUTEST_cstats

   SUBROUTINE CUTEST_cterminate( cutest_status )
   INTEGER, INTENT( OUT ) :: cutest_status
   END SUBROUTINE CUTEST_cterminate

!  rface block for threaded unconstrained tools

   SUBROUTINE CUTEST_usetup_threaded( cutest_status, input, out, threads,  &
                                      IO_BUFFERS, n, X, X_l, X_u )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: input, out, threads
   INTEGER, INTENT( INOUT ) :: n
   INTEGER, INTENT( OUT ) :: cutest_status
   INTEGER, INTENT( IN ), DIMENSION( threads ) :: IO_BUFFERS
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X, X_l, X_u
   END SUBROUTINE CUTEST_usetup_threaded

   SUBROUTINE CUTEST_ufn_threaded( cutest_status, n, X, f, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   END SUBROUTINE CUTEST_ufn_threaded

   SUBROUTINE CUTEST_ugr_threaded( cutest_status, n, X, G, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   END SUBROUTINE CUTEST_ugr_threaded

   SUBROUTINE CUTEST_uofg_threaded( cutest_status, n, X, f, G, grad,       &
                                    thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   LOGICAL, INTENT( IN ) :: grad
   END SUBROUTINE CUTEST_uofg_threaded

   SUBROUTINE CUTEST_udh_threaded( cutest_status, n, X, lh1, H, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lh1, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_udh_threaded

   SUBROUTINE CUTEST_ugrdh_threaded( cutest_status, n, X, G,               &
                                     lh1, H, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lh1, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_ugrdh_threaded

   SUBROUTINE CUTEST_ush_threaded( cutest_status, n, X,                    &
                                   nnzh, lh, H, IRNH, ICNH, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lh, thread
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_ush_threaded

   SUBROUTINE CUTEST_ugrsh_threaded( cutest_status, n, X, G,               &
                                     nnzh, lh, H, IRNH, ICNH, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, lh, thread
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_ugrsh_threaded

   SUBROUTINE CUTEST_ueh_threaded( cutest_status, n, X, ne, le, IPRNHI,    &
                                   IPRHI, lirnhi, IRNHI, lhi, Hi, byrows,  &
                                   thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, ne, le, lirnhi, lhi, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: byrows
   INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
   INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
   END SUBROUTINE CUTEST_ueh_threaded

   SUBROUTINE CUTEST_ugreh_threaded( cutest_status, n, X, G, ne, le,       &
                                     IPRNHI, IPRHI, lirnhi, IRNHI, lhi,    &
                                     Hi, byrows, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, le, lirnhi, lhi, thread
   INTEGER, INTENT( OUT ) :: cutest_status, ne
   LOGICAL, INTENT( IN ) :: byrows
   INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
   INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
   END SUBROUTINE CUTEST_ugreh_threaded

   SUBROUTINE CUTEST_uhprod_threaded( cutest_status, n, goth, X, P,        &
                                      RESULT, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: goth
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, P
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_uhprod_threaded

   SUBROUTINE CUTEST_ushprod_threaded( status, n, goth, X,                 &
                               nnz_vector, INDEX_nz_vector, VECTOR,        &
                               nnz_result, INDEX_nz_result, RESULT,        &
                               thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, nnz_vector, thread
   INTEGER, INTENT( OUT ) :: status, nnz_result
   LOGICAL, INTENT( IN ) :: goth
   INTEGER, DIMENSION( nnz_vector ), INTENT( IN ) :: INDEX_nz_vector
   INTEGER, DIMENSION( n ), INTENT( OUT ) :: INDEX_nz_result
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, VECTOR
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_ushprod_threaded

   SUBROUTINE CUTEST_ubandh_threaded( cutest_status, n, X, nsemib, BANDH,  &
                                      lbandh, maxsbw, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, nsemib, lbandh, thread
   INTEGER, INTENT( OUT ) :: cutest_status, maxsbw
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) ::  X
   REAL ( KIND = wp ), INTENT( OUT ),                                      &
                            DIMENSION( 0 : lbandh, n ) ::  BANDH
   END SUBROUTINE CUTEST_ubandh_threaded

   SUBROUTINE CUTEST_ureport_threaded( cutest_status, CALLS, CPU, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 4 ) :: CALLS
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 4 ) :: CPU
   END SUBROUTINE CUTEST_ureport_threaded

   SUBROUTINE CUTEST_csetup_threaded( cutest_status, input, out, threads,  &
                                      IO_BUFFERS, n, m, X, X_l, X_u,       &
                                      Y, C_l, C_u, EQUATN, LINEAR,         &
                                      e_order, l_order, v_order )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) ::  input, out, threads
   INTEGER, INTENT( IN ) ::  e_order, l_order, v_order
   INTEGER, INTENT( INOUT ) :: n, m
   INTEGER, INTENT( OUT ) :: cutest_status
   INTEGER, INTENT( IN ), DIMENSION( threads ) :: IO_BUFFERS
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X, X_l, X_u
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: Y, C_l, C_u
   LOGICAL, INTENT( OUT ), DIMENSION( m ) :: EQUATN, LINEAR
   END SUBROUTINE CUTEST_csetup_threaded

   SUBROUTINE CUTEST_cfn_threaded( cutest_status, n, m, X, f, C, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C
   END SUBROUTINE CUTEST_cfn_threaded

   SUBROUTINE CUTEST_cgr_threaded( cutest_status, n, m, X, Y, grlagf, G,   &
                                   jtrans, lcjac1, lcjac2, CJAC , thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac1, lcjac2, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: grlagf, jtrans
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( OUT ),                                      &
                            DIMENSION( lcjac1, lcjac2 ) :: CJAC
   END SUBROUTINE CUTEST_cgr_threaded

   SUBROUTINE CUTEST_cofg_threaded( cutest_status, n, X, f, G, grad,       &
                                    thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   LOGICAL, INTENT( IN ) :: grad
   END SUBROUTINE CUTEST_cofg_threaded

   SUBROUTINE CUTEST_csgr_threaded( cutest_status, n, m, X, Y, grlagf,     &
                                    nnzj, lcjac, CJAC, INDVAR, INDFUN,     &
                                    thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: cutest_status, nnzj
   INTEGER, INTENT( IN ) :: n, m, lcjac, thread
   LOGICAL, INTENT( IN ) :: grlagf
   INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
   END SUBROUTINE CUTEST_csgr_threaded

   SUBROUTINE CUTEST_ccfg_threaded( cutest_status, n, m, X, C, jtrans,     &
                                    lcjac1, lcjac2, CJAC, grad, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac1, lcjac2, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C
   REAL ( KIND = wp ), INTENT( OUT ),                                      &
                            DIMENSION( lcjac1, lcjac2 ) :: CJAC
   LOGICAL, INTENT( IN ) :: jtrans, grad
   END SUBROUTINE CUTEST_ccfg_threaded

   SUBROUTINE CUTEST_ccfsg_threaded( cutest_status, n, m, X, C, nnzj,      &
                                     lcjac, CJAC, INDVAR, INDFUN, grad,    &
                                     thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: cutest_status, nnzj
   INTEGER, INTENT( IN ) :: n, m, lcjac, thread
   LOGICAL, INTENT( IN ) :: grad
   INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
   END SUBROUTINE CUTEST_ccfsg_threaded

   SUBROUTINE CUTEST_ccifg_threaded( cutest_status, n, icon, X, ci, GCI,   &
                                     grad, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, icon, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: grad
   REAL ( KIND = wp ), INTENT( OUT ) :: ci
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GCI
   END SUBROUTINE CUTEST_ccifg_threaded

   SUBROUTINE CUTEST_ccifsg_threaded( cutest_status, n, icon, X, ci,       &
                                      nnzgci, lgci, GCI, INDVAR, grad,     &
                                      thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, icon, lgci, thread
   INTEGER, INTENT( OUT ) :: cutest_status, nnzgci
   LOGICAL, INTENT( IN ) :: grad
   INTEGER, INTENT( OUT ), DIMENSION( lgci ) :: INDVAR
   REAL ( KIND = wp ), INTENT( OUT ) :: ci
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lgci ) :: GCI
   END SUBROUTINE CUTEST_ccifsg_threaded

   SUBROUTINE CUTEST_cdh_threaded( cutest_status, n, m, X, Y,              &
                                   lh1, H, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lh1, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_cdh_threaded

   SUBROUTINE CUTEST_cidh_threaded( cutest_status, n, X, iprob,            &
                                    lh1, H, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, iprob, lh1, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   END SUBROUTINE CUTEST_cidh_threaded

   SUBROUTINE CUTEST_cgrdh_threaded( cutest_status, n, m, X, Y, grlagf,    &
                                     G, jtrans, lcjac1, lcjac2, CJAC,      &
                                     lh1, H, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lh1, lcjac1, lcjac2, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: grlagf, jtrans
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
   REAL ( KIND = wp ), INTENT( OUT ),                                      &
                            DIMENSION( lcjac1, lcjac2 ) :: CJAC
   END SUBROUTINE CUTEST_cgrdh_threaded

   SUBROUTINE CUTEST_csh_threaded( cutest_status, n, m, X, Y,              &
                                   nnzh, lh, H, IRNH, ICNH , thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lh, thread
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_csh_threaded

   SUBROUTINE CUTEST_cshc_threaded( cutest_status, n, m, X, Y,             &
                                    nnzh, lh, H, IRNH, ICNH , thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( IN ) :: n, m, lh, thread
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_cshc_threaded

   SUBROUTINE CUTEST_cish_threaded( cutest_status, n, X, iprob,            &
                                    nnzh, lh, H, IRNH, ICNH , thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, iprob, lh, thread
   INTEGER, INTENT( OUT ) :: cutest_status, nnzh
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   END SUBROUTINE CUTEST_cish_threaded

   SUBROUTINE CUTEST_csgrsh_threaded( cutest_status, n, m, X, Y, grlagf,   &
                                      nnzj, lcjac, CJAC, INDVAR, INDFUN,   &
                                      nnzh, lh, H, IRNH, ICNH , thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac, lh, thread
   INTEGER, INTENT( OUT ) :: cutest_status, nnzj, nnzh
   LOGICAL, INTENT( IN ) ::  grlagf
   INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
   INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
   END SUBROUTINE CUTEST_csgrsh_threaded

   SUBROUTINE CUTEST_ceh_threaded( cutest_status, n, m, X, Y, ne, le,      &
                                   IPRNHI, IPRHI, lirnhi, IRNHI, lhi,      &
                                   HI, byrows, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, le, lirnhi, lhi, thread
   INTEGER, INTENT( OUT ) :: cutest_status, ne
   LOGICAL, INTENT( IN ) :: byrows
   INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
   INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
   END SUBROUTINE CUTEST_ceh_threaded

   SUBROUTINE CUTEST_csgreh_threaded( cutest_status, n, m, X, Y, grlagf,   &
                                      nnzj, lcjac, CJAC, INDVAR, INDFUN,   &
                                      ne, le, IPRNHI, IPRHI, lirnhi,       &
                                      IRNHI, lhi, HI, byrows, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lcjac, le, lirnhi, lhi, thread
   INTEGER, INTENT( OUT ) :: cutest_status, ne, nnzj
   LOGICAL, INTENT( IN ) :: grlagf, byrows
   INTEGER, INTENT( IN ), DIMENSION( lcjac ) :: INDVAR, INDFUN
   INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
   INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
   END SUBROUTINE CUTEST_csgreh_threaded

   SUBROUTINE CUTEST_chprod_threaded( cutest_status, n, m, goth,           &
                                      X, Y, P, RESULT, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: goth
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, P
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_chprod_threaded

   SUBROUTINE CUTEST_cshprod_threaded( status, n, m, goth, X, Y,           &
                               nnz_vector, INDEX_nz_vector, VECTOR,        &
                               nnz_result, INDEX_nz_result, RESULT,        &
                               thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, nnz_vector, thread
   INTEGER, INTENT( OUT ) :: status, nnz_result
   LOGICAL, INTENT( IN ) :: goth
   INTEGER, DIMENSION( nnz_vector ), INTENT( IN ) :: INDEX_nz_vector
   INTEGER, DIMENSION( n ), INTENT( OUT ) :: INDEX_nz_result
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, VECTOR
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_cshprod_threaded

   SUBROUTINE CUTEST_chcprod_threaded( cutest_status, n, m, goth,          &
                                       X, Y, P, RESULT, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: goth
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, P
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_chcprod_threaded

   SUBROUTINE CUTEST_cshcprod_threaded( status, n, m, goth, X, Y,          &
                               nnz_vector, INDEX_nz_vector, VECTOR,        &
                               nnz_result, INDEX_nz_result, RESULT,        &
                               thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, nnz_vector, thread
   INTEGER, INTENT( OUT ) :: status, nnz_result
   LOGICAL, INTENT( IN ) :: goth
   INTEGER, DIMENSION( nnz_vector ), INTENT( IN ) :: INDEX_nz_vector
   INTEGER, DIMENSION( n ), INTENT( OUT ) :: INDEX_nz_result
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, VECTOR
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
   END SUBROUTINE CUTEST_cshcprod_threaded

   SUBROUTINE CUTEST_cjprod_threaded( cutest_status, n, m, gotj, X,        &
                                      VECTOR, lvector, RESULT, lresult,    &
                                      thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lvector, lresult, thread
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: gotj
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lvector ) :: VECTOR
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lresult ) :: RESULT
   END SUBROUTINE CUTEST_cjprod_threaded

   SUBROUTINE CUTEST_cchprods_threaded( cutest_status, n, m, goth, X,      &
                                        VECTOR, lchp, CHP_val, CHP_ind,    &
                                        CHP_ptr )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: n, m, lchp
   INTEGER, INTENT( OUT ) :: cutest_status
   LOGICAL, INTENT( IN ) :: goth
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, VECTOR
   INTEGER, INTENT( INOUT ), DIMENSION( m + 1 ) :: CHP_ptr
   INTEGER, INTENT( INOUT ), DIMENSION( lchp ) :: CHP_ind
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lchp ) :: CHP_val
   END SUBROUTINE CUTEST_cchprods_threaded

   SUBROUTINE CUTEST_creport_threaded( cutest_status, CALLS, CPU, thread )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( IN ) :: thread
   INTEGER, INTENT( OUT ) :: cutest_status
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 7 ) :: CALLS
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 4 ) :: CPU
   END SUBROUTINE CUTEST_creport_threaded
