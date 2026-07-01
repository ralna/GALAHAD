! THIS VERSION: GALAHAD 5.6 - 2026-06-30 AT 10:30 GMT.

#include "galahad_modules.h"

!-*-*-*- G A L A H A D  -  H I G H S _ I N T E R F A C E   M O D U L E -*-*-*-

!  module to provide fortran interfaces to C HiGHS functions

!  Copyright reserved, GALAHAD productions
!  Principal authors: Julian Hall and Nick Gould

!  History -
!   more "fortran-like" variant of HiGHS module highs_fortran_api
!   originally released in GALAHAD Version 5.6. June 30th 2026

  MODULE HIGHS_INTERFACE_precision

!   USE GALAHAD_KINDS_precision
    USE, INTRINSIC :: iso_c_binding
    IMPLICIT NONE
    PUBLIC :: ipc_, rpc_, c_ptr, c_char, c_bool

!  integer (HIGHSINT64) and real (double) kinds chosen by C-preprocessor

#ifdef INTEGER_64
    INTEGER, PARAMETER :: ipc_ = c_int64_t
#else
    INTEGER, PARAMETER :: ipc_ = c_int32_t
#endif
#ifdef REAL_32
    INTEGER, PARAMETER :: rpc_ = c_float
#elif REAL_128
    INTEGER, PARAMETER :: rpc_ = c_float128
#else
    INTEGER, PARAMETER :: rpc_ = c_double
#endif

!  interface blocks for C functions

    INTERFACE
      FUNCTION Highs_create_c( ) RESULT( h ) BIND( C, NAME = 'Highs_create' )
        IMPORT :: c_ptr
        TYPE( c_ptr ) :: h
      END FUNCTION Highs_create_c
    END INTERFACE
        
    INTERFACE
      subroutine Highs_destroy_c( h ) BIND( C, NAME = 'Highs_destroy' )
        IMPORT :: c_ptr
        TYPE( c_ptr ), VALUE :: h 
      END subroutine Highs_destroy_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_lpCall_c( numcol, numrow, numnz, aformat, sense, offset,  &
                               colcost, collower, colupper, rowlower, rowupper,&
                               astart, aindex, avalue, colvalue, coldual,      &
                               rowvalue, rowdual, colbasisstatus,              &
                               rowbasisstatus, modelstatus )                   &
          RESULT( s ) BIND( C, NAME = 'Highs_lpCall' )
        IMPORT :: c_ptr, ipc_, rpc_
        INTEGER ( ipc_ ), VALUE :: numcol
        INTEGER ( ipc_ ), VALUE :: numrow
        INTEGER ( ipc_ ), VALUE :: numnz
        INTEGER ( ipc_ ), VALUE :: aformat
        INTEGER ( ipc_ ), VALUE :: sense
        REAL ( rpc_ ), VALUE :: offset
        REAL ( rpc_ ) :: colcost( * )
        REAL ( rpc_ ) :: collower( * )
        REAL ( rpc_ ) :: colupper( * )
        REAL ( rpc_ ) :: rowlower( * )
        REAL ( rpc_ ) :: rowupper( * )
        INTEGER ( ipc_ ) :: astart( * )
        INTEGER ( ipc_ ) :: aindex( * )
        REAL ( rpc_ ) :: avalue( * )
        REAL ( rpc_ ) :: colvalue( * )
        REAL ( rpc_ ) :: coldual( * )
        REAL ( rpc_ ) :: rowvalue( * )
        REAL ( rpc_ ) :: rowdual( * )
        INTEGER ( ipc_ ) :: colbasisstatus( * )
        INTEGER ( ipc_ ) :: rowbasisstatus( * )
        INTEGER ( ipc_ ) :: s
        INTEGER ( ipc_ ) :: modelstatus
      END FUNCTION Highs_lpCall_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_qpCall_c( numcol, numrow, numnz, qnumnz, aformat, qformat,&
                               sense, offset, colcost, collower, colupper,     &
                               rowlower, rowupper, astart, aindex, avalue,     &
                               qstart, qindex, qvalue, colvalue, coldual,      &
                               rowvalue, rowdual, colbasisstatus,              &
                               rowbasisstatus, modelstatus )                   &
              RESULT( s ) BIND( C, NAME = 'Highs_qpCall' )
        IMPORT :: c_ptr, ipc_, rpc_
        INTEGER ( ipc_ ), VALUE :: numcol
        INTEGER ( ipc_ ), VALUE :: numrow
        INTEGER ( ipc_ ), VALUE :: numnz
        INTEGER ( ipc_ ), VALUE :: qnumnz
        INTEGER ( ipc_ ), VALUE :: aformat
        INTEGER ( ipc_ ), VALUE :: qformat
        INTEGER ( ipc_ ), VALUE :: sense
        REAL ( rpc_ ), VALUE :: offset
        REAL ( rpc_ ) :: colcost( * )
        REAL ( rpc_ ) :: collower( * )
        REAL ( rpc_ ) :: colupper( * )
        REAL ( rpc_ ) :: rowlower( * )
        REAL ( rpc_ ) :: rowupper( * )
        INTEGER ( ipc_ ) :: astart( * )
        INTEGER ( ipc_ ) :: aindex( * )
        REAL ( rpc_ ) :: avalue( * )
        INTEGER ( ipc_ ) :: qstart( * )
        INTEGER ( ipc_ ) :: qindex( * )
        REAL ( rpc_ ) :: qvalue( * )
        REAL ( rpc_ ) :: colvalue( * )
        REAL ( rpc_ ) :: coldual( * )
        REAL ( rpc_ ) :: rowvalue( * )
        REAL ( rpc_ ) :: rowdual( * )
        INTEGER ( ipc_ ) :: colbasisstatus( * )
        INTEGER ( ipc_ ) :: rowbasisstatus( * )
        INTEGER ( ipc_ ) :: s
        INTEGER ( ipc_ ) :: modelstatus
      END FUNCTION Highs_qpCall_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_passLp_c( h, numcol, numrow, numnz, aformat,              &
                               sense, offset, colcost, collower, colupper,     &
                               rowlower, rowupper, astart, aindex, avalue )    &
          RESULT( s ) BIND( C, NAME = 'Highs_passLp' )
        IMPORT :: c_ptr, ipc_, rpc_
        TYPE( c_ptr ), VALUE :: h
        INTEGER ( ipc_ ), VALUE :: numcol
        INTEGER ( ipc_ ), VALUE :: numrow
        INTEGER ( ipc_ ), VALUE :: numnz
        INTEGER ( ipc_ ), VALUE :: aformat
        INTEGER ( ipc_ ), VALUE :: sense
        REAL ( rpc_ ), VALUE :: offset
        REAL ( rpc_ ) :: colcost( * )
        REAL ( rpc_ ) :: collower( * )
        REAL ( rpc_ ) :: colupper( * )
        REAL ( rpc_ ) :: rowlower( * )
        REAL ( rpc_ ) :: rowupper( * )
        INTEGER ( ipc_ ) :: astart( * )
        INTEGER ( ipc_ ) :: aindex( * )
        REAL ( rpc_ ) :: avalue( * )
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_passLp_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_passHessian_c( h, dim, numnz, qformat, qstart, qindex,    &
                                    qvalue )                                   &
          RESULT( s ) BIND( C, NAME = 'Highs_passHessian' )
        IMPORT :: c_ptr, ipc_, rpc_
        TYPE( c_ptr ), VALUE :: h
        INTEGER ( ipc_ ), VALUE :: dim
        INTEGER ( ipc_ ), VALUE :: numnz
        INTEGER ( ipc_ ), VALUE :: qformat
        INTEGER ( ipc_ ) :: qstart( * )
        INTEGER ( ipc_ ) :: qindex( * )
        REAL ( rpc_ ) :: qvalue( * )
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_passHessian_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_run_c( h )   RESULT( s ) BIND( C, NAME = 'Highs_run' )
        IMPORT :: c_ptr, ipc_
        TYPE( c_ptr ), VALUE :: h
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_run_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_setBoolOptionValue_c( h, o, v )                           &
          RESULT( s ) BIND( C, NAME = 'Highs_setBoolOptionValue' )
        IMPORT :: c_ptr, c_char, c_bool, ipc_
        TYPE( c_ptr ), VALUE :: h
        CHARACTER( c_char ) :: o( * )
        LOGICAL ( c_bool ), VALUE :: v
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_setBoolOptionValue_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_setIntOptionValue_c( h, o, v )                            &
          RESULT( s ) BIND( C, NAME = 'Highs_setIntOptionValue' )
        IMPORT :: c_ptr, c_char, ipc_
        TYPE( c_ptr ), VALUE :: h
        CHARACTER( c_char ) :: o( * )
        INTEGER ( ipc_ ), VALUE :: v
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_setIntOptionValue_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_setDoubleOptionValue_c( h, o, v )                         &
          RESULT( s ) BIND( C, NAME = 'Highs_setDoubleOptionValue' )
        IMPORT :: c_ptr, c_char, ipc_, rpc_
        TYPE( c_ptr ), VALUE :: h
        CHARACTER( c_char ) :: o( * )
        REAL ( rpc_ ), VALUE :: v
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_setDoubleOptionValue_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_getModelStatus_c( h )                                     &
          RESULT(model_status) BIND( C, NAME = 'Highs_getModelStatus' )
        IMPORT :: c_ptr, ipc_
        TYPE( c_ptr ), VALUE :: h
        INTEGER ( ipc_ ) :: model_status
      END FUNCTION Highs_getModelStatus_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_getObjectiveValue_c( h )                                  &
          RESULT( ov ) BIND( C, NAME = 'Highs_getObjectiveValue' )
        IMPORT :: c_ptr, rpc_
        TYPE( c_ptr ), VALUE :: h
        REAL ( rpc_ ) :: ov
      END FUNCTION Highs_getObjectiveValue_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_getIterationCount_c( h )                                  &
          RESULT( ic ) BIND( C, NAME = 'Highs_getIterationCount' )
        IMPORT :: c_ptr, ipc_
        TYPE( c_ptr ), VALUE :: h
        INTEGER ( ipc_ ) :: ic
      END FUNCTION Highs_getIterationCount_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_getIntInfoValue_c( h, o, v )                              &
          RESULT( s ) BIND( C, NAME = 'Highs_getIntInfoValue' )
        IMPORT :: c_ptr, c_char, ipc_
        TYPE( c_ptr ), VALUE :: h
        CHARACTER( c_char ) :: o( * )
        INTEGER ( ipc_ ) :: v
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_getIntInfoValue_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_getDoubleInfoValue_c( h, o, v )                           &
          RESULT( s ) BIND( C, NAME = 'Highs_getDoubleInfoValue' )
        IMPORT :: c_ptr, c_char, ipc_, rpc_
        TYPE( c_ptr ), VALUE :: h
        CHARACTER( c_char ) :: o( * )
        REAL ( rpc_ ) :: v
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_getDoubleInfoValue_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_getStringOptionValue_c( h, o, v )                         &
          RESULT( s ) BIND( C, NAME = 'Highs_getStringOptionValue' )
        IMPORT :: c_ptr, c_char, ipc_
        TYPE( c_ptr ), VALUE :: h
        CHARACTER( c_char ) :: o( * )
        CHARACTER( c_char ) :: v( * )
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_getStringOptionValue_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_getSolution_c( h, cv, cd, rv, rd )                        &
          RESULT( s ) BIND( C, NAME = 'Highs_getSolution' )
        IMPORT :: c_ptr, ipc_, rpc_
        TYPE( c_ptr ), VALUE :: h
        REAL ( rpc_ ) :: cv( * )
        REAL ( rpc_ ) :: cd( * )
        REAL ( rpc_ ) :: rv( * )
        REAL ( rpc_ ) :: rd( * )
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_getSolution_c
    END INTERFACE

    INTERFACE
      FUNCTION Highs_getBasis_c( h, cbs, rbs )                                 &
          RESULT( s ) BIND( C, NAME = 'Highs_getBasis' )
        IMPORT :: c_ptr, ipc_
        TYPE( c_ptr ), VALUE :: h
        INTEGER ( ipc_ ) :: cbs( * )
        INTEGER ( ipc_ ) :: rbs( * )
        INTEGER ( ipc_ ) :: s
      END FUNCTION Highs_getBasis_c
    END INTERFACE

  CONTAINS

!  fortran subroutines corresponding to C functions

    SUBROUTINE Highs_create( h ) 
    TYPE( c_ptr ) :: h
    h = Highs_create_c( )
    END SUBROUTINE Highs_create
      
    SUBROUTINE Highs_destroy( h )
    TYPE( c_ptr ), VALUE :: h 
    CALL Highs_destroy_c( h )
    END SUBROUTINE Highs_destroy

    SUBROUTINE Highs_lpCall( numcol, numrow, numnz, aformat, sense, offset,    &
                             colcost, collower, colupper, rowlower, rowupper,  &
                             astart, aindex, avalue, colvalue, coldual,        &
                             rowvalue, rowdual, colbasisstatus,                &
                             rowbasisstatus, modelstatus, status )
    INTEGER ( ipc_ ), VALUE :: numcol
    INTEGER ( ipc_ ), VALUE :: numrow
    INTEGER ( ipc_ ), VALUE :: numnz
    INTEGER ( ipc_ ), VALUE :: aformat
    INTEGER ( ipc_ ), VALUE :: sense
    REAL ( rpc_ ), VALUE :: offset
    REAL ( rpc_ ) :: colcost( * )
    REAL ( rpc_ ) :: collower( * )
    REAL ( rpc_ ) :: colupper( * )
    REAL ( rpc_ ) :: rowlower( * )
    REAL ( rpc_ ) :: rowupper( * )
    INTEGER ( ipc_ ) :: astart( * )
    INTEGER ( ipc_ ) :: aindex( * )
    REAL ( rpc_ ) :: avalue( * )
    REAL ( rpc_ ) :: colvalue( * )
    REAL ( rpc_ ) :: coldual( * )
    REAL ( rpc_ ) :: rowvalue( * )
    REAL ( rpc_ ) :: rowdual( * )
    INTEGER ( ipc_ ) :: colbasisstatus( * )
    INTEGER ( ipc_ ) :: rowbasisstatus( * )
    INTEGER ( ipc_ ) :: status
    INTEGER ( ipc_ ) :: modelstatus
    status = Highs_lpCall_c( numcol, numrow, numnz, aformat, sense, offset,    &
                             colcost, collower, colupper, rowlower, rowupper,  &
                             astart, aindex, avalue, colvalue, coldual,        &
                             rowvalue, rowdual, colbasisstatus,                &
                             rowbasisstatus, modelstatus )
    END SUBROUTINE Highs_lpCall

    SUBROUTINE Highs_qpCall( numcol, numrow, numnz, qnumnz, aformat, qformat,  &
                             sense, offset, colcost, collower, colupper,       &
                             rowlower, rowupper, astart, aindex, avalue,       &
                             qstart, qindex, qvalue, colvalue, coldual,        &
                             rowvalue, rowdual, colbasisstatus,                &
                             rowbasisstatus, modelstatus, status )
    INTEGER ( ipc_ ), VALUE :: numcol
    INTEGER ( ipc_ ), VALUE :: numrow
    INTEGER ( ipc_ ), VALUE :: numnz
    INTEGER ( ipc_ ), VALUE :: qnumnz
    INTEGER ( ipc_ ), VALUE :: aformat
    INTEGER ( ipc_ ), VALUE :: qformat
    INTEGER ( ipc_ ), VALUE :: sense
    REAL ( rpc_ ), value :: offset
    REAL ( rpc_ ) :: colcost( * )
    REAL ( rpc_ ) :: collower( * )
    REAL ( rpc_ ) :: colupper( * )
    REAL ( rpc_ ) :: rowlower( * )
    REAL ( rpc_ ) :: rowupper( * )
    INTEGER ( ipc_ ) :: astart( * )
    INTEGER ( ipc_ ) :: aindex( * )
    REAL ( rpc_ ) :: avalue( * )
    INTEGER ( ipc_ ) :: qstart( * )
    INTEGER ( ipc_ ) :: qindex( * )
    REAL ( rpc_ ) :: qvalue( * )
    REAL ( rpc_ ) :: colvalue( * )
    REAL ( rpc_ ) :: coldual( * )
    REAL ( rpc_ ) :: rowvalue( * )
    REAL ( rpc_ ) :: rowdual( * )
    INTEGER ( ipc_ ) :: colbasisstatus( * )
    INTEGER ( ipc_ ) :: rowbasisstatus( * )
    INTEGER ( ipc_ ) :: status
    INTEGER ( ipc_ ) :: modelstatus
    status = Highs_qpCall_c( numcol, numrow, numnz, qnumnz, aformat, qformat,  &
                             sense, offset, colcost, collower, colupper,       &
                             rowlower, rowupper, astart, aindex, avalue,       &
                             qstart, qindex, qvalue, colvalue, coldual,        &
                             rowvalue, rowdual, colbasisstatus,                &
                             rowbasisstatus, modelstatus )
    END SUBROUTINE Highs_qpCall

    SUBROUTINE Highs_passLp( h, numcol, numrow, numnz, aformat, sense,         &
                             offset, colcost, collower, colupper, rowlower,    &
                             rowupper, astart, aindex, avalue, status )
    TYPE( c_ptr ), VALUE :: h
    INTEGER ( ipc_ ), VALUE :: numcol
    INTEGER ( ipc_ ), VALUE :: numrow
    INTEGER ( ipc_ ), VALUE :: numnz
    INTEGER ( ipc_ ), VALUE :: aformat
    INTEGER ( ipc_ ), VALUE :: sense
    REAL ( rpc_ ), VALUE :: offset
    REAL ( rpc_ ) :: colcost( * )
    REAL ( rpc_ ) :: collower( * )
    REAL ( rpc_ ) :: colupper( * )
    REAL ( rpc_ ) :: rowlower( * )
    REAL ( rpc_ ) :: rowupper( * )
    INTEGER ( ipc_ ) :: astart( * )
    INTEGER ( ipc_ ) :: aindex( * )
    REAL ( rpc_ ) :: avalue( * )
    INTEGER ( ipc_ ) :: status
    status = Highs_passLp_c( h, numcol, numrow, numnz, aformat,                &
                             sense, offset, colcost, collower, colupper,       &
                             rowlower, rowupper, astart, aindex, avalue )
    END SUBROUTINE Highs_passLp

    SUBROUTINE Highs_passHessian( h, dim, numnz, qformat, qstart, qindex,      &
                                  qvalue, status )
    TYPE( c_ptr ), VALUE :: h
    INTEGER ( ipc_ ), VALUE :: dim
    INTEGER ( ipc_ ), VALUE :: numnz
    INTEGER ( ipc_ ), VALUE :: qformat
    INTEGER ( ipc_ ) :: qstart( * )
    INTEGER ( ipc_ ) :: qindex( * )
    REAL ( rpc_ ) :: qvalue( * )
    INTEGER ( ipc_ ) :: status
    status = Highs_passHessian_c( h, dim, numnz, qformat, qstart, qindex,      &
                                  qvalue )
    END SUBROUTINE Highs_passHessian

    SUBROUTINE Highs_run( h, status )
    TYPE( c_ptr ), VALUE :: h
    INTEGER ( ipc_ ) :: status
    status = Highs_run_c( h )
    END SUBROUTINE Highs_run

    SUBROUTINE Highs_setBoolOptionValue( h, o, v, status )
    TYPE( c_ptr ), VALUE :: h
    CHARACTER( c_char ) :: o( * )
    LOGICAL ( c_bool ), VALUE :: v
    INTEGER ( ipc_ ) :: status
    status = Highs_setBoolOptionValue_c( h, o, v )
    END SUBROUTINE Highs_setBoolOptionValue

    SUBROUTINE Highs_setIntOptionValue( h, o, v, status ) 
    TYPE( c_ptr ), VALUE :: h
    CHARACTER( c_char ) :: o( * )
    INTEGER ( ipc_ ), VALUE :: v
    INTEGER ( ipc_ ) :: status
    status = Highs_setIntOptionValue_c( h, o, v )
    END SUBROUTINE Highs_setIntOptionValue

    SUBROUTINE Highs_setDoubleOptionValue( h, o, v, status )
    TYPE( c_ptr ), VALUE :: h
    CHARACTER( c_char ) :: o( * )
    REAL ( rpc_ ), VALUE :: v
    INTEGER ( ipc_ ) :: status
    status = Highs_setDoubleOptionValue_c( h, o, v )
    END SUBROUTINE Highs_setDoubleOptionValue

    SUBROUTINE Highs_getModelStatus( h, model_status )
    TYPE( c_ptr ), VALUE :: h
    INTEGER ( ipc_ ) :: model_status
    model_status = Highs_getModelStatus_c( h )
    END SUBROUTINE Highs_getModelStatus

    SUBROUTINE Highs_getObjectiveValue( h, ov )
    TYPE( c_ptr ), VALUE :: h
    REAL ( rpc_ ) :: ov
    ov = Highs_getObjectiveValue_c( h )
    END SUBROUTINE Highs_getObjectiveValue

    SUBROUTINE Highs_getIterationCount( h, ic )
    TYPE( c_ptr ), VALUE :: h
    INTEGER ( ipc_ ) :: ic
    ic = Highs_getIterationCount_c( h )
    END SUBROUTINE Highs_getIterationCount

    SUBROUTINE Highs_getIntInfoValue( h, o, v, status )
    TYPE( c_ptr ), VALUE :: h
    CHARACTER( c_char ) :: o( * )
    INTEGER ( ipc_ ) :: v
    INTEGER ( ipc_ ) :: status
    status = Highs_getIntInfoValue_c( h, o, v )
    END SUBROUTINE Highs_getIntInfoValue

    SUBROUTINE Highs_getDoubleInfoValue( h, o, v, status )
    TYPE( c_ptr ), VALUE :: h
    CHARACTER( c_char ) :: o( * )
    REAL ( rpc_ ) :: v
    INTEGER ( ipc_ ) :: status
    status = Highs_getDoubleInfoValue_c( h, o, v )
    END SUBROUTINE Highs_getDoubleInfoValue

    SUBROUTINE Highs_getStringOptionValue( h, o, v, status )
    TYPE( c_ptr ), VALUE :: h
    CHARACTER( c_char ) :: o( * )
    CHARACTER( c_char ) :: v( * )
    INTEGER ( ipc_ ) :: status
    status = Highs_getStringOptionValue_c( h, o, v )
    END SUBROUTINE Highs_getStringOptionValue

    SUBROUTINE Highs_getSolution( h, cv, cd, rv, rd, status )
    TYPE( c_ptr ), VALUE :: h
    REAL ( rpc_ ) :: cv( * )
    REAL ( rpc_ ) :: cd( * )
    REAL ( rpc_ ) :: rv( * )
    REAL ( rpc_ ) :: rd( * )
    INTEGER ( ipc_ ) :: status
    status = Highs_getSolution_c( h, cv, cd, rv, rd )
    END SUBROUTINE Highs_getSolution

    SUBROUTINE Highs_getBasis( h, cbs, rbs, status )
    TYPE( c_ptr ), VALUE :: h
    INTEGER ( ipc_ ) :: cbs( * )
    INTEGER ( ipc_ ) :: rbs( * )
    INTEGER ( ipc_ ) :: status
    status = Highs_getBasis_c( h, cbs, rbs )
    END SUBROUTINE Highs_getBasis

  END MODULE HIGHS_INTERFACE_precision
