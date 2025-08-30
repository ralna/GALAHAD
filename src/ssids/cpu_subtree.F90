! THIS VERSION: GALAHAD 5.3 - 2025-08-29 AT 13:00 GMT

#include "ssids_procedures.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  author: Jonathan Hogg
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

  MODULE GALAHAD_SSIDS_cpu_subtree_precision
    USE GALAHAD_KINDS_precision
    USE GALAHAD_SSIDS_cpu_iface_precision, ONLY: cpu_factor_control,           &
                                                 cpu_factor_stats,             &
                                                 cpu_copy_control_in,          &
                                                 cpu_copy_stats_out
    USE GALAHAD_SSIDS_types_precision, ONLY: SSIDS_control_type,               &
                                             SSIDS_inform_type,                &
                                             SSIDS_SUCCESS,                    &
                                             SSIDS_ERROR_ALLOCATION,           &
                                             contrib_type
    USE GALAHAD_SSIDS_subtree_precision, ONLY : symbolic_subtree_base,         &
                                                numeric_subtree_base
    IMPLICIT none

    PRIVATE
    PUBLIC :: cpu_symbolic_subtree, construct_cpu_symbolic_subtree
    PUBLIC :: cpu_numeric_subtree, cpu_free_contrib

    TYPE, EXTENDS( symbolic_subtree_base ) :: cpu_symbolic_subtree
      INTEGER( KIND = ip_ ) :: n
      type( C_PTR ) :: csubtree
    CONTAINS
      PROCEDURE :: factor
      PROCEDURE :: cleanup => symbolic_cleanup
    END type cpu_symbolic_subtree

    TYPE, EXTENDS( numeric_subtree_base ) :: cpu_numeric_subtree
      LOGICAL( C_BOOL ) :: posdef
      TYPE( cpu_symbolic_subtree ), POINTER :: symbolic
      TYPE( C_PTR ) :: csubtree
    CONTAINS
      PROCEDURE :: get_contrib
      PROCEDURE :: solve_fwd
      PROCEDURE :: solve_diag
      PROCEDURE :: solve_diag_bwd
      PROCEDURE :: solve_bwd
      PROCEDURE :: enquire_posdef
      PROCEDURE :: enquire_indef
      PROCEDURE :: alter
      PROCEDURE :: cleanup => numeric_cleanup
    END TYPE cpu_numeric_subtree

#ifdef INTEGER_64
#ifdef REAL_32
  INTERFACE
     TYPE( C_PTR ) FUNCTION c_create_symbolic_subtree( n, sa, en, sptr,        &
          sparent, rptr, rlist, nptr, nlist, ncontrib, contrib_idx, control )  &
          BIND( C, NAME = "galahad_ssids_cpu_create_symbolic_subtree_sgl_64" )
      USE GALAHAD_KINDS
      IMPORT :: cpu_factor_control
      IMPLICIT none
      INTEGER( KIND = C_IP_ ), VALUE :: n
      INTEGER( KIND = C_IP_ ), VALUE :: sa
      INTEGER( KIND = C_IP_ ), VALUE :: en
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sptr
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sparent
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: rptr
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: rlist
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nptr
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nlist
      INTEGER( KIND = C_IP_ ), VALUE :: ncontrib
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: contrib_idx
      TYPE( cpu_factor_control ), INTENT( IN ) :: control
     END FUNCTION c_create_symbolic_subtree

     SUBROUTINE c_destroy_symbolic_subtree( subtree )                          &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_symbolic_subtree_sgl_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_symbolic_subtree

     TYPE( C_PTR ) FUNCTION c_create_numeric_subtree( posdef,                  &
          symbolic_subtree, aval, scaling, child_contrib, control, stats )     &
          BIND( C, NAME = "galahad_ssids_cpu_create_num_subtree_sgl_64" )
       USE GALAHAD_KINDS_precision
       IMPORT :: cpu_factor_control, cpu_factor_stats
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: symbolic_subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: aval
       TYPE( C_PTR ), VALUE :: scaling
       TYPE( C_PTR ), DIMENSION( * ), INTENT( INOUT ) :: child_contrib
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
       TYPE( cpu_factor_stats ), INTENT( OUT ) :: stats
     END FUNCTION c_create_numeric_subtree

     SUBROUTINE c_destroy_numeric_subtree( posdef, subtree )                   &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_num_subtree_sgl_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_numeric_subtree

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_fwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_fwd_sgl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_fwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag( posdef, subtree,   &
                                                            nrhs, x, ldx )     &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_sgl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag_bwd( posdef,        &
                                                                subtree,       &
                                                                nrhs, x, ldx ) &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_bwd_sgl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag_bwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_bwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_bwd_sgl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_bwd

     SUBROUTINE c_subtree_enquire( posdef, subtree, piv_order, d )             &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_enquire_sgl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       TYPE( C_PTR ), VALUE :: piv_order
       TYPE( C_PTR ), VALUE :: d
     END SUBROUTINE c_subtree_enquire

     SUBROUTINE c_subtree_alter( posdef, subtree, d )                          &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_alter_sgl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: d
     END SUBROUTINE c_subtree_alter

     SUBROUTINE c_get_contrib( posdef, subtree, n, val, ldval, rlist, ndelay,  &
                               delay_perm, delay_val, lddelay )                &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_get_contrib_sgl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ) :: n
       TYPE( C_PTR ) :: val
       INTEGER( KIND = C_IP_ ) :: ldval
       TYPE( C_PTR ) :: rlist
       INTEGER( KIND = C_IP_ ) :: ndelay
       TYPE( C_PTR ) :: delay_perm
       TYPE( C_PTR ) :: delay_val
       INTEGER( KIND = C_IP_ ) :: lddelay
     END SUBROUTINE c_get_contrib

     SUBROUTINE c_free_contrib( posdef, subtree )                              &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_free_contrib_sgl_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_free_contrib
  END INTERFACE

#elif REAL_128
  INTERFACE
     TYPE( C_PTR ) FUNCTION c_create_symbolic_subtree( n, sa, en, sptr,        &
          sparent, rptr, rlist, nptr, nlist, ncontrib, contrib_idx, control )  &
          BIND( C, NAME = "galahad_ssids_cpu_create_symbolic_subtree_qul_64" )
      USE GALAHAD_KINDS
      IMPORT :: cpu_factor_control
      IMPLICIT none
      INTEGER( KIND = C_IP_ ), VALUE :: n
      INTEGER( KIND = C_IP_ ), VALUE :: sa
      INTEGER( KIND = C_IP_ ), VALUE :: en
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sptr
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sparent
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: rptr
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: rlist
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nptr
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nlist
      INTEGER( KIND = C_IP_ ), VALUE :: ncontrib
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: contrib_idx
      TYPE( cpu_factor_control ), INTENT( IN ) :: control
     END FUNCTION c_create_symbolic_subtree

     SUBROUTINE c_destroy_symbolic_subtree( subtree )                          &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_symbolic_subtree_qul_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_symbolic_subtree

     TYPE( C_PTR ) FUNCTION c_create_numeric_subtree( posdef,                  &
          symbolic_subtree, aval, scaling, child_contrib, control, stats )     &
          BIND( C, NAME = "galahad_ssids_cpu_create_num_subtree_qul_64" )
       USE GALAHAD_KINDS_precision
       IMPORT :: cpu_factor_control, cpu_factor_stats
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: symbolic_subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: aval
       TYPE( C_PTR ), VALUE :: scaling
       TYPE( C_PTR ), DIMENSION( * ), INTENT( INOUT ) :: child_contrib
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
       TYPE( cpu_factor_stats ), INTENT( OUT ) :: stats
     END FUNCTION c_create_numeric_subtree

     SUBROUTINE c_destroy_numeric_subtree( posdef, subtree )                   &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_num_subtree_qul_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_numeric_subtree

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_fwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_fwd_qul_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_fwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag( posdef, subtree,   &
                                                            nrhs, x, ldx )     &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_qul_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag_bwd( posdef,        &
                                                                subtree,       &
                                                                nrhs, x, ldx ) &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_bwd_qul_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag_bwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_bwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_bwd_qul_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_bwd

     SUBROUTINE c_subtree_enquire( posdef, subtree, piv_order, d )             &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_enquire_qul_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       TYPE( C_PTR ), VALUE :: piv_order
       TYPE( C_PTR ), VALUE :: d
     END SUBROUTINE c_subtree_enquire

     SUBROUTINE c_subtree_alter( posdef, subtree, d )                         &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_alter_qul_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: d
     END SUBROUTINE c_subtree_alter

     SUBROUTINE c_get_contrib( posdef, subtree, n, val, ldval, rlist, ndelay,  &
                               delay_perm, delay_val, lddelay )                &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_get_contrib_qul_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ) :: n
       TYPE( C_PTR ) :: val
       INTEGER( KIND = C_IP_ ) :: ldval
       TYPE( C_PTR ) :: rlist
       INTEGER( KIND = C_IP_ ) :: ndelay
       TYPE( C_PTR ) :: delay_perm
       TYPE( C_PTR ) :: delay_val
       INTEGER( KIND = C_IP_ ) :: lddelay
     END SUBROUTINE c_get_contrib

     SUBROUTINE c_free_contrib( posdef, subtree )                              &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_free_contrib_qul_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_free_contrib
  END INTERFACE

#else

  INTERFACE
     TYPE( C_PTR ) FUNCTION c_create_symbolic_subtree( n, sa, en, sptr,        &
          sparent, rptr, rlist, nptr, nlist, ncontrib, contrib_idx, control )  &
          BIND( C, NAME = "galahad_ssids_cpu_create_symbolic_subtree_dbl_64" )
      USE GALAHAD_KINDS
      IMPORT :: cpu_factor_control
      IMPLICIT none
      INTEGER( KIND = C_IP_ ), VALUE :: n
      INTEGER( KIND = C_IP_ ), VALUE :: sa
      INTEGER( KIND = C_IP_ ), VALUE :: en
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sptr
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sparent
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: rptr
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: rlist
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nptr
      INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nlist
      INTEGER( KIND = C_IP_ ), VALUE :: ncontrib
      INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: contrib_idx
      TYPE( cpu_factor_control ), INTENT( IN ) :: control
     END FUNCTION c_create_symbolic_subtree

     SUBROUTINE c_destroy_symbolic_subtree( subtree )                          &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_symbolic_subtree_dbl_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_symbolic_subtree

     TYPE( C_PTR ) FUNCTION c_create_numeric_subtree( posdef,                  &
          symbolic_subtree, aval, scaling, child_contrib, control, stats )     &
          BIND( C, NAME = "galahad_ssids_cpu_create_num_subtree_dbl_64" )
       USE GALAHAD_KINDS_precision
       IMPORT :: cpu_factor_control, cpu_factor_stats
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: symbolic_subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: aval
       TYPE( C_PTR ), VALUE :: scaling
       TYPE( C_PTR ), DIMENSION( * ), INTENT( INOUT ) :: child_contrib
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
       TYPE( cpu_factor_stats ), INTENT( OUT ) :: stats
     END FUNCTION c_create_numeric_subtree

     SUBROUTINE c_destroy_numeric_subtree( posdef, subtree )                   &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_num_subtree_dbl_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_numeric_subtree

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_fwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_fwd_dbl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_fwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag( posdef, subtree,   &
                                                            nrhs, x, ldx )     &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_dbl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag_bwd( posdef,        &
                                                                subtree,       &
                                                                nrhs, x, ldx ) &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_bwd_dbl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag_bwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_bwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_bwd_dbl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_bwd

     SUBROUTINE c_subtree_enquire( posdef, subtree, piv_order, d )             &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_enquire_dbl_64" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       TYPE( C_PTR ), VALUE :: piv_order
       TYPE( C_PTR ), VALUE :: d
     END SUBROUTINE c_subtree_enquire

     SUBROUTINE c_subtree_alter( posdef, subtree, d )                          &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_alter_dbl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: d
     END SUBROUTINE c_subtree_alter

     SUBROUTINE c_get_contrib( posdef, subtree, n, val, ldval, rlist, ndelay, &
                               delay_perm, delay_val, lddelay )               &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_get_contrib_dbl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ) :: n
       TYPE( C_PTR ) :: val
       INTEGER( KIND = C_IP_ ) :: ldval
       TYPE( C_PTR ) :: rlist
       INTEGER( KIND = C_IP_ ) :: ndelay
       TYPE( C_PTR ) :: delay_perm
       TYPE( C_PTR ) :: delay_val
       INTEGER( KIND = C_IP_ ) :: lddelay
     END SUBROUTINE c_get_contrib

     SUBROUTINE c_free_contrib( posdef, subtree )                              &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_free_contrib_dbl_64" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_free_contrib
  END INTERFACE
#endif
#else
#ifdef REAL_32
  INTERFACE
     TYPE( C_PTR ) FUNCTION c_create_symbolic_subtree( n, sa, en, sptr,        &
          sparent, rptr, rlist, nptr, nlist, ncontrib, contrib_idx, control )  &
          BIND( C, NAME = "galahad_ssids_cpu_create_symbolic_subtree_sgl" )
       USE GALAHAD_KINDS
       IMPORT :: cpu_factor_control
       IMPLICIT none
       INTEGER( KIND = C_IP_ ), VALUE :: n
       INTEGER( KIND = C_IP_ ), VALUE :: sa
       INTEGER( KIND = C_IP_ ), VALUE :: en
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sptr
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sparent
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: rptr
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: rlist
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nptr
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nlist
       INTEGER( KIND = C_IP_ ), VALUE :: ncontrib
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: contrib_idx
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
     END FUNCTION c_create_symbolic_subtree

     SUBROUTINE c_destroy_symbolic_subtree( subtree )                          &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_symbolic_subtree_sgl" )
       USE GALAHAD_KINDS
       IMPLICIT none
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_symbolic_subtree

     TYPE( C_PTR ) FUNCTION c_create_numeric_subtree( posdef,                  &
          symbolic_subtree, aval, scaling, child_contrib, control, stats )     &
          BIND( C, NAME = "galahad_ssids_cpu_create_num_subtree_sgl" )
       USE GALAHAD_KINDS_precision
       IMPORT :: cpu_factor_control, cpu_factor_stats
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: symbolic_subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: aval
       TYPE( C_PTR ), VALUE :: scaling
       TYPE( C_PTR ), DIMENSION( * ), INTENT( INOUT ) :: child_contrib
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
       TYPE( cpu_factor_stats ), INTENT( OUT ) :: stats
     END FUNCTION c_create_numeric_subtree

     SUBROUTINE c_destroy_numeric_subtree( posdef, subtree )                   &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_num_subtree_sgl" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_numeric_subtree

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_fwd( posdef, subtree,    &
                                                          nrhs, x,  ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_fwd_sgl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_fwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag( posdef, subtree,   &
                                                            nrhs, x, ldx )     &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_sgl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag_bwd( posdef,        &
                                                                subtree,       &
                                                                nrhs, x, ldx ) &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_bwd_sgl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag_bwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_bwd( posdef, subtree,    &
                                                           nrhs, x,  ldx )     &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_bwd_sgl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_bwd

     SUBROUTINE c_subtree_enquire( posdef, subtree, piv_order, d )             &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_enquire_sgl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       TYPE( C_PTR ), VALUE :: piv_order
       TYPE( C_PTR ), VALUE :: d
     END SUBROUTINE c_subtree_enquire

     SUBROUTINE c_subtree_alter( posdef, subtree, d )                          &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_alter_sgl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: d
     END SUBROUTINE c_subtree_alter

     SUBROUTINE c_get_contrib( posdef, subtree, n, val, ldval, rlist, ndelay,  &
                               delay_perm, delay_val, lddelay )                &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_get_contrib_sgl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ) :: n
       TYPE( C_PTR ) :: val
       INTEGER( KIND = C_IP_ ) :: ldval
       TYPE( C_PTR ) :: rlist
       INTEGER( KIND = C_IP_ ) :: ndelay
       TYPE( C_PTR ) :: delay_perm
       TYPE( C_PTR ) :: delay_val
       INTEGER( KIND = C_IP_ ) :: lddelay
     END SUBROUTINE c_get_contrib

     SUBROUTINE c_free_contrib( posdef, subtree )                              &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_free_contrib_sgl" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_free_contrib
  END INTERFACE

#elif REAL_128
  INTERFACE
     TYPE( C_PTR ) FUNCTION c_create_symbolic_subtree( n, sa, en, sptr,        &
          sparent, rptr, rlist, nptr, nlist, ncontrib, contrib_idx, control )  &
          BIND( C, NAME = "galahad_ssids_cpu_create_symbolic_subtree_qul" )
       USE GALAHAD_KINDS
       IMPORT :: cpu_factor_control
       IMPLICIT none
       INTEGER( KIND = C_IP_ ), VALUE :: n
       INTEGER( KIND = C_IP_ ), VALUE :: sa
       INTEGER( KIND = C_IP_ ), VALUE :: en
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sptr
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sparent
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: rptr
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: rlist
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nptr
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nlist
       INTEGER( KIND = C_IP_ ), VALUE :: ncontrib
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: contrib_idx
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
     END FUNCTION c_create_symbolic_subtree

     SUBROUTINE c_destroy_symbolic_subtree( subtree )                          &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_symbolic_subtree_qul" )
       USE GALAHAD_KINDS
       IMPLICIT none
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_symbolic_subtree

     TYPE( C_PTR ) FUNCTION c_create_numeric_subtree( posdef,                  &
          symbolic_subtree, aval, scaling, child_contrib, control, stats )     &
          BIND( C, NAME = "galahad_ssids_cpu_create_num_subtree_qul" )
       USE GALAHAD_KINDS_precision
       IMPORT :: cpu_factor_control, cpu_factor_stats
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: symbolic_subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: aval
       TYPE( C_PTR ), VALUE :: scaling
       TYPE( C_PTR ), DIMENSION( * ), INTENT( INOUT ) :: child_contrib
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
       TYPE( cpu_factor_stats ), INTENT( OUT ) :: stats
     END FUNCTION c_create_numeric_subtree

     SUBROUTINE c_destroy_numeric_subtree( posdef, subtree )                   &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_num_subtree_qul" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_numeric_subtree

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_fwd( posdef, subtree,    &
                                                          nrhs, x, ldx )       &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_fwd_qul" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_fwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag( posdef, subtree,   &
                                                            nrhs, x, ldx )     &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_qul" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag_bwd( posdef,        &
                                                                subtree,       &
                                                                nrhs, x, ldx ) &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_bwd_qul" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag_bwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_bwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_bwd_qul" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_bwd

     SUBROUTINE c_subtree_enquire( posdef, subtree, piv_order, d )             &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_enquire_qul" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       TYPE( C_PTR ), VALUE :: piv_order
       TYPE( C_PTR ), VALUE :: d
     END SUBROUTINE c_subtree_enquire

     SUBROUTINE c_subtree_alter( posdef, subtree, d )                          &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_alter_qul" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: d
     END SUBROUTINE c_subtree_alter

     SUBROUTINE c_get_contrib( posdef, subtree, n, val, ldval, rlist, ndelay,  &
                               delay_perm, delay_val, lddelay )                &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_get_contrib_qul" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ) :: n
       TYPE( C_PTR ) :: val
       INTEGER( KIND = C_IP_ ) :: ldval
       TYPE( C_PTR ) :: rlist
       INTEGER( KIND = C_IP_ ) :: ndelay
       TYPE( C_PTR ) :: delay_perm
       TYPE( C_PTR ) :: delay_val
       INTEGER( KIND = C_IP_ ) :: lddelay
     END SUBROUTINE c_get_contrib

     SUBROUTINE c_free_contrib( posdef, subtree )                              &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_free_contrib_qul" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_free_contrib
  END INTERFACE

#else

  INTERFACE
     TYPE( C_PTR ) FUNCTION c_create_symbolic_subtree( n, sa, en, sptr,        &
          sparent, rptr, rlist, nptr, nlist, ncontrib, contrib_idx, control )  &
          BIND( C, NAME = "galahad_ssids_cpu_create_symbolic_subtree_dbl" )
       USE GALAHAD_KINDS
       IMPORT :: cpu_factor_control
       IMPLICIT none
       INTEGER( KIND = C_IP_ ), VALUE :: n
       INTEGER( KIND = C_IP_ ), VALUE :: sa
       INTEGER( KIND = C_IP_ ), VALUE :: en
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sptr
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: sparent
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: rptr
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: rlist
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nptr
       INTEGER( KIND = CLONG_ ), DIMENSION( * ), INTENT( IN ) :: nlist
       INTEGER( KIND = C_IP_ ), VALUE :: ncontrib
       INTEGER( KIND = C_IP_ ), DIMENSION( * ), INTENT( IN ) :: contrib_idx
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
     END FUNCTION c_create_symbolic_subtree

     SUBROUTINE c_destroy_symbolic_subtree( subtree )                          &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_symbolic_subtree_dbl" )
       USE GALAHAD_KINDS
       IMPLICIT none
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_symbolic_subtree

     TYPE( C_PTR ) FUNCTION c_create_numeric_subtree( posdef,                  &
          symbolic_subtree, aval, scaling, child_contrib, control, stats )     &
          BIND( C, NAME = "galahad_ssids_cpu_create_num_subtree_dbl" )
       USE GALAHAD_KINDS_precision
       IMPORT :: cpu_factor_control, cpu_factor_stats
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: symbolic_subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: aval
       TYPE( C_PTR ), VALUE :: scaling
       TYPE( C_PTR ), DIMENSION( * ), INTENT( INOUT ) :: child_contrib
       TYPE( cpu_factor_control ), INTENT( IN ) :: control
       TYPE( cpu_factor_stats ), INTENT( OUT ) :: stats
     END FUNCTION c_create_numeric_subtree

     SUBROUTINE c_destroy_numeric_subtree( posdef, subtree )                   &
          BIND( C, NAME = "galahad_ssids_cpu_destroy_num_subtree_dbl" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_destroy_numeric_subtree

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_fwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_fwd_dbl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_fwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag( posdef, subtree,   &
                                                            nrhs, x, ldx )     &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_dbl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_diag_bwd( posdef,        &
                                                                subtree,       &
                                                                nrhs, x, ldx ) &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_diag_bwd_dbl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_diag_bwd

     INTEGER( KIND = C_IP_ ) FUNCTION c_subtree_solve_bwd( posdef, subtree,    &
                                                           nrhs, x, ldx )      &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_solve_bwd_dbl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ), VALUE :: nrhs
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( INOUT ) :: x
       INTEGER( KIND = C_IP_ ), VALUE :: ldx
     END FUNCTION c_subtree_solve_bwd

     SUBROUTINE c_subtree_enquire( posdef, subtree, piv_order, d )             &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_enquire_dbl" )
       USE GALAHAD_KINDS
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       TYPE( C_PTR ), VALUE :: piv_order
       TYPE( C_PTR ), VALUE :: d
     END SUBROUTINE c_subtree_enquire

     SUBROUTINE c_subtree_alter( posdef, subtree, d )                          &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_alter_dbl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       REAL( KIND = C_RP_ ), DIMENSION( * ), INTENT( IN ) :: d
     END SUBROUTINE c_subtree_alter

     SUBROUTINE c_get_contrib( posdef, subtree, n, val, ldval, rlist, ndelay,  &
                               delay_perm, delay_val, lddelay )                &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_get_contrib_dbl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
       INTEGER( KIND = C_IP_ ) :: n
       TYPE( C_PTR ) :: val
       INTEGER( KIND = C_IP_ ) :: ldval
       TYPE( C_PTR ) :: rlist
       INTEGER( KIND = C_IP_ ) :: ndelay
       TYPE( C_PTR ) :: delay_perm
       TYPE( C_PTR ) :: delay_val
       INTEGER( KIND = C_IP_ ) :: lddelay
     END SUBROUTINE c_get_contrib

     SUBROUTINE c_free_contrib( posdef, subtree )                              &
          BIND( C, NAME = "galahad_ssids_cpu_subtree_free_contrib_dbl" )
       USE GALAHAD_KINDS_precision
       IMPLICIT none
       LOGICAL( C_BOOL ), VALUE :: posdef
       TYPE( C_PTR ), VALUE :: subtree
     END SUBROUTINE c_free_contrib
  END INTERFACE
#endif
#endif

  CONTAINS

    FUNCTION construct_cpu_symbolic_subtree( n, sa, en, sptr, sparent, rptr,   &
       rlist, nptr, nlist, contrib_idx, control ) RESULT( this )
    IMPLICIT none
    CLASS( cpu_symbolic_subtree ), POINTER :: this
    INTEGER( KIND = ip_ ), INTENT( IN ) :: n
    INTEGER( KIND = ip_ ), INTENT( IN ) :: sa
    INTEGER( KIND = ip_ ), INTENT( IN ) :: en
    INTEGER( KIND = ip_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: sptr
    INTEGER( KIND = ip_ ), DIMENSION( * ), INTENT( IN ) :: sparent
    INTEGER( KIND = long_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: rptr
    INTEGER( KIND = ip_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: rlist
    INTEGER( KIND = long_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: nptr
    INTEGER( KIND = long_ ), DIMENSION( 2,* ), TARGET, INTENT( IN ) :: nlist
    INTEGER( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: contrib_idx
    CLASS( ssids_control_type ), INTENT( IN ) :: control

    INTEGER( KIND = ip_ ) :: st
    TYPE( cpu_factor_control ) :: ccontrol

    NULLIFY( this )

!  allocate output

    ALLOCATE( this, STAT = st )
    IF ( st /= 0 ) RETURN

!  store basic details

    this%n = n

!  call C++ subtree analyse

    CALL cpu_copy_control_in( control, ccontrol )
    this%csubtree =                                                            &
      c_create_symbolic_subtree( n, sa, en, sptr, sparent, rptr, rlist,        &
                                 nptr, nlist, INT( SIZE( contrib_idx ), ip_ ), &
                                 contrib_idx, ccontrol )
    RETURN

    END FUNCTION construct_cpu_symbolic_subtree

    SUBROUTINE symbolic_cleanup( this )
    IMPLICIT none
    CLASS( cpu_symbolic_subtree ), INTENT( INOUT ) :: this

    CALL c_destroy_symbolic_subtree( this%csubtree )
    END SUBROUTINE symbolic_cleanup

    FUNCTION factor( this, posdef, aval, child_contrib, control, inform,       &
                     scaling )
    IMPLICIT none
    CLASS( numeric_subtree_base ), POINTER :: factor
    CLASS( cpu_symbolic_subtree ), TARGET, INTENT( INOUT ) :: this
    LOGICAL, INTENT( IN ) :: posdef
    REAL( KIND = rp_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: aval
    TYPE( contrib_type ), DIMENSION( : ), TARGET,                              &
                                          INTENT( INOUT ) :: child_contrib
    TYPE( ssids_control_type ), INTENT( IN ) :: control
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform
    REAL( KIND = rp_ ), DIMENSION( * ), TARGET, OPTIONAL,                      &
                                                INTENT( IN ) :: scaling

    TYPE( cpu_numeric_subtree ), POINTER :: cpu_factor
    TYPE( cpu_factor_control ) :: ccontrol
    TYPE( cpu_factor_stats ) :: cstats
    TYPE( C_PTR ) :: cscaling
    INTEGER( KIND = ip_ ) :: i
    TYPE( C_PTR ), DIMENSION( : ), allocatable :: contrib_ptr
    INTEGER( KIND = ip_ ) :: st

!  leave output as null until successful exit

    NULLIFY( factor )

!  allocate cpu_factor for output

    ALLOCATE( cpu_factor, STAT = st )
    IF ( st /= 0 ) GO TO 10
    cpu_factor%symbolic => this

!  convert child_contrib to contrib_ptr

    ALLOCATE( contrib_ptr( size( child_contrib ) ), STAT = st )
    IF ( st /= 0 ) GO TO 10
    DO i = 1, size( child_contrib )
      contrib_ptr( i ) = C_LOC( child_contrib( i ) )
    END DO

!  call C++ factor routine

    cpu_factor%posdef = posdef
    cscaling = C_NULL_PTR
    IF ( PRESENT( scaling ) ) cscaling = C_LOC( scaling )
    CALL cpu_copy_control_in( control, ccontrol )

    cpu_factor%csubtree =                                                      &
         c_create_numeric_subtree( cpu_factor%posdef, this%csubtree,           &
         aval, cscaling, contrib_ptr, ccontrol, cstats )
    IF ( cstats%flag < 0 ) THEN
      CALL c_destroy_numeric_subtree( cpu_factor%posdef, cpu_factor%csubtree )
      DEALLOCATE( cpu_factor, STAT = st )
      inform%flag = cstats%flag
      RETURN
    END IF

!  extract to Fortran data structures

    CALL cpu_copy_stats_out( cstats, inform )

!  success, set result and return

    factor => cpu_factor
    RETURN

!  allocation error handler
10  CONTINUE
    inform%flag = SSIDS_ERROR_ALLOCATION
    inform%stat = st
    DEALLOCATE( cpu_factor, STAT = st )
    RETURN

    END FUNCTION factor

    SUBROUTINE numeric_cleanup( this )
    IMPLICIT none
    CLASS( cpu_numeric_subtree ), INTENT( INOUT ) :: this

    CALL c_destroy_numeric_subtree( this%posdef, this%csubtree )
    END SUBROUTINE numeric_cleanup

    FUNCTION get_contrib( this )
    IMPLICIT none
    TYPE( contrib_type ) :: get_contrib
    CLASS( cpu_numeric_subtree ), INTENT( IN ) :: this

    TYPE( C_PTR ) :: cval, crlist, delay_perm, delay_val

    CALL c_get_contrib( this%posdef, this%csubtree, get_contrib%n, cval,       &
         get_contrib%ldval, crlist, get_contrib%ndelay, delay_perm, delay_val, &
         get_contrib%lddelay )
    CALL c_f_pointer( cval, get_contrib%val, shape = (/ get_contrib%n ** 2 /) )
    CALL c_f_pointer( crlist, get_contrib%rlist, shape = (/ get_contrib%n /) )
    IF ( c_associated( delay_val ) ) THEN
       CALL c_f_pointer( delay_perm, get_contrib%delay_perm,                   &
            shape = (/ get_contrib%ndelay /) )
       CALL c_f_pointer( delay_val, get_contrib%delay_val,                     &
            shape = (/ get_contrib%ndelay*get_contrib%lddelay /) )
    ELSE
      NULLIFY( get_contrib%delay_perm )
      NULLIFY( get_contrib%delay_val )
    END IF
    get_contrib%owner = 0 ! cpu
    get_contrib%posdef = this%posdef
    get_contrib%owner_ptr = this%csubtree
    END FUNCTION get_contrib

    SUBROUTINE solve_fwd( this, nrhs, x, ldx, inform )
    IMPLICIT none
    CLASS( cpu_numeric_subtree ), INTENT( INOUT ) :: this
    INTEGER( KIND = ip_ ), INTENT( IN ) :: nrhs
    REAL( KIND = rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( KIND = ip_ ), INTENT( IN ) :: ldx
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

    INTEGER( KIND = C_IP_ ) :: flag

    flag = c_subtree_solve_fwd( this%posdef, this%csubtree, nrhs, x, ldx )
    IF ( flag /= SSIDS_SUCCESS ) inform%flag = flag
    RETURN

    END SUBROUTINE solve_fwd

    SUBROUTINE solve_diag( this, nrhs, x, ldx, inform )
    IMPLICIT none
    CLASS( cpu_numeric_subtree ), INTENT( INOUT ) :: this
    INTEGER( KIND = ip_ ), INTENT( IN ) :: nrhs
    REAL( KIND = rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( KIND = ip_ ), INTENT( IN ) :: ldx
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

    INTEGER( KIND = C_IP_ ) :: flag

    flag = c_subtree_solve_diag( this%posdef, this%csubtree, nrhs, x, ldx )
    IF ( flag /= SSIDS_SUCCESS ) inform%flag = flag
    RETURN

    END SUBROUTINE solve_diag

    SUBROUTINE solve_diag_bwd( this, nrhs, x, ldx, inform )
    IMPLICIT none
    CLASS( cpu_numeric_subtree ), INTENT( INOUT ) :: this
    INTEGER( KIND = ip_ ), INTENT( IN ) :: nrhs
    REAL( KIND = rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( KIND = ip_ ), INTENT( IN ) :: ldx
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

    INTEGER( KIND = C_IP_ ) :: flag

    flag = c_subtree_solve_diag_bwd( this%posdef, this%csubtree, nrhs, x, ldx )
    IF ( flag /= SSIDS_SUCCESS ) inform%flag = flag
    RETURN

    END SUBROUTINE solve_diag_bwd

    SUBROUTINE solve_bwd( this, nrhs, x, ldx, inform )
    IMPLICIT none
    CLASS( cpu_numeric_subtree ), INTENT( INOUT ) :: this
    INTEGER( KIND = ip_ ), INTENT( IN ) :: nrhs
    REAL( KIND = rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( KIND = ip_ ), INTENT( IN ) :: ldx
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

    INTEGER( KIND = C_IP_ ) :: flag

    flag = c_subtree_solve_bwd( this%posdef, this%csubtree, nrhs, x, ldx )
    IF ( flag /= SSIDS_SUCCESS ) inform%flag = flag
    RETURN

    END SUBROUTINE solve_bwd

    SUBROUTINE enquire_posdef( this, d )
    IMPLICIT none
    CLASS( cpu_numeric_subtree ), INTENT( IN ) :: this
    REAL( KIND = rp_ ), DIMENSION( * ), TARGET, INTENT( OUT ) :: d

    CALL c_subtree_enquire( this%posdef, this%csubtree, C_NULL_PTR, C_LOC( d ) )
    RETURN

    END SUBROUTINE enquire_posdef

    SUBROUTINE enquire_indef( this, piv_order, d )
    IMPLICIT none
    CLASS( cpu_numeric_subtree ), INTENT( IN ) :: this
    INTEGER( KIND = ip_ ), DIMENSION( * ), TARGET, OPTIONAL,                   &
                                                   INTENT( OUT ) :: piv_order
    REAL( KIND = rp_ ), DIMENSION( 2,* ), TARGET, OPTIONAL,                    &
                                                  INTENT( OUT ) :: d

    TYPE( C_PTR ) :: dptr, poptr

!  setup pointers

    poptr = C_NULL_PTR
    IF ( PRESENT( piv_order ) ) poptr = C_LOC( piv_order )
    dptr = C_NULL_PTR
    IF ( present( d ) ) dptr = C_LOC( d )

!  call C++ routine

    CALL c_subtree_enquire( this%posdef, this%csubtree, poptr, dptr )
    RETURN

    END SUBROUTINE enquire_indef

    SUBROUTINE alter( this, d )
    IMPLICIT none
    CLASS( cpu_numeric_subtree ), TARGET, INTENT( INOUT ) :: this
    REAL( KIND = rp_ ), DIMENSION( 2, * ), INTENT( IN ) :: d

    CALL c_subtree_alter( this%posdef, this%csubtree, d )
    RETURN

    END SUBROUTINE alter

    SUBROUTINE cpu_free_contrib( posdef, csubtree )
    IMPLICIT none
    LOGICAL( C_BOOL ), INTENT( IN ) :: posdef
    TYPE( C_PTR ), INTENT( INOUT ) :: csubtree

    CALL c_free_contrib( posdef, csubtree )
    RETURN

    END SUBROUTINE cpu_free_contrib

  END MODULE GALAHAD_SSIDS_cpu_subtree_precision
