! THIS VERSION: GALAHAD 5.3 - 2025-08-14 AT 12:10 GMT

#include "ssids_procedures.h"

! This is dummy file compiled when there is no CUDA support

MODULE GALAHAD_SSIDS_gpu_subtree_precision
  USE GALAHAD_KINDS_precision
  USE GALAHAD_SSIDS_contrib_precision, ONLY : contrib_type
  USE GALAHAD_SSIDS_types_precision
  USE GALAHAD_SSIDS_subtree_precision, ONLY : symbolic_subtree_base,           &
                                              numeric_subtree_base
  IMPLICIT none

  PRIVATE
  PUBLIC :: gpu_symbolic_subtree, construct_gpu_symbolic_subtree
  PUBLIC :: gpu_numeric_subtree, gpu_free_contrib

  TYPE, EXTENDS( symbolic_subtree_base ) :: gpu_symbolic_subtree
     INTEGER( long_ ) :: dummy
   contains
     PROCEDURE :: factor
     PROCEDURE :: cleanup => symbolic_cleanup
  end TYPE gpu_symbolic_subtree

  TYPE, EXTENDS( numeric_subtree_base ) :: gpu_numeric_subtree
    REAL( rp_ ) :: dummy ! just so we can perform dummy ops to prevent warnings
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
  END TYPE gpu_numeric_subtree

CONTAINS

    FUNCTION construct_gpu_symbolic_subtree( device, n, sa, en, sptr, sparent, &
       rptr, rlist, nptr, nlist, control ) result( this )
    IMPLICIT none
    CLASS( gpu_symbolic_subtree ), pointer :: this
    INTEGER( ip_ ), INTENT( IN ) :: device
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( ip_ ), INTENT( IN ) :: sa
    INTEGER( ip_ ), INTENT( IN ) :: en
    INTEGER( ip_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: sptr
    INTEGER( ip_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: sparent
    INTEGER( long_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: rptr
    INTEGER( ip_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: rlist
    INTEGER( long_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: nptr
    INTEGER( long_ ), DIMENSION( 2,* ), TARGET, INTENT( IN ) :: nlist
    CLASS( SSIDS_control_type ), INTENT( IN ) :: control

    NULLIFY( this )

    PRINT *, "construct_gpu_symbolic_subtree() called without GPU support."
    PRINT *, "This should never happen."
    STOP - 1

!  dummy operations to prevent warnings

    this%dummy = device + n + sa + en + sptr( 1 ) + sparent( 1 ) + rptr( 1 )   &
       + rlist( 1 ) + nptr( 1 ) + nlist( 1, 1 ) + control%print_level
    END FUNCTION construct_gpu_symbolic_subtree

    SUBROUTINE symbolic_cleanup( this )
    IMPLICIT none
    CLASS( gpu_symbolic_subtree ), INTENT( INOUT ) :: this

!  dummy operation to prevent warnings

    this%dummy = 0
    END SUBROUTINE symbolic_cleanup

    FUNCTION factor( this, posdef, aval, child_contrib, control, inform,       &
                     scaling )
    IMPLICIT none
    CLASS( numeric_subtree_base ), POINTER :: factor
    CLASS( gpu_symbolic_subtree ), TARGET, INTENT( INOUT ) :: this
    logical, INTENT( IN ) :: posdef
    REAL( rp_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: aval
    TYPE( contrib_type ), DIMENSION( : ), TARGET,                              &
                                          INTENT( INOUT ) :: child_contrib
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform
    REAL( rp_ ), DIMENSION( * ), TARGET, OPTIONAL, INTENT( IN ) :: scaling

    TYPE( gpu_numeric_subtree ), pointer :: subtree

    NULLIFY( subtree )

!  dummy operations to prevent warnings

    factor => subtree
    IF ( posdef ) subtree%dummy = REAL( this%dummy,rp_ )+aval( 1 ) +           &
         child_contrib( 1 )%val( 1 ) + control%gpu_perf_coeff
    IF ( PRESENT( scaling ) ) subtree%dummy = subtree%dummy * scaling( 1 )
    inform%flag = SSIDS_ERROR_UNKNOWN
    END FUNCTION factor

    SUBROUTINE numeric_cleanup( this )
    IMPLICIT none
    CLASS( gpu_numeric_subtree ), INTENT( INOUT ) :: this

!  dummy operations to prevent warnings

    this%dummy = 0
    END SUBROUTINE numeric_cleanup

    FUNCTION get_contrib( this )
    IMPLICIT none
    TYPE( contrib_type ) :: get_contrib
    CLASS( gpu_numeric_subtree ), INTENT( IN ) :: this

    ! Dummy operation to prevent warnings
    get_contrib%n = int( this%dummy )
    END FUNCTION get_contrib

    SUBROUTINE gpu_free_contrib( contrib )
    IMPLICIT none
    TYPE( contrib_type ), INTENT( INOUT ) :: contrib

!  dummy operation to prevent warnings

    contrib%n = 0
    END SUBROUTINE gpu_free_contrib

    SUBROUTINE solve_fwd( this, nrhs, x, ldx, inform )
    IMPLICIT none
    CLASS( gpu_numeric_subtree ), INTENT( INOUT ) :: this
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

!  dummy operations to prevent warnings

    x( nrhs + 1 * ldx ) = this%dummy
    inform%flag = SSIDS_ERROR_UNKNOWN
    END SUBROUTINE solve_fwd

    SUBROUTINE solve_diag( this, nrhs, x, ldx, inform )
    IMPLICIT none
    CLASS( gpu_numeric_subtree ), INTENT( INOUT ) :: this
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

!  dummy operations to prevent warnings

    x( nrhs + 1 * ldx ) = this%dummy
    inform%flag = SSIDS_ERROR_UNKNOWN
    END SUBROUTINE solve_diag

    SUBROUTINE solve_diag_bwd( this, nrhs, x, ldx, inform )
    IMPLICIT none
    CLASS( gpu_numeric_subtree ), INTENT( INOUT ) :: this
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

!  dummy operations to prevent warnings

    x( nrhs + 1 * ldx ) = this%dummy
    inform%flag = SSIDS_ERROR_UNKNOWN
    END SUBROUTINE solve_diag_bwd

    SUBROUTINE solve_bwd( this, nrhs, x, ldx, inform )
    IMPLICIT none
    CLASS( gpu_numeric_subtree ), INTENT( INOUT ) :: this
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

!  dummy operations to prevent warnings

    x( nrhs + 1 * ldx ) = this%dummy
    inform%flag = SSIDS_ERROR_UNKNOWN
    END SUBROUTINE solve_bwd

    SUBROUTINE enquire_posdef( this, d )
    IMPLICIT none
    CLASS( gpu_numeric_subtree ), TARGET, INTENT( IN ) :: this
    REAL( rp_ ), DIMENSION( * ), TARGET, INTENT( OUT ) :: d

!  dummy operation to prevent warnings

    d( 1 ) = this%dummy
    END SUBROUTINE enquire_posdef

    SUBROUTINE enquire_indef( this, piv_order, d )
    IMPLICIT none
    CLASS( gpu_numeric_subtree ), TARGET, INTENT( IN ) :: this
    INTEGER( ip_ ), DIMENSION( * ), TARGET, OPTIONAL, INTENT( OUT ) :: piv_order
    REAL( rp_ ), DIMENSION( 2,* ), TARGET, OPTIONAL, INTENT( OUT ) :: d

!  dummy operation to prevent warnings
    IF ( PRESENT( d ) ) d( 1, 1 ) = this%dummy
    IF ( PRESENT( piv_order ) ) piv_order( 1 ) = 1
    END SUBROUTINE enquire_indef

    SUBROUTINE alter( this, d )
    IMPLICIT none
    CLASS( gpu_numeric_subtree ), TARGET, INTENT( INOUT ) :: this
    REAL( rp_ ), DIMENSION( 2, * ), INTENT( IN ) :: d

!  dummy operation to prevent warnings

    this%dummy = d( 1,1 )
    END SUBROUTINE alter

END MODULE GALAHAD_SSIDS_gpu_subtree_precision
