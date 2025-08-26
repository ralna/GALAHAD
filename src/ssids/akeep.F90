! THIS VERSION: GALAHAD 5.3 - 2025-08-25 AT 15:20 GMT

#include "ssids_procedures.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  author: Jonathan Hogg
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

MODULE GALAHAD_SSIDS_akeep_precision
  USE GALAHAD_KINDS_precision
  USE, INTRINSIC :: iso_c_binding
  USE GALAHAD_HW, ONLY : HW_numa_region
  USE GALAHAD_SSIDS_types_precision, ONLY: SSIDS_control_type
  USE GALAHAD_SSIDS_inform_precision, ONLY: SSIDS_inform_type
  USE GALAHAD_SSIDS_subtree_precision, ONLY: symbolic_subtree_base
  IMPLICIT none

  PRIVATE
  PUBLIC :: ssids_akeep_type

  TYPE symbolic_subtree_ptr
    integer( ip_ ) :: exec_loc
    class( symbolic_subtree_base ), pointer :: ptr => null(  )
  END TYPE symbolic_subtree_ptr


! Data type for information generated in analyse phase

  TYPE SSIDS_akeep_type
     LOGICAL :: check ! copy of check as input to analyse phase
     INTEGER( ip_ ) :: n ! Dimension of matrix
     INTEGER( ip_ ) :: nnodes = -1 ! Number of nodes in assembly tree
    
!  subtree partition

     INTEGER( ip_ ) :: nparts
     INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: part
     TYPE( symbolic_subtree_ptr ), DIMENSION( : ), ALLOCATABLE :: subtree
     INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: contrib_ptr
     INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: contrib_idx
     INTEGER( ipc_ ), DIMENSION( : ), ALLOCATABLE :: invp ! inverse of pivot 
       !  order that is passed to factorize phase
     INTEGER( long_ ), DIMENSION( :,: ), ALLOCATABLE :: nlist !  map from A to
       !  factors. For nodes i, the entries nlist(1:2,nptr(i):nptr(i+1)-1)
       !  define a relationship:
       !  nodes( node )%lcol(  nlist( 2,j )  ) = val( nlist(1,j))
     INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: nptr !  entries into 
       !  nlist for nodes of the assembly tree. Has length nnodes+1
     INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: rlist 
       !  rlist( rptr( i ):rptr( i+1 )-1 )
       !  contains the row indices for node i of the assembly tree. At each
       !  node, the list is in elimination order. Allocated within mc78_analyse
     INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: rptr !  pointers into
       ! rlist for nodes of assembly tree. Has length nnodes+1.
       ! Allocated within mc78_analyse.
     INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: sparent !  sparent(i) is 
       ! parent of node i in assembly tree. sparent( i )=nnodes+1 if i is a 
       ! root. The parent is always numbered higher than each of its children.
       ! Allocated within mc78_analyse.
     INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: sptr !  (super)node 
       ! pointers. Supernode i consists of sptr(i) through sptr(i+1)-1.
       ! Allocated within mc78_analyse.
    
!  the following components are for cleaned up matrix data.
!  LOWER triangle only. We have to retain these for factorize phase
!  as used if the user wants to do scaling.
!  These components are NOT used if check is set to .false.
!  on call to ssids_analyse.

     INTEGER( long_ ), ALLOCATABLE :: ptr( : ) ! column pointers
     INTEGER( ip_ ), ALLOCATABLE :: row( : ) ! row indices
     INTEGER( long_ ) :: lmap ! length of map
     INTEGER( long_ ), ALLOCATABLE :: map( : ) ! map from old A to cleaned A
    
     ! Scaling from matching-based ordering
     REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: scaling
    
     ! Machine topology
     TYPE( HW_numa_region ), DIMENSION( : ), ALLOCATABLE :: topology
    
     ! Inform at end of analyse phase
     TYPE( ssids_inform_type ) :: inform
  CONTAINS
    PROCEDURE, pass( akeep ) :: free => free_akeep
  END TYPE SSIDS_akeep_type

CONTAINS

  SUBROUTINE free_akeep( akeep, flag )
  CLASS( ssids_akeep_type ), INTENT( INOUT ) :: akeep
  INTEGER( ip_ ), INTENT( OUT ) :: flag

  INTEGER( ip_ ) :: i
  INTEGER( ip_ ) :: st

  flag = 0

  DEALLOCATE( akeep%part, STAT = st )
  IF ( ALLOCATED( akeep%subtree ) ) THEN
    DO i = 1, size( akeep%subtree )
      IF ( associated( akeep%subtree( i )%ptr ) ) THEN
        CALL akeep%subtree( i )%ptr%cleanup(  )
        DEALLOCATE( akeep%subtree( i )%ptr )
        NULLIFY( akeep%subtree( i )%ptr )
      END IF
    END DO
    DEALLOCATE( akeep%subtree, STAT = st )
  END IF
  DEALLOCATE( akeep%contrib_ptr, STAT = st )
  DEALLOCATE( akeep%contrib_idx, STAT = st )
  DEALLOCATE( akeep%invp, STAT = st )
  DEALLOCATE( akeep%nlist, STAT = st )
  DEALLOCATE( akeep%nptr, STAT = st )
  DEALLOCATE( akeep%rlist, STAT = st )
  DEALLOCATE( akeep%rptr, STAT = st )
  DEALLOCATE( akeep%sparent, STAT = st )
  DEALLOCATE( akeep%sptr, STAT = st )
  DEALLOCATE( akeep%ptr, STAT = st )
  DEALLOCATE( akeep%row, STAT = st )
  DEALLOCATE( akeep%map, STAT = st )
  DEALLOCATE( akeep%scaling, STAT = st )
  DEALLOCATE( akeep%topology, STAT = st )
  END SUBROUTINE free_akeep

END MODULE GALAHAD_SSIDS_akeep_precision
