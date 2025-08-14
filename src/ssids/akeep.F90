! THIS VERSION: GALAHAD 5.3 - 2025-08-13 AT 15:20 GMT

#include "spral_procedures.h"

!  copyright 2016 The Science and Technology Facilities Council (STFC)
!  licence   BSD licence, see LICENCE file for details
!  author    Jonathan Hogg

MODULE GALAHAD_SSIDS_AKEEP_precision
   USE GALAHAD_SSIDS_TYPES_precision, ONLY : SSIDS_ERROR_CUDA_UNKNOWN,        &
                                             SSIDS_control_type
   USE SPRAL_KINDS_precision
   USE SPRAL_HW_TOPOLOGY, only : numa_region
   USE GALAHAD_SSIDS_INFORM_precision, ONLY : SSIDS_inform_type
   USE GALAHAD_SSIDS_SUBTREE_precision, ONLY : symbolic_subtree_base
   USE, INTRINSIC :: iso_c_binding
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
     
      ! Subtree partition
      INTEGER( ip_ ) :: nparts
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: part
      TYPE( symbolic_subtree_ptr ), DIMENSION( : ), ALLOCATABLE :: subtree
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: contrib_ptr
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: contrib_idx
      INTEGER( ipc_ ), DIMENSION( : ), ALLOCATABLE :: invp ! inverse of pivot 
         ! order that is passed to factorize phase
      INTEGER( long_ ), DIMENSION( :,: ), ALLOCATABLE :: nlist ! map from A to
         ! factors. For nodes i, the entries nlist(1:2,nptr(i):nptr(i+1)-1)
         ! define a relationship:
         ! nodes( node )%lcol(  nlist( 2,j )  ) = val( nlist(1,j))
      INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: nptr ! Entries into 
         ! nlist for nodes of the assembly tree. Has length nnodes+1
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: rlist 
         ! rlist( rptr( i ):rptr( i+1 )-1 )
         ! contains the row indices for node i of the assembly tree.
         ! At each node, the list
         ! is in elimination order. Allocated within mc78_analyse.
      INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: rptr ! Pointers into
         ! rlist for nodes of assembly tree. Has length nnodes+1.
         ! Allocated within mc78_analyse.
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: sparent ! sparent(i) is 
         ! parent of node i in assembly tree. sparent( i )=nnodes+1 if i is a 
         ! root. The parent is always numbered higher than each of its children.
         ! Allocated within mc78_analyse.
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: sptr ! (super)node 
         ! pointers. Supernode i consists of sptr(i) through sptr(i+1)-1.
         ! Allocated within mc78_analyse.
     
      ! Following components are for cleaned up matrix data.
      ! LOWER triangle only. We have to retain these for factorize phase
      ! as used if the user wants to do scaling.
      ! These components are NOT used if check is set to .false.
      ! on call to ssids_analyse.
      INTEGER( long_ ), ALLOCATABLE :: ptr( : ) ! column pointers
      INTEGER( ip_ ), ALLOCATABLE :: row( : ) ! row indices
      INTEGER( long_ ) :: lmap ! length of map
      INTEGER( long_ ), ALLOCATABLE :: map( : ) ! map from old A to cleaned A
     
      ! Scaling from matching-based ordering
      REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: scaling
     
      ! Machine topology
      TYPE( numa_region ), DIMENSION( : ), ALLOCATABLE :: topology
     
      ! Inform at end of analyse phase
      TYPE( ssids_inform_type ) :: inform
   CONTAINS
     PROCEDURE, pass( akeep ) :: free => free_akeep
   END TYPE SSIDS_akeep_type

CONTAINS

SUBROUTINE free_akeep( akeep, flag )
   CLASS( ssids_akeep ), INTENT( INOUT ) :: akeep
   INTEGER( ip_ ), INTENT( OUT ) :: flag

   INTEGER( ip_ ) :: i
   INTEGER( ip_ ) :: st

   flag = 0

   DEALLOCATE( akeep%part, stat=st )
   IF ( ALLOCATED( akeep%subtree ) ) THEN
     DO i = 1, size( akeep%subtree )
       IF ( associated( akeep%subtree( i )%ptr ) ) THEN
         CALL akeep%subtree( i )%ptr%cleanup(  )
         DEALLOCATE( akeep%subtree( i )%ptr )
         NULLIFY( akeep%subtree( i )%ptr )
       END IF
     END DO
     DEALLOCATE( akeep%subtree, stat=st )
   END IF
   DEALLOCATE( akeep%contrib_ptr, stat=st )
   DEALLOCATE( akeep%contrib_idx, stat=st )
   DEALLOCATE( akeep%invp, stat=st )
   DEALLOCATE( akeep%nlist, stat=st )
   DEALLOCATE( akeep%nptr, stat=st )
   DEALLOCATE( akeep%rlist, stat=st )
   DEALLOCATE( akeep%rptr, stat=st )
   DEALLOCATE( akeep%sparent, stat=st )
   DEALLOCATE( akeep%sptr, stat=st )
   DEALLOCATE( akeep%ptr, stat=st )
   DEALLOCATE( akeep%row, stat=st )
   DEALLOCATE( akeep%map, stat=st )
   DEALLOCATE( akeep%scaling, stat=st )
   DEALLOCATE( akeep%topology, stat=st )
END SUBROUTINE free_akeep

END MODULE GALAHAD_SSIDS_AKEEP_precision
