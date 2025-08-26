! THIS VERSION: GALAHAD 5.3 - 2025-08-14 AT 13:40 GMT

#include "ssids_procedures.h"

!  define ssids_fkeep type and associated procedures (CPU version)

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  authors: Jonathan Hogg and Florent Lopez
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

MODULE GALAHAD_SSIDS_fkeep_precision
   USE GALAHAD_KINDS_precision
   USE :: omp_lib
   USE GALAHAD_SSIDS_akeep_precision, ONLY : ssids_akeep_type
   USE GALAHAD_SSIDS_contrib_precision, ONLY : contrib_type
   USE GALAHAD_SSIDS_types_precision
   USE GALAHAD_SSIDS_inform_precision, ONLY : ssids_inform_type
   USE GALAHAD_SSIDS_subtree_precision, ONLY : numeric_subtree_base
   USE GALAHAD_SSIDS_cpu_subtree_precision, ONLY : cpu_numeric_subtree
#ifdef PROFILE
   USE GALAHAD_SSIDS_profile_precision, ONLY : profile_begin, profile_end,     &
                                               profile_add_event
#endif
   IMPLICIT none

   PRIVATE
   PUBLIC :: ssids_fkeep_type

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   TYPE numeric_subtree_ptr
      CLASS( numeric_subtree_base ), pointer :: ptr
   END TYPE numeric_subtree_ptr

   ! Data type for data generated in factorise phase

   TYPE ssids_fkeep_type

 !  stores scaling for each entry ( in original matrix order )

      REAL( rp_ ), DIMENSION( : ), allocatable :: scaling
      LOGICAL :: pos_def ! set to true if user indicates matrix pos. definite

      ! Factored subtrees
      TYPE( numeric_subtree_ptr ), DIMENSION( : ), allocatable :: subtree

      ! Copy of inform on exit from factorize
      TYPE( ssids_inform_type ) :: inform

   CONTAINS

!  do actual factorization

      PROCEDURE, PASS( fkeep ) :: inner_factor => inner_factor_cpu 

!  do actual solve

      PROCEDURE, PASS( fkeep ) :: inner_solve => inner_solve_cpu 
      PROCEDURE, PASS( fkeep ) :: enquire_posdef => enquire_posdef_cpu
      PROCEDURE, PASS( fkeep ) :: enquire_indef => enquire_indef_cpu
      PROCEDURE, PASS( fkeep ) :: alter => alter_cpu ! Alter D values
      PROCEDURE, PASS( fkeep ) :: free => free_fkeep ! Frees memory
   END TYPE ssids_fkeep_type

CONTAINS

SUBROUTINE inner_factor_cpu( fkeep, akeep, val, control, inform )
  IMPLICIT none
  TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
  CLASS( SSIDS_fkeep_type ), TARGET, INTENT( INOUT ) :: fkeep
  REAL( rp_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: val
  TYPE( SSIDS_control_type ), INTENT( IN ) :: control
  TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform

  INTEGER( ip_ ) :: i, numa_region, exec_loc, my_loc
  INTEGER( ip_ ) :: total_threads, max_gpus, to_launch, thread_num
  INTEGER( ip_ ) :: nth ! Number of threads within a region
  INTEGER( ip_ ) :: ngpus ! Number of GPUs in a given NUMA region
  LOGICAL :: abort, all_region
  TYPE( contrib_type ), DIMENSION( : ), allocatable :: child_contrib
  TYPE( ssids_inform_type ), DIMENSION( : ), allocatable :: thread_inform
#ifdef PROFILE
  ! Begin profile trace ( noop if not enabled )
  call profile_begin( akeep%topology )
#endif
!write( 6,* ) ' n ', akeep%n
!write( 6,* ) ' ptr ', akeep%ptr(  : akeep%n + 1  )
!write( 6,* ) ' row ', akeep%row(  : akeep%ptr(  akeep%n + 1  ) - 1  )
!write( 6,* ) ' val ', val(  : akeep%ptr(  akeep%n + 1  ) - 1  )
!stop

  ! Allocate space for subtrees
  allocate( fkeep%subtree( akeep%nparts ), stat=inform%stat )
  IF ( inform%stat /= 0 ) GO TO 200

  ! Determine resources
  total_threads = 0
  max_gpus = 0
  DO i = 1, size( akeep%topology )
     total_threads = total_threads + akeep%topology( i )%nproc
     max_gpus = max( max_gpus, size( akeep%topology( i )%gpus ) )
  END DO

  ! Call subtree factor routines
  allocate( child_contrib( akeep%nparts ), stat=inform%stat )
  IF ( inform%stat /= 0 ) GO TO 200
  ! Split into numa regions; parallelism within a region is responsibility
  ! of subtrees.
  to_launch = size( akeep%topology )*( 1+max_gpus )
  allocate( thread_inform( to_launch ), stat=inform%stat )
  IF ( inform%stat /= 0 ) GO TO 200
  all_region = .false.

  !$omp parallel proc_bind( spread ) num_threads( to_launch ) &
  !$omp    default( none ) &
  !$omp    private( abort, i, exec_loc, numa_region, my_loc, thread_num ) &
  !$omp    private( nth, ngpus ) &
  !$omp    shared( akeep, fkeep, val, control, thread_inform, child_contrib, &
  !$omp           all_region ) &
  !$omp    if ( to_launch.gt.1 )

  thread_num = 0
  !$ thread_num = omp_get_thread_num(  )
  numa_region = mod( thread_num, size( akeep%topology ) ) + 1
  my_loc = thread_num + 1
  IF ( thread_num .lt. size( akeep%topology ) ) THEN
     ngpus = size( akeep%topology( numa_region )%gpus,1 )
     ! CPU, control number of inner threads ( not needed for gpu )
     nth = akeep%topology( numa_region )%nproc
     ! nth = nth - ngpus
  ELSE
     nth = 1
  END IF

  !$ call omp_set_num_threads( int( nth ) )
  ! Split into threads for this NUMA region ( unless we're running a GPU )
  exec_loc = -1 ! avoid compiler warning re uninitialized
  abort = .false.

  !$omp parallel proc_bind( close ) default( shared ) &
  !$omp    num_threads( nth ) &
  !$omp    if ( my_loc.le.size( akeep%topology ) )

  !$omp single
  !$omp taskgroup

  DO i = 1, akeep%nparts
     exec_loc = akeep%subtree( i )%exec_loc
     IF ( numa_region.eq.1 .and. exec_loc.eq.-1 ) all_region = .true.
     IF ( exec_loc /= my_loc ) cycle

     !$omp task untied default( shared ) firstprivate( i, exec_loc ) &
     !$omp    if ( my_loc.le.size( akeep%topology ) )
     IF ( abort ) GO TO 10
     IF ( allocated( fkeep%scaling ) ) THEN
        fkeep%subtree( i )%ptr => akeep%subtree( i )%ptr%factor(  &
             fkeep%pos_def, val, &
             child_contrib( akeep%contrib_ptr( i ):akeep%contrib_ptr( i+1 )-1 ), &
             control, thread_inform( my_loc ), scaling=fkeep%scaling &
              )
     ELSE
        fkeep%subtree( i )%ptr => akeep%subtree( i )%ptr%factor(  &
             fkeep%pos_def, val, &
             child_contrib( akeep%contrib_ptr( i ):akeep%contrib_ptr( i+1 )-1 ), &
             control, thread_inform( my_loc ) &
              )
     END IF
     IF ( thread_inform( my_loc )%flag .lt. 0 ) THEN
        abort = .true.
        GO TO 10
        ! !$omp    cancel taskgroup
     END IF
     IF ( akeep%contrib_idx( i ) .le. akeep%nparts ) THEN
        ! There is a parent subtree to contribute to
        child_contrib( akeep%contrib_idx( i ) ) = &
             fkeep%subtree( i )%ptr%get_contrib(  )
        !$omp    flush
        child_contrib( akeep%contrib_idx( i ) )%ready = .true.
     END IF
10   continue ! jump target for abort
     !$omp end task

  end DO

!write( 99,* ) ' all_region ', all_region
!stop
  !$omp end taskgroup

  !$omp end single

  !$omp end parallel
  !$omp end parallel

  DO i = 1, size( thread_inform )
     call inform%reduce( thread_inform( i ) )
  END DO
  IF ( inform%flag.lt.0 ) GO TO 100 ! cleanup and exit

  IF ( all_region ) THEN

#ifdef PROFILE
     ! At least some all region subtrees exist
     call profile_add_event( "EV_ALL_REGIONS", &
                            "Starting processing root subtree", 0 )
#endif

     !$omp parallel num_threads( total_threads ) default( shared )
     !$omp single
     DO i = 1, akeep%nparts
        exec_loc = akeep%subtree( i )%exec_loc
        IF ( exec_loc /= - 1 ) cycle
        IF ( allocated( fkeep%scaling ) ) THEN
           fkeep%subtree( i )%ptr => akeep%subtree( i )%ptr%factor(  &
                fkeep%pos_def, val, &
                child_contrib( akeep%contrib_ptr( i ):akeep%contrib_ptr( i+1 )-1 ), &
                control, inform, scaling=fkeep%scaling &
                 )
        ELSE
           fkeep%subtree( i )%ptr => akeep%subtree( i )%ptr%factor(  &
                fkeep%pos_def, val, &
                child_contrib( akeep%contrib_ptr( i ):akeep%contrib_ptr( i+1 )-1 ), &
                control, inform &
                 )
        END IF
        IF ( akeep%contrib_idx( i ) .gt. akeep%nparts ) cycle ! part is a root
        child_contrib( akeep%contrib_idx( i ) ) = &
             fkeep%subtree( i )%ptr%get_contrib(  )
        !$omp    flush
        child_contrib( akeep%contrib_idx( i ) )%ready = .true.
     END DO
     !$omp end single
     !$omp end parallel
  END if

100 continue ! cleanup and exit

#ifdef PROFILE
  ! End profile trace ( noop if not enabled )
  call profile_end(  )
#endif

  return

200 continue
  inform%flag = SSIDS_ERROR_ALLOCATION
  GO TO 100 ! cleanup and exit
end SUBROUTINE inner_factor_cpu

SUBROUTINE inner_solve_cpu( local_job, nrhs, x, ldx, akeep, fkeep, inform )
   TYPE( ssids_akeep_type ), INTENT( IN ) :: akeep
   CLASS( ssids_fkeep_type ), INTENT( INOUT ) :: fkeep
   INTEGER( ip_ ), INTENT( INOUT ) :: local_job
   INTEGER( ip_ ), INTENT( IN ) :: nrhs
   INTEGER( ip_ ), INTENT( IN ) :: ldx
   REAL( rp_ ), DIMENSION( ldx,nrhs ), TARGET, INTENT( INOUT ) :: x
   TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

   INTEGER( ip_ ) :: i, r, part
   INTEGER( ip_ ) :: n
   REAL( rp_ ), DIMENSION( :,: ), allocatable :: x2

   n = akeep%n

   allocate( x2( n, nrhs ), stat=inform%stat )
   IF ( inform%stat /= 0 ) GO TO 100

   ! Permute/scale
   IF ( ALLOCATED( fkeep%scaling ) .AND.                                       &
         ( local_job == SSIDS_SOLVE_JOB_ALL .OR.                               &
           local_job == SSIDS_SOLVE_JOB_FWD ) ) THEN
      ! Copy and scale
      DO r = 1, nrhs
         DO i = 1, n
            x2( i,r ) = x( akeep%invp( i ),r ) * fkeep%scaling( i )
         END DO
      END DO

!  just copy

   ELSE
      DO r = 1, nrhs
         x2( 1:n, r ) = x( akeep%invp( 1:n ), r )
      END DO
   END if

   ! Perform relevant solves
   IF ( local_job.eq.SSIDS_SOLVE_JOB_FWD .OR.                                  &
        local_job.eq.SSIDS_SOLVE_JOB_ALL ) THEN
     DO part = 1, akeep%nparts
       CALL fkeep%subtree( part )%ptr%solve_fwd( nrhs, x2, n, inform )
       IF ( inform%stat /= 0 ) GO TO 100
     END DO
   END IF

   IF ( local_job.eq.SSIDS_SOLVE_JOB_DIAG ) THEN
     DO part = 1, akeep%nparts
       CALL fkeep%subtree( part )%ptr%solve_diag( nrhs, x2, n, inform )
       IF ( inform%stat /= 0 ) GO TO 100
     END DO
   END IF

   IF ( local_job.eq.SSIDS_SOLVE_JOB_BWD ) THEN
     DO part = akeep%nparts, 1, -1
       CALL fkeep%subtree( part )%ptr%solve_bwd( nrhs, x2, n, inform )
       IF ( inform%stat /= 0 ) GO TO 100
     END DO
   END IF

   IF ( local_job.eq.SSIDS_SOLVE_JOB_DIAG_BWD .OR.                             &
        local_job.eq.SSIDS_SOLVE_JOB_ALL ) THEN
     DO part = akeep%nparts, 1, -1
       CALL fkeep%subtree( part )%ptr%solve_diag_bwd( nrhs, x2, n, inform )
       IF ( inform%stat /= 0 ) GO TO 100
     END DO
   END IF

!  unscale/unpermute

   IF ( ALLOCATED( fkeep%scaling ) .AND.                                       &
         (  local_job == SSIDS_SOLVE_JOB_ALL .OR.                              &
            local_job == SSIDS_SOLVE_JOB_BWD .OR.                              &
            local_job == SSIDS_SOLVE_JOB_DIAG_BWD ) ) THEN

!  copy and scale

      DO r = 1, nrhs
        DO i = 1, n
          x( akeep%invp( i ),r ) = x2( i,r ) * fkeep%scaling( i )
        END DO
      END DO

!  just copy

   ELSE
     DO r = 1, nrhs
       x( akeep%invp( 1:n ), r ) = x2( 1:n, r )
     END DO
   END IF

   RETURN

   100 continue
   inform%flag = SSIDS_ERROR_ALLOCATION
   RETURN

   END SUBROUTINE inner_solve_cpu

!****************************************************************************

   SUBROUTINE enquire_posdef_cpu( akeep, fkeep, d )
   TYPE( ssids_akeep_type ), INTENT( IN ) :: akeep
   CLASS( ssids_fkeep_type ), TARGET, INTENT( IN ) :: fkeep
   REAL( rp_ ), DIMENSION( * ), INTENT( OUT ) :: d

   INTEGER( ip_ ) :: n
   INTEGER( ip_ ) :: part, sa, en

   n = akeep%n
   ! ensure d is not returned undefined
   d( 1:n ) = 0.0 ! ensure do not returned with this undefined

   DO part = 1, akeep%nparts
      sa = akeep%part( part )
      en = akeep%part( part+1 )-1
      associate( subtree => fkeep%subtree( part )%ptr )
         select TYPE( subtree )
         TYPE IS ( cpu_numeric_subtree )
            call subtree%enquire_posdef( d( sa:en ) )
         END select
      END associate
   END DO

   END SUBROUTINE enquire_posdef_cpu

!****************************************************************************

   SUBROUTINE enquire_indef_cpu( akeep, fkeep, inform, piv_order, d )
   TYPE( ssids_akeep_type ), INTENT( IN ) :: akeep
   CLASS( ssids_fkeep_type ), TARGET, INTENT( IN ) :: fkeep
   TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform
   INTEGER( ip_ ), DIMENSION( akeep%n ), optional, INTENT( OUT ) :: piv_order
      ! If i is used to index a variable, its position in the pivot sequence
      ! will be placed in piv_order( i ), with its sign negative if it is
      ! part of a 2 x 2 pivot; otherwise, piv_order( i ) will be set to zero.
   REAL( rp_ ), DIMENSION( 2,akeep%n ), optional, INTENT( OUT ) :: d ! The diagonal
      ! entries of D^{-1} will be placed in d( 1,:i ) and the off-diagonal
      ! entries will be placed in d( 2,: ). The entries are held in pivot order.

   INTEGER( ip_ ) :: part, sa
   INTEGER( ip_ ) :: i, n
   INTEGER( ip_ ), DIMENSION( : ), allocatable :: po

   n = akeep%n

!  ensure d is not returned undefined

   IF ( PRESENT( d ) ) THEN
      d( 1:2,1:n ) = 0.0
   END if

!  we need to apply the invp externally to piv_order

   IF ( PRESENT( piv_order ) ) THEN
      allocate( po( akeep%n ), stat=inform%stat )
      IF ( inform%stat /= 0 ) THEN
         inform%flag = SSIDS_ERROR_ALLOCATION
         return
      END IF
   END IF

!  FIXME: should probably return nelim from each part, due to delays passing
!  between them

   DO part = 1, akeep%nparts
     sa = akeep%part( part )
     ASSOCIATE( subtree => fkeep%subtree( 1 )%ptr )
       SELECT TYPE( subtree )
       TYPE IS ( cpu_numeric_subtree )
         IF ( PRESENT( d ) ) THEN
           IF ( PRESENT( piv_order ) ) THEN
             CALL subtree%enquire_indef( piv_order=po( sa:n ), d=d( 1:2,sa:n ) )
           ELSE
              CALL subtree%enquire_indef( d=d( 1:2,sa: ) )
           END IF
         ELSE
           IF ( PRESENT( piv_order ) ) THEN
              CALL subtree%enquire_indef( piv_order=po( sa:akeep%n ) )
           ELSE
              ! No-op
              ! FIXME: should we report an error here? ( or done higher up? )
           END IF
         END IF
       END SELECT
     END ASSOCIATE
   END DO

!  apply invp to piv_order

   IF ( PRESENT( piv_order ) ) THEN
      DO i = 1, akeep%n
         piv_order(  akeep%invp( i )  ) = po( i )
!        piv_order(  i  ) = po( i )
      END DO
   END IF

   END SUBROUTINE enquire_indef_cpu

! Alter D values

   SUBROUTINE alter_cpu( d, akeep, fkeep )
   REAL( rp_ ), DIMENSION( 2, * ), INTENT( IN ) :: d  ! The required diagonal entries
     ! of D^{-1} must be placed in d( 1,i ) ( i = 1,...n )
     ! and the off-diagonal entries must be placed in d( 2,i ) ( i = 1,...n-1 ).
   TYPE( ssids_akeep_type ), INTENT( IN ) :: akeep
   CLASS( ssids_fkeep_type ), TARGET, INTENT( INOUT ) :: fkeep

   INTEGER( ip_ ) :: part

   DO part = 1, akeep%nparts
      ASSOCIATE( subtree => fkeep%subtree( 1 )%ptr )
         SELECT TYPE( subtree )
         TYPE IS ( cpu_numeric_subtree )
            call subtree%alter( d( 1:2,akeep%part( part ):akeep%part( part+1 )-1 ) )
         END select
      END associate
   END DO
   END SUBROUTINE alter_cpu

!****************************************************************************

   SUBROUTINE free_fkeep( fkeep, flag )
   CLASS( ssids_fkeep_type ), INTENT( INOUT ) :: fkeep
   INTEGER( ip_ ), INTENT( OUT ) :: flag ! not used for cpu version, set to 0

   INTEGER( ip_ ) :: i, st

   flag = 0 ! Not used for basic SSIDS, just set to zero

   deallocate( fkeep%scaling, stat=st )
   IF ( allocated( fkeep%subtree ) ) THEN
      DO i = 1, size( fkeep%subtree )
         IF ( associated( fkeep%subtree( i )%ptr ) ) THEN
            call fkeep%subtree( i )%ptr%cleanup(  )
            deallocate( fkeep%subtree( i )%ptr )
            nullify( fkeep%subtree( i )%ptr )
         END IF
      END DO
      deallocate( fkeep%subtree )
   END IF
   END SUBROUTINE free_fkeep

END MODULE GALAHAD_SSIDS_fkeep_precision
