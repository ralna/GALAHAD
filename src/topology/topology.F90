! THIS VERSION: GALAHAD 5.5 - 2026-07-27 AT 11:25 GMT.

#include "galahad_modules.h"

!-*-*-*-*-  G A L A H A D _ T O P O L O G Y   M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Alexis Montoison

!  Pure-Fortran hardware-topology detection.
!  On Linux the NUMA layout is read straight from sysfs
! (/sys/devices/system/node/nodeN/cpulist): one region per NUMA node,
! with nproc = number of physical cores in that node -- counted from
!  each CPU's topology/thread_siblings_list, so hyperthreads collapse onto their
!  core, matching hwloc's former core count. If the per-CPU topology cannot be
!  read it falls back to the logical-CPU count, and if sysfs is unavailable
!  (non-Linux, or no NUMA info) to a single region spanning all OpenMP threads.

      MODULE GALAHAD_TOPOLOGY

!  provides routines for detecting and/or specifying hardware topology for
!  topology-aware routines

        USE GALAHAD_KINDS, ONLY: ip_
!$      USE omp_lib, ONLY: omp_get_max_threads
        IMPLICIT NONE

        PRIVATE
        PUBLIC :: TOPOLOGY_numa_region, TOPOLOGY_guess_topology

!  derived type describing regions

        TYPE :: TOPOLOGY_numa_region

!  number of processors in region

          INTEGER ( KIND = ip_ ) :: nproc
        END TYPE TOPOLOGY_numa_region

      CONTAINS

!-*-*-*-*-   H W _ G U E S S _ T O P O L O G Y   S U B R O U T I N E   -*-*-*-*

        SUBROUTINE TOPOLOGY_guess_topology( regions, st )

!  return best guess for machine topology: one region per Linux NUMA node
!  (from sysfs), else a single region spanning all OpenMP threads

!  upon return allocated to have size equal to the number of NUMA regions.
!  The members describe each region

        TYPE ( TOPOLOGY_numa_region ), DIMENSION( : ), ALLOCATABLE,                  &
                                                 INTENT( OUT ) :: regions

!  status return from allocate. If non-zero upon return, an allocation failed

        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: st

        INTEGER ( KIND = ip_ ) :: i, nnode, nproc
        LOGICAL :: ex

        st = 0

!  count NUMA nodes exposed by the kernel (probe nodeN until one is missing)

        nnode = 0
        DO i = 0, 8191
          INQUIRE( FILE = node_cpulist_path( i ), EXIST = ex )
          IF ( .NOT. ex ) EXIT
          nnode = nnode + 1
        END DO

!  build one region per NUMA node from its cpulist

        IF ( nnode > 0 ) THEN
          ALLOCATE ( regions( nnode ), STAT = st )
          IF ( st /= 0 ) RETURN
          DO i = 1, nnode
            nproc = count_cores( node_cpulist_path( i - 1 ) )    ! physical cores
            IF ( nproc <= 0 )                                                   &
              nproc = count_cpulist( node_cpulist_path( i - 1 ) )  ! -> logical
            IF ( nproc <= 0 ) THEN            ! unreadable/empty -> give up on sysfs
              DEALLOCATE ( regions, STAT = st )
              nnode = 0
              EXIT
            END IF
            regions( i )%nproc = nproc
          END DO
        END IF

!  fallback: a single region spanning all available OpenMP threads

        IF ( nnode <= 0 ) THEN
          nproc = 1
!$        nproc = INT( omp_get_max_threads( ), ip_ )
          ALLOCATE ( regions( 1 ), STAT = st )
          IF ( st /= 0 ) RETURN
          regions( 1 )%nproc = nproc
        END IF

        RETURN
        END SUBROUTINE TOPOLOGY_guess_topology

!-*-*-*-*-*-*-*-*-*-*-*-   h e l p e r s   -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  sysfs path of NUMA node `node`'s cpulist

        FUNCTION node_cpulist_path( node ) RESULT( path )
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: node
        CHARACTER ( LEN = 64 ) :: path
        CHARACTER ( LEN = 16 ) :: num
        WRITE( num, '( I0 )' ) node
        path = '/sys/devices/system/node/node' // TRIM( num ) // '/cpulist'
        END FUNCTION node_cpulist_path

!  count the logical CPUs listed in a Linux cpulist file, whose contents look
!  like "0-15" or "0-7,16-23" or "3" (comma-separated ranges / singletons);
!  returns 0 if the file cannot be read

        FUNCTION count_cpulist( path ) RESULT( ncpu )
        CHARACTER ( LEN = * ), INTENT( IN ) :: path
        INTEGER ( KIND = ip_ ) :: ncpu
        CHARACTER ( LEN = 8192 ) :: line
        INTEGER :: unit, ios, i, n, lo, hi, dash, p
        CHARACTER ( LEN = 64 ) :: tok

        ncpu = 0
        OPEN( NEWUNIT = unit, FILE = path, STATUS = 'OLD', ACTION = 'READ',    &
              IOSTAT = ios )
        IF ( ios /= 0 ) RETURN
        line = ''
        READ( unit, '( A )', IOSTAT = ios ) line
        CLOSE( unit )
        IF ( ios /= 0 ) RETURN

!  split on commas, each token is either "lo-hi" or "v"

        n = LEN_TRIM( line )
        i = 1
        DO WHILE ( i <= n )
          p = INDEX( line( i : n ), ',' )
          IF ( p == 0 ) THEN
            tok = line( i : n ); i = n + 1
          ELSE
            tok = line( i : i + p - 2 ); i = i + p
          END IF
          tok = ADJUSTL( tok )
          IF ( LEN_TRIM( tok ) == 0 ) CYCLE
          dash = INDEX( tok, '-' )
          IF ( dash == 0 ) THEN
            ncpu = ncpu + 1
          ELSE
            READ( tok( 1 : dash - 1 ), * , IOSTAT = ios ) lo
            IF ( ios /= 0 ) CYCLE
            READ( tok( dash + 1 : ), * , IOSTAT = ios ) hi
            IF ( ios /= 0 ) CYCLE
            ncpu = ncpu + ( hi - lo + 1 )
          END IF
        END DO
        END FUNCTION count_cpulist

!  count the physical cores among the logical CPUs listed in NUMA-node cpulist
!  file `path`. Each core is counted once, via its lowest-numbered thread (the
!  first entry of that CPU's topology/thread_siblings_list). Returns 0 if the
!  cpulist or any sibling file cannot be read, so the caller can fall back to
!  the logical-CPU count

        FUNCTION count_cores( path ) RESULT( ncore )
        CHARACTER ( LEN = * ), INTENT( IN ) :: path
        INTEGER ( KIND = ip_ ) :: ncore
        CHARACTER ( LEN = 8192 ) :: line
        INTEGER ( KIND = ip_ ), ALLOCATABLE :: ids( : )
        INTEGER ( KIND = ip_ ) :: nid, k, c, smin
        INTEGER :: unit, ios
        LOGICAL :: ok

        ncore = 0
        OPEN( NEWUNIT = unit, FILE = path, STATUS = 'OLD', ACTION = 'READ',     &
              IOSTAT = ios )
        IF ( ios /= 0 ) RETURN
        line = ''
        READ( unit, '( A )', IOSTAT = ios ) line
        CLOSE( unit )
        IF ( ios /= 0 ) RETURN

        CALL expand_cpulist( line, ids, nid )
        IF ( nid <= 0 ) RETURN

        DO k = 1, nid
          c = ids( k )
          smin = core_first_sibling( c, ok )
          IF ( .NOT. ok ) THEN         ! no per-CPU topology -> caller falls back
            ncore = 0
            RETURN
          END IF
          IF ( smin == c ) ncore = ncore + 1
        END DO
        END FUNCTION count_cores

!  expand a Linux cpulist string ("0-7,16-23" or "3") into the array `ids` of
!  individual CPU numbers (allocated here); nid is the number of entries

        SUBROUTINE expand_cpulist( line, ids, nid )
        CHARACTER ( LEN = * ), INTENT( IN ) :: line
        INTEGER ( KIND = ip_ ), ALLOCATABLE, INTENT( OUT ) :: ids( : )
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nid
        INTEGER ( KIND = ip_ ) :: lo, hi, v
        INTEGER :: n, i, p, dash, ios, pass, cnt
        CHARACTER ( LEN = 64 ) :: tok

!  pass 1 counts the ids, pass 2 fills the allocated array

        DO pass = 1, 2
          cnt = 0
          n = LEN_TRIM( line )
          i = 1
          DO WHILE ( i <= n )
            p = INDEX( line( i : n ), ',' )
            IF ( p == 0 ) THEN
              tok = line( i : n ); i = n + 1
            ELSE
              tok = line( i : i + p - 2 ); i = i + p
            END IF
            tok = ADJUSTL( tok )
            IF ( LEN_TRIM( tok ) == 0 ) CYCLE
            dash = INDEX( tok, '-' )
            IF ( dash == 0 ) THEN
              READ( tok, *, IOSTAT = ios ) v
              IF ( ios /= 0 ) CYCLE
              cnt = cnt + 1
              IF ( pass == 2 ) ids( cnt ) = v
            ELSE
              READ( tok( 1 : dash - 1 ), *, IOSTAT = ios ) lo
              IF ( ios /= 0 ) CYCLE
              READ( tok( dash + 1 : ), *, IOSTAT = ios ) hi
              IF ( ios /= 0 ) CYCLE
              DO v = lo, hi
                cnt = cnt + 1
                IF ( pass == 2 ) ids( cnt ) = v
              END DO
            END IF
          END DO
          IF ( pass == 1 ) THEN
            nid = cnt
            ALLOCATE ( ids( MAX( nid, 1 ) ) )
            IF ( nid <= 0 ) RETURN
          END IF
        END DO
        END SUBROUTINE expand_cpulist

!  return the lowest-numbered logical CPU sharing a physical core with `cpu`,
!  read from /sys/devices/system/cpu/cpuN/topology/thread_siblings_list (the
!  kernel lists siblings in ascending order, so the first integer is the
!  minimum). ok is .FALSE. if the file cannot be read

        FUNCTION core_first_sibling( cpu, ok ) RESULT( smin )
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: cpu
        LOGICAL, INTENT( OUT ) :: ok
        INTEGER ( KIND = ip_ ) :: smin
        CHARACTER ( LEN = 128 ) :: path
        CHARACTER ( LEN = 16 ) :: num
        CHARACTER ( LEN = 8192 ) :: line
        INTEGER :: unit, ios, e, p

        smin = cpu ; ok = .FALSE.
        WRITE( num, '( I0 )' ) cpu
        path = '/sys/devices/system/cpu/cpu' // TRIM( num ) //                  &
               '/topology/thread_siblings_list'
        OPEN( NEWUNIT = unit, FILE = path, STATUS = 'OLD', ACTION = 'READ',     &
              IOSTAT = ios )
        IF ( ios /= 0 ) RETURN
        line = ''
        READ( unit, '( A )', IOSTAT = ios ) line
        CLOSE( unit )
        IF ( ios /= 0 ) RETURN
        line = ADJUSTL( line )
        e = LEN_TRIM( line )
        IF ( e < 1 ) RETURN
        p = SCAN( line( 1 : e ), ',-' )    ! first integer = up to first sep
        IF ( p > 0 ) e = p - 1
        IF ( e < 1 ) RETURN
        READ( line( 1 : e ), *, IOSTAT = ios ) smin
        IF ( ios /= 0 ) THEN
          smin = cpu
          RETURN
        END IF
        ok = .TRUE.
        END FUNCTION core_first_sibling

    END MODULE GALAHAD_TOPOLOGY
