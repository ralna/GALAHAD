! THIS VERSION: GALAHAD 5.2 - 2025-02-20 AT 08:25 GMT.

#ifdef INTEGER_64
#define galahad_kinds galahad_kinds_64
#define galahad_metis4 galahad_metis4_64
#define galahad_metis4_setup galahad_metis4_setup_64
#define METIS_NodeND_4 METIS_NodeND_4_64
#endif

!  MeTiS 4 interface

  SUBROUTINE galahad_metis4( n, iptr, irn, metftn, metopt, invprm, perm )
    USE galahad_kinds, ONLY: ip_
    IMPLICIT NONE
    INTEGER( KIND = ip_ ), iNTENT( IN ) :: n
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: iptr
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( * ) :: irn
    INTEGER( KIND = ip_ ), INTENT( IN ) :: metftn
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( 8 ) :: metopt
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: invprm
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: perm
!EXTERNAL  METIS_NodeND_4
    ! C interface for MeTiS_NodeND
    INTERFACE
      SUBROUTINE galahad_metis4_adapter( nvtxs, xadj, adjncy, numflag,         &
!     SUBROUTINE METIS_NodeND( nvtxs, xadj, adjncy, numflag,         &
                                options, perm, iperm ) BIND(C)
        USE galahad_kinds, ONLY: ipc_
        IMPLICIT NONE
        INTEGER( KIND = ipc_ ), INTENT( IN ) :: nvtxs, numflag
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: xadj, adjncy
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: options
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( OUT ) :: perm, iperm
      END SUBROUTINE galahad_metis4_adapter
!     END SUBROUTINE METIS_NodeND
    END INTERFACE

    ! Call MeTiS 4 to get ordering
    CALL galahad_metis4_adapter( n, iptr, irn, metftn, metopt, invprm, perm )
!   CALL METIS_NodeND( n, iptr, irn, metftn, metopt, invprm, perm )
    RETURN
  END SUBROUTINE galahad_metis4

!  MeTiS 4 and 5 default options

  SUBROUTINE galahad_metis4_setopt( metopt )
    USE galahad_kinds, ONLY: ip_
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( 8 ) :: metopt
    metopt = (/ 1,  & ! OPTION_PTYPE
                3,  & ! OPTION_CTYPE
                1,  & ! OPTION_ITYPE
                1,  & ! OPTION_RTYPE
                0,  & ! OPTION_DBGLVL (not default)
                1,  & ! OPTION_OFLAGS
                -1, & ! OPTION_PFACTOR
                1  /) ! OPTION_NSEPS
    RETURN
  END SUBROUTINE galahad_metis4_setopt
