! THIS VERSION: GALAHAD 5.0 - 2024-03-25 AT 08:25 GMT.

#ifdef INTEGER_64
#define galahad_kinds galahad_kinds_64
#define galahad_metis galahad_metis_64
#define galahad_metis_setup galahad_metis_setup_64
#define galahad_metis5_adapter galahad_metis5_adapter_64
#endif

!  MeTiS 4 and 5 interfaces

  SUBROUTINE galahad_metis( n, ptr, irn, f_based, options, invprm, perm )
    USE galahad_kinds, ONLY: ip_
    IMPLICIT NONE
    INTEGER( KIND = ip_ ), iNTENT( IN ) :: n
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: ptr
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( * ) :: irn
    INTEGER( KIND = ip_ ), INTENT( IN ) :: f_based
    INTEGER( KIND = ip_ ), INTENT( IN ), DIMENSION( 8 ) :: options
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: invprm
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: perm

#ifdef METIS4
    ! Call MeTiS 4 to get ordering
    CALL metis_nodend( n, ptr, irn, fbased, options, invprm, perm )
#else
    ! C interface for MeTiS 4 to 5 adapter
    INTEGER :: n_options
    INTERFACE
      SUBROUTINE galahad_metis5_adapter( nvtxs, xadj, adjncy, numflag,         &
                                         options, perm, iperm ) BIND(C)
        USE galahad_kinds, ONLY: ipc_
        IMPLICIT NONE
        INTEGER( KIND = ipc_ ), INTENT( IN ) :: nvtxs, numflag
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: xadj, adjncy
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: options
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( OUT ) :: perm, iperm
      END SUBROUTINE galahad_metis5_adapter
    END INTERFACE
    ! call MeTiS 5 to get ordering via C MeTiS 4 to 5 adapter
    n_options = SIZE( options )
!write(6,*) ' n ', n
!write(6,*) ' ptr ', ptr
!write(6,*) ' irn ', irn( : ptr( n + 1 ) - 1 )
!write(6,*) ' fbased ', f_based
!write(6,*) ' n_options ', n_options
!write(6,*) ' options ', options
    CALL galahad_metis5_adapter( n, ptr, irn, f_based, options, invprm, perm )
!write(6,*) ' invprm ', invprm
!write(6,*) ' perm ', perm
#endif
  RETURN
  END SUBROUTINE galahad_metis

!  MeTiS 4 and 5 default options

  SUBROUTINE galahad_metis_setopt( options )
    USE galahad_kinds, ONLY: ip_
    INTEGER( KIND = ip_ ), INTENT( OUT ), DIMENSION( 8 ) :: options
#ifdef METIS4
    options( 1 ) = 0   ! OPTION_PTYPE
    options( 2 ) = 3   ! OPTION_CTYPE               
    options( 3 ) = 1   ! OPTION_ITYPE               
    options( 4 ) = 1   ! OPTION_RTYPE               
    options( 5 ) = 0   ! OPTION_DBGLVL (not default)
    options( 6 ) = 1   ! OPTION_OFLAGS              
    options( 7 ) = - 1 ! OPTION_PFACTOR             
    options( 8 ) = 1   ! OPTION_NSEPS  
#else
    options( 1 ) = 1 ! set options                       
    options( 2 ) = 0 ! METIS_OPTION_CTYPE                 
    options( 3 ) = 1 ! METIS_OPTION_IPTYPE                
    options( 4 ) = 0 ! METIS_OPTION_RTYPE                 
    options( 5 ) = 0 ! METIS_OPTION_DBGLVL (not default) 
    options( 6 ) = 1 ! METIS_OPTION_COMPRESS and _CCORDER 
    options( 7 ) = 0 ! METIS_OPTION_PFACTOR               
    options( 8 ) = 1 ! METIS_OPTION_NSEPS  
#endif
  RETURN
  END SUBROUTINE galahad_metis_setopt
