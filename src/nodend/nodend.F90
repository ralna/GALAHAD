! THIS VERSION: GALAHAD 5.2 - 2025-03-23 AT 09:45 GMT

#include "galahad_modules.h"
#undef METIS_DBG_INFO

#ifdef INTEGER_64
#define galahad_nodend4_adapter galahad_nodend4_adapter_64
#define galahad_nodend51_adapter galahad_nodend51_adapter_64
#define galahad_nodend52_adapter galahad_nodend52_adapter_64
#define METIS_NodeND_4 METIS_NodeND_4_64
#endif

!-*-*-*-*-*-*-*-*- G A L A H A D _ M E T I S    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 5.2. February 28th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_NODEND_precision
     USE GALAHAD_KINDS, ONLY: i4_, i8_, ip_, ipc_
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SMT_precision
     USE GALAHAD_SORT_precision

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: NODEND_initialize, NODEND_read_specfile, NODEND_order,          &
               NODEND_order_adjacency, NODEND_half_order,                      &
               NODEND_full_initialize, NODEND_information,                     &
               SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE NODEND_half_order
       MODULE PROCEDURE NODEND_half_order_i4, NODEND_half_order_i8
     END INTERFACE NODEND_half_order

     INTERFACE NODEND_initialize
       MODULE PROCEDURE NODEND_initialize, NODEND_full_initialize
     END INTERFACE NODEND_initialize

!  MeTiS 4 option addresses

     INTEGER, PARAMETER :: METIS4_OPTION_PTYPE = 1
     INTEGER, PARAMETER :: METIS4_OPTION_CTYPE = 2
     INTEGER, PARAMETER :: METIS4_OPTION_ITYPE = 3
     INTEGER, PARAMETER :: METIS4_OPTION_RTYPE = 4
     INTEGER, PARAMETER :: METIS4_OPTION_DBGLVL = 5
     INTEGER, PARAMETER :: METIS4_OPTION_OFLAGS = 6
     INTEGER, PARAMETER :: METIS4_OPTION_PFACTOR = 7
     INTEGER, PARAMETER :: METIS4_OPTION_NSEPS = 8

!  MeTiS 5.1 (non-command-line) option addresses

     INTEGER, PARAMETER :: METIS51_OPTION_PTYPE = 1
     INTEGER, PARAMETER :: METIS51_OPTION_OBJTYPE = 2
     INTEGER, PARAMETER :: METIS51_OPTION_CTYPE = 3
     INTEGER, PARAMETER :: METIS51_OPTION_IPTYPE = 4
     INTEGER, PARAMETER :: METIS51_OPTION_RTYPE = 5
     INTEGER, PARAMETER :: METIS51_OPTION_DBGLVL = 6
     INTEGER, PARAMETER :: METIS51_OPTION_NITER = 7
     INTEGER, PARAMETER :: METIS51_OPTION_NCUTS = 8
     INTEGER, PARAMETER :: METIS51_OPTION_SEED = 9
     INTEGER, PARAMETER :: METIS51_OPTION_NO2HOP = 10 
     INTEGER, PARAMETER :: METIS51_OPTION_MINCONN = 11
     INTEGER, PARAMETER :: METIS51_OPTION_CONTIG = 12
     INTEGER, PARAMETER :: METIS51_OPTION_COMPRESS = 13
     INTEGER, PARAMETER :: METIS51_OPTION_CCORDER = 14
     INTEGER, PARAMETER :: METIS51_OPTION_PFACTOR = 15
     INTEGER, PARAMETER :: METIS51_OPTION_NSEPS = 16
     INTEGER, PARAMETER :: METIS51_OPTION_UFACTOR = 17
     INTEGER, PARAMETER :: METIS51_OPTION_NUMBERING = 18

!  MeTiS 5.2 (non-command-line) option addresses

     INTEGER, PARAMETER :: METIS52_OPTION_PTYPE = 1
     INTEGER, PARAMETER :: METIS52_OPTION_OBJTYPE = 2 
     INTEGER, PARAMETER :: METIS52_OPTION_CTYPE = 3
     INTEGER, PARAMETER :: METIS52_OPTION_IPTYPE = 4
     INTEGER, PARAMETER :: METIS52_OPTION_RTYPE = 5
     INTEGER, PARAMETER :: METIS52_OPTION_DBGLVL = 6
     INTEGER, PARAMETER :: METIS52_OPTION_NIPARTS = 7
     INTEGER, PARAMETER :: METIS52_OPTION_NITER = 8
     INTEGER, PARAMETER :: METIS52_OPTION_NCUTS = 9
     INTEGER, PARAMETER :: METIS52_OPTION_SEED = 10
     INTEGER, PARAMETER :: METIS52_OPTION_ONDISK = 11
     INTEGER, PARAMETER :: METIS52_OPTION_MINCONN = 12
     INTEGER, PARAMETER :: METIS52_OPTION_CONTIG = 13
     INTEGER, PARAMETER :: METIS52_OPTION_COMPRESS = 14
     INTEGER, PARAMETER :: METIS52_OPTION_CCORDER = 15
     INTEGER, PARAMETER :: METIS52_OPTION_PFACTOR = 16
     INTEGER, PARAMETER :: METIS52_OPTION_NSEPS = 17
     INTEGER, PARAMETER :: METIS52_OPTION_UFACTOR = 18
     INTEGER, PARAMETER :: METIS52_OPTION_NUMBERING = 19
     INTEGER, PARAMETER :: METIS52_OPTION_DROPEDGES = 20
     INTEGER, PARAMETER :: METIS52_OPTION_NO2HOP = 21
     INTEGER, PARAMETER :: METIS52_OPTION_TWOHOP = 22
     INTEGER, PARAMETER :: METIS52_OPTION_FAST = 23

!  default control values

     INTEGER, PARAMETER :: metis4_ptype_default = 0
     INTEGER, PARAMETER :: metis4_ctype_default = 3
     INTEGER, PARAMETER :: metis4_itype_default = 1
     INTEGER, PARAMETER :: metis4_rtype_default = 1
     INTEGER, PARAMETER :: metis4_dbglvl_default = 0
     INTEGER, PARAMETER :: metis4_oflags_default = 1
     INTEGER, PARAMETER :: metis4_pfactor_default = - 1
     INTEGER, PARAMETER :: metis4_nseps_default = 1
     INTEGER, PARAMETER :: metis5_ptype_default = 0
     INTEGER, PARAMETER :: metis5_objtype_default = 2
     INTEGER, PARAMETER :: metis5_ctype_default = 1
     INTEGER, PARAMETER :: metis5_iptype_default = 2
     INTEGER, PARAMETER :: metis5_rtype_default = 2
     INTEGER, PARAMETER :: metis5_dbglvl_default = 0
     INTEGER, PARAMETER :: metis5_niter_default = 10
     INTEGER, PARAMETER :: metis5_ncuts_default = - 1
     INTEGER, PARAMETER :: metis5_seed_default = - 1
     INTEGER, PARAMETER :: metis5_no2hop_default = 0
     INTEGER, PARAMETER :: metis5_minconn_default = - 1
     INTEGER, PARAMETER :: metis5_contig_default = - 1
     INTEGER, PARAMETER :: metis5_compress_default = 1
     INTEGER, PARAMETER :: metis5_ccorder_default = 0
     INTEGER, PARAMETER :: metis5_pfactor_default = 0
     INTEGER, PARAMETER :: metis5_nseps_default = 1
     INTEGER, PARAMETER :: metis5_ufactor_default = 200
     INTEGER, PARAMETER :: metis5_niparts_default = -1
     INTEGER, PARAMETER :: metis5_ondisk_default = 0
     INTEGER, PARAMETER :: metis5_dropedges_default = 0
     INTEGER, PARAMETER :: metis5_twohop_default = - 1
     INTEGER, PARAMETER :: metis5_fast_default = - 1

!----------------------
!   I n t e r f a c e s
!----------------------

    INTERFACE
      SUBROUTINE galahad_nodend4_adapter( n, PTR, ROW, options,                &
                                          perm, iperm ) BIND( C )
        USE GALAHAD_KINDS, ONLY: ipc_
        IMPLICIT NONE
        INTEGER( KIND = ipc_ ), INTENT( IN ) :: n
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: PTR, ROW
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: options
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( OUT ) :: perm, iperm
      END SUBROUTINE galahad_nodend4_adapter
    END INTERFACE

    INTERFACE
      INTEGER( KIND = ipc_ )                                                   &
        FUNCTION galahad_nodend51_adapter( n, PTR, ROW, options,               &
                                           perm, iperm ) BIND( C )
        USE GALAHAD_KINDS, ONLY: ipc_
        IMPLICIT NONE
        INTEGER( KIND = ipc_ ), INTENT( IN ) :: n
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: PTR, ROW
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: options
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( OUT ) :: perm, iperm
      END FUNCTION galahad_nodend51_adapter
    END INTERFACE

    INTERFACE
      INTEGER( KIND = ipc_ )                                                   &
        FUNCTION galahad_nodend52_adapter( n, PTR, ROW, options,               &
                                           perm, iperm ) BIND( C )
        USE GALAHAD_KINDS, ONLY: ipc_
        IMPLICIT NONE
        INTEGER( KIND = ipc_ ), INTENT( IN ) :: n
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: PTR, ROW
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( IN ) :: options
        INTEGER( KIND = ipc_ ), DIMENSION( * ), INTENT( OUT ) :: perm, iperm
      END FUNCTION galahad_nodend52_adapter
    END INTERFACE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: NODEND_control_type

!  version of MeTiS to use: one of '4.0', '5.1' or '5.2'

       CHARACTER ( LEN = 30 ) :: version = '5.2'  // REPEAT( ' ', 27 )

!  unit for error output

       INTEGER ( KIND = ip_ ) :: error = 6

!  unit for monitor output

       INTEGER ( KIND = ip_ ) :: out = 6

!  controls level of diagnostic output

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  if MeTiS 4.0 is not availble, should we use Metis 5.2 instead?

       LOGICAL :: no_metis_4_use_5_instead

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""' // REPEAT( ' ', 28 )

!  -------------------------------------------
!  MeTiS 4 default options (with alternatives)
!  -------------------------------------------

!  the partitioning method: 0 = multilevel recursive bisectioning, 
!   1 = multilevel k-way partitioning

       INTEGER ( KIND = ip_ ) :: metis4_ptype = metis4_ptype_default

!  the matching scheme to be used during coarsening: 1 = random matching, 
!   2 = heavy-edge matching, 3 = sorted heavy-edge matching, 
!   4 = k-way sorted heavy-edge matching

       INTEGER ( KIND = ip_ ) :: metis4_ctype = metis4_ctype_default

!  the algorithm used during initial partitioning: 
!   1 = edge-based region growing, 2 = node-based region growing

       INTEGER ( KIND = ip_ ) :: metis4_itype = metis4_itype_default

!  the algorithm used for refinement: 
!   1 = two-sided node Fiduccia-Mattheyses (FM) refinement,
!   2 = one-sided node FM refinement

       INTEGER ( KIND = ip_ ) :: metis4_rtype = metis4_rtype_default

!  the amount of progress/debugging information printed: 0 = nothing, > 0 more

       INTEGER ( KIND = ip_ ) :: metis4_dbglvl = metis4_dbglvl_default

!  select whether or not to compress the graph, and to order connected 
!   components separately: 0 = do neither, 1 = try to compress the graph, 
!   2 = order each connected component separately, 3 = do both

       INTEGER ( KIND = ip_ ) :: metis4_oflags = metis4_oflags_default

!  the minimum degree of the vertices that will be ordered last: < 1 = none

       INTEGER ( KIND = ip_ ) :: metis4_pfactor = metis4_pfactor_default

!  the number of different separators that the algorithm will compute
!  at each level of nested dissection:

       INTEGER ( KIND = ip_ ) :: metis4_nseps = metis4_nseps_default

!  -------------------------------------------
!  MeTiS 5 default options (with alternatives)
!  -------------------------------------------

!  the partitioning method: 0 = multilevel recursive bisectioning, 
!   1 = multilevel k-way partitioning

       INTEGER ( KIND = ip_ ) :: metis5_ptype = metis5_ptype_default

!  the type of the objective: 2 = node-based nested disection

       INTEGER ( KIND = ip_ ) :: metis5_objtype = metis5_objtype_default

!  the matching scheme to be used during coarsening: 0 = random matching, 
!   1 = sorted heavy-edge matching

       INTEGER ( KIND = ip_ ) :: metis5_ctype = metis5_ctype_default

!  the algorithm used during initial partitioning:
!   2 = derive separators from edge cuts,
!   3 = grow bisections using a greedy node-based strategy

       INTEGER ( KIND = ip_ ) :: metis5_iptype = metis5_iptype_default

!  the algorithm used for refinement: 2 = Two-sided node FM refinement,
!   3 = One-sided node FM refinement.

       INTEGER ( KIND = ip_ ) :: metis5_rtype = metis5_rtype_default

!  the amount of progress/debugging information printed: 0 = nothing, 
!   1 = diagnostics, 2 = + timings, > 2 + more

       INTEGER ( KIND = ip_ ) :: metis5_dbglvl = metis5_dbglvl_default

!  the number of iterations for the refinement algorithm:

       INTEGER ( KIND = ip_ ) :: metis5_niter = metis5_niter_default

!  the number of different partitionings that it will compute: -1 = not used

       INTEGER ( KIND = ip_ ) :: metis5_ncuts = metis5_ncuts_default

!  the seed for the random number generator:

       INTEGER ( KIND = ip_ ) :: metis5_seed = metis5_seed_default

!  specify that the coarsening will not perform any 2–hop matchings when 
!  the standard matching approach fails to sufficiently coarsen the graph:
!  0 = no, 1 = yes

       INTEGER ( KIND = ip_ ) :: metis5_no2hop = metis5_no2hop_default

!  specify that the partitioning routines should try to minimize the maximum 
!  degree of the subdomain graph: 0 = no, 1 = yes, -1 = not used

       INTEGER ( KIND = ip_ ) :: metis5_minconn = metis5_minconn_default

!  specify that the partitioning routines should try to produce partitions 
!  that are contiguous: 0 = no, 1 = yes, -1 = not used

       INTEGER ( KIND = ip_ ) :: metis5_contig = metis5_contig_default

!  specify that the graph should be compressed by combining together vertices 
!  that have identical adjacency lists: 0 = no, 1 = yes

       INTEGER ( KIND = ip_ ) :: metis5_compress = metis5_compress_default

!  specify if the connected components of the graph should first be
!  identified and ordered separately: 0 = no, 1 = yes

       INTEGER ( KIND = ip_ ) :: metis5_ccorder = metis5_ccorder_default

!  the minimum degree of the vertices that will be ordered last:

       INTEGER ( KIND = ip_ ) :: metis5_pfactor = metis5_pfactor_default

!  the number of different separators that the algorithm will compute
!  at each level of nested dissection:

       INTEGER ( KIND = ip_ ) :: metis5_nseps = metis5_nseps_default

!  the maximum allowed load imbalance among the partitions:

       INTEGER ( KIND = ip_ ) :: metis5_ufactor = metis5_ufactor_default

!  --------------------------------------------------------
!  additional MeTiS 5.2 default options (with alternatives)
!  --------------------------------------------------------

!  the number of initial partitions:

       INTEGER ( KIND = ip_ ) :: metis5_niparts = metis5_niparts_default

!  on disk storage: 0 = no, 1 = yes

       INTEGER ( KIND = ip_ ) :: metis5_ondisk = metis5_ondisk_default

!  drop edges: 0 = no, 1 = yes

       INTEGER ( KIND = ip_ ) :: metis5_dropedges = metis5_dropedges_default

!  mentioned but not used:

       INTEGER ( KIND = ip_ ) :: metis5_twohop = metis5_twohop_default
       INTEGER ( KIND = ip_ ) :: metis5_fast = metis5_fast_default

     END TYPE NODEND_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: NODEND_inform_type

!  reported return status:
!     0  success
!   -ve  failure
!   +ve  warnings

       INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the version of MeTiS used

       CHARACTER ( LEN = 3 ) :: version = '5.2'

     END TYPE NODEND_inform_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: NODEND_full_data_type
        LOGICAL :: f_indexing = .TRUE.
!       TYPE ( NODEND_data_type ) :: NODEND_data
        TYPE ( NODEND_control_type ) :: NODEND_control
        TYPE ( NODEND_inform_type ) :: NODEND_inform
      END TYPE NODEND_full_data_type

   CONTAINS

!-*-*-*-*-   N O D E N D _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*

      SUBROUTINE NODEND_initialize( control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for NODEND. This routine should be called before
!  other NODEND procedres
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( NODEND_control_type ), INTENT( OUT ) :: control
      TYPE ( NODEND_inform_type ), INTENT( OUT ) :: inform

!  local variables

      TYPE ( NODEND_control_type ) :: control_local
      TYPE ( NODEND_inform_type ) :: inform_local

!  set default values

      control = control_local
      inform = inform_local

!  End of NODEND_initialize

      END SUBROUTINE NODEND_initialize

!- G A L A H A D - N O D E N D  _ F U L L _ I N I T I A L I Z E  S U B ROUTINE -

      SUBROUTINE NODEND_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for NODEND controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( NODEND_full_data_type ), INTENT( INOUT ) :: data
      TYPE ( NODEND_control_type ), INTENT( OUT ) :: control
      TYPE ( NODEND_inform_type ), INTENT( OUT ) :: inform

      CALL NODEND_initialize( control, inform )

      RETURN

!  End of subroutine NODEND_full_initialize

      END SUBROUTINE NODEND_full_initialize

!-*-*-*-   M E T I S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE NODEND_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by NODEND_initialize could (roughly)
!  have been set as:

! BEGIN NODEND SPECIFICATIONS (DEFAULT)
!  version                                           5.2
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  metis4-ptype                                      0
!  metis4-ctype                                      3
!  metis4-itype                                      1
!  metis4-rtype                                      1
!  metis4-dbglvl                                     0
!  metis4-oflags                                     1
!  metis4-pfactor                                    -1 
!  metis4-nseps                                      1
!  metis5-ptype                                      0
!  metis5-objtype                                    2 
!  metis5-ctype                                      1
!  metis5-iptype                                     2
!  metis5-rtype                                      2
!  metis5-dbglvl                                     0
!  metis5-niparts                                    -1  
!  metis5-niter                                      10
!  metis5-ncuts                                      -1
!  metis5-seed                                       -1
!  metis5-ondisk                                     0
!  metis5-minconn                                    -1  
!  metis5-contig                                     1
!  metis5-compress                                   1   
!  metis5-ccorder                                    0  
!  metis5-pfactor                                    0  
!  metis5-nseps                                      1
!  metis5-ufactor                                    200
!  metis5-dropedges                                  0
!  metis5-no2hop                                     0
!  metis5-twohop                                     -1
!  metis5-fast                                       -1
!  no-metis-4-use-5-instead                          YES
!  output-line-prefix                                ""
! END NODEND SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( NODEND_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: version = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: error = version + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis4_ptype = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis4_ctype = metis4_ptype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis4_itype = metis4_ctype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis4_rtype = metis4_itype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis4_dbglvl = metis4_rtype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis4_oflags = metis4_dbglvl + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis4_pfactor = metis4_oflags + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis4_nseps = metis4_pfactor + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_ptype = metis4_nseps + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_objtype = metis5_ptype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_ctype = metis5_objtype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_iptype = metis5_ctype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_rtype = metis5_iptype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_dbglvl = metis5_rtype + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_niparts = metis5_dbglvl + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_niter = metis5_niparts + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_ncuts = metis5_niter + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_seed = metis5_ncuts + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_ondisk = metis5_seed + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_minconn = metis5_ondisk + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_contig = metis5_minconn + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_compress = metis5_contig + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_ccorder = metis5_compress + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_pfactor = metis5_ccorder + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_nseps = metis5_pfactor + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_ufactor = metis5_nseps + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_dropedges = metis5_ufactor + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_no2hop = metis5_dropedges + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_twohop = metis5_no2hop + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: metis5_fast = metis5_twohop + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: no_metis_4_use_5_instead            &
                                             = metis5_fast + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = no_metis_4_use_5_instead + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 6 ), PARAMETER :: specname = 'NODEND'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( metis4_ptype )%keyword = 'metis4-ptype'
      spec( metis4_ctype )%keyword = 'metis4-ctype'
      spec( metis4_itype )%keyword = 'metis4-itype'
      spec( metis4_rtype )%keyword = 'metis4-rtype'
      spec( metis4_dbglvl )%keyword = 'metis4-dbglvl'
      spec( metis4_oflags )%keyword = 'metis4-oflags'
      spec( metis4_pfactor )%keyword = 'metis4-pfactor'
      spec( metis4_nseps )%keyword = 'metis4-nseps'
      spec( metis5_ptype )%keyword = 'metis5-ptype'
      spec( metis5_objtype )%keyword = 'metis5-objtype'
      spec( metis5_ctype )%keyword = 'metis5-ctype'
      spec( metis5_iptype )%keyword = 'metis5-iptype'
      spec( metis5_rtype )%keyword = 'metis5-rtype'
      spec( metis5_dbglvl )%keyword = 'metis5-dbglvl'
      spec( metis5_niparts )%keyword = 'metis5-niparts'
      spec( metis5_niter )%keyword = 'metis5-niter'
      spec( metis5_ncuts )%keyword = 'metis5-ncuts'
      spec( metis5_seed )%keyword = 'metis5-seed'
      spec( metis5_ondisk )%keyword = 'metis5-ondisk'
      spec( metis5_minconn )%keyword = 'metis5-minconn'
      spec( metis5_contig )%keyword = 'metis5-contig'
      spec( metis5_compress )%keyword = 'metis5-compress'
      spec( metis5_ccorder )%keyword = 'metis5-ccorder'
      spec( metis5_pfactor )%keyword = 'metis5-pfactor'
      spec( metis5_nseps )%keyword = 'metis5-nseps'
      spec( metis5_ufactor )%keyword = 'metis5-ufactor'
      spec( metis5_dropedges )%keyword = 'metis5-dropedges'
      spec( metis5_no2hop )%keyword = 'metis5-no2hop'
      spec( metis5_twohop )%keyword = 'metis5-twohop'
      spec( metis5_fast )%keyword = 'metis5-fast'

!  Logicalr key-words

      spec( no_metis_4_use_5_instead )%keyword = 'no-metis-4-use-5-instead'

!  Character key-words

      spec( version )%keyword = 'version'
      spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( metis4_ptype ),                        &
                                  control%metis4_ptype,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis4_ctype ),                        &
                                  control%metis4_ctype,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis4_itype ),                        &
                                  control%metis4_itype,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis4_rtype ),                        &
                                  control%metis4_rtype,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis4_dbglvl ),                       &
                                  control%metis4_dbglvl,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis4_oflags ),                       &
                                  control%metis4_oflags,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis4_pfactor ),                      &
                                  control%metis4_pfactor,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis4_nseps ),                        &
                                  control%metis4_nseps,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_ptype ),                        &
                                  control%metis5_ptype,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_objtype ),                      &
                                  control%metis5_objtype,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_ctype ),                        &
                                  control%metis5_ctype,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_iptype ),                       &
                                  control%metis5_iptype,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_rtype ),                        &
                                  control%metis5_rtype,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_dbglvl ),                       &
                                  control%metis5_dbglvl,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_niparts ),                      &
                                  control%metis5_niparts,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_niter ),                        &
                                  control%metis5_niter,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_ncuts ),                        &
                                  control%metis5_ncuts,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_seed ),                         &
                                  control%metis5_seed,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_ondisk ),                       &
                                  control%metis5_ondisk,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_minconn ),                      &
                                  control%metis5_minconn,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_contig ),                       &
                                  control%metis5_contig,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_compress ),                     &
                                  control%metis5_compress,                     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_ccorder ),                      &
                                  control%metis5_ccorder,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_pfactor ),                      &
                                  control%metis5_pfactor,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_nseps ),                        &
                                  control%metis5_nseps,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_ufactor ),                      &
                                  control%metis5_ufactor,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_dropedges ),                    &
                                  control%metis5_dropedges,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_no2hop ),                       &
                                  control%metis5_no2hop,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_twohop ),                       &
                                  control%metis5_twohop,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( metis5_fast ),                         &
                                  control%metis5_fast,                         &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( no_metis_4_use_5_instead ),            &
                                  control%no_metis_4_use_5_instead,            &
                                  control%error )

!  Set character values

      CALL SPECFILE_assign_value( spec( version ),                             &
                                  control%version,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )
      RETURN

!  end of subroutine NODEND_read_specfile

      END SUBROUTINE NODEND_read_specfile

!- - -  -  G A L A H A D -  M E T I S _ O R D E R   S U B R O U T I N E - - - -

!  MeTiS 4 and 5 interfaces with various compact (symmetric) structue

      SUBROUTINE NODEND_order( A, PERM, control, inform )
      TYPE ( SMT_type ), INTENT( IN ) :: A
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( A%n ) :: PERM
      TYPE ( NODEND_control_type ), INTENT( IN ) :: control
      TYPE ( NODEND_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, n, ne
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check for input data errors and trivial permutations

      n = A%n
      IF ( n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        WRITE( control%error, "( A, ' A%n = ', I0, ' < 0' )" ) prefix, n
        RETURN
      ELSE IF ( n == 1 ) THEN
        PERM( 1 ) = 1
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  copy from the symmetric (one triangle) input order to full (both
!  triangles) order, but without the diagonals. First, compute the required
!  space to hold the full matrix

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'COORDINATE' )
        ne = 2 * COUNT( A%row( : A%ne ) /= A%col( : A%ne ) )
      CASE ( 'SPARSE_BY_ROWS' )
        ne = 0
        DO i = 1, n
          DO k = A%ptr( i ), A%ptr( i + 1 ) - 1
            IF ( A%col( k ) /= i ) ne = ne + 2 
          END DO
        END DO
      CASE ( 'DENSE' )
        ne =  n * ( n - 1 )
      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        WRITE( control%error, "( A, ' matrix type ', A, ' unknown' )" )        &
          prefix, SMT_get( A%type ) 
        RETURN
      END SELECT

!  if the full matrix is diagonal, return the trivial permutation

      IF ( ne == 0 ) THEN
        PERM( : n ) = (/ ( i, i = 1, n ) /)
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  next allocate workspace for the full matrix

      ALLOCATE( A_row( ne ), A_col( ne ), STAT = inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        WRITE( control%error, "( A, ' allocation error ', I0, ' for A_*' )" )  &
          prefix, inform%alloc_status
        inform%status = GALAHAD_error_allocate ; inform%bad_alloc = 'A_*'
        RETURN
      END IF

!  now copy the symmetrix matrix to the full one

      ne = 0
      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'COORDINATE' )
        DO k = 1, A%ne
          i = A%row( k ) ; j = A%col( k ) 
          IF ( i /= j ) THEN
            ne = ne + 1
            A_row( ne ) = i ; A_col( ne ) = j
            ne = ne + 1
            A_row( ne ) = j ; A_col( ne ) = i
          END IF
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, n
          DO k = A%ptr( i ), A%ptr( i + 1 ) - 1
            j = A%col( k )
            IF ( i /= j ) THEN
              ne = ne + 1
              A_row( ne ) = i ; A_col( ne ) = j
              ne = ne + 1
              A_row( ne ) = j ; A_col( ne ) = i
            END IF
          END DO
        END DO
      CASE ( 'DENSE' )
        DO i = 1, n
          DO j = 1, n
            IF ( i /= j ) THEN
              ne = ne + 1
              A_row( ne ) = i ; A_col( ne ) = j
            END IF
          END DO
        END DO
      END SELECT

!  find the ordering

      CALL NODEND_order_main( n, ne, A_row, A_col, PERM, control, inform )

!  deallocate workspace arrays

      DEALLOCATE( A_row, A_col, STAT = inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        WRITE( control%error, "( A, ' deallocation error ', I0, ' for A_*' )") &
          prefix, inform%alloc_status
        inform%status = GALAHAD_error_deallocate ; inform%bad_alloc = 'A_*'
      END IF

      RETURN

!  End of subroutine NODEND_order

      END SUBROUTINE NODEND_order

!- G A L A H A D -  M E T I S _ H A L F _ O R D E R _ I 4  S U B R O U T I N E -

!  MeTiS 4 and 5 interface with compact (symmetric) sparse-by-row structue

      SUBROUTINE NODEND_half_order_i4( n, A_half_ptr, A_half_col, PERM,        &
                                       control, inform )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
      INTEGER ( KIND = i4_ ), INTENT( IN ), DIMENSION( n + 1 ) :: A_half_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: A_half_col
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: PERM
      TYPE ( NODEND_control_type ), INTENT( IN ) :: control
      TYPE ( NODEND_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, ne
      INTEGER ( KIND = i4_ ) :: k
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check for input data errors and trivial permutations

      IF ( n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        WRITE( control%error, "( A, ' n = ', I0, ' < 0' )" ) prefix, n
        RETURN
      ELSE IF ( n == 1 ) THEN
        PERM( 1 ) = 1
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  copy from the half (one triangle) input order to full (both triangles) 
!  order, but without the diagonals. First, compute the required space to 
!  hold the full matrix

      ne = 0
      DO i = 1, n
        DO k = A_half_ptr( i ), A_half_ptr( i + 1 ) - 1
          IF ( A_half_col( k ) /= i ) ne = ne + 2 
        END DO
      END DO

!  if the full matrix is diagonal, return the trivial permutation

      IF ( ne == 0 ) THEN
        PERM( : n ) = (/ ( i, i = 1, n ) /)
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  next allocate workspace for the full matrix

      ALLOCATE( A_row( ne ), A_col( ne ), STAT = inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        WRITE( control%error, "( A, ' allocation error ', I0, ' for A_*' )" )  &
          prefix, inform%alloc_status
        inform%status = GALAHAD_error_allocate ; inform%bad_alloc = 'A_*'
        RETURN
      END IF

!  now copy the half matrix to the full one

      ne = 0
      DO i = 1, n
        DO k = A_half_ptr( i ), A_half_ptr( i + 1 ) - 1
          j = A_half_col( k )
          IF ( i /= j ) THEN
            ne = ne + 1
            A_row( ne ) = i ; A_col( ne ) = j
            ne = ne + 1
            A_row( ne ) = j ; A_col( ne ) = i
          END IF
        END DO
      END DO

!  find the ordering

      CALL NODEND_order_main( n, ne, A_row, A_col, PERM, control, inform )

!  deallocate workspace arrays

      DEALLOCATE( A_row, A_col, STAT = inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        WRITE( control%error, "( A, ' deallocation error ', I0, ' for A_*' )") &
          prefix, inform%alloc_status
        inform%status = GALAHAD_error_deallocate ; inform%bad_alloc = 'A_*'
      END IF

      RETURN

!  End of subroutine NODEND_half_order_i4

      END SUBROUTINE NODEND_half_order_i4

!- G A L A H A D -  M E T I S _ H A L F _ O R D E R _ I 8  S U B R O U T I N E -

!  MeTiS 4 and 5 interface with compact (symmetric) sparse-by-row structue

      SUBROUTINE NODEND_half_order_i8( n, A_half_ptr, A_half_col, PERM,        &
                                       control, inform )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
      INTEGER ( KIND = i8_ ), INTENT( IN ), DIMENSION( n + 1 ) :: A_half_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: A_half_col
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: PERM
      TYPE ( NODEND_control_type ), INTENT( IN ) :: control
      TYPE ( NODEND_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, ne
      INTEGER ( KIND = i8_ ) :: k
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check for input data errors and trivial permutations

      IF ( n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        WRITE( control%error, "( A, ' n = ', I0, ' < 0' )" ) prefix, n
        RETURN
      ELSE IF ( n == 1 ) THEN
        PERM( 1 ) = 1
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  copy from the half (one triangle) input order to full (both triangles) 
!  order, but without the diagonals. First, compute the required space to 
!  hold the full matrix

      ne = 0
      DO i = 1, n
        DO k = A_half_ptr( i ), A_half_ptr( i + 1 ) - 1
          IF ( A_half_col( k ) /= i ) ne = ne + 2 
        END DO
      END DO

!  if the full matrix is diagonal, return the trivial permutation

      IF ( ne == 0 ) THEN
        PERM( : n ) = (/ ( i, i = 1, n ) /)
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  next allocate workspace for the full matrix

      ALLOCATE( A_row( ne ), A_col( ne ), STAT = inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        WRITE( control%error, "( A, ' allocation error ', I0, ' for A_*' )" )  &
          prefix, inform%alloc_status
        inform%status = GALAHAD_error_allocate ; inform%bad_alloc = 'A_*'
        RETURN
      END IF

!  now copy the half matrix to the full one

      ne = 0
      DO i = 1, n
        DO k = A_half_ptr( i ), A_half_ptr( i + 1 ) - 1
          j = A_half_col( k )
          IF ( i /= j ) THEN
            ne = ne + 1
            A_row( ne ) = i ; A_col( ne ) = j
            ne = ne + 1
            A_row( ne ) = j ; A_col( ne ) = i
          END IF
        END DO
      END DO

!  find the ordering

      CALL NODEND_order_main( n, ne, A_row, A_col, PERM, control, inform )

!  deallocate workspace arrays

      DEALLOCATE( A_row, A_col, STAT = inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        WRITE( control%error, "( A, ' deallocation error ', I0, ' for A_*' )") &
          prefix, inform%alloc_status
        inform%status = GALAHAD_error_deallocate ; inform%bad_alloc = 'A_*'
      END IF

      RETURN

!  End of subroutine NODEND_half_order_i8

      END SUBROUTINE NODEND_half_order_i8

!- - G A L A H A D -  M E T I S _ O R D E R _ M A I N   S U B R O U T I N E - -

!  MeTiS 4 and 5 interface with compact (symmetric) sparse-by-row structue

      SUBROUTINE NODEND_order_main( n, ne, A_row, A_col, PERM, control, inform )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ne
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( ne ) :: A_row, A_col
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: PERM
      TYPE ( NODEND_control_type ), INTENT( IN ) :: control
      TYPE ( NODEND_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: i, status
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr, IW
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check for input data errors and trivial permutations

      IF ( n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        WRITE( control%error, "( A, ' n = ', I0, ' < 0' )" ) prefix, n
        RETURN
      ELSE IF ( n == 1 ) THEN
        PERM( 1 ) = 1
        inform%status = GALAHAD_ok
        RETURN
      ELSE IF ( ne == 0 ) THEN
        PERM( : n ) = (/ ( i, i = 1, n ) /)
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  allocate further workspace for the full matrix

      ALLOCATE( A_ptr( n + 1 ), IW( n + 1 ), STAT = inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        WRITE( control%error, "( A, ' allocation error ', I0, ' for IW' )" )   &
          prefix, inform%alloc_status
        inform%status = GALAHAD_error_allocate ; inform%bad_alloc = 'IW'
        RETURN
      END IF

!  reorder the full matrix to column order

      CALL SORT_reorder_by_cols( n, n, ne, A_row, A_col, ne, A_ptr, n + 1,     &
                                 IW, n + 1, control%error, control%out, status )
      IF ( status > 0 ) THEN
        WRITE( control%error, "( A, ' sort error = ', I0 )" ) prefix, status
        inform%status = GALAHAD_error_sort
      ELSE

!  call the nodend ordering packages

        CALL NODEND_order_adjacency( n, A_ptr, A_row, PERM, control, inform )
      END IF

!  deallocate workspace arrays

      DEALLOCATE( A_ptr, IW, STAT = inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        WRITE( control%error, "( A, ' deallocation error ', I0, ' for A_*' )") &
          prefix, inform%alloc_status
        inform%status = GALAHAD_error_deallocate ; inform%bad_alloc = 'A_*'
      END IF

      RETURN

!  End of subroutine NODEND_order_main

      END SUBROUTINE NODEND_order_main

! G A L A H A D - M E T I S _ O R D E R _ A D J A C E N C Y  S U B R O U T I N E

!  MeTiS 4 and 5 interfaces with full (non-symmetric) structure

      SUBROUTINE NODEND_order_adjacency( n, PTR, IND, PERM, control, inform )
      INTEGER ( KIND = ip_ ), iNTENT( IN ) :: n
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: PTR
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( * ) :: IND
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: PERM
      TYPE ( NODEND_control_type ), INTENT( IN ) :: control
      TYPE ( NODEND_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: i, out, metis_status
      INTEGER ( KIND = ip_ ), DIMENSION( 40 ) :: options
      INTEGER ( KIND = ip_ ), DIMENSION( n ) :: INVERSE_PERM
      INTEGER ( KIND = ip_ ), PARAMETER :: n_dummy = 2
      INTEGER ( KIND = ip_ ), DIMENSION( n_dummy + 1 ) ::                      &
                                PTR_dummy = (/ 1, 2, 3 /)
      INTEGER ( KIND = ip_ ), DIMENSION( n_dummy ) :: IND_dummy = (/ 2, 1 /)

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check for input data errors and trivial permutations

      IF ( n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        WRITE( control%error, "( A, ' n = ', I0, ' < 0' )" ) prefix, n
        RETURN
      ELSE IF ( n == 1 ) THEN
        PERM( 1 ) = 1
        inform%status = GALAHAD_ok
        RETURN
      ELSE IF ( PTR( n + 1 ) == 1 ) THEN
        PERM( : n ) = (/ ( i, i = 1, n ) /)
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  use 1-based arrays

      options = - 1_ip_
      out = control%out
      inform%version = control%version( 1 : 3 )

!  print input data if required

      IF ( out > 0 .AND. control%print_level > 0 ) THEN
        WRITE( out, "( A, ' n = ', I0 )" ) prefix, n
        WRITE( out, "( ' PTR: ', 10I7, /, ( 6X, 10I7 ) ) " ) PTR
        WRITE( out, "( ' IND: ', 10I7, /, ( 6X, 10I7 ) ) " ) &
          IND( : PTR( n + 1 ) - 1 )
        WRITE( out, "( ' options: ', 10I6, /, ( 10X, 10I6 ) ) " ) options
      END IF

!  call the appropriate version of metis_nodend

  100 CONTINUE
      SELECT CASE ( inform%version )

!  MeTiS 4.0

      CASE ( '4.0' )

!  test to see if MeTiS 4.0 is available

        options( METIS4_OPTION_PTYPE ) = metis4_ptype_default
        options( METIS4_OPTION_CTYPE ) = metis4_ctype_default
        options( METIS4_OPTION_ITYPE ) = metis4_itype_default
        options( METIS4_OPTION_RTYPE ) = metis4_rtype_default
        options( METIS4_OPTION_DBGLVL ) = metis4_dbglvl_default
        options( METIS4_OPTION_OFLAGS ) = metis4_oflags_default
        options( METIS4_OPTION_PFACTOR ) = metis4_pfactor_default
        options( METIS4_OPTION_NSEPS ) = metis4_nseps_default

        CALL galahad_nodend4_adapter( n_dummy, PTR_dummy, IND_dummy,           &
                                      options, INVERSE_PERM, PERM )

!  MeTiS is not available, chose whether to try with MeTiS 5 or to exit

        IF ( PERM( 1 ) <= 0 ) THEN
          IF ( out > 0 .AND. control%print_level > 0 ) WRITE( out,             &
           "( A, ' MeTiS version ', A3, ' not availble' )" )                   &
             prefix, control%version
          IF ( control%no_metis_4_use_5_instead ) THEN
            IF ( out > 0 .AND. control%print_level > 0 ) WRITE( out,           &
             "( A, ' Switching to MeTiS version 5.2' )" ) prefix
            inform%version = '5.2'
            GO TO 100
          END IF
          GO TO 910
        END IF

!  record options, overriding inappropriate choices with defaults

        IF ( control%metis4_ptype == 0 .OR. control%metis4_ptype == 1 ) THEN
          options( METIS4_OPTION_PTYPE ) = control%metis4_ptype
        ELSE
          options( METIS4_OPTION_PTYPE ) = metis4_ptype_default
        END IF
        IF ( control%metis4_ctype >= 1 .AND. control%metis4_ctype <= 4 ) THEN
          options( METIS4_OPTION_CTYPE ) = control%metis4_ctype
        ELSE
          options( METIS4_OPTION_CTYPE ) = metis4_ctype_default
        END IF
        IF ( control%metis4_itype == 1 .OR. control%metis4_itype == 2 ) THEN
          options( METIS4_OPTION_ITYPE ) = control%metis4_itype
        ELSE
          options( METIS4_OPTION_ITYPE ) = metis4_itype_default
        END IF
        IF ( control%metis4_rtype == 1 .OR. control%metis4_rtype == 2 ) THEN
          options( METIS4_OPTION_RTYPE ) = control%metis4_rtype
        ELSE
          options( METIS4_OPTION_RTYPE ) = metis4_rtype_default
        END IF
        IF ( control%metis4_dbglvl >= 0 ) THEN
          options( METIS4_OPTION_DBGLVL ) = control%metis4_dbglvl
        ELSE
          options( METIS4_OPTION_DBGLVL ) = metis4_dbglvl_default
        END IF
        IF ( control%metis4_oflags >=0 .AND. control%metis4_oflags <= 3 ) THEN
          options( METIS4_OPTION_OFLAGS ) = control%metis4_oflags
        ELSE
          options( METIS4_OPTION_OFLAGS ) = metis4_oflags_default
        END IF
        options( METIS4_OPTION_PFACTOR ) = control%metis4_pfactor
        IF ( control%metis4_nseps > 1 ) THEN
          options( METIS4_OPTION_NSEPS ) = control%metis4_nseps
        ELSE
          options( METIS4_OPTION_NSEPS ) = metis4_nseps_default
        END IF

!  call MeTiS 4 to get the ordering

        CALL galahad_nodend4_adapter( n, PTR, IND, options,                    &
                                      INVERSE_PERM, PERM )
        IF ( PERM( 1 ) == - 2 ) THEN
!         write(6,*) ' bailed out of MeTiS 4'
          inform%status = GALAHAD_error_metis
          RETURN
        END IF

!  MeTiS 5.1

      CASE ( '5.1' )

!  record options, overriding inappropriate choices with defaults

        IF ( control%metis5_ptype == 0 .OR. control%metis5_ptype == 1 ) THEN
          options( METIS51_OPTION_PTYPE ) = control%metis5_ptype
        ELSE
          options( METIS51_OPTION_PTYPE ) = metis5_ptype_default
        END IF
!       options( METIS52_OPTION_OBJTYPE ) = control%metis5_objtype
        options( METIS52_OPTION_OBJTYPE ) = 2  ! no choice
        IF ( control%metis5_ctype == 0 .OR. control%metis5_ctype == 1 ) THEN
          options( METIS51_OPTION_CTYPE ) = control%metis5_ctype
        ELSE
          options( METIS51_OPTION_CTYPE ) = metis5_ctype_default
        END IF
        IF ( control%metis5_iptype == 2 .OR.control%metis5_iptype == 3 ) THEN
          options( METIS51_OPTION_IPTYPE ) = control%metis5_iptype
        ELSE
          options( METIS51_OPTION_IPTYPE ) = metis5_iptype_default
        END IF
        IF ( control%metis5_rtype == 2 .OR. control%metis5_rtype == 3 ) THEN
          options( METIS51_OPTION_RTYPE ) = control%metis5_rtype
        ELSE
          options( METIS51_OPTION_RTYPE ) = metis5_rtype_default
        END IF
        IF ( control%metis5_dbglvl >= 0 ) THEN
          options( METIS51_OPTION_DBGLVL ) = control%metis5_dbglvl
        ELSE
          options( METIS51_OPTION_DBGLVL ) = metis5_dbglvl_default
        END IF
        IF ( control%metis5_niter >= 0 ) THEN
          options( METIS51_OPTION_NITER ) = control%metis5_niter
        ELSE
          options( METIS51_OPTION_NITER ) = metis5_niter_default
        END IF
        IF ( control%metis5_ncuts > 0 ) THEN
          options( METIS51_OPTION_NCUTS ) = control%metis5_ncuts
        ELSE
          options( METIS51_OPTION_NCUTS ) = metis5_ncuts_default
        END IF
        options( METIS51_OPTION_SEED ) = control%metis5_seed
        IF ( control%metis5_no2hop == 0 .OR.control%metis5_no2hop == 1 ) THEN
          options( METIS51_OPTION_NO2HOP ) = control%metis5_no2hop
        ELSE
          options( METIS51_OPTION_NO2HOP ) = metis5_no2hop_default
        END IF
        IF ( control%metis5_minconn >= - 1 .AND.                               &
             control%metis5_minconn <= 1 ) THEN
          options( METIS51_OPTION_MINCONN ) = control%metis5_minconn
        ELSE
          options( METIS51_OPTION_MINCONN ) = metis5_minconn_default
        END IF
        IF ( control%metis5_contig  >= - 1 .AND.                               &
             control%metis5_contig  <= 1 ) THEN
          options( METIS51_OPTION_CONTIG ) = control%metis5_contig
        ELSE
          options( METIS51_OPTION_CONTIG ) = metis5_contig_default
        END IF
        IF ( control%metis5_compress == 0 .OR.                                 &
             control%metis5_compress == 1 ) THEN
          options( METIS51_OPTION_COMPRESS ) = control%metis5_compress
        ELSE
          options( METIS51_OPTION_COMPRESS ) = metis5_compress_default
        END IF
        IF ( control%metis5_ccorder == 0 .OR. control%metis5_ccorder == 1 ) THEN
          options( METIS51_OPTION_CCORDER ) = control%metis5_ccorder
        ELSE
          options( METIS51_OPTION_CCORDER ) = metis5_ccorder_default
        END IF
        IF ( control%metis5_pfactor > 0 ) THEN
          options( METIS51_OPTION_PFACTOR ) = control%metis5_pfactor
        ELSE
          options( METIS51_OPTION_PFACTOR ) = metis5_pfactor_default
        END IF
        IF ( control%metis5_nseps > 0 ) THEN
          options( METIS51_OPTION_NSEPS ) = control%metis5_nseps
        ELSE
          options( METIS51_OPTION_NSEPS ) = metis5_nseps_default
        END IF
        IF ( control%metis5_ufactor > 0 ) THEN
          options( METIS51_OPTION_UFACTOR ) = control%metis5_ufactor
        ELSE
          options( METIS51_OPTION_UFACTOR ) = metis5_ufactor_default
        END IF
        options( METIS51_OPTION_NUMBERING ) = 1  ! no choice

!  call MeTiS 5.1 to get ordering via C MeTiS 4 to 5.1 adapter

        metis_status = galahad_nodend51_adapter( n, PTR, IND, options,         &
                                                 INVERSE_PERM, PERM )
        SELECT CASE( metis_status )
        CASE ( 1 )
          inform%status = GALAHAD_ok
        CASE ( - 2 )
          inform%status = GALAHAD_error_restrictions
        CASE ( - 3 )
          inform%status = GALAHAD_error_metis_memory
        CASE DEFAULT
          inform%status = GALAHAD_error_metis
        END SELECT

!  MeTiS 5.2

      CASE ( '5.2' )

!  record options, overriding inappropriate choices with defaults

        IF ( control%metis5_ptype == 0 .OR. control%metis5_ptype == 1 ) THEN
          options( METIS52_OPTION_PTYPE ) = control%metis5_ptype
        ELSE
          options( METIS52_OPTION_PTYPE ) = metis5_ptype_default
        END IF
!       options( METIS52_OPTION_OBJTYPE ) = control%metis5_objtype
        options( METIS52_OPTION_OBJTYPE ) = 2  ! no choice
        IF ( control%metis5_ctype == 0 .OR. control%metis5_ctype == 1 ) THEN
          options( METIS52_OPTION_CTYPE ) = control%metis5_ctype
        ELSE
          options( METIS52_OPTION_CTYPE ) = metis5_ctype_default
        END IF
        IF ( control%metis5_iptype == 2 .OR.control%metis5_iptype == 3 ) THEN
          options( METIS52_OPTION_IPTYPE ) = control%metis5_iptype
        ELSE
          options( METIS52_OPTION_IPTYPE ) = metis5_iptype_default
        END IF
        IF ( control%metis5_rtype == 2 .OR. control%metis5_rtype == 3 ) THEN
          options( METIS52_OPTION_RTYPE ) = control%metis5_rtype
        ELSE
          options( METIS52_OPTION_RTYPE ) = metis5_rtype_default
        END IF
        IF ( control%metis5_dbglvl >= 0 ) THEN
          options( METIS52_OPTION_DBGLVL ) = control%metis5_dbglvl
        ELSE
          options( METIS52_OPTION_DBGLVL ) = metis5_dbglvl_default
        END IF
        IF ( control%metis5_niparts < 1 ) THEN
          options( METIS52_OPTION_NIPARTS ) = control%metis5_niparts
        ELSE
          options( METIS52_OPTION_NIPARTS ) = metis5_niparts_default
        END IF
        IF ( control%metis5_niter >= 0 ) THEN
          options( METIS52_OPTION_NITER ) = control%metis5_niter
        ELSE
          options( METIS52_OPTION_NITER ) = metis5_niter_default
        END IF
        IF ( control%metis5_ncuts > 0 ) THEN
          options( METIS52_OPTION_NCUTS ) = control%metis5_ncuts
        ELSE
          options( METIS52_OPTION_NCUTS ) = metis5_ncuts_default
        END IF
        options( METIS52_OPTION_SEED ) = control%metis5_seed
        IF ( control%metis5_ondisk == 0 .OR. control%metis5_ondisk == 1 ) THEN
          options( METIS52_OPTION_ONDISK ) = control%metis5_ondisk
        ELSE
          options( METIS52_OPTION_ONDISK ) = metis5_ondisk_default
        END IF
        IF ( control%metis5_no2hop == 0 .OR.control%metis5_no2hop == 1 ) THEN
          options( METIS52_OPTION_NO2HOP ) = control%metis5_no2hop
        ELSE
          options( METIS52_OPTION_NO2HOP ) = metis5_no2hop_default
        END IF
        IF ( control%metis5_minconn >= - 1 .AND.                               &
             control%metis5_minconn <= 1 ) THEN
          options( METIS52_OPTION_MINCONN ) = control%metis5_minconn
        ELSE
          options( METIS52_OPTION_MINCONN ) = metis5_minconn_default
        END IF
        IF ( control%metis5_contig  >= - 1 .AND.                               &
             control%metis5_contig  <= 1 ) THEN
          options( METIS52_OPTION_CONTIG ) = control%metis5_contig
        ELSE
          options( METIS52_OPTION_CONTIG ) = metis5_contig_default
        END IF
        IF ( control%metis5_compress == 0 .OR.                                 &
             control%metis5_compress == 1 ) THEN
          options( METIS52_OPTION_COMPRESS ) = control%metis5_compress
        ELSE
          options( METIS52_OPTION_COMPRESS ) = metis5_compress_default
        END IF
        IF ( control%metis5_ccorder == 0 .OR. control%metis5_ccorder == 1 ) THEN
          options( METIS52_OPTION_CCORDER ) = control%metis5_ccorder
        ELSE
          options( METIS52_OPTION_CCORDER ) = metis5_ccorder_default
        END IF
        IF ( control%metis5_pfactor > 0 ) THEN
          options( METIS52_OPTION_PFACTOR ) = control%metis5_pfactor
        ELSE
          options( METIS52_OPTION_PFACTOR ) = metis5_pfactor_default
        END IF
        IF ( control%metis5_nseps > 0 ) THEN
          options( METIS52_OPTION_NSEPS ) = control%metis5_nseps
        ELSE
          options( METIS52_OPTION_NSEPS ) = metis5_nseps_default
        END IF
        IF ( control%metis5_ufactor > 0 ) THEN
          options( METIS52_OPTION_UFACTOR ) = control%metis5_ufactor
        ELSE
          options( METIS52_OPTION_UFACTOR ) = metis5_ufactor_default
        END IF
        options( METIS52_OPTION_NUMBERING ) = 1  ! no choice
        IF ( control%metis5_dropedges == 0 .OR.                                &
             control%metis5_dropedges == 1 ) THEN
          options( METIS52_OPTION_DROPEDGES ) = control%metis5_dropedges
        ELSE
          options( METIS52_OPTION_DROPEDGES ) = metis5_dropedges_default
        END IF
        IF ( control%metis5_no2hop == 0 .OR.control%metis5_no2hop == 1 ) THEN
          options( METIS52_OPTION_NO2HOP ) = control%metis5_no2hop
        ELSE
          options( METIS52_OPTION_NO2HOP ) = metis5_no2hop_default
        END IF
        options( METIS52_OPTION_TWOHOP ) = metis5_twohop_default ! unused
        options( METIS52_OPTION_FAST ) = metis5_fast_default ! unused

!  call MeTiS 5.2 to get ordering via C MeTiS 4 to 5.2 adapter

        metis_status = galahad_nodend52_adapter( n, PTR, IND, options,         &
                                                 INVERSE_PERM, PERM )
        SELECT CASE( metis_status )
        CASE ( 1 )
          inform%status = GALAHAD_ok
        CASE ( - 2 )
          inform%status = GALAHAD_error_restrictions
        CASE ( - 3 )
          inform%status = GALAHAD_error_metis_memory
        CASE DEFAULT
          inform%status = GALAHAD_error_metis
        END SELECT

!  MeTiS version not known

      CASE DEFAULT
        IF ( out > 0 .AND. control%print_level > 0 ) WRITE( out,               &
           "( A, ' MeTiS version ', A3, ' not supported' )" )                  &
             prefix, control%version
        GO TO 910
      END SELECT

!  print output data if required

      IF ( out > 0 .AND. control%print_level > 0 ) THEN
        WRITE( out, "( ' PERM:   ', 10I7, /, ( 10X, 10I7 ) ) " ) PERM
        WRITE( out, "( ' INVPRM: ', 10I7, /, ( 10X, 10I7 ) ) " ) INVERSE_PERM
      END IF

      inform%status = GALAHAD_ok
      RETURN

 910  CONTINUE
      inform%status = GALAHAD_error_metis

      RETURN

!  End of subroutine NODEND_order_adjacency

      END SUBROUTINE NODEND_order_adjacency

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================

! ----------------------------------------------------------------------------
!- G A L A H A D -  N O D E N D _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE NODEND_information( data, inform, status )

!  return solver information during or after solution by NODEND
!  See NODEND_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NODEND_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NODEND_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%nodend_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine NODEND_information

     END SUBROUTINE NODEND_information

!  end of module GALAHAD_NODEND

   END MODULE GALAHAD_NODEND_precision

