! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*- G A L A H A D _ A M D    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Timothy A. Davis, Patrick Amestoy, Iain S. Duff,
!  John K. Reid and Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1. October 11th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      MODULE GALAHAD_AMD_precision

!  This module is based on

!-------------------------------------------------------------------------
! AMD:  approximate minimum degree, with and without aggressive absorption
!-------------------------------------------------------------------------

! Authors, and Copyright (C) 1995 by:
!  Timothy A. Davis, Patrick Amestoy, Iain S. Duff, & John K. Reid.
!  Nick Gould, used by permission and translated to modern fortran, 2022

! AMD License:

!    Your use or distribution of AMD or any modified version of
!    AMD implies that you agree to this License.

!    THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
!    EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.

!    Permission is hereby granted to use or copy this program, provided
!    that the Copyright, this License, and the Availability of the original
!    version is retained on all copies.  User documentation of any code that
!    uses AMD or any modified version of AMD code must cite the
!    Copyright, this License, the Availability note, and "Used by permission."
!    Permission to modify the code and to distribute modified code is granted,
!    provided the Copyright, this License, and the Availability note are
!    retained, and a notice that the code was modified is included.  This
!    software was developed with support from the National Science Foundation,
!    and is provided to you free of charge.

!  Availability:

!   http://www.cise.ufl.edu/research/sparse/amd

!  Original paper and source:

!  Algorithm 837: AMD, an approximate minimum degree ordering algorithm
!  Patrick R. Amestoy, Timothy A. Davis and Iain S. Duff.
!  ACM Transactions on Mathematical Software, Volume 30,
!  Issue 3 September 2004 pp 381–388.

!  https://doi.org/10.1145/1024074.1024081

        USE GALAHAD_KINDS_precision
        USE GALAHAD_SYMBOLS
        USE GALAHAD_SPECFILE_precision

        IMPLICIT NONE

        PRIVATE
        PUBLIC :: AMD_initialize, AMD_read_specfile, AMD_order, AMD_main,      &
                  AMD_terminate

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

        TYPE, PUBLIC :: AMD_control_type

!  unit for error messages

          INTEGER ( KIND = ip_ ) :: error = 6

!  unit for warning messages

          INTEGER ( KIND = ip_ ) :: warning = 6

!  unit for monitor output

          INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

          INTEGER ( KIND = ip_ ) :: print_level = 0

!  fraction of extra storage used as workspace relative to that needed to store
!  the whole (lower and upper triangular minus diagonal) of the input matrix

          REAL ( KIND = rp_ ):: expansion = 1.2_rp_

!  use aggressive absorption to tighten the bound on the degree?

          LOGICAL :: aggressive = .TRUE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

          CHARACTER ( LEN = 30 ) :: prefix = '""                            '

        END TYPE AMD_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

        TYPE, PUBLIC :: AMD_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  faulty data faulty (%n < 1, etc)

          INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

          INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the number of workspace compressions required (ideally 0, otherwise
!  consider increasing control%expansion)

          INTEGER ( KIND = ip_ ) :: compresses = - 1

        END TYPE AMD_inform_type

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!   data derived type (empty at present, but included for future extensions)
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        TYPE, PUBLIC :: AMD_data_type
          PRIVATE
        END TYPE AMD_data_type

      CONTAINS

!-*-*-*-*-*-   A M D _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

        SUBROUTINE AMD_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for AMD. This routine should be called before AMD_order
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        TYPE ( AMD_data_type ), INTENT( INOUT ) :: data
        TYPE ( AMD_control_type ), INTENT( OUT ) :: control
        TYPE ( AMD_inform_type ), INTENT( OUT ) :: inform

        inform%status = GALAHAD_ok

        RETURN

!  End of AMD_initialize

        END SUBROUTINE AMD_initialize

!-*-*-*-*-   S H A _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

        SUBROUTINE AMD_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by AMD_initialize could (roughly)
!  have been set as:

! BEGIN AMD SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  print-level                                     0
!  workspace-expansion-factor                      1.2
!  aggressive-absorption                           .TRUE.
! END AMD SPECIFICATIONS

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

        TYPE ( AMD_control_type ), INTENT( INOUT ) :: control
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
        CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

        INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
        INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
        INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
        INTEGER ( KIND = ip_ ), PARAMETER :: expansion = print_level + 1
        INTEGER ( KIND = ip_ ), PARAMETER :: aggressive = expansion + 1
        INTEGER ( KIND = ip_ ), PARAMETER :: prefix = aggressive + 1
        INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
        CHARACTER( LEN = 4 ), PARAMETER :: specname = 'AMD '
        TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

        spec%keyword = ''

!  Integer key-words

        spec( error )%keyword = 'error-printout-device'
        spec( out )%keyword = 'printout-device'
        spec( print_level )%keyword = 'print-level'

!  Real key-words

        spec( expansion )%keyword = 'workspace-expansion-factor'

!  Logical key-words

        spec( aggressive )%keyword = 'aggressive-absorption'

!  Character key-words

        spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

        IF ( PRESENT( alt_specname ) ) THEN
          CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
        ELSE
          CALL SPECFILE_read( device, specname, spec, lspec, control%error )
        END IF

!  Interpret the result

!  Set integer values

        CALL SPECFILE_assign_value( spec( error ),                             &
                                    control%error,                             &
                                    control%error )
        CALL SPECFILE_assign_value( spec( out ),                               &
                                    control%out,                               &
                                    control%error )
        CALL SPECFILE_assign_value( spec( print_level ),                       &
                                    control%print_level,                       &
                                    control%error )

!  Set real values

        CALL SPECFILE_assign_value( spec( expansion ),                         &
                                    control%expansion,                         &
                                    control%error )

!  Set logical values

        CALL SPECFILE_assign_value( spec( aggressive ),                        &
                                    control%aggressive,                        &
                                    control%error )

!  Set character values

        CALL SPECFILE_assign_value( spec( prefix ),                            &
                                    control%prefix,                            &
                                    control%error )

        RETURN

!  End of subroutine AMD_read_specfile

        END SUBROUTINE AMD_read_specfile

!-*-*-*-*-*-*-*-*-   A M D _ o r d e r   S U B R O U T I N E   -*-*-*-*-*-*-*-

        SUBROUTINE AMD_order( n, PTR, ROW, PERM, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  compute a symmetric permutation of the rows and columns of a given
!  matrix A to reduce the fill in when using symmetric Gaussian elimination.
!  This is based on the approximate minimum degree heuristic of Amestoy,
!  Davis and Duff, see below
!
!  Arguments:
!
!  n        the dimension of the matrix A
!  PTR      integer array of length n+1, whose i-th entry gives the postition
!           in ROW of the first nonzero in column of ONE TRIANGULAR
!           PART (lower or uper) of the matrix A for i = 1, .., n. PTR(n+1)
!           gives the postition of the last entry in column n plus 1
!  ROW      integer array of length PTR(n+1)-1, whose entries ROW(j),
!           j = PTR(i), .. , PTR(i+1)-1 give the indices of the row entries
!           in the ONE TRIANGULAR PART of the matrix A for i = 1, .., n.
!           The entries withinin each column may appear in any order
!  PERM     integer array of length n that on output holds the symmetric
!           permutation, that is, if i = PERM(j), i is the j-th row
!           in the permuted matrix, PAP^T.
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


!  A, (excluding the diagonal) perform an approximate minimum

        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
        INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ), INTENT( IN ) :: PTR
        INTEGER ( KIND = ip_ ), DIMENSION( ptr( n + 1 ) - 1 ),                 &
                                   INTENT( IN ) :: ROW
        INTEGER ( KIND = ip_ ), DIMENSION( n ), INTENT( OUT ) :: PERM
        TYPE ( AMD_data_type ), INTENT( INOUT ) :: data
        TYPE ( AMD_control_type ), INTENT( IN ) :: control
        TYPE ( AMD_inform_type ), INTENT( OUT ) :: inform

!  local variables

        INTEGER ( KIND = ip_ ) :: len_whole, pfree, nnz
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PTR_whole
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ROW_whole
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: LEN, NEXT, HEAD
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ELEN, DEGREE
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: NV, W

!  allocate space to hold the structure of the whole of the matrix

        ALLOCATE( PTR_whole( n + 1 ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) GO TO 900

!  build the structure of the whole of the matrix

        CALL AMD_build_whole_matrix( n, PTR, ROW, PTR_whole,                   &
                                     ROW_whole, control%expansion,             &
                                     nnz, len_whole, inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) GO TO 900

!  allocate space required fto find the AMD permutation

        ALLOCATE( LEN( n ), NV( n ), ELEN( n ), DEGREE( n ), NEXT( n ),        &
                  HEAD( n ), W( n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) GO TO 900

!  set up input date

        LEN = PTR_whole( 2 : n + 1 ) - PTR_whole( 1 : n )
        pfree = nnz + 1

!  find the AMD permutation

        CALL AMD_main( control%aggressive, n, PTR_whole, ROW_whole, LEN,       &
                       len_whole, pfree, NV, NEXT, PERM, HEAD, ELEN, DEGREE,   &
                       inform%compresses, W )

!  deallocate all workspace

        DEALLOCATE( PTR_whole, ROW_whole, LEN, NV, NEXT, HEAD, ELEN, DEGREE,   &
                    W, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) GO TO 910

!  successful return

        inform%status = GALAHAD_ok
        RETURN

!  allocation failure

  900   CONTINUE
        inform%status = GALAHAD_error_allocate
        RETURN

!  deallocation failure

  910   CONTINUE
        inform%status = GALAHAD_error_deallocate
        RETURN

!  End of AMD_initialize

        END SUBROUTINE AMD_order

!-*-*-*-*-*-*-   A M D _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

        SUBROUTINE AMD_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Destroy control data for AMD. This routine should be called after AMD_order
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        TYPE ( AMD_data_type ), INTENT( INOUT ) :: data
        TYPE ( AMD_control_type ), INTENT( OUT ) :: control
        TYPE ( AMD_inform_type ), INTENT( OUT ) :: inform

        inform%status = GALAHAD_ok

        RETURN

!  End of AMD_terminate

        END SUBROUTINE AMD_terminate

!-*-*-*-*-*-*-*-*-*-   A M D _ m a i n   S U B R O U T I N E   -*-*-*-*-*-*-*-

        SUBROUTINE AMD_main( aggressive, n, PE, IW, LEN, iwlen, PFREE,         &
                             NV, NEXT, LAST, HEAD, ELEN, DEGREE, ncmpa, W )
        LOGICAL, INTENT( IN ) :: aggressive
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, iwlen
        INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: pfree
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ncmpa
        INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: PE( n ), LEN( n )
        INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: IW( iwlen )
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: NV( n ), ELEN( n )
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: LAST( n ), DEGREE( n )
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: NEXT( n ), HEAD( n ), W( n )

!  Given a representation of the nonzero pattern of a symmetric matrix,
!  A, (excluding the diagonal) perform an approximate minimum
!  (UMFPACK/MA38-style) degree ordering to compute a pivot order
!  such that the introduction of nonzeros (fill-in) in the Cholesky
!  factors A = LL^T are kept low.  At each step, the pivot
!  selected is the one with the minimum UMFPACK/MA38-style
!  upper-bound on the external degree.
!
!  Aggresive absorption is optionally used

! **********************************************************************
! ***** CAUTION:  ARGUMENTS ARE NOT CHECKED FOR ERRORS ON INPUT.  ******
! **********************************************************************

!  References:
!
!  [1] Timothy A. Davis and Iain Duff, "An unsymmetric-pattern
!      multifrontal method for sparse LU factorization", SIAM J.
!      Matrix Analysis and Applications, vol. 18, no. 1, pp.
!      140-158.  Discusses UMFPACK / MA38, which first introduced
!      the approximate minimum degree used by this routine.
!
!  [2] Patrick Amestoy, Timothy A. Davis, and Iain S. Duff, "An
!      approximate degree ordering algorithm," SIAM J. Matrix
!      Analysis and Applications, vol. 17, no. 4, pp. 886-905,
!      1996.  Discusses AMD, AMDBAR, and MC47B.
!
!  [3] Alan George and Joseph Liu, "The evolution of the minimum
!      degree ordering algorithm," SIAM Review, vol. 31, no. 1,
!      pp. 1-19, 1989.  We list below the features mentioned in
!      that paper that this code includes:
!
!  mass elimination:
!          Yes.  MA27 relied on supervariable detection for mass
!          elimination.
!  indistinguishable nodes:
!          Yes (we call these "supervariables").  This was also in
!          the MA27 code - although we modified the method of
!          detecting them (the previous hash was the true degree,
!          which we no longer keep track of).  A supervariable is
!          a set of rows with identical nonzero pattern.  All
!          variables in a supervariable are eliminated together.
!          Each supervariable has as its numerical name that of
!          one of its variables (its principal variable).
!  quotient graph representation:
!          Yes.  We use the term "element" for the cliques formed
!          during elimination.  This was also in the MA27 code.
!          The algorithm can operate in place, but it will work
!          more efficiently if given some "elbow room."
!  element absorption:
!          Yes.  This was also in the MA27 code.
!  external degree:
!          Yes.  The MA27 code was based on the true degree.
!  incomplete degree update and multiple elimination:
!          No.  This was not in MA27, either.  Our method of
!          degree update within MC47B/BD is element-based, not
!          variable-based.  It is thus not well-suited for use
!          with incomplete degree update or multiple elimination.

!-----------------------------------------------------------------------
! Authors, and Copyright (C) 1995 by:
!  Timothy A. Davis, Patrick Amestoy, Iain S. Duff, & John K. Reid.
!
! Acknowledgements:
!  This work (and the UMFPACK package) was supported by the
!  National Science Foundation (ASC-9111263 and DMS-9223088).
!  The UMFPACK/MA38 approximate degree update algorithm, the
!  unsymmetric analog which forms the basis of MC47B/BD, was
!  developed while Tim Davis was supported by CERFACS (Toulouse,
!  France) in a post-doctoral position.
!
! Date:  September, 1995
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
! INPUT ARGUMENTS (unaltered):
!-----------------------------------------------------------------------

! aggressive: use aggressive absorption (.TRUE.) or not (.FALSE.).
!
! n: The matrix order.
!
!  Restriction:  1 <= n .lt. (iovflo/2)-2, where iovflo is
!  the largest positive integer that your computer can represent.

! iwlen: The length of iw (1..iwlen).  On input, the matrix is
!  stored in iw (1..pfree-1).  However, iw (1..iwlen) should be
!  slightly larger than what is required to hold the matrix, at
!  least iwlen >= pfree + n is recommended.  Otherwise,
!  excessive compressions will take place.
!  *** We do not recommend running this algorithm with ***
!  ***      iwlen .lt. pfree + n.                      ***
!  *** Better performance will be obtained if          ***
!  ***      iwlen >= pfree + n                         ***
!  *** or better yet                                   ***
!  ***      iwlen .gt. 1.2 * pfree                     ***
!  *** (where pfree is its value on input).            ***
!  The algorithm will not run at all if iwlen .lt. pfree-1.
!
!  Restriction: iwlen >= pfree-1

!-----------------------------------------------------------------------
! INPUT/OUPUT ARGUMENTS:
!-----------------------------------------------------------------------

! pe: On input, pe (i) is the index in iw of the start of row i, or
!  zero if row i has no off-diagonal non-zeros.
!
!  During execution, it is used for both supervariables and
!  elements:
!
!  * Principal supervariable i:  index into iw of the
!          description of supervariable i.  A supervariable
!          represents one or more rows of the matrix
!          with identical nonzero pattern.
!  * Non-principal supervariable i:  if i has been absorbed
!          into another supervariable j, then pe (i) = -j.
!          That is, j has the same pattern as i.
!          Note that j might later be absorbed into another
!          supervariable j2, in which case pe (i) is still -j,
!          and pe (j) = -j2.
!  * Unabsorbed element e:  the index into iw of the description
!          of element e, if e has not yet been absorbed by a
!          subsequent element.  Element e is created when
!          the supervariable of the same name is selected as
!          the pivot.
!  * Absorbed element e:  if element e is absorbed into element
!          e2, then pe (e) = -e2.  This occurs when the pattern of
!          e (that is, Le) is found to be a subset of the pattern
!          of e2 (that is, Le2).  If element e is "null" (it has
!          no nonzeros outside its pivot block), then pe (e) = 0.
!
!  On output, pe holds the assembly tree/forest, which implicitly
!  represents a pivot order with identical fill-in as the actual
!  order (via a depth-first search of the tree).
!
!  On output:
!  If nv (i) .gt. 0, then i represents a node in the assembly tree,
!  and the parent of i is -pe (i), or zero if i is a root.
!  If nv (i) = 0, then (i,-pe (i)) represents an edge in a
!  subtree, the root of which is a node in the assembly tree.

! pfree: On input the tail end of the array, iw (pfree..iwlen),
!  is empty, and the matrix is stored in iw (1..pfree-1).
!  During execution, additional data is placed in iw, and pfree
!  is modified so that iw (pfree..iwlen) is always the unused part
!  of iw.  On output, pfree is set equal to the size of iw that
!  would have been needed for no compressions to occur.  If
!  ncmpa is zero, then pfree (on output) is less than or equal to
!  iwlen, and the space iw (pfree+1 ... iwlen) was not used.
!  Otherwise, pfree (on output) is greater than iwlen, and all the
!  memory in iw was used.

!-----------------------------------------------------------------------
! INPUT/MODIFIED (undefined on output):
!-----------------------------------------------------------------------

! len: On input, len (i) holds the number of entries in row i of the
!  matrix, excluding the diagonal.  The contents of len (1..n)
!  are undefined on output.

! iw: On input, iw (1..pfree-1) holds the description of each row i
!  in the matrix.  The matrix must be symmetric, and both upper
!  and lower triangular parts must be present.  The diagonal must
!  not be present.  Row i is held as follows:
!
!          len (i):  the length of the row i data structure
!          iw (pe (i) ... pe (i) + len (i) - 1):
!                  the list of column indices for nonzeros
!                  in row i (simple supervariables), excluding
!                  the diagonal.  All supervariables start with
!                  one row/column each (supervariable i is just
!                  row i).
!          if len (i) is zero on input, then pe (i) is ignored
!          on input.
!
!          Note that the rows need not be in any particular order,
!          and there may be empty space between the rows.
!
!  During execution, the supervariable i experiences fill-in.
!  This is represented by placing in i a list of the elements
!  that cause fill-in in supervariable i:
!
!          len (i):  the length of supervariable i
!          iw (pe (i) ... pe (i) + elen (i) - 1):
!                  the list of elements that contain i.  This list
!                  is kept short by removing absorbed elements.
!          iw (pe (i) + elen (i) ... pe (i) + len (i) - 1):
!                  the list of supervariables in i.  This list
!                  is kept short by removing nonprincipal
!                  variables, and any entry j that is also
!                  contained in at least one of the elements
!                  (j in Le) in the list for i (e in row i).
!
!  When supervariable i is selected as pivot, we create an
!  element e of the same name (e=i):
!
!          len (e):  the length of element e
!          iw (pe (e) ... pe (e) + len (e) - 1):
!                  the list of supervariables in element e.
!
!  An element represents the fill-in that occurs when supervariable
!  i is selected as pivot (which represents the selection of row i
!  and all non-principal variables whose principal variable is i).
!  We use the term Le to denote the set of all supervariables
!  in element e.  Absorbed supervariables and elements are pruned
!  from these lists when computationally convenient.
!
!  CAUTION:  THE INPUT MATRIX IS OVERWRITTEN DURING COMPUTATION.
!  The contents of iw are undefined on output.

!-----------------------------------------------------------------------
! OUTPUT (need not be set on input):
!-----------------------------------------------------------------------

! nv:  During execution, abs (nv (i)) is equal to the number of rows
!  that are represented by the principal supervariable i.  If i is
!  a nonprincipal variable, then nv (i) = 0.  Initially,
!  nv (i) = 1 for all i.  nv (i) .lt. 0 signifies that i is a
!  principal variable in the pattern Lme of the current pivot
!  element me.  On output, nv (e) holds the true degree of element
!  e at the time it was created (including the diagonal part).

! ncmpa: The number of times iw was compressed.  If this is
!  excessive, then the execution took longer than what could have
!  been.  To reduce ncmpa, try increasing iwlen to be 10% or 20%
!  larger than the value of pfree on input (or at least
!  iwlen >= pfree + n).  The fastest performance will be
!  obtained when ncmpa is returned as zero.  If iwlen is set to
!  the value returned by pfree on *output*, then no compressions
!  will occur.

! elen: See the description of iw above.  At the start of execution,
!  elen (i) is set to zero.  During execution, elen (i) is the
!  number of elements in the list for supervariable i.  When e
!  becomes an element, elen (e) = -nel is set, where nel is the
!  current step of factorization.  elen (i) = 0 is done when i
!  becomes nonprincipal.
!
!  For variables, elen (i) >= 0 holds until just before the
!  permutation vectors are computed.  For elements,
!  elen (e) .lt. 0 holds.
!
!  On output elen (1..n) holds the inverse permutation (the same
!  as the 'INVP' argument in Sparspak).  That is, if k = elen (i),
!  then row i is the kth pivot row.  Row i of A appears as the
!  (elen(i))-th row in the permuted matrix, PAP^T.

! last: In a degree list, last (i) is the supervariable preceding i,
!  or zero if i is the head of the list.  In a hash bucket,
!  last (i) is the hash key for i.  last (head (hash)) is also
!  used as the head of a hash bucket if head (hash) contains a
!  degree list (see head, below).
!
!  On output, last (1..n) holds the permutation (the same as the
!  'PERM' argument in Sparspak).  That is, if i = last (k), then
!  row i is the kth pivot row.  Row last (k) of A is the k-th row
!  in the permuted matrix, PAP^T.

!-----------------------------------------------------------------------
! LOCAL (not input or output - used only during execution):
!-----------------------------------------------------------------------

! degree: If i is a supervariable, then degree (i) holds the
!  current approximation of the external degree of row i (an upper
!  bound).  The external degree is the number of nonzeros in row i,
!  minus abs (nv (i)) (the diagonal part).  The bound is equal to
!  the external degree if elen (i) is less than or equal to two.
!
!  We also use the term "external degree" for elements e to refer
!  to |Le \ Lme|.  If e is an element, then degree (e) holds |Le|,
!  which is the degree of the off-diagonal part of the element e
!  (not including the diagonal part).

! head: head is used for degree lists.  head (deg) is the first
!  supervariable in a degree list (all supervariables i in a
!  degree list deg have the same approximate degree, namely,
!  deg = degree (i)).  If the list deg is empty then
!  head (deg) = 0.
!
!  During supervariable detection head (hash) also serves as a
!  pointer to a hash bucket.
!  If head (hash) .gt. 0, there is a degree list of degree hash.
!          The hash bucket head pointer is last (head (hash)).
!  If head (hash) = 0, then the degree list and hash bucket are
!          both empty.
!  If head (hash) .lt. 0, then the degree list is empty, and
!          -head (hash) is the head of the hash bucket.
!  After supervariable detection is complete, all hash buckets
!  are empty, and the (last (head (hash)) = 0) condition is
!  restored for the non-empty degree lists.

! next: next (i) is the supervariable following i in a link list, or
!  zero if i is the last in the list.  Used for two kinds of
!  lists:  degree lists and hash buckets (a supervariable can be
!  in only one kind of list at a time).

! w: The flag array w determines the status of elements and
!  variables, and the external degree of elements.
!
!  for elements:
!     if w (e) = 0, then the element e is absorbed
!     if w (e) >= wflg, then w (e) - wflg is the size of
!          the set |Le \ Lme|, in terms of nonzeros (the
!          sum of abs (nv (i)) for each principal variable i that
!          is both in the pattern of element e and NOT in the
!          pattern of the current pivot element, me).
!     if wflg .gt. w (e) .gt. 0, then e is not absorbed and has
!          not yet been seen in the scan of the element lists in
!          the computation of |Le\Lme| in loop in Scan 1 below.
!
!  for variables:
!     during supervariable detection, if w (j) .ne. wflg then j is
!     not in the pattern of variable i
!
!  The w array is initialized by setting w (i) = 1 for all i,
!  and by setting wflg = 2.  It is reinitialized if wflg becomes
!  too large (to ensure that wflg+n does not cause integer
!  overflow).

!-----------------------------------------------------------------------
! LOCAL INTEGERS:
!-----------------------------------------------------------------------

        INTEGER ( KIND = ip_ ) :: deg, degme, dext, dmax, e, elenme, eln
        INTEGER ( KIND = ip_ ) :: hash, hmod, i, ilast, inext, j, jlast
        INTEGER ( KIND = ip_ ) :: jnext, k, knt1, knt2, knt3, lenj, ln, maxmem
        INTEGER ( KIND = ip_ ) :: me, mem, mindeg, nel, newmem, nleft
        INTEGER ( KIND = ip_ ) :: nvi, nvj, nvpiv, slenme, we, wflg, wnvi, x
        LOGICAL :: nothing

! deg:          the degree of a variable or element
! degme:        size, |Lme|, of the current element, me (= degree (me))
! dext:         external degree, |Le \ Lme|, of some element e
! dmax:         largest |Le| seen so far
! e:            an element
! elenme:       the length, elen (me), of element list of pivotal var.
! eln:          the length, elen (...), of an element list
! hash:         the computed value of the hash function
! hmod:         the hash function is computed modulo hmod = max (1,n-1)
! i:            a supervariable
! ilast:        the entry in a link list preceding i
! inext:        the entry in a link list following i
! j:            a supervariable
! jlast:        the entry in a link list preceding j
! jnext:        the entry in a link list, or path, following j
! k:            the pivot order of an element or variable
! knt1:         loop counter used during element construction
! knt2:         loop counter used during element construction
! knt3:         loop counter used during compression
! lenj:         len (j)
! ln:           length of a supervariable list
! maxmem:       amount of memory needed for no compressions
! me:           current supervariable being eliminated, and the
!                  current element created by eliminating that
!                  supervariable
! mem:          memory in use assuming no compressions have occurred
! mindeg:       current minimum degree
! nel:          number of pivots selected so far
! newmem:       amount of new memory needed for current pivot element
! nleft:        n - nel, the number of nonpivotal rows/columns remaining
! nvi:          the number of variables in a supervariable i (= nv (i))
! nvj:          the number of variables in a supervariable j (= nv (j))
! nvpiv:        number of pivots in current element
! slenme:       number of variables in variable list of pivotal variable
! we:           w (e)
! wflg:         used for flagging the w array.  See description of iw.
! wnvi:         wflg - nv (i)
! x:            either a supervariable or an element

!-----------------------------------------------------------------------
! LOCAL POINTERS:
!-----------------------------------------------------------------------

        INTEGER ( KIND = ip_ ) :: p, p1, p2, p3, pdst, pend, pj
        INTEGER ( KIND = ip_ ) :: pme, pme1, pme2, pn, psrc

!          Any parameter (pe (...) or pfree) or local variable
!          starting with "p" (for Pointer) is an index into iw,
!          and all indices into iw use variables starting with
!          "p."  The only exception to this rule is the iwlen
!          input argument.

! p:            pointer into lots of things
! p1:           pe (i) for some variable i (start of element list)
! p2:           pe (i) + elen (i) -  1 for some var. i (end of el. list)
! p3:           index of first supervariable in clean list
! pdst:         destination pointer, for compression
! pend:         end of memory to compress
! pj:           pointer into an element or variable
! pme:          pointer into the current element (pme1...pme2)
! pme1:         the current element, me, is stored in iw (pme1...pme2)
! pme2:         the end of the current element
! pn:           pointer into a "clean" variable, also used to compress
! psrc:         source pointer, for compression

!=======================================================================
!  INITIALIZATIONS
!=======================================================================

!  scalars

        wflg = 2
        mindeg = 1
        ncmpa = 0
        nel = 0
        hmod = MAX( 1, n - 1 )
        dmax = 0
        mem = pfree - 1
        maxmem = mem
        me = 0

!  arrays

        LAST = 0
        HEAD = 0
        NV = 1
        W = 1
        ELEN = 0
        DEGREE = LEN

!  initialize degree lists and eliminate rows with no off-diagonal nomzeros

        DO i = 1, n
          deg = DEGREE( i )

!  place i in the degree list corresponding to its degree

          IF ( deg > 0 ) THEN
            inext = HEAD( deg )
            IF ( inext /= 0 ) LAST( inext ) = i
            NEXT( i ) = inext
            HEAD( deg ) = i

!  we have a variable that can be eliminated at once because there is no
!  off-diagonal non-zero in its row

          ELSE
            nel = nel + 1
            ELEN( i ) = - nel
            PE( i ) = 0
            W( i ) = 0
          END IF
        END DO

!=======================================================================
!  pivot selection loop
!=======================================================================

        DO
          IF ( nel >= n ) EXIT

!=======================================================================
!  GET PIVOT OF MINIMUM DEGREE
!=======================================================================

!  find next supervariable for elimination

           DO deg = mindeg, n
             me = HEAD( deg )
             IF ( me > 0 ) EXIT
           END DO
           mindeg = deg

!  remove chosen variable from link list

           inext = NEXT( me )
           IF ( inext /= 0 ) LAST( inext ) = 0
           HEAD( deg ) = inext

!  me represents the elimination of pivots nel+1 to nel+nv(me ). Place me
!  itself as the first in this set. It will be moved to the nel+nv(me)
!  position when the permutation vectors are computed

           elenme = ELEN( me )
           ELEN( me ) = - ( nel + 1 )
           nvpiv = NV( me )
           nel = nel + nvpiv

!=======================================================================
!  CONSTRUCT NEW ELEMENT
!=======================================================================

!  At this point, me is the pivotal supervariable.  It will be converted into
!  the current element.  Scan list of the pivotal supervariable, me, setting
!  tree pointers and constructing new list of supervariables for the new
!  element,  me. p is a pointer to the current position in the old list.

!  flag the variable "me" as being in Lme by negating nv(me)

           NV( me ) = - nvpiv
           degme = 0

!  construct the new element in place

           IF ( elenme == 0 ) THEN
             pme1 = PE( me )
             pme2 = pme1 - 1
             DO p = pme1, pme1 + LEN( me ) - 1
               i = IW( p )
               nvi = NV( i )

!  i is a principal variable not yet placed in Lme.
!  store i in new list

               IF ( nvi > 0 ) THEN
                 degme = degme + nvi

!  flag i as being in Lme by negating nv( i )

                 NV( i ) = - nvi
                 pme2 = pme2 + 1
                 IW( pme2 ) = i

!  remove variable i from degree list.

                 ilast = LAST( i )
                 inext = NEXT( i )
                 IF ( inext /= 0 ) LAST( inext ) = ilast
                 IF ( ilast /= 0 ) THEN
                    NEXT( ilast ) = inext

!  i is at the head of the degree list

                 ELSE
                   HEAD( DEGREE( i ) ) = inext
                 END IF
               END IF
             END DO

!  this element takes no new memory in iw

             newmem = 0

!  construct the new element in empty space, iw(pfree ...)

           ELSE
             p = PE( me )
             pme1 = pfree
             slenme = LEN( me ) - elenme

             DO knt1 = 1, elenme + 1

!  search the supervariables in me

               IF ( knt1 > elenme ) THEN
                 e = me
                 pj = p
                 ln = slenme

!  search the elements in me

               ELSE
                 e = IW( p )
                 p = p + 1
                 pj = PE( e )
                 ln = LEN( e )
               END IF

!  search for different supervariables and add them to the new list,
!  compressing when necessary. this loop is executed once for each element
!  in the list and once for all the supervariables in the list

               DO knt2 = 1, ln
                 i = IW( pj )
                 pj = pj + 1
                 nvi = NV( i )

!  compress iw, if necessary

                 IF ( nvi > 0 ) THEN

!  prepare for compressing iw by adjusting  pointers and lengths so that
!  the lists being searched in the inner and outer loops contain only the
!  remaining entries

                   IF ( pfree > iwlen ) THEN
                     PE( me ) = p
                     LEN( me ) = LEN( me ) - knt1

!  nothing left of supervariable me

                     IF ( LEN( me ) == 0 ) PE( me ) = 0
                     PE( e ) = pj
                     LEN( e ) = ln - knt2

!  nothing left of element e

                     IF ( LEN( e ) == 0 ) PE( e ) = 0
                     ncmpa = ncmpa + 1

!  store first item in pe, and  set first entry to -item

                     DO j = 1, n
                       pn = PE( j )
                       IF ( pn > 0 ) THEN
                         PE( j ) = IW( pn )
                         IW( pn ) = - j
                       END IF
                     END DO

!  psrc/pdst point to source/destination

                     pdst = 1
                     psrc = 1
                     pend = pme1 - 1

!  while loop:  search for next negative entry

                     DO
                       IF ( psrc > pend ) EXIT
                       j = - IW( psrc )
                       psrc = psrc + 1
                       IF ( j > 0 ) THEN
                         IW( pdst ) = PE( j )
                         PE( j ) = pdst
                         pdst = pdst + 1

!  copy from source to destination

                          lenj = LEN( j )
                          DO knt3 = 0, lenj - 2
                            IW( pdst + knt3 ) = IW( psrc + knt3 )
                          END DO
                          pdst = pdst + lenj - 1
                          psrc = psrc + lenj - 1
                       END IF
                     END DO

!  move the new partially-constructed element

                     p1 = pdst
                     DO psrc = pme1, pfree - 1
                       IW( pdst ) = IW( psrc )
                       pdst = pdst + 1
                     END DO
                     pme1 = p1
                     pfree = pdst
                     pj = PE( e )
                     p = PE( me )
                   END IF

!  i is a principal variable not yet placed in Lme, and store i in new list

                   degme = degme + nvi

!  flag i as being in Lme by negating nv(i)

                   NV( i ) = - nvi
                   IW( pfree ) = i
                   pfree = pfree + 1

!  remove variable i from degree link list

                   ilast = LAST( i )
                   inext = NEXT( i )
                   IF ( inext /= 0 ) LAST( inext ) = ilast
                   IF ( ilast /= 0 ) THEN
                     NEXT( ilast ) = inext

!  i is at the head of the degree list

                   ELSE
                     HEAD( DEGREE( i ) ) = inext
                   END IF
                 END IF
               END DO

!  set tree pointer and flag to indicate element e is
!  absorbed into new element me( the parent of e is me )

               IF ( e /= me ) THEN
                 PE( e ) = - me
                 W( e ) = 0
               END IF
             END DO
             pme2 = pfree - 1

!  this element takes newmem new memory in iw( possibly zero )

             newmem = pfree - pme1
             mem = mem + newmem
             maxmem = MAX( maxmem, mem )
           END IF

!  -------------------------------------------------------------
!  me has now been converted into an element in iw( pme1..pme2 )
!  -------------------------------------------------------------


!  degme holds the external degree of new element

           DEGREE( me ) = degme
           PE( me ) = pme1
           LEN( me ) = pme2 - pme1 + 1

!  make sure that wflg is not too large.  With the current value of wflg,
!  wflg+n must not cause integer overflow

           IF ( wflg + n <= wflg ) THEN
             DO x = 1, n
               IF ( W( x ) /= 0 ) W( x ) = 1
             END DO
             wflg = 2
           END IF

!=======================================================================
!  COMPUTE( w( e ) - wflg ) = |Le\Lme| FOR ALL ELEMENTS
!=======================================================================

!   Scan 1:  compute the external degrees of previous elements  with respect
!   to the current element.  That is: ( w( e ) - wflg ) = |Le \ Lme| for each
!   element e that appears in any supervariable in Lme. The notation Le refers
!   to the pattern( list of supervariables ) of a previous element e, where
!   e is not yet absorbed, stored in iw(pe(e)+1 ... pe(e)+iw(pe(e))). The
!   notation Lme refers to the pattern of the current element (stored in
!   iw( pme1..pme2) ). If (w(e) - wflg) becomes zero, then the element e
!   will be absorbed in Scan 2

           DO pme = pme1, pme2
             i = IW( pme )
             eln = ELEN( i )

!  note that nv(i) has been negated to denote i in Lme

             IF ( eln > 0 ) THEN
               nvi = - NV( i )
               wnvi = wflg - nvi
               DO p = PE( i ), PE( i ) + eln - 1
                 e = IW( p )
                 we = W( e )

!  unabsorbed element e has been seen in this loop

                 IF ( we >= wflg ) THEN
                   we = we - nvi

!  e is an unabsorbed element. Tthis is the first we have seen e in all of
!  Scan 1
                 ELSE IF ( we /= 0 ) THEN
                   we = DEGREE( e ) + wnvi
                 END IF
                 W( e ) = we
               END DO
             END IF
           END DO

!=======================================================================
!  DEGREE UPDATE AND ELEMENT ABSORPTION
!=======================================================================

!  Scan 2:  for each i in Lme, sum up the degree of Lme (which is degme), plus
!  the sum of the external degrees of each Le for the elements e appearing
!  within i, plus the supervariables in i.  Place i in hash list.


           DO pme = pme1, pme2
             i = IW( pme )
             p1 = PE( i )
             p2 = p1 + ELEN( i ) - 1
             pn = p1
             hash = 0
             deg = 0

!  ----------------------------------------------------------
!  scan the element list associated with supervariable i
!  ----------------------------------------------------------

!  aggressive approximate degree

             IF (  aggressive  ) THEN
               DO p = p1, p2
                 e = IW( p )

!  dext = |Le \ Lme|

                 dext = W( e ) - wflg
                 IF ( dext > 0 ) THEN
                   deg = deg + dext
                   IW( pn ) = e
                   pn = pn + 1
                   hash = hash + e

!  aggressive absorption: e is not adjacent to me, but the |Le \ Lme| is 0,
!  so absorb it into me

                 ELSE IF ( dext == 0 ) THEN
                   PE( e ) = -me
                   W( e ) = 0

!  element e has already been absorbed, due to regular absorption, in do loop
!  above. Ignore it

                 ELSE
                 END IF
               END DO

!  UMFPACK/MA38-style approximate degree

             ELSE
               DO p = p1, p2
                 e = IW( p )
                 we = W( e )

!  e is an unabsorbed element

                 IF ( we /= 0 ) THEN
                   deg = deg + we - wflg
                   IW( pn ) = e
                   pn = pn + 1
                   hash = hash + e
                 END IF
               END DO
             END IF

!  count the number of elements in i( including me ):

             ELEN( i ) = pn - p1 + 1

!  scan the supervariables in the list associated with i

             p3 = pn
             DO p = p2 + 1, p1 + LEN( i ) - 1
               j = IW( p )
               nvj = NV( j )

!  j is unabsorbed, and not in Lme. Add to degree and add to new list

               IF ( nvj > 0 ) THEN
                 deg = deg + nvj
                 IW( pn ) = j
                 pn = pn + 1
                 hash = hash + j
               END IF
             END DO

!  update the degree and check for mass elimination

             IF ( aggressive ) THEN
               nothing = deg == 0
             ELSE
               nothing = ELEN( i ) == 1 .AND. p3 == pn
             END IF

!  mass elimination: there is nothing left of this node except for an  edge to
!  the current pivot element.  elen(i) is 1, and there are no variables
!  adjacent to node i. Absorb i into the current pivot element, me

             IF ( nothing ) THEN
               PE( i ) = - me
               nvi = -NV( i )
               degme = degme - nvi
               nvpiv = nvpiv + nvi
               nel = nel + nvi
               NV( i ) = 0
               ELEN( i ) = 0

!  update the upper-bound degree of i

             ELSE

!  the following degree does not yet include the size of the current element,
!  which is added later:

               DEGREE( i ) = MIN( DEGREE( i ), deg )

!  add me to the list for i. Move first supervariable to end of list

               IW( pn ) = IW( p3 )

!  move first element to end of element part of list

               IW( p3 ) = IW( p1 )

!  add new element to front of list.

               IW( p1 ) = me

!  store the new length of the list in len( i )

               LEN( i ) = pn - p1 + 1

!  place in hash bucket.  Save hash key of i in last( i ).

               hash = MOD( hash, hmod ) + 1
               j = HEAD( hash )

!  the degree list is empty, hash head is -j

               IF ( j <= 0 ) THEN
                 NEXT( i ) = - j
                 HEAD( hash ) = - i

!  degree list is not empty. Use last( head( hash ) ) as hash head

               ELSE
                 NEXT( i ) = LAST( j )
                 LAST( j ) = i
               END IF
               LAST( i ) = hash
             END IF
           END DO
           DEGREE( me ) = degme

!  Clear the counter array, w(...), by incrementing wflg.

           dmax = MAX( dmax, degme )
           wflg = wflg + dmax

!  make sure that wflg+n does not cause integer overflow

           IF ( wflg + n <= wflg ) THEN
             DO x = 1, n
               IF ( W( x ) /= 0 ) W( x ) = 1
             END DO
             wflg = 2
           END IF

!  at this point, w( 1..n ) < wflg holds

!=======================================================================
!  SUPERVARIABLE DETECTION
!=======================================================================

           DO pme = pme1, pme2
             i = IW( pme )

!  i is a principal variable in Lme

             IF ( NV( i ) < 0 ) THEN

!  examine all hash buckets with 2 or more variables. Do this by examing all
!  unique hash keys for super-variables in the pattern Lme of the current
!  element, me

               hash = LAST( i )

!  let i = head of hash bucket, and empty the hash bucket

               j = HEAD( hash )
               IF ( j == 0 ) CYCLE

!  degree list is empty

               IF ( j < 0 ) THEN
                 i = - j
                 HEAD( hash ) = 0

!  degree list is not empty, restore last() of head

               ELSE
                 i = LAST( j )
                 LAST( j ) = 0
               END IF
               IF ( i == 0 ) CYCLE
               DO
                 IF ( NEXT( i ) /= 0 ) THEN

!  this bucket has one or more variables following i. Sscan all of them to see
!  if i can absorb any entries that follow i in hash bucket.  Scatter i into w

                   ln = LEN( i )
                   eln = ELEN( i )

!  do not flag the first element in the list( me )

                   DO p = PE( i ) + 1, PE( i ) + ln - 1
                     W( IW( p ) ) = wflg
                   END DO

!  scan every other entry j following i in bucket

                   jlast = i
                   j = NEXT( i )

!  loop over bucket
                   DO

!  check if j and i have identical nonzero pattern

                     IF ( j == 0 ) EXIT

!  i and j do not have same size data structure or number of adjacent elements

                     IF ( LEN( j ) /= ln ) GO TO 240
                     IF ( ELEN( j ) /= eln ) GO TO 240

!  do not flag the first element in the list( me )

                     DO p = PE( j ) + 1, PE( j ) + ln - 1

!  an entry( iw(p) ) is in j but not in i

                       IF ( W( IW( p ) ) /= wflg ) GO TO 240
                     END DO

!  found it!  j can be absorbed into i

                     PE( j ) = - i

!  both nv( i ) and nv( j ) are negated since they
!  are in Lme, and the absolute values of each
!  are the number of variables in i and j:

                     NV( i ) = NV( i ) + NV( j )
                     NV( j ) = 0
                     ELEN( j ) = 0

!  delete j from hash bucket

                     j = NEXT( j )
                     NEXT( jlast ) = j
                     CYCLE

240                  CONTINUE

!  j cannot be absorbed into i

                     jlast = j
                     j = NEXT( j )
                   END DO
                 END IF

!  no more variables can be absorbed into i. Go to next i in bucket and
!  clear flag array

                 wflg = wflg + 1
                 i = NEXT( i )
                 IF ( i == 0 ) EXIT
               END DO
             END IF
           END DO

!=======================================================================
!  RESTORE DEGREE LISTS AND REMOVE NONPRINCIPAL SUPERVAR. FROM ELEMENT
!=======================================================================

           p = pme1
           nleft = n - nel
           DO pme = pme1, pme2
             i = IW( pme )
             nvi = - NV( i )

!  i is a principal variable in Lme. Restore nv( i ) to signify that
!  i is principal

             IF ( nvi > 0 ) THEN
               NV( i ) = nvi

!  compute the external degree (add size of current elememt)

               IF (  aggressive  ) THEN
                 deg = MIN( DEGREE( I ) + degme - nvi, nleft - nvi )
               ELSE
                 deg = MAX( 1, MIN( DEGREE( i ) + degme - nvi, nleft - nvi ) )
               END IF

!  place the supervariable at the head of the degree list

               inext = HEAD( deg )
               IF ( inext /= 0 ) LAST( inext ) = i
               NEXT( i ) = inext
               LAST( i ) = 0
               HEAD( deg ) = i

!  save the new degree, and find the minimum degree

               mindeg = MIN( mindeg, deg )
               DEGREE( i ) = deg

!  place the supervariable in the element pattern

               IW( p ) = i
               p = p + 1
             END IF
           END DO

!=======================================================================
!  FINALIZE THE NEW ELEMENT
!=======================================================================

           NV( me ) = nvpiv + degme

!     nv( me ) is now the degree of pivot( including diagonal part )
!     save the length of the list for the new element me

           LEN( me ) = p - pme1

!    there is nothing left of the current pivot element

           IF ( LEN( me ) == 0 ) THEN
             PE( me ) = 0
             W( me ) = 0
           END IF

!    element was not constructed in place: deallocate part
!    of it( final size is less than or equal to newmem,
!    since newly nonprincipal variables have been removed ).

           IF ( newmem /= 0 ) THEN
             pfree = p
             mem = mem - newmem + LEN( me )
           END IF

!  end of pivot selection loop

        END DO

!=======================================================================
!  COMPUTE THE PERMUTATION VECTORS
!=======================================================================

!  The time taken by the following code is O(n). At this point,
!  elen(e) = -k has been done for all elements e, and elen(i) = 0
!  has been done for all nonprincipal variables i. There are no
!  principal supervariables left, and all elements are absorbed

!  compute the ordering of unordered nonprincipal variables

        DO i = 1, n
          IF ( ELEN( i ) == 0 ) THEN

!  i is an un-ordered row.  Traverse the tree from i until reaching an
!  element, e.  The element, e, was the principal supervariable of i
!  and all nodes in the path from i to when e was selected as pivot.

            j = - PE( i )

!    while( j is a variable ) do

            DO
              IF ( ELEN( j ) < 0 ) EXIT
              j = - PE( j )
            END DO
            e = j

!  get the current pivot ordering of e

            k = - ELEN( e )

!  traverse the path again from i to e, and compress the path (all nodes
!  point to e).  Path compression allows this code to compute in O(n) time.
!  Order the unordered nodes in the path, and place the element e at the end

            j = i

!  while( j is a variable ) do:

            DO
              IF ( ELEN( J ) < 0 ) EXIT
              jnext = - PE( j )
              PE( j ) = -e

!  j is an unordered row

              IF ( ELEN( j ) == 0 ) THEN
                 ELEN( j ) = k
                 k = k + 1
              END IF
              j = jnext
            END DO

!  leave elen(e) negative, so we know it is an element

            ELEN( e ) = - k
          END IF
        END DO

!  ----------------------------------------------------------------
!  reset the inverse permutation( elen( 1..n ) ) to be positive,
!  and compute the permutation( last( 1..n ) ).
!  ----------------------------------------------------------------

        DO i = 1, n
          k = ABS( ELEN( i ) )
          LAST( k ) = i
          ELEN( i ) = k
        END DO

!=======================================================================
!  RETURN THE MEMORY USAGE IN IW
!=======================================================================

!  If maxmem is less than or equal to iwlen, then no compressions
!  occurred, and iw( maxmem+1 ... iwlen ) was unused.  Otherwise
!  compressions did occur, and iwlen would have had to have been
!  greater than or equal to maxmem for no compressions to occur.
!  Return the value of maxmem in the pfree argument.

        pfree = maxmem

        RETURN

!  end of subroutine AMD_main

        END SUBROUTINE AMD_main

!  Expand a given symmetric matrix whose lower triangle is specified in
!  CSC format to one in which the whole (lower and upper triangular) CSC
!  form without its diagonal is specified

        SUBROUTINE AMD_build_whole_matrix( n, PTR, ROW, PTR_whole, ROW_whole,  &
                                           expansion, nnz, len_whole, status )

!  dummy arguments

        INTEGER ( KIND = ip_ ), INTENT( IN  ) :: n
        INTEGER ( KIND = ip_ ), INTENT( OUT  ) :: nnz, len_whole, status
        REAL ( KIND = rp_ ), INTENT( IN ) :: expansion
        INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ), INTENT( IN ) :: PTR
        INTEGER ( KIND = ip_ ), DIMENSION( ptr( n + 1 ) - 1 ),                &
                                  INTENT( IN ) :: ROW
        INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ), INTENT( OUT ) :: PTR_whole
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ),                  &
                                  INTENT( OUT ) :: ROW_whole

!  local variables

        INTEGER ( KIND = ip_ ) :: i, j, k

!  compute the numbers of nonzeros in column j of the whole matrix

        PTR_whole( 1 : n + 1 ) = 0
        DO j = 1, n
          DO k = PTR( j ), PTR( j + 1 ) - 1
            i = row( k )
            IF ( i /= j ) THEN
              PTR_whole( i ) = PTR_whole( i ) + 1
              PTR_whole( j ) = PTR_whole( j ) + 1
            END IF
          END DO
        END DO
        nnz = SUM( PTR_whole( 1 : n ) )

!  compute pointers to the ends of each column in the whole matrix

        PTR_whole( 1 ) = PTR_whole( 1 ) + 1
        DO j = 2, n
          PTR_whole( j ) = PTR_whole( j ) + PTR_whole( j - 1 )
        END DO
        nnz = PTR_whole( n ) - 1
        PTR_whole( n + 1 ) = nnz + 1

!  allocate space for the whole matrix, providing extra expansion room

        len_whole = MAX( nnz + n, INT( REAL( nnz, KIND = rp_ ) * expansion ) )
        ALLOCATE( ROW_whole( len_whole ), STAT = status )
        IF ( status /= 0 ) RETURN

!  fill in the row indices in each column  of the whole matrix

        DO j = 1, n
          DO k = PTR( j ), PTR( j + 1 ) - 1
            i = ROW( k )
            IF ( i /= j ) THEN
              PTR_whole( i ) = PTR_whole( i ) - 1
              PTR_whole( j ) = PTR_whole( j ) - 1
              ROW_whole( PTR_whole( i ) ) = j
              ROW_whole( PTR_whole( j ) ) = i
            END IF
          END DO
        END DO

!       DO j = 1, n - 1
!         PTR_whole( j ) = PTR_whole( j + 1 )
!       END DO
!        PTR_whole( n ) = nnz


!        ROW_whole( 1 ) = PTR_whole( 1 )
!        PTR_whole( 1 ) = PTR_whole( 1 ) + 1
!        DO j = 2, n
!          ROW_whole( PTR_whole( j - 1 ) + 1 ) = PTR_whole( j )
!          PTR_whole( j ) = PTR_whole( j - 1 ) + PTR_whole( j ) + 1
!        END DO
!        PTR_whole( n + 1 ) = PTR_whole( n ) + 1
!write(6,"(' ptr_whole = ', 6I3 )" ) PTR_whole
!
!  fill in the row indices in each column  of the whole matrix
!
!        DO j = 1, n
!          DO k = PTR( j ), PTR( j + 1 ) - 1
!            i = ROW( k )
!            IF ( i /= j ) THEN
!              ROW_whole( PTR_whole( i ) ) = j
!              ROW_whole( PTR_whole( j ) ) = i
!              PTR_whole( i ) = PTR_whole( i ) - 1
!              PTR_whole( j ) = PTR_whole( j ) - 1
!            END IF
!          END DO
!        END DO

        RETURN

!  end of subroutine AMD_build_whole_matrix

        END SUBROUTINE AMD_build_whole_matrix

!  end of module GALAHAD_AMD

      END MODULE GALAHAD_AMD_precision
