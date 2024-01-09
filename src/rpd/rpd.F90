! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ R P D   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.0. January 22nd 2006

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_RPD_precision

!      --------------------------------------------------
!     |                                                  |
!     | Read and write data for the linear program (LP)  |
!     |                                                  |
!     |    minimize           g(T) x + f                 |
!     |    subject to     c_l <= A x <= c_u              |
!     |                   x_l <=  x  <= x_u              |
!     |                                                  |
!     | the linear program with quadratic                |
!     | constraints (QCP)                                |
!     |                                                  |
!     |    minimize       g(T) x + f                     |
!     |    subject to c_l <= A x +                       |
!     |            1/2 vec( x . H_c . x ) <= c_u         |
!     |                   x_l <=  x  <= x_u              |
!     |                                                  |
!     | the bound-constrained quadratic program (BQP)    |
!     |                                                  |
!     |    minimize     1/2 x(T) H x + g(T) x + f        |
!     |    subject to     x_l <=  x  <= x_u              |
!     |                                                  |
!     | the quadratic program (QP)                       |
!     |                                                  |
!     |    minimize     1/2 x(T) H x + g(T) x + f        |
!     |    subject to     c_l <= A x <= c_u              |
!     |                   x_l <=  x  <= x_u              |
!     |                                                  |
!     | or the quadratic program with quadratic          |
!     | constraints (QCQP)                               |
!     |                                                  |
!     |    minimize     1/2 x(T) H x + g(T) x + f        |
!     |    subject to c_l <= A x +                       |
!     |            1/2 vec( x . H_c . x ) <= c_u         |
!     |                   x_l <=  x  <= x_u              |
!     |                                                  |
!     | where vec( x . H_c . x ) is the vector whose     |
!     | i-th component is  x(T) (H_c)_i x for the i-th   |
!     | constraint, from and to a QPLIB-format data file |
!     |                                                  |
!      --------------------------------------------------

      USE GALAHAD_KINDS_precision
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SMT_precision, ONLY: SMT_put
      USE GALAHAD_QPT_precision
      USE GALAHAD_SPACE_precision
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_STRING, ONLY: STRING_trim_real_24, STRING_trim_integer_16,   &
                                STRING_lower_word
      USE GALAHAD_SORT_precision, ONLY: SORT_heapsort_build,                   &
                                        SORT_heapsort_smallest
      USE GALAHAD_LMS_precision, ONLY: LMS_apply_lbfgs

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: RPD_read_problem_data, RPD_write_qp_problem_data,              &
                RPD_get_stats, RPD_initialize, RPD_get_g, RPD_get_f,           &
                RPD_get_xlu, RPD_get_clu,                                      &
                RPD_get_H, RPD_get_A, RPD_get_H_c, RPD_get_x_type,             &
                RPD_get_x, RPD_get_y, RPD_get_z, RPD_terminate,                &
                RPD_full_initialize, RPD_full_terminate, RPD_information,      &
                RPD_read_specfile, QPT_problem_type

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE RPD_initialize
       MODULE PROCEDURE RPD_initialize, RPD_full_initialize
     END INTERFACE RPD_initialize

     INTERFACE RPD_terminate
       MODULE PROCEDURE RPD_terminate, RPD_full_terminate
     END INTERFACE RPD_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      INTEGER ( KIND = ip_ ), PARAMETER :: input_line_length = 256
      INTEGER ( KIND = ip_ ), PARAMETER :: qp = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qcqp = 2
      INTEGER ( KIND = ip_ ), PARAMETER :: bqp = 3
      INTEGER ( KIND = ip_ ), PARAMETER :: lp = 4
      INTEGER ( KIND = ip_ ), PARAMETER :: qcp = 5
      INTEGER ( KIND = ip_ ), PARAMETER :: out_debug = 6
      LOGICAL, PARAMETER :: debug = .FALSE.
!     LOGICAL, PARAMETER :: debug = .TRUE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: RPD_control_type

!   QPLIB file input on stream qplib

        INTEGER ( KIND = ip_ ) :: qplib = 21

!   error and warning diagnostics occur on stream error

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

        INTEGER ( KIND = ip_ ) :: print_level = 0

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

     END TYPE RPD_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: RPD_inform_type

!  return status:
!    0 = OK,
!   -2 = allocation failure,
!   -3 = end of file,
!   -4 = other read error,
!   -5 = unrecognised type

        INTEGER ( KIND = ip_ ) :: status = 0

!  status from last allocation attempt

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  status from last read attempt

        INTEGER ( KIND = ip_ ) :: io_status = 0

!  number of last line read from i/o file

        INTEGER ( KIND = ip_ ) :: line = 0

!  problem type

        CHARACTER ( LEN = 3 ) :: p_type = REPEAT( ' ', 3 )

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )
      END TYPE

!  - - - - - - - - - - - -
!   full data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: RPD_full_data_type
        LOGICAL :: f_indexing
        TYPE ( RPD_control_type ) :: RPD_control
        TYPE ( RPD_inform_type ) :: RPD_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE RPD_full_data_type

   CONTAINS

!- G A L A H A D -  R P D _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE RPD_initialize( control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for RPD controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_control_type ), INTENT( OUT ) :: control
     TYPE ( RPD_inform_type ), INTENT( OUT ) :: inform

     RETURN

!  End of subroutine RPD_initialize

     END SUBROUTINE RPD_initialize

!- G A L A H A D -  R P D _ F U L L I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE RPD_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for RPD controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( RPD_control_type ), INTENT( OUT ) :: control
     TYPE ( RPD_inform_type ), INTENT( OUT ) :: inform

     RETURN

!  End of subroutine RPD_full_initialize

     END SUBROUTINE RPD_full_initialize

!-*-*-*-*-   R P D _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE RPD_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by RPD_initialize could (roughly)
!  have been set as:

! BEGIN RPD SPECIFICATIONS (DEFAULT)
!  qplib-file-device                                 21
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  space-critical                                    F
!  deallocate-error-fatal                            F
! END RPD SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( RPD_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: qplib = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: error = qplib + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = deallocate_error_fatal + 1
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'RPD'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( qplib )%keyword = 'qplib-file-device'
      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'

!  Logical key-words

      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' rpd: ', alt_specname

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( qplib ),                                &
                                 control%qplib,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

      RETURN

      END SUBROUTINE RPD_read_specfile

!-*-*-   R P D _ R E A D _ P R O B L E M _ D A T A   S U B R O U T I N E   -*-*-

      SUBROUTINE RPD_read_problem_data( prob, control, inform )

!  Read the QPLIB-format data file from unit qplib into the derived type prob
!  (see above for components of inform, and GALAHAD_qpt for those of prob)

!  ****************************************************************************

!  For the linear program (LP)

!    minimize           g(T) x + f

!    subject to     c_l <= A x <= c_u
!                   x_l <= x <= x_u

!  the linear program with quadratic constraints (QCP)

!    minimize           g(T) x + f

!    subject to     c_l <= A x + 1/2 vec( x . H_c . x ) <= c_u
!                   x_l <= x <= x_u

!  the bound-constrained quadratic program (BQP)

!    minimize     1/2 x(T) H x + g(T) x + f

!    subject to     x_l <= x <= x_u

!  the quadratic program (QP)

!    minimize     1/2 x(T) H x + g(T) x + f

!    subject to     c_l <= A x <= c_u
!                   x_l <= x <= x_u

!  or the quadratic program with quadratic constraints (QCQP)

!    minimize     1/2 x(T) H x + g(T) x + f

!    subject to     c_l <= A x + 1/2 vec( x . H_c . x ) <= c_u
!                   x_l <= x <= x_u

!  where vec( x . H_c . x ) is the vector whose ith component is  x(T) H_c x
!  for the i-th constraint. Variables may be continuous, binary or integer

!  ****************************************************************************

!  The data should be input in a file on unit input. The data is in free format
!  (blanks separate values), but must occur in the order given here (depending
!  on the precise form of problem under consideration, certain data is not
!  required and should not be provided, see below). Any blank lines, or lines
!  starting with any of the  characters "!", "%" or "#" are ignored. Each term
!  in "quotes" denotes a required value. Any strings beyond those required on
!  a given lines will be regarded as comments and ignored.

!  "problem name"
!  "problem type"
!  "problem sense" i.e. one of the words minimize or maximize (case irrelevant)
!  "number variables, n"
!  "number general linear constraints, m"                                   [1]
!  "number of nonzeros in upper triangle of H"                              [2]
!  "row" "column" "value" for each entry of H (if any), one triple on each line
!  "default value for entries in g"
!  "number of non-default entries in g"
!  "index" "value" for each non-default term in g (if any), one pair per line
!  "value of f"
!  "number of nonzeros in upper triangles of H_c"                         [1,3]
!  "constraint" "row" "column" "value" for each entry of H_c (if any),
!    one quadruple on each line
!  "number of nonzeros in A"                                                [1]
!  "row" "column" "value" for each entry of A (if any), one triple on each line
!  "value for infinity" for bounds - any bound exceeding this in absolute value
!     is infinite
!  "default value for entries in c_l"                                       [1]
!  "number of non-default entries in c_l"                                   [1]
!  "index" "value" for each non-default term in c_l (if any), one pair per line
!  "default value for entries in c_u"                                       [1]
!  "number of non-default entries in c_u"                                   [1]
!  "index" "value" for each non-default term in c_u (if any), one pair per line
!  "default value for entries in x_l"                                       [4]
!  "number of non-default entries in x_l"                                   [4]
!  "index" "value" for each non-default term in x_l (if any), one pair per
!          line                                                             [4]
!  "default value for entries in x_u"                                       [4]
!  "number of non-default entries in x_u"                                   [4]
!  "index" "value" for each non-default term in x_u (if any), one pair per
!          line                                                             [4]
!  "default variable type"  (0 for a continuous variable, 1 for an integer one
!     and 2 for a binary one)                                               [5]
!  "number of non-default variables"                                        [5]
!  "index" "value" for each non-default variable type (if any), one pair/line
!  "default value for starting value for variables x"
!  "number of non-default starting entries in x"
!  "index" "value" for each non-default term in x (if any), one pair per line
!  "default value for starting value for Lagrange multipliers y for constraints"
!                                                                           [1]
!  "number of non-default starting entries in y"                            [1]
!  "index" "value" for each non-default term in y (if any), one pair per line
!  "default value for starting value for dual variables z for simple bounds"
!  "number of non-default starting entries in z"
!  "index" "value" for each non-default term in z (if any), one pair per line
!  "number of non-default names of variables" - default for variable i is "xi"
!  "index" "name" for each non-default name for variable x_i with index i
!    (if any)
!  "number of non-default names of constraints" - default for constraint i is
!    "ci"
!  "index" "name" for each non-default name for constraint with index i (if any)

!  The "problem type" is a string of three characters.

!  The first character indicates the type of objective function used.
!  It must be one of the following:

!   L  a linear objective function
!   D  a convex quadratic objective function whose Hessian is a diagonal matrix
!   C  a convex quadratic objective function
!   Q  a quadratic objective function whose Hessian may be indefinite

!  The second character indicates the types of variables that are present.
!  It must be one of the following:

!   C  all the variables are continuous
!   B  all the variables are binary (0-1)
!   M  the variables are a mix of continuous and binary
!   I  all the variables are integer
!   G  the variables are a mix of continuous, binary and integer

!  The third character indicates the type of the (most extreme)
!  constraint function used; other constraints may be of a lesser type.
!  It must be one of the following:

!   N  there are no constraints
!   B  some of the variables lie between lower and upper bounds (box constraint)
!   L  the constraint functions are linear
!   D  the constraint functions are convex quadratics with diagonal Hessians
!   C  the constraint functions are convex quadratics
!   Q  the constraint functions are quadratics whose Hessians may be indefinite

!  Thus for continuous problems, we would have

!    LCL            a linear program
!    LCC or LCQ     a linear program with quadratic constraints
!    CCB or QCB     a bound-constrained quadratic program
!    CCL or QCL     a quadratic program
!    CCC or CCQ or  a quadratic program with quadratic constraints
!    QCC or QCQ

!  For integer problems, the second character would be I rather than C,
!  and for mixed integer problems, the second character would by M or G.

!  [1] for bound-constrained QPs, these sections are omitted
!  [2] for linear program with quadratic constraints, this section is omitted
!  [3] for problems without quadratic constraints, this section is omitted
!  [4] for purely binary problems, these section are omitted.
!  [5] for purely-continuous, purely-binary or purely-integer problems,
!      this section is omitted. Lower and upper bounds on binary variables
!      will be 0 and 1, and this will override any other settings

!  *****************************************************************************

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( RPD_control_type ), INTENT( IN ) :: control
      TYPE ( RPD_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l  V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, ic, j, k, A_ne, H_ne, H_c_ne, ip, i_default
     INTEGER ( KIND = ip_ ) :: nnzg, nnzc_l, nnzc_u, nnzx_l, nnzx_u, smt_stat
     INTEGER ( KIND = ip_ ) :: problem_type, input, nnzx_0, nnzy_0, nnzz_0
     REAL ( KIND = rp_ ) :: rv, default
     LOGICAL :: objmax
     CHARACTER ( LEN = 2 ) :: oc
     CHARACTER ( LEN = 10 ) :: pname, cv
     CHARACTER ( LEN = 24 ) :: p_type
     CHARACTER ( LEN = input_line_length ) :: input_line, blank_line

     IF ( debug ) WRITE( out_debug, * ) 'in rpd'
     input = control%qplib
     inform%line = 0
     inform%alloc_status = 0
     inform%p_type = '   '

!    DO i = 1, input_line_length
!      blank_line( i : i ) = ' '
!    END DO
     blank_line = REPEAT( ' ', input_line_length )

!  Determine the problem name

     IF ( debug ) WRITE( out_debug, * ) 'pname'
!    pname = '          '
     pname = REPEAT( ' ', 10 )
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) pname
     ALLOCATE( prob%name( 10 ) )
     prob%name = TRANSFER( pname, prob%name )

!  Determine the problem type

     IF ( debug ) WRITE( out_debug, * ) 'p_type'
     p_type = REPEAT( ' ', 24 )
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) p_type
     p_type = ADJUSTL( p_type ) ; inform%p_type = p_type( 1 : 3 )
     IF ( p_type( 2 : 2 ) == 'M' .OR. p_type( 2 : 2 ) == 'G' ) THEN
       ip = 3
     ELSE IF ( p_type( 2 : 2 ) == 'B' ) THEN
       ip = 2
     ELSE IF ( p_type( 2 : 2 ) == 'I' ) THEN
       ip = 1
     ELSE
       ip = 0
     END IF
     oc( 1 : 1 ) = p_type( 1 : 1 ) ; oc( 2 : 2 ) = p_type( 3 : 3 )
     IF ( oc == 'QQ' .OR. oc == 'QC' .OR. oc == 'QD' .OR. oc == 'CQ' .OR.      &
          oc == 'CC' .OR. oc == 'CD' .OR. oc == 'DQ' .OR. oc == 'DC' .OR.      &
          oc == 'DD' ) THEN
       problem_type = qcqp
     ELSE IF ( oc == 'LQ' .OR. oc == 'LC' .OR. oc == 'LD' ) THEN
       problem_type = qcp
     ELSE IF ( oc == 'QB' .OR. oc == 'CB' .OR. oc == 'DB' ) THEN
       problem_type = bqp
     ELSE IF ( oc == 'QL' .OR. oc == 'CL' .OR. oc == 'DL' ) THEN
       problem_type = qp
     ELSE IF ( oc == 'LL' ) THEN
       problem_type = lp
     ELSE
       GO TO 950
     END IF

!  Determine if the problem is a minimization or maximization one

     IF ( debug ) WRITE( out_debug, * ) 'minmax'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO

     CALL STRING_lower_word( input_line( 1 : 8 ) )
     IF ( input_line( 1 : 8 ) == 'maximize' ) THEN
       objmax = .TRUE.
     ELSE
       objmax = .FALSE.
     END IF

!  Determine the number of variables and constraints

     IF ( debug ) WRITE( out_debug, * ) 'nm'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) prob%n
     IF ( problem_type /= bqp ) THEN
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) prob%m
     ELSE
       prob%m = 0
     END IF

!  Allocate suitable arrays

     ALLOCATE( prob%X( prob%n ), prob%X_l( prob%n ), prob%X_u( prob%n ),       &
               prob%G( prob%n ), prob%Z( prob%n ), STAT = inform%alloc_status )
     IF ( inform%alloc_status /= 0 ) THEN
       inform%status = - 2 ; inform%bad_alloc = 'X' ; RETURN
     END IF

     ALLOCATE( prob%C_l( prob%m ), prob%C_u( prob%m ), prob%Y( prob%m ),       &
               prob%C( prob%m ), STAT = inform%alloc_status )
     IF ( inform%alloc_status /= 0 ) THEN
       inform%status = - 2 ; inform%bad_alloc = 'Y' ; RETURN
     END IF

!  Fill component H

     IF ( problem_type == qp .OR. problem_type == bqp .OR.                     &
           problem_type == qcqp ) THEN
       IF ( debug ) WRITE( out_debug, * ) 'H_ne'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO

       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) prob%H%ne
       ALLOCATE( prob%H%row( prob%H%ne + prob%n ),                             &
                 prob%H%col( prob%H%ne + prob%n ),                             &
                 prob%H%val( prob%H%ne + prob%n ), STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = - 2 ; inform%bad_alloc = 'H' ; RETURN
       END IF

       IF ( debug ) WRITE( out_debug, * ) 'H'
       H_ne = 0
       DO k = 1, prob%H%ne
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, j, rv
         IF ( rv == zero ) CYCLE
         IF ( objmax ) rv = - rv
         H_ne = H_ne + 1 ; prob%H%val( H_ne ) = rv
         IF ( i >= j ) THEN
           prob%H%row( H_ne ) = i
           prob%H%col( H_ne ) = j
         ELSE
           prob%H%row( H_ne ) = j
           prob%H%col( H_ne ) = i
         END IF
       END DO
     ELSE
       H_ne = 0
     END IF
     prob%H%ne = H_ne ; prob%H%m = prob%n ; prob%H%n = prob%n
     IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
     CALL SMT_put( prob%H%type, 'COORDINATE', smt_stat )

!  Fill component g

     IF ( debug ) WRITE( out_debug, * ) 'g_default'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) default
     IF ( objmax ) default = - default
     prob%G = default
     IF ( debug ) WRITE( out_debug, * ) 'g'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) nnzg
     DO k = 1, nnzg
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) i, rv
       IF ( objmax ) rv = - rv
       prob%G( i ) = rv
     END DO

!  Fill component f

     IF ( debug ) WRITE( out_debug, * ) 'f'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) prob%f
     IF ( objmax ) prob%f = - prob%f

!  Fill component H_c

     IF ( problem_type == qcqp .OR. problem_type == qcp ) THEN
       IF ( debug ) WRITE( out_debug, * ) 'H_cne'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO

       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) prob%H_c%ne
       ALLOCATE( prob%H_c%row( prob%H_c%ne ), prob%H_c%col( prob%H_c%ne ),     &
                 prob%H_c%ptr( prob%H_c%ne ), prob%H_c%val( prob%H_c%ne ),     &
                 STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = - 2 ; inform%bad_alloc = 'H_c' ; RETURN
       END IF

       IF ( debug ) WRITE( out_debug, * ) 'H_c'
       H_c_ne = 0
       DO k = 1, prob%H_c%ne
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) ic, i, j, rv
         IF ( rv == zero ) CYCLE
         H_c_ne = H_c_ne + 1 ; prob%H_c%val( H_c_ne ) = rv
         prob%H_c%ptr( H_c_ne ) = ic
         IF ( i >= j ) THEN
           prob%H_c%row( H_c_ne ) = i
           prob%H_c%col( H_c_ne ) = j
         ELSE
           prob%H_c%row( H_c_ne ) = j
           prob%H_c%col( H_c_ne ) = i
         END IF
       END DO
     ELSE
       H_c_ne = 0
     END IF
     prob%H_c%ne = H_c_ne ; prob%H_c%m = prob%n ; prob%H_c%n = prob%n
     IF ( ALLOCATED( prob%H_c%type ) ) DEALLOCATE( prob%H_c%type )
     CALL SMT_put( prob%H_c%type, 'COORDINATE', smt_stat )

!  Fill component A

     IF ( problem_type /= bqp ) THEN
       IF ( debug ) WRITE( out_debug, * ) 'A_ne'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) prob%A%ne
       ALLOCATE( prob%A%row( prob%A%ne ), prob%A%col( prob%A%ne ),             &
                 prob%A%val( prob%A%ne ), STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = - 2 ; inform%bad_alloc = 'A' ; RETURN
       END IF

       IF ( debug ) WRITE( out_debug, * ) 'A'
       A_ne = 0
       DO k = 1, prob%A%ne
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, j, rv
         IF ( rv == zero ) CYCLE
         A_ne = A_ne + 1 ; prob%A%val( A_ne ) = rv
         prob%A%row( A_ne ) = i ; prob%A%col( A_ne ) = j
       END DO
     ELSE
       A_ne = 0
       ALLOCATE( prob%A%row( A_ne ), prob%A%col( A_ne ),                       &
                 prob%A%val( A_ne ), STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = - 2 ; inform%bad_alloc = 'A' ; RETURN
       END IF
     END IF
     prob%A%ne = A_ne ; prob%A%m = prob%m ; prob%A%n = prob%n
     IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
     CALL SMT_put( prob%A%type, 'COORDINATE', smt_stat )

!  Fill component infinity

     IF ( debug ) WRITE( out_debug, * ) 'infinity'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) prob%infinity

!  Fill component c_l

     IF ( problem_type /= bqp ) THEN
       IF ( debug ) WRITE( out_debug, * ) 'default_c_l'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) default
       prob%C_l = default
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) nnzc_l
       IF ( debug ) WRITE( out_debug, * ) 'c_l'
       DO k = 1, nnzc_l
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, rv
         prob%C_l( i ) = rv
       END DO

!  Fill component c_u

       IF ( debug ) WRITE( out_debug, * ) 'default_c_u'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) default
       prob%C_u = default
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) nnzc_u
       IF ( debug ) WRITE( out_debug, * ) 'c_u'
       DO k = 1, nnzc_u
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, rv
         prob%C_u( i ) = rv
       END DO
     END IF

     IF ( ip == 2 ) THEN
       prob%X_l = zero ; prob%X_u = one
     ELSE

!  Fill component x_l

       IF ( debug ) WRITE( out_debug, * ) 'default_x_l'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) default
       prob%X_l = default
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) nnzx_l
       IF ( debug ) WRITE( out_debug, * ) 'x_l'
       DO k = 1, nnzx_l
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, rv
         prob%X_l( i ) = rv
       END DO

!  Fill component x_u

       IF ( debug ) WRITE( out_debug, * ) 'default_x_u'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) default
       prob%X_u = default
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) nnzx_u
       IF ( debug ) WRITE( out_debug, * ) 'x_u'
       DO k = 1, nnzx_u
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, rv
         prob%X_u( i ) = rv
       END DO
     END IF

!  Fill component x_type

     ALLOCATE( prob%X_type( prob%n ), STAT = inform%alloc_status )
     IF ( inform%alloc_status /= 0 ) THEN
       inform%status = - 2 ; inform%bad_alloc = 'X_type' ; RETURN
     END IF
     IF ( ip == 3 ) THEN
       IF ( debug ) WRITE( out_debug, * ) 'i_default'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) i_default
       prob%X_type = i_default
       IF ( debug ) WRITE( out_debug, * ) 'nnzx_0'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) nnzx_0
       IF ( debug ) WRITE( out_debug, * ) 'x_type'
       DO k = 1, nnzx_0
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, j
         prob%X_type( i ) = j
       END DO
     ELSE IF ( ip == 2 ) THEN
       prob%X_type = 2
     ELSE IF ( ip == 1 ) THEN
       prob%X_type = 1
       DO i = 1, prob%n
         IF ( prob%X_l( i ) == zero .AND. prob%X_u( i ) == one )               &
           prob%X_type( i ) = 2
       END DO
     ELSE
       prob%X_type = 0
     END IF

!  Fill component x

     IF ( debug ) WRITE( out_debug, * ) 'x0_default'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) default
     prob%X = default
     IF ( debug ) WRITE( out_debug, * ) 'nnzx_0'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) nnzx_0
     IF ( debug ) WRITE( out_debug, * ) 'x0'
     DO k = 1, nnzx_0
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) i, rv
       prob%X( i ) = rv
     END DO

!  Fill component y

     IF ( problem_type /= bqp ) THEN
       IF ( debug ) WRITE( out_debug, * ) 'y0_default'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) default
       prob%Y = default
       IF ( debug ) WRITE( out_debug, * ) 'nnzy_0'
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) nnzy_0
       IF ( debug ) WRITE( out_debug, * ) 'y0'
       DO k = 1, nnzy_0
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, rv
         prob%Y( i ) = rv
       END DO
     END IF

!  Fill component z

     IF ( debug ) WRITE( out_debug, * ) 'z0_default'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) default
     prob%Z = default
     IF ( debug ) WRITE( out_debug, * ) 'nnzz_0'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) nnzz_0
     IF ( debug ) WRITE( out_debug, * ) 'z0'
     DO k = 1, nnzz_0
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) i, rv
       prob%Z( i ) = rv
     END DO

!  Fill component x_names

     IF ( debug ) WRITE( out_debug, * ) 'x_names_default'
     ALLOCATE( prob%X_names( prob%n ), STAT = inform%alloc_status )
     IF ( inform%alloc_status /= 0 ) THEN
       inform%status = - 2 ; inform%bad_alloc = 'X_names' ; RETURN
     END IF

     DO i = 1, prob%n
       prob%X_names( i ) = 'x' // REPEAT( ' ', 9 )
       WRITE( prob%X_names( i )( 2 : 10 ), "( I0 )" ) i
     END DO
     IF ( debug ) WRITE( out_debug, * ) 'x_names'
     DO
       inform%line = inform%line + 1
       input_line = blank_line
       READ( input, "( A )", END = 930, ERR = 940 ) input_line
       IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
     END DO
     READ( input_line, *, IOSTAT = inform%io_status,                           &
           END = 930, ERR = 940 ) nnzx_0
     DO k = 1, nnzx_0
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) i, cv
       prob%X_names( i ) = cv
     END DO

!  Fill component c_names

     IF ( problem_type /= bqp ) THEN
       IF ( debug ) WRITE( out_debug, * ) 'c_names_default'
       ALLOCATE( prob%C_names( prob%m ), STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = - 2 ; inform%bad_alloc = 'C_names' ; RETURN
       END IF

       DO i = 1, prob%m
         prob%C_names( i ) = 'c' // REPEAT( ' ', 9 )
         WRITE( prob%C_names( i )( 2 : 10 ), "( I0 )" ) i
       END DO
       DO
         inform%line = inform%line + 1
         input_line = blank_line
         READ( input, "( A )", END = 930, ERR = 940 ) input_line
         IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
       END DO
       READ( input_line, *, IOSTAT = inform%io_status,                         &
             END = 930, ERR = 940 ) nnzy_0
       IF ( debug ) WRITE( out_debug, * ) 'c_names'
       DO k = 1, nnzy_0
         DO
           inform%line = inform%line + 1
           input_line = blank_line
           READ( input, "( A )", END = 930, ERR = 940 ) input_line
           IF ( .NOT. RPD_ignore_string( input_line ) ) EXIT
         END DO
         READ( input_line, *, IOSTAT = inform%io_status,                       &
               END = 930, ERR = 940 ) i, cv
         prob%C_names( i ) = cv
       END DO
     END IF

!  - successful execution

     inform%status = GALAHAD_ok
     IF ( debug ) WRITE( out_debug, * ) 'out ok'
     RETURN

!  Error returns

!  - end of file encountered

 930 CONTINUE
     inform%status = GALAHAD_error_input_status
     IF ( debug ) WRITE( out_debug, * ) 'out end of file'
     RETURN

!  - other error encountered

 940 CONTINUE
     inform%status = GALAHAD_error_io
     IF ( debug ) WRITE( out_debug, * ) 'out io error'
     RETURN

!  - problem type unrecognised

 950 CONTINUE
     IF ( debug ) WRITE( out_debug, * ) 'out unrecognised'
     inform%status = GALAHAD_unavailable_option
     RETURN

!  End of RPD_read_problem_data

     END SUBROUTINE RPD_read_problem_data

!-   R P D _ W R I T E _ Q P _ P R O B L E M _ D A T A   S U B R O U T I N E  -

     SUBROUTINE RPD_write_qp_problem_data( prob, file_name, qplib, inform )

!  Write the QP data contained in the derived type prob as QP problem-data on
!  unit qplib (see above for components of inform, and GALAHAD_qpt for those
!  of prob). Partially extracted from the QPLIB package in CUTEst

!  Dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: qplib
     CHARACTER ( LEN = 30 ) :: file_name
     TYPE ( RPD_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER ( KIND = ip_ ) :: i, j, l, m, n, A_ne, H_ne, iores, problem_type
     REAL ( KIND = rp_ ) :: infinity_used,  mode_v, val
     LOGICAL :: filexx
     CHARACTER ( len = 10 ) :: name
     CHARACTER ( len = 16 ) :: char_i, char_j, char_l
     CHARACTER ( len = 24 ) :: char_val
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: DX, WORK_n

!  check if the file is old or new

     INQUIRE( FILE = file_name, EXIST = filexx )
     IF ( filexx ) THEN
        OPEN( qplib, FILE = file_name, FORM = 'FORMATTED',                     &
              STATUS = 'OLD', IOSTAT = iores )
     ELSE
        OPEN( qplib, FILE = file_name, FORM = 'FORMATTED',                     &
               STATUS = 'NEW', IOSTAT = iores )
     END IF

     IF ( iores /= 0 ) THEN
       inform%status = GALAHAD_error_io
       GO TO 900
     END IF

!  record common values

     m = prob%m ; n = prob%n
     name = REPEAT( ' ', 10 )
     IF ( ALLOCATED( prob%name ) ) name = SMT_get( prob%name )
     IF ( TRIM( name ) == '' ) THEN
       DO l = 1, 10
         IF ( file_name( l + 1 : l + 1 ) == '.' ) EXIT
       END DO
       name( 1 : l ) =  file_name( 1 : l )
     END IF

!  compute how many nonzero Hessian entries there are

     H_ne = 0
     IF ( prob%Hessian_kind < 0 ) THEN
       IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
         H_ne = prob%n
       ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
         IF ( prob%H%val( 1 ) /= zero ) H_ne = prob%n
       ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
         DO i = 1, prob%n
           IF ( prob%H%val( i ) /= zero ) H_ne = H_ne + 1
         END DO
       ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
         l = 0
         DO i = 1, prob%n
           DO j = 1, i
             l = l + 1
             IF ( prob%H%val( l ) /= zero ) H_ne = H_ne + 1
           END DO
         END DO
       ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
         IF (  prob%H%ptr( prob%n + 1 ) > 1 ) THEN
           DO i = 1, prob%n
             DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
               IF ( prob%H%val( l ) /= zero ) H_ne = H_ne + 1
             END DO
           END DO
         END IF
       ELSE IF ( SMT_get( prob%H%type ) == 'COORDINATE' ) THEN
         IF (  prob%H%ne > 0 ) THEN
           DO l = 1, prob%H%ne
             IF ( prob%H%val( l ) /= zero ) H_ne = H_ne + 1
           END DO
         END IF
       ELSE
         H_ne = prob%H_lm%n_restriction * ( prob%H_lm%n_restriction + 1 ) / 2
       END IF
     ELSE IF ( prob%Hessian_kind == 1 ) THEN
       H_ne = prob%n
     ELSE IF ( prob%Hessian_kind >= 2 ) THEN
       DO i = 1, prob%n
         IF ( prob%WEIGHT( i ) /= zero ) H_ne = H_ne + 1
       END DO
     END IF

!  compute how many nonzero Jacobian entries there are

     IF ( m > 0 ) THEN
       A_ne = 0
       IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
         l = 0
         DO i = 1, prob%m
           DO j = 1, prob%n
             l = l + 1
             IF ( prob%A%val( l ) /= zero ) A_ne = A_ne + 1
           END DO
         END DO
       ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
         DO i = 1, prob%m
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             IF ( prob%A%val( l ) /= zero ) A_ne = A_ne + 1
           END DO
         END DO
       ELSE
         DO l = 1, prob%A%ne
           val = prob%A%val( l )
           IF ( prob%A%val( l ) /= zero ) A_ne = A_ne + 1
         END DO
       END IF
     END IF

!  set header

     IF ( TRIM( name ) == '' ) THEN
       WRITE( qplib, "( 'problem generated by GALAHAD ',                       &
      &    'RPD_write_qp_problem_data' )" )
     ELSE
       WRITE( qplib, "( A, ' generated by GALAHAD ',                           &
      &    'RPD_write_qp_problem_data' )" ) TRIM( name )
     END IF

     problem_type = qp
     IF ( m == 0 ) problem_type = bqp
     IF ( H_ne == 0 ) problem_type = lp

!    IF ( int_var == 0 ) THEN
       SELECT CASE ( problem_type )
!      CASE ( qcqp )
!        WRITE( qplib, "( 'QCQ                      a quadratic program',      &
!       &               ' with quadratic constraints' )" )
       CASE ( bqp )
         WRITE( qplib, "( 'QCB                      a bound-constrained',      &
        &               ' quadratic program' )" )
       CASE ( lp )
         WRITE( qplib, "( 'LCL                      a linear program' )" )
!      CASE ( qcp )
!        WRITE( qplib, "( 'LCQ                      a linear program',         &
!       &                ' with quadratic constraints' )" )
       CASE DEFAULT
         WRITE( qplib, "( 'QCL                      a quadratic program' )")
       END SELECT
!    ELSE IF ( bin_var == n ) THEN
!      SELECT CASE ( problem_type )
!      CASE ( qcqp )
!        WRITE( qplib, "( 'QBQ                      a binary',                 &
!       &    ' QP with quadratic constraints' )" )
!      CASE ( bqp )
!        WRITE( qplib, "( 'QBB                      a binary',                 &
!       &    ' bound-constrained quadratic program' )" )
!      CASE ( lp )
!        WRITE( qplib, "( 'LBL                      a binary',                 &
!     &     ' linear program' )" )
!      CASE ( qcp )
!        WRITE( qplib, "( 'LBQ                      a binary',                 &
!       &    ' LP with quadratic constraints' )" )
!      CASE DEFAULT
!        WRITE( qplib, "( 'QBL                      a binary',                 &
!       &   ' quadratic program' )")
!      END SELECT
!    ELSE IF ( int_var == n ) THEN
!      SELECT CASE ( problem_type )
!      CASE ( qcqp )
!        WRITE( qplib, "( 'QIQ                      an integer',               &
!       &    ' QP with quadratic constraints' )" )
!      CASE ( bqp )
!        WRITE( qplib, "( 'QIB                      an integer',               &
!       &    ' bound-constrained quadratic program' )" )
!      CASE ( lp )
!        WRITE( qplib, "( 'LIL                      an integer',               &
!     &     ' linear program' )" )
!      CASE ( qcp )
!        WRITE( qplib, "( 'LIQ                      an integer',               &
!       &    ' LP with quadratic constraints' )" )
!      CASE DEFAULT
!        WRITE( qplib, "( 'QIL                      an integer',               &
!       &   ' quadratic program' )")
!      END SELECT
!    ELSE
!       SELECT CASE ( problem_type )
!      CASE ( qcqp )
!        WRITE( qplib, "( 'QGQ                      a mixed-integer',          &
!       &    ' QP with quadratic constraints' )" )
!      CASE ( bqp )
!        WRITE( qplib, "( 'QGB                      a mixed-integer',          &
!       &    ' bound-constrained quadratic program' )" )
!      CASE ( lp )
!        WRITE( qplib, "( 'QGL                      a mixed-integer',          &
!     &     ' linear program' )" )
!      CASE ( qcp )
!        WRITE( qplib, "( 'LGQ                      a mixed-integer',          &
!       &    ' LP with quadratic constraints' )" )
!      CASE DEFAULT
!        WRITE( qplib, "( 'QGL                      a mixed-integer',          &
!       &   ' quadratic program' )")
!      END SELECT
!    END IF
     WRITE( qplib, "( 'Minimize' )" )
     char_l = STRING_trim_integer_16( n )
     WRITE( qplib, "( A16, 8X, ' # variables ' )" ) char_l
     IF ( problem_type /= bqp ) THEN
       char_l = STRING_trim_integer_16( m )
       WRITE( qplib, "( A16, 8X, ' # general linear constraints ' )" ) char_l
     END IF

!  Hessian values

     IF ( problem_type == qp .OR. problem_type == bqp .OR.                     &
          problem_type == qcqp ) THEN
       char_l = STRING_trim_integer_16( H_ne )
       IF ( H_ne == 0 ) THEN
         WRITE( qplib, "( /, A16, 8X, ' # nonzeros in upper triangle of H')" ) &
           char_l
       ELSE
         WRITE( qplib, "( /, A16, 8X, ' # nonzeros in upper triangle of H:',   &
        &   ' row,column,value' )" ) char_l
         IF ( prob%Hessian_kind < 0 ) THEN
           IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
             DO i = 1, prob%n
               val = 1.0_rp_
               char_i = STRING_trim_integer_16( i )
               char_val = STRING_trim_real_24( val )
               WRITE( qplib, 2000 ) char_i, char_i, char_val
             END DO
           ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
             val = prob%H%val( 1 )
             IF ( val /= zero ) THEN
               char_val = STRING_trim_real_24( val )
               DO i = 1, prob%n
                 char_i = STRING_trim_integer_16( i )
                 WRITE( qplib, 2000 ) char_i, char_i, char_val
               END DO
             END IF
           ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
             DO i = 1, prob%n
               val = prob%H%val( i )
               IF ( val /= zero ) THEN
                 char_i = STRING_trim_integer_16( i )
                 char_val = STRING_trim_real_24( val )
                 WRITE( qplib, 2000 ) char_i, char_i, char_val
               END IF
             END DO
           ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
             l = 0
             DO i = 1, prob%n
               DO j = 1, i
                 l = l + 1
                 val = prob%H%val( l )
                 IF ( val /= zero ) THEN
                   char_i = STRING_trim_integer_16( i )
                   char_j = STRING_trim_integer_16( j )
                   char_val = STRING_trim_real_24( val )
                   WRITE( qplib, 2000 ) char_j, char_i, char_val
                 END IF
               END DO
             END DO
           ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
             IF (  prob%H%ptr( prob%n + 1 ) > 1 ) THEN
               DO i = 1, prob%n
                 DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                   val = prob%H%val( l )
                   IF ( val /= zero ) THEN
                     char_i = STRING_trim_integer_16( i )
                     char_j = STRING_trim_integer_16( prob%H%col( l ) )
                     char_val = STRING_trim_real_24( val )
                     IF ( i <= j ) THEN
                       WRITE( qplib, 2000 ) char_i, char_j, char_val
                     ELSE
                       WRITE( qplib, 2000 ) char_j, char_i, char_val
                     END IF
                   END IF
                 END DO
               END DO
             END IF
           ELSE IF ( SMT_get( prob%H%type ) == 'COORDINATE' ) THEN
             IF ( prob%H%ne > 0 ) THEN
               DO l = 1, prob%H%ne
                 val = prob%H%val( l )
                 IF ( val /= zero ) THEN
                   char_i = STRING_trim_integer_16( prob%H%row( l ) )
                   char_j = STRING_trim_integer_16( prob%H%col( l ) )
                   char_val = STRING_trim_real_24( val )
                   IF ( i <= j ) THEN
                     WRITE( qplib, 2000 ) char_i, char_j, char_val
                   ELSE
                     WRITE( qplib, 2000 ) char_j, char_i, char_val
                   END IF
                 END IF
               END DO
             END IF
           ELSE
             ALLOCATE( DX( prob%H_lm%n_restriction ),                          &
                       WORK_n( prob%H_lm%n_restriction ) )
             DX = zero
             DO j = 1, prob%H_lm%n_restriction
               DX( j ) = 1.0_rp_
               CALL LMS_apply_lbfgs( DX, prob%H_lm, i, RESULT = WORK_n )
               DO i = 1, j
                 val = WORK_n( i )
!                IF ( val /= zero ) THEN
                   char_i = STRING_trim_integer_16( i )
                   char_j = STRING_trim_integer_16( j )
                   char_val = STRING_trim_real_24( val )
                   WRITE( qplib, 2000 ) char_i, char_j, char_val
!                END IF
               END DO
               DX( j ) = zero
             END DO
             DEALLOCATE( DX, WORK_n )
           END IF
         ELSE IF ( prob%Hessian_kind == 1 ) THEN
           val = 1.0_rp_
           char_val = STRING_trim_real_24( val )
           DO i = 1, prob%n
             char_i = STRING_trim_integer_16( i )
             WRITE( qplib, 2000 ) char_i, char_i, char_val
           END DO
         ELSE IF ( prob%Hessian_kind >= 2 ) THEN
           DO i = 1, prob%n
             val = prob%WEIGHT( i ) ** 2
             IF ( val /= zero ) THEN
               char_i = STRING_trim_integer_16( i )
               char_val = STRING_trim_real_24( val )
               WRITE( qplib, 2000 ) char_i, char_i, char_val
             END IF
           END DO
         END IF
       END IF
     END IF

!  gradient values

     mode_v = MODE( n, prob%G )
     l = COUNT( prob%G( : n ) /= mode_v )
     char_l = STRING_trim_integer_16( l )
     char_val = STRING_trim_real_24( mode_v )
     WRITE( qplib, "( /, A24, ' default value for entries in g' )" ) char_val
     IF ( l == 0 ) THEN
       WRITE( qplib, "( A16, 8X, ' # non default entries in g' )" ) char_l
     ELSE
       WRITE( qplib, "( A16, 8X, ' # non default entries in g: index,value')") &
         char_l
       DO i = 1, n
         IF ( prob%G( i ) /= mode_v ) THEN
           char_i = STRING_trim_integer_16( i )
           char_val = STRING_trim_real_24( prob%G( i ) )
           WRITE( qplib, 2010 ) char_i, char_val
         END IF
       END DO
     END IF

!  function value

     char_val = STRING_trim_real_24( prob%f )
     WRITE( qplib, "( /, A24, ' value of f' )" ) char_val

!  Hessian values for constraints

!    IF ( problem_type == qcqp .OR. problem_type == qcp ) THEN

!      ALLOCATE( X0( n ), STAT = status )
!      IF ( status /= 0 ) GO TO 900
!      X0 = zero
!      nnzh_i = 0 ; lh = SIZE( H_val )

!  open a dummy file to store the Hessian values temporarily

!      OPEN( qplib_out_dummy,  IOSTAT = status )
!      IF ( status /= 0 ) GO TO 900

!  write the Hessian values for the ith constraint to the dummy file

!      DO i = 1, m
!        CALL CUTEST_cish( status, n, X0, i, nehi, lh, H_val, H_row, H_col )
!        IF ( status /= 0 ) GO TO 900
!        char_l = STRING_trim_integer_16( i )
!        DO l = 1, nehi
!           IF ( H_val( l ) /= zero ) THEN
!           nnzh_i = nnzh_i + 1
!           char_i = STRING_trim_integer_16( H_row( l ) )
!           char_j = STRING_trim_integer_16( H_col( l ) )
!           char_val = STRING_trim_real_24( H_val( l ) )
!           WRITE( qplib_out_dummy, "( A16, 1X, A16, 1X, A16, 1X, A24 )" )    &
!             char_l, char_i, char_j, char_val
!          END IF
!        END DO
!      END DO
!      DEALLOCATE( X0, stat = status )
!      CALL CUTEST_cterminate( status )

!  record the total number of constraintg Hessian values

!      char_l = STRING_trim_integer_16( nnzh_i )
!      WRITE( qplib, "( /, A16, 8X, ' # nonzeros in upper triangle',          &
!     &  ' of the H_i')") char_l

!  append the constraint Hessian values to the qplib file

!      IF ( nnzh_i > 0 ) THEN
!        REWIND( qplib_out_dummy )
!        DO l = 1, nnzh_i
!          READ( qplib_out_dummy, "( A75 )" ) qplib_hi
!          WRITE( qplib_qplib, "( A75 )" ) qplib_hi
!        END DO
!      END IF
!      CLOSE( qplib_out_dummy )
!    END IF

!  constraint Jacobian values

     IF ( problem_type /= bqp ) THEN
       char_l = STRING_trim_integer_16( A_ne )
       IF ( A_ne == 0 ) THEN
         WRITE( qplib, "( /, A16, 8X, ' # nonzeros in A' )" ) char_l
       ELSE
         WRITE( qplib, "( /, A16, 8X, ' # nonzeros in A:',                     &
        &   ' row,column,value' )" ) char_l
       END IF

       IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
         l = 0
         DO i = 1, prob%m
           char_i = STRING_trim_integer_16( i )
           DO j = 1, prob%n
             l = l + 1
             val = prob%A%val( l )
             IF ( val /= zero ) THEN
               char_j = STRING_trim_integer_16( j )
               char_val = STRING_trim_real_24( val )
               WRITE( qplib, 2000 ) char_i, char_j, char_val
             END IF
           END DO
         END DO
       ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
         DO i = 1, prob%m
           char_i = STRING_trim_integer_16( i )
           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             val = prob%A%val( l )
             IF ( val /= zero ) THEN
               char_j = STRING_trim_integer_16( prob%A%col( l ) )
               char_val = STRING_trim_real_24( val )
               WRITE( qplib, 2000 ) char_i, char_j, char_val
             END IF
           END DO
         END DO
       ELSE
         DO l = 1, prob%A%ne
           val = prob%A%val( l )
           IF ( val /= zero ) THEN
             char_i = STRING_trim_integer_16( prob%A%row( l ) )
             char_j = STRING_trim_integer_16( prob%A%col( l ) )
             char_val = STRING_trim_real_24( val )
             WRITE( qplib, 2000 ) char_i, char_j, char_val
           END IF
         END DO
       END IF
     END IF

!  infinity

      char_val = STRING_trim_real_24( prob%infinity )
      WRITE( qplib, "( /, A24, ' value of infinite bounds' )" ) char_val
      infinity_used = 10.0_rp_ * prob%infinity

!  constraint lower bounds

     IF ( problem_type /= bqp ) THEN
       IF ( m > 0 ) THEN
         mode_v = MODE( m, prob%C_l )
         l = COUNT( prob%C_l( : m ) /= mode_v )
         char_l = STRING_trim_integer_16( l )
         char_val = STRING_trim_real_24( mode_v )
         WRITE( qplib, "( /, A24, ' default value for entries in c_l' )" )     &
           char_val
         IF ( l == 0 ) THEN
           WRITE( qplib, "( A16, 8X, ' # non default entries in c_l')" ) char_l
         ELSE
           WRITE( qplib, "( A16, 8X, ' # non default entries in c_l:',         &
          &  ' index,value' )" ) char_l
           DO i = 1, m
             IF ( prob%C_l( i ) /= mode_v ) THEN
               char_i = STRING_trim_integer_16( i )
               char_val = STRING_trim_real_24( prob%C_l( i ) )
               WRITE( qplib, 2010 ) char_i, char_val
             END IF
           END DO
         END IF
       ELSE
         mode_v = - infinity_used
         l = 0
         char_l = STRING_trim_integer_16( l )
         char_val = STRING_trim_real_24( mode_v )
         WRITE( qplib, "( /, A24, ' default value for entries in c_l' )" )     &
           char_val
         WRITE( qplib, "( A16, 8X, ' # non default entries in c_l' )" ) char_l
       END IF

!  constraint upper bounds

       IF ( m > 0 ) THEN
         mode_v = MODE( m, prob%C_u )
         l = COUNT( prob%C_u( : m ) /= mode_v )
         char_l = STRING_trim_integer_16( l )
         char_val = STRING_trim_real_24( mode_v )
         WRITE( qplib, "( /, A24, ' default value for entries in c_u' )" )     &
           char_val
         IF ( l == 0 ) THEN
           WRITE( qplib, "( A16, 8X, ' # non default entries in c_u')" ) char_l
         ELSE
           WRITE( qplib, "( A16, 8X, ' # non default entries in c_u:',         &
         &   ' index,value' )" ) char_l
           DO i = 1, m
             IF ( prob%C_u( i ) /= mode_v ) THEN
               char_i = STRING_trim_integer_16( i )
               char_val = STRING_trim_real_24( prob%C_u( i ) )
               WRITE( qplib, 2010 ) char_i, char_val
             END IF
           END DO
         END IF
       ELSE
         mode_v = infinity_used
         l = 0
         char_l = STRING_trim_integer_16( l )
         char_val = STRING_trim_real_24( mode_v )
         WRITE( qplib, "( /, A24, ' default value for entries in c_u' )" )     &
           char_val
         WRITE( qplib, "( A16, 8X, ' # non default entries in c_u' )" ) char_l
       END IF
     END IF

!  variable lower bounds

     mode_v = MODE( n, prob%X_l )
     l = COUNT( prob%X_l( : n ) /= mode_v )
     char_l = STRING_trim_integer_16( l )
     char_val = STRING_trim_real_24( mode_v )
     WRITE( qplib, "( /, A24, ' default value for entries in x_l' )" ) char_val
     IF ( l == 0 ) THEN
       WRITE( qplib, "( A16, 8X, ' # non default entries in x_l' )" ) char_l
     ELSE
       WRITE( qplib, "( A16, 8X, ' # non default entries in x_l:',             &
      &  ' index,value' )" ) char_l
       DO i = 1, n
         IF ( prob%X_l( i ) /= mode_v ) THEN
           char_i = STRING_trim_integer_16( i )
           char_val = STRING_trim_real_24( prob%X_l( i ) )
           WRITE( qplib, 2010 ) char_i, char_val
         END IF
       END DO
     END IF

!  variable upper bounds

     mode_v = MODE( n, prob%X_u )
     l = COUNT( prob%X_u( : n ) /= mode_v )
     char_l = STRING_trim_integer_16( l )
     char_val = STRING_trim_real_24( mode_v )
     WRITE( qplib, "( /, A24, ' default value for entries in x_u' )" ) char_val
     IF ( l == 0 ) THEN
       WRITE( qplib, "( A16, 8X, ' # non default entries in x_u' )" ) char_l
     ELSE
       WRITE( qplib, "( A16, 8X, ' # non default entries in x_u:',             &
      &  ' index,value' )") char_l
       DO i = 1, n
         IF ( prob%X_u( i ) /= mode_v ) THEN
           char_i = STRING_trim_integer_16( i )
           char_val = STRING_trim_real_24( prob%X_u( i ) )
           WRITE( qplib, 2010 ) char_i, char_val
         END IF
       END DO
     END IF

!  variable types

!    IF ( int_var > 0 .AND. int_var < n ) THEN
!      IF ( n >= 2 * int_var ) THEN
!        char_l = STRING_trim_integer_16( 0 )
!        WRITE( qplib, "( /, A16, 8X, ' default variable type',                &
!       &  ' (0 for continuous, 1 for integer)' )" ) char_l
!        char_j = STRING_trim_integer_16( int_var )
!        IF ( int_var == 0 ) THEN
!          WRITE( qplib, "( A16, 8X, ' # non default variables' )" ) char_j
!        ELSE
!          WRITE( qplib, "( A16, 8X, ' # non default variables: index,type')")&
!         &  char_j
!          DO i = 1, n
!            IF (  X_type( i ) /= 0 ) THEN
!              char_i = STRING_trim_integer_16( i )
!              char_j = STRING_trim_integer_16( X_type( i ) )
!              WRITE( qplib, 2010 ) char_i, char_j
!            END IF
!          END DO
!        END IF
!      ELSE
!        char_l = STRING_trim_integer_16( 1 )
!        WRITE( qplib, "( /, A16, 8X, ' default variable type',                &
!       & ' (0 for continuous, 1 for integer)' )" ) char_l
!        char_j = STRING_trim_integer_16( n - int_var )
!        IF ( int_var == n ) THEN
!          WRITE( qplib, "( A16, 8X, ' # non default variables' )" ) char_j
!        ELSE
!          WRITE( qplib, "( A16, 8X, ' # non default variables: index,type')") &
!            char_j
!          DO i = 1, n
!            IF (  X_type( i ) == 0 ) THEN
!              char_i = STRING_trim_integer_16( i )
!              char_j = STRING_trim_integer_16( X_type( i ) )
!              WRITE( qplib, 2010 ) char_i, char_j
!            END IF
!          END DO
!        END IF
!      END IF
!    END IF

!  initial primal variables

     mode_v = MODE( n, prob%X )
     l = COUNT( prob%X( : n ) /= mode_v )
     char_l = STRING_trim_integer_16( l )
     char_val = STRING_trim_real_24( mode_v )
     WRITE( qplib, "( /, A24, ' default value for entries in initial x' )" )   &
       char_val
     IF ( l == 0 ) THEN
       WRITE( qplib, "( A16, 8X, ' # non default entries in x' )" ) char_l
     ELSE
       WRITE( qplib, "( A16, 8X, ' # non default entries in x: index,value')") &
         char_l
       DO i = 1, n
         IF ( prob%X( i ) /= mode_v ) THEN
           char_i = STRING_trim_integer_16( i )
           char_val = STRING_trim_real_24( prob%X( i ) )
           WRITE( qplib, 2010 ) char_i, char_val
         END IF
       END DO
     END IF

!  initial Lagrange multipliers

     IF ( problem_type /= bqp ) THEN
       IF ( m > 0 ) THEN
         mode_v = MODE( m, prob%Y )
         l = COUNT( prob%Y( : m ) /= mode_v )
         char_l = STRING_trim_integer_16( l )
         char_val = STRING_trim_real_24( mode_v )
         WRITE( qplib, "( /, A24, ' default value for entries in initial y')")&
           char_val
         IF ( l == 0 ) THEN
           WRITE( qplib, "( A16, 8X, ' # non default entries in y' )" ) char_l
         ELSE
           WRITE( qplib, "( A16, 8X, ' # non default entries in y:',           &
          &  ' index,value' )" ) char_l
           DO i = 1, m
             IF ( prob%Y( i ) /= mode_v ) THEN
               char_i = STRING_trim_integer_16( i )
               char_val = STRING_trim_real_24( prob%Y( i ) )
               WRITE( qplib, 2010 ) char_i, char_val
             END IF
           END DO
         END IF
       ELSE
         mode_v = zero
         l = 0
         char_l = STRING_trim_integer_16( l )
         char_val = STRING_trim_real_24( mode_v )
         WRITE( qplib, "( /, A24, ' default value for entries in initial y')") &
           char_val
         WRITE( qplib, "( A16, 8X, ' # non default entries in y' )" ) char_l
       END IF
     END IF

!  initial dual variables

     mode_v = MODE( n, prob%Z )
     l = COUNT( prob%Z( : n ) /= mode_v )
     char_l = STRING_trim_integer_16( l )
     char_val = STRING_trim_real_24( mode_v )
     WRITE( qplib, "( /, A24, ' default value for entries in initial z' )" )   &
       char_val
     IF ( l == 0 ) THEN
       WRITE( qplib, "( A16, 8X, ' # non default entries in z' )" ) char_l
     ELSE
       WRITE( qplib, "( A16, 8X, ' # non default entries in z: index,value')")&
         char_l
       DO i = 1, n
         IF ( prob%Z( i ) /= mode_v ) THEN
           char_i = STRING_trim_integer_16( i )
           char_val = STRING_trim_real_24( prob%Z( i ) )
           WRITE( qplib, 2010 ) char_i, char_val
         END IF
       END DO
     END IF

!  variable names

     l = 0
     IF ( ALLOCATED( prob%X_names ) ) THEN
       DO i = 1, n
         name = prob%X_names( i )
         IF ( TRIM( name ) /= '' ) THEN
           char_i = STRING_trim_integer_16( i )
           IF ( name /= 'x' // TRIM( char_i ) ) THEN
             l = l + 1
           END IF
         END IF
       END DO
     END IF

     char_l = STRING_trim_integer_16( l )
     WRITE( qplib, "( /, A16, 8X, ' # non default names for variables:',       &
    &   ' index,name' )" ) char_l
     IF ( l > 0 ) THEN
       DO i = 1, n
         name = prob%X_names( i )
         IF ( TRIM( name ) /= '' ) THEN
           char_i = STRING_trim_integer_16( i )
           IF ( name /= 'x' // TRIM( char_i ) )                                &
             WRITE( qplib, "( A16, 1X, A10 )" ) char_i, name
         END IF
       END DO
     END IF

!  constraint names

     IF ( problem_type /= bqp ) THEN
       IF ( m > 0 ) THEN
         l = 0
         IF ( ALLOCATED( prob%C_names ) ) THEN
           DO i = 1, m
             name = prob%C_names( i )
             IF ( TRIM( name ) /= '' ) THEN
               char_i = STRING_trim_integer_16( i )
               IF ( name /= 'c' // TRIM( char_i ) ) THEN
                 l = l + 1
               END IF
             END IF
           END DO
         END IF

         char_l = STRING_trim_integer_16( l )
         WRITE( qplib, "( /, A16, 8X, ' # non default names for constraints:', &
        &   ' index,name' )" ) char_l
         IF ( l > 0 ) THEN
           DO i = 1, m
             name = prob%C_names( i )
             IF ( TRIM( name ) /= '' ) THEN
               char_i = STRING_trim_integer_16( i )
               IF ( name /= 'c' // TRIM( char_i ) )                            &
                 WRITE( qplib, "( A16, 1X, A10 )" ) char_i, name
             END IF
           END DO
         END IF
       ELSE
         char_l = STRING_trim_integer_16( 0_ip_ )
         WRITE( qplib, "( /, A16, 8X, ' # non default names for constraints:', &
        &   ' index,name' )" ) char_l
       END IF
     END IF

     CLOSE( qplib )
     inform%status = GALAHAD_ok
     RETURN

!  error exits

 900 CONTINUE
!    WRITE( out, "( ' error status = ', I0 )" ) status
     CLOSE( qplib )
     RETURN

!  non-executable statements

2000 FORMAT( A16, 1X, A16, 1X, A24 )
2010 FORMAT( A16, 1X, A24 )

!  ------------------------ M O D E  F U N C T I O N --------------------------

     CONTAINS

       FUNCTION MODE( n, V )
       IMPLICIT NONE
       REAL ( KIND = rp_ ) :: MODE
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: V

!  find the "mode", i.e., the most commonly-occuring value, of a vector v

       INTEGER ( KIND = ip_ ) :: i, mode_start, max_len, same, len, m, inform

       REAL ( KIND = rp_ ), DIMENSION( n ) :: V_sorted

!  sort a copy of v into increasing order

       IF ( n > 0 ) THEN
         V_sorted = V
         CALL SORT_heapsort_build( n, V_sorted, inform ) !  build the heap
         DO i = 1, n
           m = n - i + 1
           CALL SORT_heapsort_smallest( m, V_sorted, inform ) !  reorder v
         END DO

!  run through the sorted values, finding adjacent entries that are identical

         mode_start = 1 ; max_len = 1
         same = 1 ; len = 1
         DO i = 2, n
           IF ( V_sorted( i ) /= V_sorted( same ) ) THEN
             IF ( len > max_len ) THEN
               mode_start = same
               max_len = len
             END IF
             same = i ; len = 1
           ELSE
             len = len + 1
           END IF
         END DO
         IF ( len > max_len ) THEN
           mode_start = same
           max_len = len
         END IF
         MODE = V_sorted( mode_start )
       ELSE
         MODE = zero
       END IF
       RETURN

       END FUNCTION MODE

!  End of RPD_write_qp_problem_data

     END SUBROUTINE RPD_write_qp_problem_data

!-*-*-*-*-*-   R P D _ I G N O R E _ S T R I N G   F U N C T I O N   -*-*-*-*-*-

     FUNCTION RPD_ignore_string( input_line )
     LOGICAL :: RPD_ignore_string

!  Ignore a string if it is (a) blank or (b) starts with "!", "%" or "#"

     CHARACTER ( LEN = input_line_length ), INTENT( IN ) :: input_line

!  Local variables

     INTEGER ( KIND = ip_ ) :: i, length_string

     length_string = LEN_TRIM( input_line )
     IF ( length_string <= 0 ) THEN
       RPD_ignore_string = .TRUE.
       RETURN
     END IF

     DO i = 1, length_string
       IF ( input_line( i : i ) == ' ' ) CYCLE
       IF ( input_line( i : i ) == '!' .OR. input_line( i : i ) == '#' .OR.    &
            input_line( i : i ) == '%' .OR. input_line( i : i ) == '|' ) THEN
         RPD_ignore_string = .TRUE.
         RETURN
       END IF
       EXIT
     END DO
     RPD_ignore_string = .FALSE.

     RETURN

!  End of RPD_ignore_string

     END FUNCTION RPD_ignore_string

!-*-*-*-*-*-*-   R P D _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE RPD_terminate( prob, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   prob    see Subroutine RPD_read_problem_data
!   control see Subroutine RPD_initialize
!   inform  see Subroutine RPD_initialize

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     TYPE ( RPD_control_type ), INTENT( IN ) :: control
     TYPE ( RPD_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate any internal problem arrays

     array_name = 'rpd: prob%name'
     CALL SPACE_dealloc_array( prob%name,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%X_names'
     CALL SPACE_dealloc_array( prob%X_names,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%C_names'
     CALL SPACE_dealloc_array( prob%C_names,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%X_type'
     CALL SPACE_dealloc_array( prob%X_type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%X'
     CALL SPACE_dealloc_array( prob%X,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%X_l'
     CALL SPACE_dealloc_array( prob%X_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%X_u'
     CALL SPACE_dealloc_array( prob%X_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%G'
     CALL SPACE_dealloc_array( prob%G,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%Y'
     CALL SPACE_dealloc_array( prob%Y,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%Z'
     CALL SPACE_dealloc_array( prob%Z,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%C_l'
     CALL SPACE_dealloc_array( prob%C_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%C_u'
     CALL SPACE_dealloc_array( prob%C_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%C'
     CALL SPACE_dealloc_array( prob%C,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H%row'
     CALL SPACE_dealloc_array( prob%H%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H%col'
     CALL SPACE_dealloc_array( prob%H%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H%val'
     CALL SPACE_dealloc_array( prob%H%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H%type'
     CALL SPACE_dealloc_array( prob%H%type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H_c%ptr'
     CALL SPACE_dealloc_array( prob%H_c%ptr,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H_c%row'
     CALL SPACE_dealloc_array( prob%H_c%row,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H_c%col'
     CALL SPACE_dealloc_array( prob%H_c%col,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H_c%val'
     CALL SPACE_dealloc_array( prob%H_c%val,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%H_c%type'
     CALL SPACE_dealloc_array( prob%H_c%type,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%A%row'
     CALL SPACE_dealloc_array( prob%A%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%A%col'
     CALL SPACE_dealloc_array( prob%A%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%A%val'
     CALL SPACE_dealloc_array( prob%A%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'rpd: prob%A%type'
     CALL SPACE_dealloc_array( prob%A%type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine RPD_terminate

     END SUBROUTINE RPD_terminate

! -  G A L A H A D -  R P D _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE RPD_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( RPD_control_type ), INTENT( IN ) :: control
     TYPE ( RPD_inform_type ), INTENT( INOUT ) :: inform

     CALL RPD_terminate( data%prob, control, inform )
     RETURN

!  End of subroutine RPD_full_terminate

     END SUBROUTINE RPD_full_terminate


! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

! - G A L A H A D - R P D _ g e t _ s t a t s   S U B R O U T I N E -

     SUBROUTINE RPD_get_stats( control, data, status, p_type,                  &
                               n, m, h_ne, a_ne, h_c_ne )

!  determine the type of quadratic program in the QPLIB file provuded  on unit
!  input as well as the numbers of variables and general constraints involved,
!  as well as the numbers of nonzeros in the objective and constraint Hessians
!  and the constraint Jacobian

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_control_type ), INTENT( IN ) :: control
     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, n, m, h_ne, a_ne, h_c_ne
     CHARACTER ( LEN = 3 ), INTENT( OUT ) :: p_type

     data%RPD_control = control
     CALL RPD_read_problem_data( data%prob, data%RPD_control, data%RPD_inform )

!  recover the problem type and its dimensions

     p_type = data%RPD_inform%p_type
     n = data%prob%n
     m = data%prob%m
     h_ne = data%prob%H%ne
     a_ne = data%prob%A%ne
     h_c_ne = data%prob%H_c%ne

     status = data%RPD_inform%status
     RETURN

!  end of subroutine RPD_get_stats

     END SUBROUTINE RPD_get_stats

! - G A L A H A D - R P D _ g e t _ g  S U B R O U T I N E -

     SUBROUTINE RPD_get_g( data, status, G )

!  recover the linear term from the objective function, g

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: n

!  copy g

     IF ( .NOT. ALLOCATED( data%prob%G ) ) GO TO 900
     n = data%prob%n
     G( : n ) = data%prob%G( : n )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_g

     END SUBROUTINE RPD_get_g

! - G A L A H A D - R P D _ g e t _ f  S U B R O U T I N E -

     SUBROUTINE RPD_get_f( data, status, f )

!  recover the constant term from the objective function, f

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), INTENT( OUT ) :: f

!  copy f

     f = data%prob%f

     status = GALAHAD_ok
     RETURN

!  end of subroutine RPD_get_f

     END SUBROUTINE RPD_get_f

! - G A L A H A D - R P D _ g e t _ x l u  S U B R O U T I N E -

     SUBROUTINE RPD_get_xlu( data, status, X_l, X_u )

!  recover the lower and upper bounds on the constraints, x_l and x_u

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: X_l, X_u

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: n

!  copy x_l and x_u

     IF ( .NOT. ALLOCATED( data%prob%X_l ) .OR.                                &
          .NOT. ALLOCATED( data%prob%X_u ) ) GO TO 900
     n = data%prob%n
     X_l( : n ) = data%prob%X_l( : n )
     X_u( : n ) = data%prob%X_u( : n )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_xlu

     END SUBROUTINE RPD_get_xlu

! - G A L A H A D - R P D _ g e t _ c l u  S U B R O U T I N E -

     SUBROUTINE RPD_get_clu( data, status, C_l, C_u )

!  recover the lower and upper bounds on the constraints, c_l and c_u

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: C_l, C_u

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: m

!  copy c_l and x_c

     IF ( .NOT. ALLOCATED( data%prob%C_l ) .OR.                                &
          .NOT. ALLOCATED( data%prob%C_u ) ) GO TO 900
     m = data%prob%m
     C_l( : m ) = data%prob%C_l( : m )
     C_u( : m ) = data%prob%C_u( : m )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_clu

     END SUBROUTINE RPD_get_clu

! - G A L A H A D - R P D _ g e t _ H  S U B R O U T I N E -

     SUBROUTINE RPD_get_H( data, status, H_row, H_col, H_val )

!  recover the objective Hessian, H. The row and column indices and values
!  of the lower triangle of H are in the array triplet (H_row, H_col, H_val)


!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: H_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: H_col
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_val

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: ne

!  copy H

     IF ( .NOT. ALLOCATED( data%prob%H%row ) .OR.                              &
          .NOT. ALLOCATED( data%prob%H%col ) .OR.                              &
          .NOT. ALLOCATED( data%prob%H%val ) ) GO TO 900
     ne = data%prob%H%ne
     H_row( : ne ) = data%prob%H%row( : ne )
     H_col( : ne ) = data%prob%H%col( : ne )
     H_val( : ne ) = data%prob%H%val( : ne )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_H

     END SUBROUTINE RPD_get_H

! - G A L A H A D - R P D _ g e t _ A  S U B R O U T I N E -

     SUBROUTINE RPD_get_A( data, status, A_row, A_col, A_val )

!  recover the constraint Jacobian, A. The row and column indices and values
!  of A are in the array triplet (A_row, A_col, A_val)

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: A_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: A_col
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: A_val

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: ne

!  copy A

     IF ( .NOT. ALLOCATED( data%prob%A%row ) .OR.                              &
          .NOT. ALLOCATED( data%prob%A%col ) .OR.                              &
          .NOT. ALLOCATED( data%prob%A%val ) ) GO TO 900
     ne = data%prob%A%ne
     A_row( : ne ) = data%prob%A%row( : ne )
     A_col( : ne ) = data%prob%A%col( : ne )
     A_val( : ne ) = data%prob%A%val( : ne )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_A

     END SUBROUTINE RPD_get_A

! - G A L A H A D - R P D _ g e t _ H _ c  S U B R O U T I N E -

     SUBROUTINE RPD_get_H_c( data, status, H_c_ptr, H_c_row, H_c_col, H_c_val )

!  recover the Hessians of the constraints, H_c. The constraint, row and
!  column indices and values of the lower triangles of the H_c  are in the
!  quadruplet (H_c_ptr, H_crow, H_c_col, H_c_val)

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: H_c_ptr
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: H_c_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: H_c_col
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_c_val

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: ne

!  copy H_c

     IF ( .NOT. ALLOCATED( data%prob%H_c%ptr ) .OR.                            &
          .NOT. ALLOCATED( data%prob%H_c%row ) .OR.                            &
          .NOT. ALLOCATED( data%prob%H_c%col ) .OR.                            &
          .NOT. ALLOCATED( data%prob%H_c%val ) ) GO TO 900
     ne = data%prob%H_c%ne
     H_c_ptr( : ne ) = data%prob%H_c%ptr( : ne )
     H_c_row( : ne ) = data%prob%H_c%row( : ne )
     H_c_col( : ne ) = data%prob%H_c%col( : ne )
     H_c_val( : ne ) = data%prob%H_c%val( : ne )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_H_c

     END SUBROUTINE RPD_get_H_c

! - G A L A H A D - R P D _ g e t _ x_type  S U B R O U T I N E -

     SUBROUTINE RPD_get_x_type( data, status, X_type )

!  recover the variable type, x_type

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( : ) :: X_type

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: n

!  copy x_type

     IF ( .NOT. ALLOCATED( data%prob%X_type ) ) GO TO 900
     n = data%prob%n
     X_type( : n ) = data%prob%X_type( : n )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_x_type

     END SUBROUTINE RPD_get_x_type

! - G A L A H A D - R P D _ g e t _ x  S U B R O U T I N E -

     SUBROUTINE RPD_get_x( data, status, X )

!  recover the primal starting point, x

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: X

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: n

!  copy x

     IF ( .NOT. ALLOCATED( data%prob%X ) ) GO TO 900
     n = data%prob%n
     X( : n ) = data%prob%X( : n )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_x

     END SUBROUTINE RPD_get_x

! - G A L A H A D - R P D _ g e t _ y  S U B R O U T I N E -

     SUBROUTINE RPD_get_y( data, status, Y )

!  recover the initial Lagrange multiplier, y

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Y

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: m

!  copy y

     IF ( .NOT. ALLOCATED( data%prob%Y ) ) GO TO 900
     m = data%prob%m
     Y( : m ) = data%prob%Y( : m )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_y

     END SUBROUTINE RPD_get_y

! - G A L A H A D - R P D _ g e t _ z  S U B R O U T I N E -

     SUBROUTINE RPD_get_z( data, status, Z )

!  recover the dual starting point, z

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Z

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: n

!  copy z

     IF ( .NOT. ALLOCATED( data%prob%Z ) ) GO TO 900
     n = data%prob%n
     Z( : n ) = data%prob%Z( : n )

     status = GALAHAD_ok
     RETURN

!  unallocated error return

 900 CONTINUE
     status = GALAHAD_error_unallocated
     RETURN

!  end of subroutine RPD_get_z

     END SUBROUTINE RPD_get_z

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-  G A L A H A D -  R P D _ i n f o r m a t i o n   S U B R O U T I N E  -*-

     SUBROUTINE RPD_information( data, inform, status )

!  return solver information during or after solution by RPD
!  See RPD_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( RPD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( RPD_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%rpd_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine RPD_information

     END SUBROUTINE RPD_information

!  End of module RPD

   END MODULE GALAHAD_RPD_precision
