! COPYRIGHT (c) 2000,2010,2013,2016 Science and Technology Facilities Council
! Authors: Jonathan Hogg and Iain Duff
!
! Based on modified versions of MC56 and HSL_MC56.
module spral_rutherford_boeing

  use spral_matrix_util
  use spral_random, only : random_state, random_real
  implicit none

  integer, parameter :: wp = kind(0d0)
  integer, parameter :: long = selected_int_kind(18)
  real(wp), parameter :: zero = 0.0_wp

  private
  public :: rb_peek, &         ! Peeks at the header of a RB file
       rb_read,      &         ! Reads a RB file
       rb_write                ! Writes a RB file
  public :: rb_read_options, & ! Options that control what rb_read returns
       rb_write_options        ! Options that control what rb_write does

  ! Possible values options%lwr_upr_full
  integer, parameter :: TRI_LWR  = 1 ! Lower triangle
  integer, parameter :: TRI_UPR  = 2 ! Upper triangle
  integer, parameter :: TRI_FULL = 3 ! Both lower and upper triangles

  ! Possible values of options%values
  integer, parameter :: VALUES_FILE       = 0 ! As per file
  integer, parameter :: VALUES_PATTERN    = 1 ! Pattern only
  integer, parameter :: VALUES_SYM        = 2 ! Random values, symmetric
  integer, parameter :: VALUES_DIAG_DOM   = 3 ! Random vals, diag dominant
  integer, parameter :: VALUES_UNSYM      = 4 ! Random values, unsymmetric

  ! Possible error returns
  integer, parameter :: SUCCESS           =  0 ! No errors
  integer, parameter :: ERROR_BAD_FILE    = -1 ! Failed to open file
  integer, parameter :: ERROR_NOT_RB      = -2 ! Header not valid for RB
  integer, parameter :: ERROR_IO          = -3 ! Error return from io
  integer, parameter :: ERROR_TYPE        = -4 ! Tried to read bad type
  integer, parameter :: ERROR_ELT_ASM     = -5 ! Read elt as asm or v/v
  integer, parameter :: ERROR_MATRIX_TYPE = -6 ! Bad value of matrix_type
  integer, parameter :: ERROR_EXTRA_SPACE = -10 ! options%extra_space<1.0
  integer, parameter :: ERROR_LWR_UPR_FULL= -11 ! options%lwr_up_full oor
  integer, parameter :: ERROR_VALUES      = -13 ! options%values oor
  integer, parameter :: ERROR_ALLOC       = -20 ! failed on allocate

  ! Possible warnings
  integer, parameter :: WARN_AUX_FILE     = 1 ! values in auxiliary file

  type rb_read_options
     logical  :: add_diagonal = .false.        ! Add missing diagonal entries
     real     :: extra_space = 1.0             ! Array sizes are mult by this
     integer  :: lwr_upr_full = TRI_LWR   ! Ensure entries in lwr/upr tri
     integer  :: values = VALUES_FILE     ! As per file
  end type rb_read_options

  type rb_write_options
     character(len=20) :: val_format = "(3e24.16)"
  end type rb_write_options

  interface rb_peek
     module procedure rb_peek_file, rb_peek_unit
  end interface rb_peek

  interface rb_read
     module procedure rb_read_double_int32, rb_read_double_int64
  end interface rb_read

  interface rb_write
     module procedure rb_write_double_int32, rb_write_double_int64
  end interface rb_write
contains
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> Read header information from file (filename version).
  subroutine rb_peek_file(filename, info, m, n, nelt, nvar, nval, &
       matrix_type, type_code, title, identifier)
    implicit none
    character(len=*), intent(in) :: filename     !< File to peek at
    integer, intent(out) :: info                 !< Return code
    integer, optional, intent(out) :: m          !< # rows
    integer, optional, intent(out) :: n          !< # columns
    integer(long), optional, intent(out) :: nelt !< # elements (0 if asm)
    integer(long), optional, intent(out) :: nvar !< # indices in file
    integer(long), optional, intent(out) :: nval !< # values in file
    integer, optional, intent(out) :: matrix_type !< SPRAL matrix type
    character(len=3), optional, intent(out) :: type_code !< eg "rsa"
    character(len=72), optional, intent(out) :: title  !< title field of file
    character(len=8), optional, intent(out) :: identifier !< id field of file

    integer :: iunit ! unit file is open on
    integer :: iost ! stat parameter for io calls

    info = SUCCESS

    ! Find a free unit and open file on it
    open(newunit=iunit, file=filename, status="old", action="read", &
         iostat=iost)
    if (iost .ne. 0) then
       info = ERROR_BAD_FILE
       return
    end if

    ! Call unit version to do hard work, no need to rewind as we will close
    ! file immediately
    call rb_peek_unit(iunit, info, m=m, n=n, nelt=nelt, nvar=nvar, &
         nval=nval, matrix_type=matrix_type, type_code=type_code, title=title, &
         identifier=identifier, no_rewind=.true.)

    ! Close file
    close(iunit, iostat=iost)
    if ((iost .ne. 0) .and. (info .eq. SUCCESS)) then
       ! Note: we ignore close errors if info indicates a previous error
       info = ERROR_IO
       return
    end if
  end subroutine rb_peek_file

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> Read header information from file (unit version).
  subroutine rb_peek_unit(iunit, info, m, n, nelt, nvar, nval, &
       matrix_type, type_code, title, identifier, no_rewind)
    implicit none
    integer, intent(in) :: iunit                 ! unit file is open on
    integer, intent(out) :: info                 ! return code
    integer, optional, intent(out) :: m          ! # rows
    integer, optional, intent(out) :: n          ! # columns
    integer(long), optional, intent(out) :: nelt ! # elements (0 if asm)
    integer(long), optional, intent(out) :: nvar ! # indices in file
    integer(long), optional, intent(out) :: nval ! # values in file
    integer, optional, intent(out) :: matrix_type ! spral matrix type code
    character(len=3), optional, intent(out) :: type_code ! eg "rsa"
    character(len=72), optional, intent(out) :: title ! title field of file
    character(len=8), optional, intent(out) :: identifier ! id field of file
    logical, optional, intent(in) :: no_rewind ! If present and true, don't
      ! backspace unit to start

    ! "shadow" versions of file data - can't rely on arguments being present
    ! so data is read into these and copied to arguments if required
    integer :: r_m
    integer :: r_n
    integer :: r_nelt
    integer :: r_nvar
    integer :: r_nval
    character(len=3) :: r_type_code
    character(len=72) :: r_title
    character(len=8) :: r_identifier
    logical :: r_rewind

    ! Other local variables
    character(len=80) :: buffer1, buffer2 ! Buffers for reading char data
    integer :: t1, t2, t3, t4 ! Temporary variables for reading integer data
    integer :: iost ! stat parameter for io ops

    info = SUCCESS

    r_rewind = .true.
    if (present(no_rewind)) r_rewind = .not. no_rewind

    ! Nibble top of file to find desired information, then return to original
    ! position if required
    read (iunit, '(a72,a8/a80/a80)', iostat=iost) &
         r_title, r_identifier, buffer1, buffer2
    if (iost .ne. 0) then
       info = ERROR_IO
       return
    end if
    if (r_rewind) then
       backspace(iunit); backspace(iunit); backspace(iunit)
    end if

    read(buffer2, '(a3,11x,4(1x,i13))') r_type_code, t1, t2, t3, t4

    !
    ! Validate type_code code, remap data depending on value of type_code(3:3)
    !
    select case (r_type_code(1:1))
    case("r", "c", "i", "p", "q")
       ! Good, do nothing
    case default
       ! Not a matrix in RB format
       info = ERROR_NOT_RB
       return
    end select

    select case (r_type_code(2:2))
    case("s", "u", "h", "z", "r")
       ! Good, do nothing
    case default
       ! Not a matrix in RB format
       info = ERROR_NOT_RB
       return
    end select

    select case (r_type_code(3:3))
    case("a")
       ! Assembled format
       r_m = t1
       r_n = t2
       r_nvar = t3
       if (t4 .ne. 0) then
          ! RB format requires t4 to be an explicit zero
          info = ERROR_NOT_RB
          return
       end if
       r_nval = r_nvar ! one-to-one correspondence between integers and reals
       r_nelt = 0 ! no elemental matrices
    case("e")
       ! Elemental format
       r_m = t1
       r_n = r_m ! Elemental matrices are square
       r_nelt = t2
       r_nvar = t3
       r_nval = t4
    case default
       ! Not a valid RB letter code
       info = ERROR_NOT_RB
       return
    end select

    !
    ! Copy out data if requested
    !
    if (present(m)) m = r_m
    if (present(n)) n = r_n
    if (present(nelt)) nelt = r_nelt
    if (present(nvar)) nvar = r_nvar
    if (present(nval)) nval = r_nval
    if (present(matrix_type)) &
         matrix_type = sym_to_matrix_type(r_type_code(2:2))
    if (present(type_code)) type_code = r_type_code
    if (present(title)) title = r_title
    if (present(identifier)) identifier = r_identifier
  end subroutine rb_peek_unit

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> Read a matrix from a Rutherford Boeing file
  subroutine rb_read_double_int32(filename, m, n, ptr, row, val, &
       options, info, matrix_type, type_code, title, identifier, state)
    implicit none
    character(len=*), intent(in) :: filename ! File to read
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, dimension(:), allocatable, intent(out) :: ptr
    integer, dimension(:), allocatable, target, intent(out) :: row
    real(wp), dimension(:), allocatable, target, intent(out) :: val
    type(rb_read_options), intent(in) :: options ! control variables
    integer, intent(out) :: info ! return code
    integer, optional, intent(out) :: matrix_type ! spral matrix type code
    character(len=3), optional, intent(out) :: type_code ! file data type
    character(len=72), optional, intent(out) :: title ! file title
    character(len=8), optional, intent(out) :: identifier ! file identifier
    type(random_state), optional, intent(inout) :: state ! state to use for
      ! random number generation

    integer(long), dimension(:), allocatable :: ptr64
    integer :: st

    call rb_read_double_int64(filename, m, n, ptr64, row, val, &
         options, info, matrix_type=matrix_type, type_code=type_code, &
         title=title, identifier=identifier, state=state)

    ! FIXME: Add an error code if ne > maxint
    if (allocated(ptr64)) then
       deallocate(ptr, stat=st)
       allocate(ptr(n+1), stat=st)
       if (st .ne. 0) then
          info = ERROR_ALLOC
          return
       end if
       ptr(1:n+1) = int(ptr64(1:n+1)) ! Forced conversion, FIXME: add guard
    end if
  end subroutine rb_read_double_int32

  subroutine rb_read_double_int64(filename, m, n, ptr, row, val, &
       options, info, matrix_type, type_code, title, identifier, state)
    implicit none
    character(len=*), intent(in) :: filename ! File to read
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer(long), dimension(:), allocatable, intent(out) :: ptr
    integer, dimension(:), allocatable, target, intent(out) :: row
    real(wp), dimension(:), allocatable, target, intent(out) :: val
    type(rb_read_options), intent(in) :: options ! control variables
    integer, intent(out) :: info ! return code
    integer, optional, intent(out) :: matrix_type ! spral matrix type code
    character(len=3), optional, intent(out) :: type_code ! file data type
    character(len=72), optional, intent(out) :: title ! file title
    character(len=8), optional, intent(out) :: identifier ! file identifier
    type(random_state), optional, intent(inout) :: state ! state to use for
    ! random number generation

    ! Below variables are required for calling f77 MC56
    integer, dimension(:), allocatable :: ival

    ! Shadow variable for type_code (actual argument is optional)
    character(len=3) :: r_type_code

    ! Pointers to simplify which array we are reading in to.
    integer, pointer, dimension(:), contiguous :: rcptr => null()
    real(wp), pointer, dimension(:), contiguous :: vptr => null()

    real(wp), target :: temp(1) ! place holder array
    integer :: k ! loop indices
    integer(long) :: j ! loop indices
    integer :: r, c ! loop indices
    integer(long) :: nnz ! number of non-zeroes
    integer(long) :: nelt ! number of elements in file, should be 0
    integer(long) :: len, len2 ! length of arrays to allocate
    integer :: iunit ! unit we open the file on
    integer :: st, iost ! error codes from allocate and file operations
    logical :: symmetric ! .true. if file claims to be (skew) symmetric or H
    logical :: skew ! .true. if file claims to be skew symmetric
    logical :: read_val ! .true. if we are only reading pattern from file
    logical :: expanded ! .true. if pattern has been expanded
    type(random_state) :: state2 ! random state used if state not present
    integer, dimension(:), allocatable :: iw34 ! work array used by mc34
    integer, dimension(:), allocatable, target :: col ! work array in case we
      ! need to flip from lwr to upr.
    integer, dimension(:), allocatable :: nzrow ! number of entries in row

    info = SUCCESS

    ! Initialize variables to avoid compiler warnings
    symmetric = .false.
    skew = .false.

    ! Validate options paramters
    if (options%extra_space .lt. 1.0) then
       info = ERROR_EXTRA_SPACE
       return
    end if
    if ((options%lwr_upr_full .lt. 1) .or. (options%lwr_upr_full .gt. 3)) then
       info = ERROR_LWR_UPR_FULL
       return
    end if
    if ((options%values .eq. -1) .or. (abs(options%values) .gt. 4)) then
       info = ERROR_VALUES
       return
    end if

    ! Find a free unit and open file on it
    open(newunit=iunit, file=filename, status="old", action="read", &
         iostat=iost)
    if (iost .ne. 0) then
       info = ERROR_BAD_FILE
       return
    end if

    ! Read top of file (and rewind) to determine space required
    call rb_peek_unit(iunit, info, m=m, n=n, nelt=nelt, &
         nval=nnz, matrix_type=matrix_type, type_code=r_type_code, &
         title=title, identifier=identifier)
    if (info .ne. 0) goto 100

    if (nelt .ne. 0) then
       ! Attempting to read element file as assembled
       info = ERROR_ELT_ASM
       goto 100
    end if

    !
    ! Allocate space for matrix
    !

    ! ptr
    len = n + 1
    len = max(len, int(real(len,wp) * options%extra_space, long))
    allocate(ptr(len), stat=st)
    if (st .ne. 0) goto 200

    ! row and/or col
    len = nnz
    select case (r_type_code(2:2))
    case("s", "h", "z")
       symmetric = .true.
       skew = (r_type_code(2:2) .eq. "z")
       ! Do we need to allow for expansion?
       ! (a) to get both upper and lower triangles
       if (options%lwr_upr_full .eq. TRI_FULL) len = len * 2
       ! (b) to add additional diagonal entries
       if (options%add_diagonal .or. &
               (options%values .eq. -VALUES_DIAG_DOM) .or. &
               ((options%values .eq. VALUES_DIAG_DOM) .and. &
               ((r_type_code(1:1) .eq. "p") .or. (r_type_code(1:1) .eq. "q")))) then
          len = len + n
       end if
    case("u", "r")
       symmetric = .false.
       ! Unsymmetric or rectangular, no need to worry about upr/lwr, but
       ! may need to add diagonal.
       if (options%add_diagonal) len = len + n
    end select
    len2 = len
    len = max(len, int(real(len,wp) * options%extra_space, long))
    allocate(row(len), stat=st)
    if (st .ne. 0) goto 200
    rcptr => row
    if (symmetric .and. (options%lwr_upr_full .eq. TRI_UPR)) then
       ! We need to read into col then copy into row as we flip
       ! from lwr to upr
       allocate(col(len2), stat=st)
       rcptr => col
    end if
    if (st .ne. 0) goto 200

    ! Allocate val if required
    if ((abs(options%values) .ge. VALUES_SYM) .or. &
         ((options%values .eq. 0) .and. (r_type_code(1:1) .ne. "p") .and. &
         (r_type_code(1:1) .ne. "q"))) then
       ! We are actually going to store some values
       allocate(val(len), stat=st)
       if (st .ne. 0) goto 200
       vptr => val
    else
       ! Use a place holder in call to mc56
       vptr => temp
    end if

    !
    ! Read matrix in its native format (real/integer)
    !
    if (r_type_code(1:1) .eq. "q") info = WARN_AUX_FILE

    ! Determine whether we are reading values from file or not
    read_val = ((options%values .ge. 0) .and. (options%values .ne. VALUES_PATTERN))
    read_val = read_val .and. (r_type_code(1:1) .ne. "p")
    read_val = read_val .and. (r_type_code(1:1) .ne. "q")

    select case(r_type_code(1:1))
    case ("r") ! Real
       if (read_val) then
          ! Want pattern and values
          call read_data_real(iunit, n, nnz, ptr, rcptr, iost, val=vptr)
       else
          ! Want pattern only
          call read_data_real(iunit, n, nnz, ptr, rcptr, iost)
       end if
    case ("c") ! Complex
       info = ERROR_TYPE
       goto 100
    case ("i") ! Integer
       if (read_val) then
          allocate(ival(nnz), stat=st)
          if (st .ne. 0) goto 200
          call read_data_integer(iunit, n, nnz, ptr, rcptr, iost, val=ival)
          if (iost .ne. 0) val(1:nnz) = real(ival)
       else
          call read_data_integer(iunit, n, nnz, ptr, rcptr, iost)
       end if
    case ("p", "q") ! Pattern only
       call read_data_real(iunit, n, nnz, ptr, rcptr, iost)
    end select
    if(iost .ne. 0) then
       info = ERROR_IO
       goto 100 ! error
    end if

    !
    ! Add any missing diagonal entries
    !
    if (options%add_diagonal .or. &
         (symmetric .and. (.not. read_val) .and. (abs(options%values) .eq. 3))) then
       if (read_val) then
          call add_missing_diag(m, n, ptr, rcptr, val=val)
       else
          call add_missing_diag(m, n, ptr, rcptr)
       end if
    end if

    !
    ! Expand pattern if we need to generate unsymmetric values for it
    !
    if (((.not. read_val) .and. (abs(options%values) .eq. VALUES_UNSYM)) &
         .and. symmetric .and. (options%lwr_upr_full .eq. TRI_FULL)) then
       allocate(iw34(n),stat=st)
       if (st .ne. 0) goto 200
       call half_to_full(n, rcptr, ptr, iw34)
       expanded = .true.
    else
       expanded = .false.
    end if

    !
    ! Generate values if required
    !
    if ((.not. read_val) .and. ((options%values .lt. 0) .or. (options%values .ge. 2))) then
       if (abs(options%values) .eq. 3) then
          allocate(nzrow(n), stat=st)
          if (st .ne. 0) goto 200
          nzrow(:) = 0
       end if
       do c = 1, n
          k = int( ptr(c+1) - ptr(c) )
          if (present(state)) then
             do j = ptr(c), ptr(c+1)-1
                val(j) = random_real(state, .false.)
                r = rcptr(j)
                if ((abs(options%values) .eq. 3) .and. symmetric) then
                   nzrow(r) = nzrow(r) + 1
                   if (r .eq. c) val(j) = max(100, 10*(k+nzrow(r)-1))
                end if
             end do
          else
             do j = ptr(c), ptr(c+1)-1
                val(j) = random_real(state2, .false.)
                r = rcptr(j)
                if ((abs(options%values) .eq. 3) .and. symmetric) then
                   nzrow(r) = nzrow(r) + 1
                   if (r .eq. c) val(j) = max(100, 10*(k+nzrow(r)-1))
                end if
             end do
          end if
       end do
    end if

    !
    ! Expand to full storage or flip lwr/upr as required
    !
    if (symmetric) then
       select case (options%lwr_upr_full)
       case(TRI_LWR)
          ! No-op
       case(TRI_UPR)
          ! Only need to flip from upr to lwr if want to end up as CSC
          if (allocated(val)) then
             call flip_lwr_upr(n, ptr, col, row, st, val=val)
          else
             call flip_lwr_upr(n, ptr, col, row, st)
          end if
          if (st .ne. 0) goto 200
          if (skew .and. associated(vptr, val)) then
             call sym_to_skew(n, ptr, row, val)
          end if
       case(TRI_FULL)
          if (.not. allocated(iw34)) allocate(iw34(n),stat=st)
          if (st .ne. 0) goto 200
          if (.not. expanded) then
             if (allocated(val)) then
                call half_to_full(n, rcptr, ptr, iw34, a=val)
             else
                call half_to_full(n, rcptr, ptr, iw34)
             end if
             expanded = .true.
             if (skew .and. allocated(val)) then
                ! HSL_MC34 doesn't cope with skew symmetry, need to flip
                ! -ive all entries in upper triangle.
                call sym_to_skew(n, ptr, row, val)
             end if
          end if
       end select
    end if

100 continue

    if (present(type_code)) type_code = r_type_code

    close(iunit, iostat=iost)
    if ((iost .ne. 0) .and. (info .eq. SUCCESS)) then
       ! Note: we ignore close errors if info indicates a previous error
       info = ERROR_IO
       return
    end if

    !!!!!!!!!!!!!!!!
    return
    !!!!!!!!!!!!!!!!

    !
    ! Error handlers
    !
200 continue
    ! Allocation error
    info = ERROR_ALLOC
    goto 100
  end subroutine rb_read_double_int64

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> @brief Write a CSC matrix to the specified file
   !> @param filename File to write to. If it already exists, it will be
   !>        overwritten.
   !> @param matrix_type SPRAL matrix type, as defined in matrix_utils.
   !> @param m Number of rows in matrix.
   !> @param n Number of columns in matrix.
   !> @param ptr Column pointers for matrix. Column i has entries corresponding
   !>        to row(ptr(i):ptr(i+1)-1) and val(ptr(i):ptr(i+1)-1).
   !> @param row Row indices for matrix.
   !> @param val Floating point values for matrix.
   !> @param options User-specifyable options.
   !> @param info Status on output, 0 for success.
  subroutine rb_write_double_int32(filename, matrix_type, m, n, ptr, row, &
       options, inform, val, title, identifier)
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(in) :: matrix_type
    integer, intent(in) :: m
    integer, intent(in) :: n
    integer, dimension(n+1), intent(in) :: ptr
    integer, dimension(ptr(n+1)-1), intent(in) :: row
    type(rb_write_options), intent(in) :: options
    integer, intent(out) :: inform
    real(wp), dimension(ptr(n+1)-1), optional, intent(in) :: val
    character(len=*), optional, intent(in) :: title
    character(len=*), optional, intent(in) :: identifier

    integer(long), dimension(:), allocatable :: ptr64
    integer :: st

    ! Copy from 32-bit to 64-bit ptr array and call 64-bit version.
    allocate(ptr64(n+1), stat=st)
    if (st .ne. 0) then
       inform = ERROR_ALLOC
       return
    end if
    ptr64(:) = ptr(:)

    call rb_write_double_int64(filename, matrix_type, m, n, ptr64, row, &
         options, inform, val=val, title=title, identifier=identifier)
  end subroutine rb_write_double_int32

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> @brief Write a CSC matrix to the specified file
   !> @param filename File to write to. If it already exists, it will be
   !>        overwritten.
   !> @param matrix_type SPRAL matrix type, as defined in matrix_utils.
   !> @param m Number of rows in matrix.
   !> @param n Number of columns in matrix.
   !> @param ptr Column pointers for matrix. Column i has entries corresponding
   !>        to row(ptr(i):ptr(i+1)-1) and val(ptr(i):ptr(i+1)-1).
   !> @param row Row indices for matrix.
   !> @param val Floating point values for matrix.
   !> @param options User-specifyable options.
   !> @param inform Status on output, 0 for success.
   !> @param title Title to use in file, defaults to "Matrix"
   !> @param id Matrix name/identifyer to use in file, defaults to "0"
  subroutine rb_write_double_int64(filename, matrix_type, m, n, ptr, row, &
       options, inform, val, title, identifier)
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(in) :: matrix_type
    integer, intent(in) :: m
    integer, intent(in) :: n
    integer(long), dimension(n+1), intent(in) :: ptr
    integer, dimension(ptr(n+1)-1), intent(in) :: row
    type(rb_write_options), intent(in) :: options
    integer, intent(out) :: inform
    real(wp), dimension(ptr(n+1)-1), optional, intent(in) :: val
    character(len=*), optional, intent(in) :: title
    character(len=*), optional, intent(in) :: identifier

    character(len=3) :: type
    integer :: i, iunit
    integer(long) :: ptr_lines, row_lines, val_lines, total_lines
    integer(long) :: max_ptr
    integer :: max_row, ptr_prec, row_prec
    integer :: ptr_per_line, row_per_line, val_per_line
    character(len=16) :: ptr_format, row_format
    character(len=72) :: the_title
    character(len=8) :: the_id
    integer :: st

    inform = 0 ! by default, success

    ! Check arguments
    if ((matrix_type .lt. 0) .or. (matrix_type .gt. 6) .or. (matrix_type .eq. 5)) then
       inform = ERROR_MATRIX_TYPE
       return
    end if

    ! Open file
    open(file=filename, newunit=iunit, status='replace', iostat=st)
    if (st .ne. 0) then
       inform = ERROR_BAD_FILE
       return
    end if

    ! Determine formats
    max_ptr = maxval(ptr(1:n+1))
    ptr_prec = int(log10(real(max_ptr, wp)))+2
    ptr_per_line = 80 / ptr_prec ! 80 character per line
    ptr_format = create_format(ptr_per_line, ptr_prec)
    max_row = maxval(row(1:ptr(n+1)-1))
    row_prec = int(log10(real(max_row, wp)))+2
    row_per_line = 80 / row_prec ! 80 character per line
    row_format = create_format(row_per_line, row_prec)

    ! Calculate lines
    ! First find val_per_line
    do i = 2, len(options%val_format)
       if ((options%val_format(i:i) .eq. 'e') .or. (options%val_format(i:i) .eq. 'f')) &
            exit
    end do
    read(options%val_format(2:i-1), *) val_per_line
    ptr_lines = (size(ptr)-1) / ptr_per_line + 1
    row_lines = (size(row)-1) / row_per_line + 1
    if (present(val)) then
       val_lines = (size(val)-1) / val_per_line + 1
    else
       val_lines = 0
    end if
    total_lines = ptr_lines + row_lines + val_lines

    ! Determine type string
    if (present(val)) then
       type(1:1) = 'r' ! real
    else
       type(1:1) = 'p' ! pattern
    end if
    type(2:2) = matrix_type_to_sym(matrix_type)
    type(3:3) = 'a' ! assembled

    ! Write header
    the_title = "Matrix"
    if (present(title)) the_title = title
    the_id = "0"
    if (present(identifier)) the_id = identifier
    write(iunit, "(a72,a8)") the_title, the_id
    write(iunit, "(i14, 1x, i13, 1x, i13, 1x, i13)") &
         total_lines, ptr_lines, row_lines, val_lines
    write(iunit, "(a3, 11x, i14, 1x, i13, 1x, i13, 1x, i13)") &
         type, m, n, ptr(n+1)-1, 0 ! last entry is explicitly zero by RB spec
    write(iunit, "(a16, a16, a20)") &
         ptr_format, row_format, options%val_format

    ! Write matrix
    write(iunit, ptr_format) ptr(:)
    write(iunit, row_format) row(:)
    if (present(val)) &
         write(iunit, options%val_format) val(:)

    ! Close file
    close(iunit)
  end subroutine rb_write_double_int64

  character(len=16) function create_format(per_line, prec)
    implicit none
    integer, intent(in) :: per_line
    integer, intent(in) :: prec

    ! We assume inputs are both < 100
    if (per_line .lt. 10) then
       if (prec .lt. 10) then
          write(create_format, "('(',i1,'i',i1,')')") per_line, prec
       else ! prec >= 10
          write(create_format, "('(',i1,'i',i2,')')") per_line, prec
       end if
    else ! per_line >= 10
       if (prec .lt. 10) then
          write(create_format, "('(',i2,'i',i1,')')") per_line, prec
       else ! prec >= 10
          write(create_format, "('(',i2,'i',i2,')')") per_line, prec
       end if
    end if
  end function create_format

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> \brief Convert symmetric matrix to skew symmetric.
   !>
   !> Sets all entries in the upper triangle to minus their original value.
   !> (i.e. this is a no-op if matrix is only stored as lower triangle).
  subroutine sym_to_skew(n, ptr, row, val)
    implicit none
    integer, intent(inout) :: n
    integer(long), dimension(n+1), intent(inout) :: ptr
    integer, dimension(:), allocatable, intent(inout) :: row
    real(wp), dimension(ptr(n+1)-1), intent(inout) :: val

    integer :: i
    integer(long) :: j

    ! CSC format
    do i = 1, n
       do j = ptr(i), ptr(i+1)-1
          if (row(j) .ge. i) cycle ! in lower triangle
          val(j) = -val(j)
       end do
    end do
  end subroutine sym_to_skew

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> \brief Transpose a symmetric matrix
   !>
   !> To reduce copying we supply the destination integer matrix distinct from
   !> the source. The destination val and ptr arrays is the same as the source
   !> (if required). The matrix must be symmetric.
  subroutine flip_lwr_upr(n, ptr, row, col, st, val)
    implicit none
    integer, intent(in) :: n ! Number of rows.columns in matrix (is symmetric)
    integer(long), dimension(n+1), intent(inout) :: ptr ! ptrs into row/col
    integer, dimension(ptr(n+1)-1), intent(in) :: row ! source index array
    integer, dimension(ptr(n+1)-1), intent(out) :: col ! destination index a.
    integer, intent(out) :: st ! stat parameter for allocates
    real(wp), dimension(ptr(n+1)-1), optional, intent(inout) :: val ! numeric
      ! values can be flipped as well, if required (indiciated by presence)

    integer(long) :: i ! loop indices
    integer :: r, c ! loop indices
    integer, dimension(:), allocatable :: wptr ! working copy of ptr
    real(wp), dimension(:), allocatable :: wval ! working copy of val

    ! Allocate memory
    allocate(wptr(n+2), stat=st)
    if (st .ne. 0) return
    if (present(val)) allocate(wval(ptr(n+1)-1), stat=st)
    if (st .ne. 0) return

    ! Count number of entries in row r as wptr(r+2)
    wptr(2:n+2) = 0
    do c = 1, n
       do i = ptr(c), ptr(c+1)-1
          r = row(i)
          wptr(r+2) = wptr(r+2) + 1
       end do
    end do

    ! Determine insert point for row r as wptr(r+1)
    wptr(1:2) = 1
    do r = 1, n
       wptr(r+2) = wptr(r+1) + wptr(r+2)
    end do

    ! Now loop over matrix inserting entries at correct points
    if (present(val)) then
       do c = 1, n
          do i = ptr(c), ptr(c+1)-1
             r = row(i)
             col(wptr(r+1)) = c
             wval(wptr(r+1)) = val(i)
             wptr(r+1) = wptr(r+1) + 1
          end do
       end do
    else
       do c = 1, n
          do i = ptr(c), ptr(c+1)-1
             r = row(i)
             col(wptr(r+1)) = c
             wptr(r+1) = wptr(r+1) + 1
          end do
       end do
    end if

    ! Finally copy data back to where it needs to be
    ptr(1:n+1) = wptr(1:n+1)
    if (present(val)) val(1:ptr(n+1)-1) = wval(1:ptr(n+1)-1)
  end subroutine flip_lwr_upr

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> Add any missing values to matrix
  subroutine add_missing_diag(m, n, ptr, row, val)
    implicit none
    integer, intent(in) :: m
    integer, intent(in) :: n
    integer(long), dimension(n+1), intent(inout) :: ptr
    integer, dimension(:), intent(inout) :: row
    real(wp), dimension(*), optional, intent(inout) :: val

    integer :: col
    integer(long) :: i
    integer :: ndiag
    logical :: found

    ! Count number of missing diagonal entries
    ndiag = 0
    do col = 1, min(m,n)
       do i = ptr(col), ptr(col+1)-1
          if (row(i) .eq. col) ndiag = ndiag + 1
       end do
    end do

    ndiag = min(m,n) - ndiag ! Determine number missing

    ! Process matrix, adding diagonal entries as first entry in column if
    ! not otherwise present
    do col = n, 1, -1
       if (ndiag .eq. 0) return
       found = .false.
       if (present(val)) then
          do i = ptr(col+1)-1, ptr(col), -1
             found = ( found .or. (row(i) .eq. col) )
             row(i+ndiag) = row(i)
             val(i+ndiag) = val(i)
          end do
       else
          do i = ptr(col+1)-1, ptr(col), -1
             found = ( found .or. (row(i) .eq. col) )
             row(i+ndiag) = row(i)
          end do
       end if
       ptr(col+1) = ptr(col+1) + ndiag
       if ((.not. found) .and. (col .le. m)) then
          ! Note: only adding diagonal if we're in the square submatrix!
          ndiag = ndiag - 1
          i = ptr(col) + ndiag
          row(i) = col
          if (present(val)) val(i) = zero
       end if
    end do
  end subroutine add_missing_diag

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !>  Read data from file: Real-valued version
  subroutine read_data_real(lunit, n, nnz, ptr, row, iost, val)
    implicit none
    integer, intent(in) :: lunit !< unit from which to read data
    integer, intent(in) :: n !< Number of columns to read
    integer(long), intent(in) :: nnz ! Number of entries to read
    integer(long), dimension(*), intent(out) :: ptr ! Column pointers
    integer, dimension(*), intent(out) :: row ! Row indices
    integer, intent(out) :: iost ! iostat from failed op, or 0
    real(wp), dimension(*), optional, intent(out) :: val ! If present,
      ! returns the numerical data.

    character(len=80) :: buffer1, buffer2, buffer3
    character(len=16) :: ptr_format, row_format
    character(len=20) :: val_format

    ! Skip past header information that isn't formats
    read (lunit,'(a80/a80/a80)', iostat=iost) buffer1, buffer2, buffer3
    if (iost .ne. 0) return

    ! Read formats
    read(lunit,'(2a16,a20)',iostat=iost) ptr_format, row_format, val_format
    if (iost .ne. 0) return

    ! Read column pointers
    read(lunit,ptr_format,iostat=iost) ptr(1:n+1)
    if (iost .ne. 0) return

    ! Read row indices
    read(lunit,row_format,iostat=iost) row(1:nnz)
    if (iost .ne. 0) return

    ! Read values if desired
    if (present(val)) then
       read(lunit,val_format,iostat=iost) val(1:nnz)
       if (iost .ne. 0) return
    end if
  end subroutine read_data_real

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !>  Read data from file: Integer-valued version
  subroutine read_data_integer(lunit, n, nnz, ptr, row, iost, val)
    implicit none
    integer, intent(in) :: lunit !< unit from which to read data
    integer, intent(in) :: n !< Number of columns to read
    integer(long), intent(in) :: nnz ! Number of entries to read
    integer(long), dimension(*), intent(out) :: ptr ! Column pointers
    integer, dimension(*), intent(out) :: row ! Row indices
    integer, intent(out) :: iost ! iostat from failed op, or 0
    integer, dimension(*), optional, intent(out) :: val ! If present,
      ! returns the numerical data.

    character(len=80) :: buffer1, buffer2, buffer3
    character(len=16) :: ptr_format, row_format
    character(len=20) :: val_format

    ! Skip past header information that isn't formats
    read (lunit,'(a80/a80/a80)', iostat=iost) buffer1, buffer2, buffer3
    if (iost .ne. 0) return

    ! Read formats
    read(lunit,'(2a16,a20)',iostat=iost) ptr_format, row_format, val_format
    if (iost .ne. 0) return

    ! Read column pointers
    read(lunit,ptr_format,iostat=iost) ptr(1:n+1)
    if (iost .ne. 0) return

    ! Read row indices
    read(lunit,row_format,iostat=iost) row(1:nnz)
    if (iost .ne. 0) return

    ! Read values if desired
    if (present(val)) then
       read(lunit,val_format,iostat=iost) val(1:nnz)
       if(iost .ne. 0) return
    end if
  end subroutine read_data_integer

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> Convert SPRAL matrix type code to type_code(2:2) character
  character(len=1) function matrix_type_to_sym(matrix_type)
    implicit none
    integer, intent(in) :: matrix_type

    select case (matrix_type)
    case(SPRAL_MATRIX_UNSPECIFIED)
       matrix_type_to_sym = "r"
    case(SPRAL_MATRIX_REAL_RECT)
       matrix_type_to_sym = "r"
    case(SPRAL_MATRIX_REAL_UNSYM)
       matrix_type_to_sym = "u"
    case(SPRAL_MATRIX_REAL_SYM_PSDEF)
       matrix_type_to_sym = "s"
    case(SPRAL_MATRIX_REAL_SYM_INDEF)
       matrix_type_to_sym = "s"
    case(SPRAL_MATRIX_REAL_SKEW)
       matrix_type_to_sym = "z"
    end select
  end function matrix_type_to_sym

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !> Convert type_code(2:2) character to SPRAL matrix type code
  integer function sym_to_matrix_type(sym)
    implicit none
    character(len=1), intent(in) :: sym

    select case (sym)
    case("r")
       sym_to_matrix_type = SPRAL_MATRIX_REAL_RECT
    case("s")
       sym_to_matrix_type = SPRAL_MATRIX_REAL_SYM_INDEF
    case("u")
       sym_to_matrix_type = SPRAL_MATRIX_REAL_UNSYM
    case("z")
       sym_to_matrix_type = SPRAL_MATRIX_REAL_SKEW
    case default
       ! This should never happen
       sym_to_matrix_type = SPRAL_MATRIX_UNSPECIFIED
    end select
  end function sym_to_matrix_type

end module spral_rutherford_boeing
