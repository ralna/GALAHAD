! COPYRIGHT (c) 2014 Science and Technology Facilities Council
! Authors: Jonathan Hogg
!
! Implementation of simple LCG PRNG
! Parameters as in glibc
module spral_random
  implicit none

  private
  public :: random_real,& ! Returns random real
       random_integer,  & ! Returns random integer
       random_logical,  & ! Returns random logical
       random_get_seed, & ! Get seed of generator
       random_set_seed    ! Set seed of generator
  public :: random_state  ! State type

  integer, parameter :: wp = kind(0d0)
  integer, parameter :: long = selected_int_kind(18)

  ! LCG data
  integer(long), parameter :: a = 1103515245
  integer(long), parameter :: c = 12345
  integer(long), parameter :: m = 2**31_long

  ! Store random generator state
  type :: random_state
     private
     integer :: x = 486502
  end type random_state

  interface random_integer
     module procedure random_integer32, random_integer64
  end interface random_integer

contains

  !
  ! Get random seed
  !
  integer function random_get_seed(state)
    implicit none
    type(random_state), intent(in) :: state

    random_get_seed = state%x
  end function random_get_seed

  !
  ! Set random seed
  !
  subroutine random_set_seed(state, seed)
    implicit none
    type(random_state), intent(inout) :: state
    integer, intent(in) :: seed

    state%x = seed
  end subroutine random_set_seed


  !
  !  Real random number in the range
  !  [ 0, 1] (if positive is present and .TRUE.); or
  !  [-1, 1] (otherwise)
  !
  real(wp) function random_real(state, positive)
    implicit none
    type(random_state), intent(inout) :: state
    logical, optional, intent(in) :: positive

    logical :: pos

    pos = .false.
    if (present(positive)) pos = positive

    ! X_{n+1} = (aX_n + c) mod m
    state%x = int(mod(a*state%x+c, m))

    ! Convert to a random real
    if (pos) then
       random_real = real(state%x,wp) / real(m,wp)
    else
       random_real = 1.0 - 2.0*real(state%x,wp)/real(m,wp)
    end if
  end function random_real

  !
  !  Integer random number in the range [1,n] if n > 1.
  !  otherwise, the value n is returned
  !
  integer(long) function random_integer64(state, n)
    implicit none
    type(random_state), intent(inout) :: state
    integer(long), intent(in) :: n

    if (n .le. 0) then
       random_integer64 = n
       return
    end if

    ! X_{n+1} = (aX_n + c) mod m
    state%x = int(mod(a*state%x+c, m))

    ! Take modulo n for return value
    random_integer64 = int(state%x * (real(n,wp)/real(m,wp)), long) + 1
  end function random_integer64

  !
  !  Integer random number in the range [1,n] if n > 1.
  !  otherwise, the value n is returned
  !
  integer function random_integer32(state, n)
    implicit none
    type(random_state), intent(inout) :: state
    integer, intent(in) :: n

    ! Just call 64-bit version with type casts
    random_integer32 = int(random_integer64(state, int(n,long)))
  end function random_integer32

  !
  !  Generate a random logical value
  !
  logical function random_logical(state)
    implicit none
    type(random_state), intent(inout) :: state

    integer :: test

    test = random_integer(state, 2)
    random_logical = (test .eq. 1)
  end function random_logical

end module spral_random
