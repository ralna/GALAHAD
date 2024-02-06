! THIS VERSION: GALAHAD 4.3 - 2024-02-06 AT 09:50 GMT.

#include "spral_procedures.h"

module spral_pgm
   use spral_kinds, only: ip_
   implicit none

   private
   public :: writePGM,           & ! Grayscale
             writePPM,           & ! Colour
             writeMatrixPattern    ! Utility for writing out pattern as PGM
contains

subroutine writeMatrixPattern(filename, n, ptr, row, lp, rperm, cperm)
   character(len=*), intent(in) :: filename
   integer(ip_), intent(in) :: n
   integer(ip_), dimension(n+1), intent(in) :: ptr
   integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row
   integer(ip_), optional, intent(in) :: lp
   integer(ip_), dimension(n), optional, intent(in) :: rperm
   integer(ip_), dimension(n), optional, intent(in) :: cperm

   integer(ip_), parameter :: funit = 13
   integer(ip_), parameter :: maxxy = 600

   integer(ip_) :: nper, xy
   integer(ip_) :: i, j, p, q, bw, r,s
   integer(ip_) :: img(maxxy, maxxy)
   integer(ip_) :: llp

   llp = 0
   if(present(lp)) llp = lp

   if(llp.ne.0) write(llp, '(2a)') "Writing ", filename

   nper = (n-1)/maxxy + 1
   xy = (n-1)/nper + 1

   do i = 1, xy
      do j = 1, xy
         img(i,j) = nper ! maxgray is white, so set to white initially
      end do
   end do

   bw = 0
   do i = 1, n
      do j = ptr(i), ptr(i+1)-1
         if(present(rperm)) then
            p = (rperm(row(j))-1)/nper + 1
            r = rperm(row(j))
         else
            p = (row(j)-1)/nper + 1
            r = row(j)
         endif
         if(present(cperm)) then
            q = (cperm(i)-1)/nper + 1
            s =  cperm(i)
         else
            q = (i-1)/nper + 1
            s = i
         endif
         !img(p,q) = img(p,q) - 1
         img(p,q) = 1
         img(q,p) = 1
         bw = max(bw, abs(r-s))
      end do
   end do
   !print *, "bw = ", bw

   open(file=filename, unit=funit, status="replace")
   call writePGM(funit, img(1:xy,1:xy))
   close(funit)
end subroutine writeMatrixPattern

!
! Write out a Portable Grey Map (.pgm) file
! values of the array bitmap specify grey level:
! 0              is black
! maxval(bitmap) is white
!
subroutine writePGM(funit, bitmap)
   integer(ip_), intent(in) :: funit
   integer(ip_), dimension(:,:), intent(in) :: bitmap

   integer(ip_) :: m, n, nlvl
   integer(ip_) :: i, j

   m = size(bitmap, 1)
   n = size(bitmap, 2)
   nlvl = maxval(bitmap(:,:))

   write(funit, "(a)") "P2"
   write(funit, "(3i5)") n, m, nlvl
   do i = 1, m ! loop over rows
      do j = 1, n
         write(funit, "(i5)") bitmap(i,j)
      end do
   end do
end subroutine writePGM

!
! Write out a Portable Pixel Map (.ppm) file
! values of the array bitmap(:,:) specify an index of the array color(:,:)
! color(:,:) should have size colours(3, ncolor) where ncolor is the maximum
! number of colors. For a given color i:
! color(1,i) gives the red   component with value between 0 and 255
! color(2,i) gives the green component with value between 0 and 255
! color(3,i) gives the blue  component with value between 0 and 255
!
subroutine writePPM(funit, bitmap, color, scale)
   integer(ip_), intent(in) :: funit
   integer(ip_), dimension(:,:), intent(in) :: bitmap
   integer(ip_), dimension(:,:), intent(in) :: color
   integer(ip_), optional, intent(in) :: scale ! how many pixels  point occupies

   integer(ip_) :: m, n, nlvl
   integer(ip_) :: i, j, c, s1, s2
   integer(ip_) :: scale2

   scale2 = 1
   if(present(scale)) scale2 = scale

   m = size(bitmap, 1)
   n = size(bitmap, 2)
   nlvl = maxval(bitmap(:,:))

   write(funit, "(a)") "P3"
   write(funit, "(3i5)") n*scale, m*scale, 255
   do i = 1, m ! loop over rows
      do s1 = 1, scale2
         do j = 1, n
            c = bitmap(i,j)
            do s2 = 1, scale2
               write(funit, "(3i5)") color(:,c)
            end do
         end do
      end do
   end do
end subroutine writePPM

end module spral_pgm
