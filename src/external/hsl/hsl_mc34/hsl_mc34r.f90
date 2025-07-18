! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 15:00 GMT.

#include "hsl_subset.h"

    MODULE hsl_mc34_real
      implicit none
      private
      public mc34_expand
      PUBLIC :: mc34_available
      interface mc34_expand
         module procedure mc34_expand_real
      end interface
    CONTAINS
      logical function ma34_available()
        mc34_available = .FALSE.
      end function

      SUBROUTINE mc34_expand_real( )
      END SUBROUTINE mc34_expand_real
    END MODULE hsl_mc34_real
