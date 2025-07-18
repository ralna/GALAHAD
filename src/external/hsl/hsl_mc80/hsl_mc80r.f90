! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 15:00 GMT.

#include "hsl_subset.h"

    MODULE hsl_mc80_real
      PUBLIC :: mc80_available
    CONTAINS
      logical function mc80_available()
        mc80_available = .FALSE.
      end function

      SUBROUTINE mc80r( )
      END SUBROUTINE mc80r
    END MODULE hsl_mc80_real
