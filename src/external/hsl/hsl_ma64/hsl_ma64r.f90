! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 14:30 GMT.

#include "hsl_subset.h"

    MODULE hsl_ma64_real
      PUBLIC :: ma64_available
    CONTAINS
      logical function ma64_available()
        ma64_available = .FALSE.
      end function

      SUBROUTINE ma64r( )
      END SUBROUTINE ma64r
    END MODULE hsl_ma64_real
