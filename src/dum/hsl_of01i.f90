#include "galahad_modules.h"

    module hsl_of01_integer
      USE GALAHAD_KINDS
      private
      public of01_data
      type of01_data
        integer ( kind = ip_ ) :: dum = 0
      end type of01_data
    end module hsl_of01_integer
