.. index:: pair: table; lsrt_control_type
.. _doxid-structlsrt__control__type:

lsrt_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lsrt_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          print_gap::INT
          itmin::INT
          itmax::INT
          bitmax::INT
          extra_vectors::INT
          stopping_rule::INT
          freq::INT
          stop_relative::T
          stop_absolute::T
          fraction_opt::T
          time_limit::T
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}

.. _details-structlsrt__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlsrt__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlsrt__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structlsrt__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structlsrt__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structlsrt__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structlsrt__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structlsrt__control__type_print_gap:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_gap

the number of iterations between printing

.. index:: pair: variable; itmin
.. _doxid-structlsrt__control__type_itmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT itmin

the minimum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; itmax
.. _doxid-structlsrt__control__type_itmax:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT itmax

the maximum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; bitmax
.. _doxid-structlsrt__control__type_bitmax:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT bitmax

the maximum number of Newton inner iterations per outer iteration allowed (-ve = no bound)

.. index:: pair: variable; extra_vectors
.. _doxid-structlsrt__control__type_extra_vectors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT extra_vectors

the number of extra work vectors of length n used

.. index:: pair: variable; stopping_rule
.. _doxid-structlsrt__control__type_stopping_rule:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stopping_rule

the stopping rule used: 0=1.0, 1=norm step, 2=norm step/sigma (NOT USED)

.. index:: pair: variable; freq
.. _doxid-structlsrt__control__type_freq:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT freq

frequency for solving the reduced tri-diagonal problem (NOT USED)

.. index:: pair: variable; stop_relative
.. _doxid-structlsrt__control__type_stop_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_relative

the iteration stops successfully when \|\|A^Tr\|\| is less than max( stop_relative \* \|\|A^Tr initial \|\|, stop_absolute )

.. index:: pair: variable; stop_absolute
.. _doxid-structlsrt__control__type_stop_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_absolute

see stop_relative

.. index:: pair: variable; fraction_opt
.. _doxid-structlsrt__control__type_fraction_opt:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T fraction_opt

an estimate of the solution that gives at least .fraction_opt times the optimal objective value will be found

.. index:: pair: variable; time_limit
.. _doxid-structlsrt__control__type_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T time_limit

the maximum elapsed time allowed (-ve means infinite)

.. index:: pair: variable; space_critical
.. _doxid-structlsrt__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlsrt__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structlsrt__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

