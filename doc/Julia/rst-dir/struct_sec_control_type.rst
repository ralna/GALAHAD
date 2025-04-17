.. index:: pair: table; sec_control_type
.. _doxid-structsec__control__type:

sec_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sec_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          h_initial::T
          update_skip_tol::T
          prefix::NTuple{31,Cchar}

.. _details-structsec__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsec__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsec__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structsec__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structsec__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required. <= 0 gives no output, >= 1 warning message

.. index:: pair: variable; h_initial
.. _doxid-structsec__control__type_h_initial:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T h_initial

the initial Hessian approximation will be h_initial \* $I$

.. index:: pair: variable; update_skip_tol
.. _doxid-structsec__control__type_update_skip_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T update_skip_tol

an update is skipped if the resulting matrix would have grown too much; specifically it is skipped when y^T s / y^T y <= update_skip_tol.

.. index:: pair: variable; prefix
.. _doxid-structsec__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

