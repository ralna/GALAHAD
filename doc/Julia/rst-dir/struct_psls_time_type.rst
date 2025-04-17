.. index:: pair: table; psls_time_type
.. _doxid-structpsls__time__type:

psls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct psls_time_type{T}
          total::Float32
          assemble::Float32
          analyse::Float32
          factorize::Float32
          solve::Float32
          update::Float32
          clock_total::T
          clock_assemble::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T
          clock_update::T
	
.. _details-structpsls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structpsls__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

total time

.. index:: pair: variable; assemble
.. _doxid-structpsls__time__type_assemble:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 assemble

time to assemble the preconditioner prior to factorization

.. index:: pair: variable; analyse
.. _doxid-structpsls__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 analyse

time for the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structpsls__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 factorize

time for the factorization phase

.. index:: pair: variable; solve
.. _doxid-structpsls__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 solve

time for the linear solution phase

.. index:: pair: variable; update
.. _doxid-structpsls__time__type_update:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 update

time to update the factorization

.. index:: pair: variable; clock_total
.. _doxid-structpsls__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_assemble
.. _doxid-structpsls__time__type_clock_assemble:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_assemble

clock time to assemble the preconditioner prior to factorization

.. index:: pair: variable; clock_analyse
.. _doxid-structpsls__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

clock time for the analysis phase

.. index:: pair: variable; clock_factorize
.. _doxid-structpsls__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

clock time for the factorization phase

.. index:: pair: variable; clock_solve
.. _doxid-structpsls__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

clock time for the linear solution phase

.. index:: pair: variable; clock_update
.. _doxid-structpsls__time__type_clock_update:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_update

clock time to update the factorization

