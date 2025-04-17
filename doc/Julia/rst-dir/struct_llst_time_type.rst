.. index:: pair: table; llst_time_type
.. _doxid-structllst__time__type:

llst_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct llst_time_type{T}
          total::T
          assemble::T
          analyse::T
          factorize::T
          solve::T
          clock_total::T
          clock_assemble::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T

.. _details-structllst__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structllst__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

total CPU time spent in the package

.. index:: pair: variable; assemble
.. _doxid-structllst__time__type_assemble:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T assemble

CPU time assembling $K(\lambda)$ in (1)

.. index:: pair: variable; analyse
.. _doxid-structllst__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

CPU time spent analysing $K(\lambda)$.

.. index:: pair: variable; factorize
.. _doxid-structllst__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

CPU time spent factorizing $K(\lambda)$.

.. index:: pair: variable; solve
.. _doxid-structllst__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

CPU time spent solving linear systems inolving $K(\lambda)$.

.. index:: pair: variable; clock_total
.. _doxid-structllst__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_assemble
.. _doxid-structllst__time__type_clock_assemble:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_assemble

clock time assembling $K(\lambda)$

.. index:: pair: variable; clock_analyse
.. _doxid-structllst__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

clock time spent analysing $K(\lambda)$

.. index:: pair: variable; clock_factorize
.. _doxid-structllst__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

clock time spent factorizing $K(\lambda)$

.. index:: pair: variable; clock_solve
.. _doxid-structllst__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

clock time spent solving linear systems inolving $K(\lambda)$
