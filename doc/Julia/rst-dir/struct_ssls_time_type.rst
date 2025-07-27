.. index:: pair: table; ssls_time_type
.. _doxid-structssls__time__type:

ssls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ssls_time_type{T}
          total::T
          analyse::T
          factorize::T
          solve::T
          clock_total::T
          clock_analyse::T
          clock_factorize::T
          clock_solve::T
	
.. _details-structssls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structssls__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

total cpu time spent in the package

.. index:: pair: variable; analyse
.. _doxid-structssls__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

cpu time spent forming and analysing $K$

.. index:: pair: variable; factorize
.. _doxid-structssls__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

cpu time spent factorizing $K$

.. index:: pair: variable; solve
.. _doxid-structssls__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

cpu time spent solving linear systems inolving $K$

.. index:: pair: variable; clock_total
.. _doxid-structssls__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_analyse
.. _doxid-structssls__time__type_clock_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

clock time spent forming and analysing $K$

.. index:: pair: variable; clock_factorize
.. _doxid-structssls__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

clock time spent factorizing $K$

.. index:: pair: variable; clock_solve
.. _doxid-structssls__time__type_clock_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

clock time spent solving linear systems inolving $K$

