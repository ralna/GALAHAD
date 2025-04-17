.. index:: pair: table; sbls_time_type
.. _doxid-structsbls__time__type:

sbls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sbls_time_type{T}
          total::T
          form::T
          factorize::T
          apply::T
          clock_total::T
          clock_form::T
          clock_factorize::T
          clock_apply::T
	
.. _details-structsbls__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structsbls__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

total cpu time spent in the package

.. index:: pair: variable; form
.. _doxid-structsbls__time__type_form:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T form

cpu time spent forming the preconditioner $K_G$

.. index:: pair: variable; factorize
.. _doxid-structsbls__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

cpu time spent factorizing $K_G$

.. index:: pair: variable; apply
.. _doxid-structsbls__time__type_apply:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T apply

cpu time spent solving linear systems inolving $K_G$

.. index:: pair: variable; clock_total
.. _doxid-structsbls__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_form
.. _doxid-structsbls__time__type_clock_form:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_form

clock time spent forming the preconditioner $K_G$

.. index:: pair: variable; clock_factorize
.. _doxid-structsbls__time__type_clock_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

clock time spent factorizing $K_G$

.. index:: pair: variable; clock_apply
.. _doxid-structsbls__time__type_clock_apply:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_apply

clock time spent solving linear systems inolving $K_G$

