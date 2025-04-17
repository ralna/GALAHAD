.. index:: pair: table; lms_time_type
.. _doxid-structlms__time__type:

lms_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lms_time_type{T}
          total::T
          setup::T
          form::T
          apply::T
          clock_total::T
          clock_setup::T
          clock_form::T
          clock_apply::T

.. _details-structlms__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structlms__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

total cpu time spent in the package

.. index:: pair: variable; setup
.. _doxid-structlms__time__type_setup:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T setup

cpu time spent setting up space for the secant approximation

.. index:: pair: variable; form
.. _doxid-structlms__time__type_form:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T form

cpu time spent updating the secant approximation

.. index:: pair: variable; apply
.. _doxid-structlms__time__type_apply:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T apply

cpu time spent applying the secant approximation

.. index:: pair: variable; clock_total
.. _doxid-structlms__time__type_clock_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_setup
.. _doxid-structlms__time__type_clock_setup:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_setup

clock time spent setting up space for the secant approximation

.. index:: pair: variable; clock_form
.. _doxid-structlms__time__type_clock_form:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_form

clock time spent updating the secant approximation

.. index:: pair: variable; clock_apply
.. _doxid-structlms__time__type_clock_apply:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_apply

clock time spent applying the secant approximation

