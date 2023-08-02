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
.. _doxid-structsbls__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

total cpu time spent in the package

.. index:: pair: variable; form
.. _doxid-structsbls__time__type_1a8ac63de5e103d8e01b0e0f88bb7d230d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T form

cpu time spent forming the preconditioner $K_G$

.. index:: pair: variable; factorize
.. _doxid-structsbls__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

cpu time spent factorizing $K_G$

.. index:: pair: variable; apply
.. _doxid-structsbls__time__type_1a9d8129bf5b1a9f21dfcc24dc5c706274:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T apply

cpu time spent solving linear systems inolving $K_G$

.. index:: pair: variable; clock_total
.. _doxid-structsbls__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_form
.. _doxid-structsbls__time__type_1ab275f3b71b8e019aa35acf43c3fd7473:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_form

clock time spent forming the preconditioner $K_G$

.. index:: pair: variable; clock_factorize
.. _doxid-structsbls__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

clock time spent factorizing $K_G$

.. index:: pair: variable; clock_apply
.. _doxid-structsbls__time__type_1afbbb1dd5fc63c640620fbd32a0481493:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_apply

clock time spent solving linear systems inolving $K_G$

