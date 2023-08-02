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
.. _doxid-structllst__time__type_1ad3803b3bb79c5c74d9300520fbe733f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T total

total CPU time spent in the package

.. index:: pair: variable; assemble
.. _doxid-structllst__time__type_1ae84d232eee798a974ebaeb9c82d623f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T assemble

CPU time assembling $K(\lambda)$ in (1)

.. index:: pair: variable; analyse
.. _doxid-structllst__time__type_1a9c5b9155e1665977103d8c32881d9f00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T analyse

CPU time spent analysing $K(\lambda)$.

.. index:: pair: variable; factorize
.. _doxid-structllst__time__type_1a79e62dbb4cbb6e99d82167e60c703015:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorize

CPU time spent factorizing $K(\lambda)$.

.. index:: pair: variable; solve
.. _doxid-structllst__time__type_1a4c971b10c915041b89daa05a29125376:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T solve

CPU time spent solving linear systems inolving $K(\lambda)$.

.. index:: pair: variable; clock_total
.. _doxid-structllst__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_assemble
.. _doxid-structllst__time__type_1a4df2b92cea9269b8f8cad7024b83a10d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_assemble

clock time assembling $K(\lambda)$

.. index:: pair: variable; clock_analyse
.. _doxid-structllst__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

clock time spent analysing $K(\lambda)$

.. index:: pair: variable; clock_factorize
.. _doxid-structllst__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

clock time spent factorizing $K(\lambda)$

.. index:: pair: variable; clock_solve
.. _doxid-structllst__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

clock time spent solving linear systems inolving $K(\lambda)$
