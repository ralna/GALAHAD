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
.. _doxid-structpsls__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

total time

.. index:: pair: variable; assemble
.. _doxid-structpsls__time__type_1a10940eefc2f59c72a3ecc6cb4f44e233:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 assemble

time to assemble the preconditioner prior to factorization

.. index:: pair: variable; analyse
.. _doxid-structpsls__time__type_1a0ca2b20748c7749a77d684124011c531:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 analyse

time for the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structpsls__time__type_1ab7eecce4b013c87e490b8984c74c59c3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 factorize

time for the factorization phase

.. index:: pair: variable; solve
.. _doxid-structpsls__time__type_1a6356532c25755a6e5fedee1a7d703949:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 solve

time for the linear solution phase

.. index:: pair: variable; update
.. _doxid-structpsls__time__type_1abe9f0d8cfe95c5d6b3fb64a0c1e6e55f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 update

time to update the factorization

.. index:: pair: variable; clock_total
.. _doxid-structpsls__time__type_1ae9145eea8e19f9cae77904d3d00c5d1f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_total

total clock time spent in the package

.. index:: pair: variable; clock_assemble
.. _doxid-structpsls__time__type_1a4df2b92cea9269b8f8cad7024b83a10d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_assemble

clock time to assemble the preconditioner prior to factorization

.. index:: pair: variable; clock_analyse
.. _doxid-structpsls__time__type_1a3394e706afb175d930c81c4b86fe8f4b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_analyse

clock time for the analysis phase

.. index:: pair: variable; clock_factorize
.. _doxid-structpsls__time__type_1ad3f0f50628260b90d6cf974e02f86192:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_factorize

clock time for the factorization phase

.. index:: pair: variable; clock_solve
.. _doxid-structpsls__time__type_1af569df4b8828eb7ac8a05ef1030d1358:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_solve

clock time for the linear solution phase

.. index:: pair: variable; clock_update
.. _doxid-structpsls__time__type_1acb90c4a0e3e2434c815d1428316c8ee9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_update

clock time to update the factorization

