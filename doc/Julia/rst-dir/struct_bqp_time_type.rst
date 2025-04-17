.. index:: pair: struct; bqp_time_type
.. _doxid-structbqp__time__type:

bqp_time_type structure
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct bqp_time_type
          total::Float32
          analyse::Float32
          factorize::Float32
          solve::Float32

.. _details-structbqp__time__type:

detailed documentation
----------------------

time derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structbqp__time__type_total:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

total time

.. index:: pair: variable; analyse
.. _doxid-structbqp__time__type_analyse:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 analyse

time for the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structbqp__time__type_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 factorize

time for the factorization phase

.. index:: pair: variable; solve
.. _doxid-structbqp__time__type_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 solve

time for the linear solution phase

