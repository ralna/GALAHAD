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
.. _doxid-structbqp__time__type_1aa7b2ccce10ffc8ef240d5be56ec1fbbc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 total

total time

.. index:: pair: variable; analyse
.. _doxid-structbqp__time__type_1a0ca2b20748c7749a77d684124011c531:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 analyse

time for the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structbqp__time__type_1ab7eecce4b013c87e490b8984c74c59c3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 factorize

time for the factorization phase

.. index:: pair: variable; solve
.. _doxid-structbqp__time__type_1a6356532c25755a6e5fedee1a7d703949:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Float32 solve

time for the linear solution phase

