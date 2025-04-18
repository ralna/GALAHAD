.. index:: pair: table; sha_inform_type
.. _doxid-structsha__inform__type:

sha_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sha_inform_type{T,INT}
          status::INT
          alloc_status::INT
          max_degree::INT
          differences_needed::INT
          max_reduced_degree::INT
          approximation_algorith_used::INT
          bad_row::INT
          max_off_diagonal_difference::T
          bad_alloc::NTuple{81,Cchar}

.. _details-structsha__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsha__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See SHA_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structsha__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; max_degree
.. _doxid-structsha__inform__type_max_degree:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_degree

the maximum degree in the adgacency graph.

.. index:: pair: variable; differences_needed
.. _doxid-structsha__inform__type_differences_needed:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT differences_needed

the number of differences that will be needed.

.. index:: pair: variable; max_reduced_degree
.. _doxid-structsha__inform__type_max_reduced_degree:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_reduced_degree

the maximum reduced degree in the adgacency graph.

.. index:: pair: variable; approximation_algorithm_used
.. _doxid-structsha__inform__type_approximation_algorithm_used:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT approximation_algorithm_used

the approximation algorithm actually used

.. index:: pair: variable; bad_row
.. _doxid-structsha__inform__type_bad_row:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT bad_row

a failure occured when forming the bad_row-th row (0 = no failure).


.. index:: pair: variable; max_off_diagonal_difference
.. _doxid-structsha__inform__type_max_off_diagonal_difference:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T max_off_diagonal_difference

the maximum difference between estimated Hessian off-diagonal 
pairs if approximation algorithm 1, 3 or 4 has been employed and
control.average_off_diagonals is true. It will be zero otherwise.

.. index:: pair: variable; bad_alloc
.. _doxid-structsha__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred.
