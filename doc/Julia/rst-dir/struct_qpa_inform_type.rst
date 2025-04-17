.. index:: pair: struct; qpa_inform_type
.. _doxid-structqpa__inform__type:

qpa_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct qpa_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          major_iter::INT
          iter::INT
          cg_iter::INT
          factorization_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          nfacts::INT
          nmods::INT
          num_g_infeas::INT
          num_b_infeas::INT
          obj::T
          infeas_g::T
          infeas_b::T
          merit::T
          time::qpa_time_type{T}
          sls_inform::sls_inform_type{T,INT}

.. _details-structqpa__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structqpa__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See QPA_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structqpa__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structqpa__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; major_iter
.. _doxid-structqpa__inform__type_major_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT major_iter

the total number of major iterations required

.. index:: pair: variable; iter
.. _doxid-structqpa__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structqpa__inform__type_cg_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structqpa__inform__type_factorization_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structqpa__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structqpa__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structqpa__inform__type_nfacts:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nfacts

the total number of factorizations performed

.. index:: pair: variable; nmods
.. _doxid-structqpa__inform__type_nmods:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nmods

the total number of factorizations which were modified to ensure that th matrix was an appropriate preconditioner

.. index:: pair: variable; num_g_infeas
.. _doxid-structqpa__inform__type_num_g_infeas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT num_g_infeas

the number of infeasible general constraints

.. index:: pair: variable; num_b_infeas
.. _doxid-structqpa__inform__type_num_b_infeas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT num_b_infeas

the number of infeasible simple-bound constraints

.. index:: pair: variable; obj
.. _doxid-structqpa__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by QPA_solve

.. index:: pair: variable; infeas_g
.. _doxid-structqpa__inform__type_infeas_g:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infeas_g

the 1-norm of the infeasibility of the general constraints

.. index:: pair: variable; infeas_b
.. _doxid-structqpa__inform__type_infeas_b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infeas_b

the 1-norm of the infeasibility of the simple-bound constraints

.. index:: pair: variable; merit
.. _doxid-structqpa__inform__type_merit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T merit

the merit function value = obj + rho_g \* infeas_g + rho_b \* infeas_b

.. index:: pair: variable; time
.. _doxid-structqpa__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`qpa_time_type<doxid-structqpa__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structqpa__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters for SLS

