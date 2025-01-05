.. index:: pair: struct; nls_inform_type
.. _doxid-structnls__inform__type:

nls_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct nls_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          bad_eval::NTuple{13,Cchar}
          iter::INT
          cg_iter::INT
          c_eval::INT
          j_eval::INT
          h_eval::INT
          factorization_max::INT
          factorization_status::INT
          max_entries_factors::Int64
          factorization_integer::Int64
          factorization_real::Int64
          factorization_average::T
          obj::T
          norm_c::T
          norm_g::T
          weight::T
          time::nls_time_type{T}
          rqs_inform::rqs_inform_type{T,INT}
          glrt_inform::glrt_inform_type{T,INT}
          psls_inform::psls_inform_type{T,INT}
          bsc_inform::bsc_inform_type{T,INT}
          roots_inform::roots_inform_type{INT}
          subproblem_inform::nls_subproblem_inform_type{T,INT}

.. _details-structnls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structnls__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See NLS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structnls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structnls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; bad_eval
.. _doxid-structnls__inform__type_1a184c27298dc565470437c213a2bd2f3e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char bad_eval[13]

the name of the user-supplied evaluation routine for which an error occurred

.. index:: pair: variable; iter
.. _doxid-structnls__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structnls__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of CG iterations performed

.. index:: pair: variable; c_eval
.. _doxid-structnls__inform__type_1ab8312e1defeefffdcc0b5956bcb31ad4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT c_eval

the total number of evaluations of the residual function c(x)

.. index:: pair: variable; j_eval
.. _doxid-structnls__inform__type_1a47a079918ad01b32fd15ed6a0b8bd581:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT j_eval

the total number of evaluations of the Jacobian J(x) of c(x)

.. index:: pair: variable; h_eval
.. _doxid-structnls__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT h_eval

the total number of evaluations of the scaled Hessian H(x,y) of c(x)

.. index:: pair: variable; factorization_max
.. _doxid-structnls__inform__type_1a97dadabf3b7bdf921c4dcd1f43129f05:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; factorization_status
.. _doxid-structnls__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; max_entries_factors
.. _doxid-structnls__inform__type_1a177e429e737cfa2cd3df051a65fcfb68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structnls__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structnls__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; factorization_average
.. _doxid-structnls__inform__type_1a42d0c89df887685f68327d07c6e92f05:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorization_average

the average number of factorizations per sub-problem solve

.. index:: pair: variable; obj
.. _doxid-structnls__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function $\frac{1}{2}\|c(x)\|^2_W$ at the best estimate the solution, x, determined by NLS_solve

.. index:: pair: variable; norm_c
.. _doxid-structnls__inform__type_1a4969b17b30edb63a6bbcb89c7c10a340:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_c

the norm of the residual $\|c(x)\|_W$ at the best estimate of the solution x, determined by NLS_solve

.. index:: pair: variable; norm_g
.. _doxid-structnls__inform__type_1ae1bc0a751c6ede62421bbc49fbe7d9fe:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_g

the norm of the gradient of $\|c(x)\|_W$ of the objective function at the best estimate, x, of the solution determined by NLS_solve

.. index:: pair: variable; weight
.. _doxid-structnls__inform__type_1adcd20aeaf7042e972ddab56f3867ce70:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight

the final regularization weight used

.. index:: pair: variable; time
.. _doxid-structnls__inform__type_1a44dc03b1a33bf900f668c713cbac9498:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`nls_time_type<doxid-structnls__time__type>` time

timings (see above)

.. index:: pair: variable; rqs_inform
.. _doxid-structnls__inform__type_1a68497e7bbd1695ac9b830fc8fe594d60:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` rqs_inform

inform parameters for RQS

.. index:: pair: variable; glrt_inform
.. _doxid-structnls__inform__type_1aa5a47a840c1f9680ac8b9e4db3eb9e88:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` glrt_inform

inform parameters for GLRT

.. index:: pair: variable; psls_inform
.. _doxid-structnls__inform__type_1a57ca5ed37882eb917736f845d3cdb8ee:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; bsc_inform
.. _doxid-structnls__inform__type_1ab95c5e6786b9d93eb147f64fbf14da17:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` bsc_inform

inform parameters for BSC

.. index:: pair: variable; roots_inform
.. _doxid-structnls__inform__type_1a68574d04a336f7be88a151fa8b975885:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`roots_inform_type<doxid-structroots__inform__type>` roots_inform

inform parameters for ROOTS

.. index:: pair: variable; subproblem_inform
.. _doxid-structnls__inform__type_1afe321f4b9cfc27d8927047e53e2f288a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`nls_subproblem_inform_type<doxid-structnls__subproblem__inform__type>` subproblem_inform

inform parameters for subproblem


