.. index:: pair: struct; dgo_inform_type
.. _doxid-structdgo__inform__type:

dgo_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct dgo_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          f_eval::INT
          g_eval::INT
          h_eval::INT
          obj::T
          norm_pg::T
          length_ratio::T
          f_gap::T
          why_stop::NTuple{2,Cchar}
          time::dgo_time_type{T}
          hash_inform::hash_inform_type{INT}
          ugo_inform::ugo_inform_type{T,INT}
          trb_inform::trb_inform_type{T,INT}

.. _details-structdgo__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure


components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structdgo__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See DGO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structdgo__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structdgo__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structdgo__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations performed

.. index:: pair: variable; f_eval
.. _doxid-structdgo__inform__type_f_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structdgo__inform__type_g_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structdgo__inform__type_h_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; obj
.. _doxid-structdgo__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by DGO_solve

.. index:: pair: variable; norm_pg
.. _doxid-structdgo__inform__type_norm_pg:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_pg

the norm of the projected gradient of the objective function at the best estimate of the solution determined by DGO_solve

.. index:: pair: variable; length_ratio
.. _doxid-structdgo__inform__type_length_ratio:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T length_ratio

the ratio of the final to the initial box lengths

.. index:: pair: variable; f_gap
.. _doxid-structdgo__inform__type_f_gap:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T f_gap

the gap between the best objective value found and the lowest bound

.. index:: pair: variable; why_stop
.. _doxid-structdgo__inform__type_why_stop:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char why_stop[2]

why did the iteration stop? This wil be 'D' if the box length is small enough, 'F' if the objective gap is small enough, and ' ' otherwise

.. index:: pair: variable; time
.. _doxid-structdgo__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`dgo_time_type<doxid-structdgo__time__type>` time

timings (see above)

.. index:: pair: variable; hash_inform
.. _doxid-structdgo__inform__type_hash_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`hash_inform_type<doxid-structhash__inform__type>` hash_inform

inform parameters for HASH

.. index:: pair: variable; ugo_inform
.. _doxid-structdgo__inform__type_ugo_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ugo_inform_type<doxid-structugo__inform__type>` ugo_inform

inform parameters for UGO

.. index:: pair: variable; trb_inform
.. _doxid-structdgo__inform__type_trb_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`trb_inform_type<doxid-structtrb__inform__type>` trb_inform

inform parameters for TRB

