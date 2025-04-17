.. index:: pair: struct; bgo_inform_type
.. _doxid-structbgo__inform__type:

bgo_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

	#include <galahad_bgo.h>
	
        struct bgo_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          f_eval::INT
          g_eval::INT
          h_eval::INT
          obj::T
          norm_pg::T
          time::bgo_time_type{T}
          ugo_inform::ugo_inform_type{T,INT}
          lhs_inform::lhs_inform_type{INT}
          trb_inform::trb_inform_type{T,INT}

.. _details-structbgo__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structbgo__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See BGO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structbgo__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structbgo__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; f_eval
.. _doxid-structbgo__inform__type_f_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structbgo__inform__type_g_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structbgo__inform__type_h_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; obj
.. _doxid-structbgo__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by BGO_solve

.. index:: pair: variable; norm_pg
.. _doxid-structbgo__inform__type_norm_pg:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_pg

the norm of the projected gradient of the objective function at the best estimate of the solution determined by BGO_solve

.. index:: pair: variable; time
.. _doxid-structbgo__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`bgo_time_type<doxid-structbgo__time__type>` time

timings (see above)

.. index:: pair: variable; ugo_inform
.. _doxid-structbgo__inform__type_ugo_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ugo_inform_type<doxid-structugo__inform__type>` ugo_inform

inform parameters for UGO

.. index:: pair: variable; lhs_inform
.. _doxid-structbgo__inform__type_lhs_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lhs_inform_type<doxid-structlhs__inform__type>` lhs_inform

inform parameters for LHS

.. index:: pair: variable; trb_inform
.. _doxid-structbgo__inform__type_trb_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`trb_inform_type<doxid-structtrb__inform__type>` trb_inform

inform parameters for TRB

