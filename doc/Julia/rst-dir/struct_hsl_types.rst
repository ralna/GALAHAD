.. index:: pair: struct; hsl_control
.. index:: pair: struct; hsl_info
.. _doxid-structhsl__controlinfo:

.. _details-structma48__control:

ma48_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma48_control{T,INT}
          f_arrays::INT
          multiplier::T
          u::T
          switch_::T
          drop::T
          tolerance::T
          cgce::T
          lp::INT
          wp::INT
          mp::INT
          ldiag::INT
          btf::INT
          struct_::INT
          maxit::INT
          factor_blocking::INT
          solve_blas::INT
          pivoting::INT
          diagonal_pivoting::INT
          fill_in::INT
          switch_mode::INT

.. _details-structma48__ainfo:

ma48_ainfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma48_ainfo{T,INT}
          ops::T
          flag::INT
          more::INT
          lena_analyse::Clong
          lenj_analyse::Clong
          lena_factorize::Clong
          leni_factorize::Clong
          ncmpa::INT
          rank::INT
          drop::Clong
          struc_rank::INT
          oor::Clong
          dup::Clong
          stat::INT
          lblock::INT
          sblock::INT
          tblock::Clong

.. _details-structma48__finfo:

ma48_finfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma48_finfo{T,INT}
          ops::T
          flag::INT
          more::INT
          size_factor::Clong
          lena_factorize::Clong
          leni_factorize::Clong
          drop::Clong
          rank::INT
          stat::INT

.. _details-structma48__sinfo:

ma48_sinfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma48_sinfo{INT}
          flag::INT
          more::INT
          stat::INT

.. _details-structma57__control:

ma57_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma57_control{T,INT}
          f_arrays::INT
          multiplier::T
          reduce::T
          u::T
          static_tolerance::T
          static_level::T
          tolerance::T
          convergence::T
          consist::T
          lp::INT
          wp::INT
          mp::INT
          sp::INT
          ldiag::INT
          nemin::INT
          factorblocking::INT
          solveblocking::INT
          la::INT
          liw::INT
          maxla::INT
          maxliw::INT
          pivoting::INT
          thresh::INT
          ordering::INT
          scaling::INT
          rank_deficient::INT
          ispare::NTuple{5,INT}
          rspare::NTuple{10,T}

.. _details-structma57__ainfo:

ma57_ainfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma57_ainfo{T,INT}
          opsa::T
          opse::T
          flag::INT
          more::INT
          nsteps::INT
          nrltot::INT
          nirtot::INT
          nrlnec::INT
          nirnec::INT
          nrladu::INT
          niradu::INT
          ncmpa::INT
          ordering::INT
          oor::INT
          dup::INT
          maxfrt::INT
          stat::INT
          ispare::NTuple{5,INT}
          rspare::NTuple{10,T}

.. _details-structma57__finfo:

ma57_finfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma57_finfo{T,INT}
          opsa::T
          opse::T
          opsb::T
          maxchange::T
          smin::T
          smax::T
          flag::INT
          more::INT
          maxfrt::INT
          nebdu::INT
          nrlbdu::INT
          nirbdu::INT
          nrltot::INT
          nirtot::INT
          nrlnec::INT
          nirnec::INT
          ncmpbr::INT
          ncmpbi::INT
          ntwo::INT
          neig::INT
          delay::INT
          signc::INT
          static_::INT
          modstep::INT
          rank::INT
          stat::INT
          ispare::NTuple{5,INT}
          rspare::NTuple{10,T}

.. _details-structma57__sinfo:

ma57_sinfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma57_sinfo{T,INT}
          cond::T
          cond2::T
          berr::T
          berr2::T
          error::T
          flag::INT
          stat::INT
          ispare::NTuple{5,INT}
          rspare::NTuple{10,T}

.. _details-structma77__control:

ma77_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma77_control{T,INT}
          f_arrays::INT
          print_level::INT
          unit_diagnostics::INT
          unit_error::INT
          unit_warning::INT
          bits::INT
          buffer_lpage::NTuple{2,INT}
          buffer_npage::NTuple{2,INT}
          file_size::Clong
          maxstore::Clong
          storage::NTuple{3,Clong}
          nemin::INT
          maxit::INT
          infnorm::INT
          thresh::T
          nb54::INT
          action::INT
          multiplier::T
          nb64::INT
          nbi::INT
          small::T
          static_::T
          storage_indef::Clong
          u::T
          umin::T
          consist_tol::T
          ispare::NTuple{5,INT}
          lspare::NTuple{5,Clong}
          rspare::NTuple{5,T}

.. _details-structma77__info:

ma77_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma77_info{T,INT}
          detlog::T
          detsign::INT
          flag::INT
          iostat::INT
          matrix_dup::INT
          matrix_rank::INT
          matrix_outrange::INT
          maxdepth::INT
          maxfront::INT
          minstore::Clong
          ndelay::INT
          nfactor::Clong
          nflops::Clong
          niter::INT
          nsup::INT
          num_neg::INT
          num_nothresh::INT
          num_perturbed::INT
          ntwo::INT
          stat::INT
          index::NTuple{4,INT}
          nio_read::NTuple{2,Clong}
          nio_write::NTuple{2,Clong}
          nwd_read::NTuple{2,Clong}
          nwd_write::NTuple{2,Clong}
          num_file::NTuple{4,INT}
          storage::NTuple{4,Clong}
          tree_nodes::INT
          unit_restart::INT
          unused::INT
          usmall::T
          ispare::NTuple{5,INT}
          lspare::NTuple{5,Clong}
          rspare::NTuple{5,T}

.. _details-structma86__control:

ma86_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma86_control{T,INT}
          f_arrays::INT
          diagnostics_level::INT
          unit_diagnostics::INT
          unit_error::INT
          unit_warning::INT
          nemin::INT
          nb::INT
          action::INT
          nbi::INT
          pool_size::INT
          small_::T
          static_::T
          u::T
          umin::T
          scaling::INT

.. _details-structma86__info:

ma86_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma86_info{T,INT}
          detlog::T
          detsign::INT
          flag::INT
          matrix_rank::INT
          maxdepth::INT
          num_delay::INT
          num_factor::Clong
          num_flops::Clong
          num_neg::INT
          num_nodes::INT
          num_nothresh::INT
          num_perturbed::INT
          num_two::INT
          pool_size::INT
          stat::INT
          usmall::T

.. _details-structma87__control:

ma87_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma87_control{T,INT}
          f_arrays::INT
          diagnostics_level::INT
          unit_diagnostics::INT
          unit_error::INT
          unit_warning::INT
          nemin::INT
          nb::INT
          pool_size::INT
          diag_zero_minus::T
          diag_zero_plus::T
          unused::NTuple{40,Cchar}

.. _details-structma87__info:

ma87_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma87_info{T,INT}
          detlog::T
          flag::INT
          maxdepth::INT
          num_factor::Clong
          num_flops::Clong
          num_nodes::INT
          pool_size::INT
          stat::INT
          num_zero::INT
          unused::NTuple{40,Cchar}

.. _details-structma97__control:

ma97_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma97_control{T,INT}
          f_arrays::INT
          action::INT
          nemin::INT
          multiplier::T
          ordering::INT
          print_level::INT
          scaling::INT
          small::T
          u::T
          unit_diagnostics::INT
          unit_error::INT
          unit_warning::INT
          factor_min::Clong
          solve_blas3::INT
          solve_min::Clong
          solve_mf::INT
          consist_tol::T
          ispare::NTuple{5,INT}
          rspare::NTuple{10,T}

.. _details-structma97__info:

ma97_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma97_info{T,INT}
          flag::INT
          flag68::INT
          flag77::INT
          matrix_dup::INT
          matrix_rank::INT
          matrix_outrange::INT
          matrix_missing_diag::INT
          maxdepth::INT
          maxfront::INT
          num_delay::INT
          num_factor::Clong
          num_flops::Clong
          num_neg::INT
          num_sup::INT
          num_two::INT
          ordering::INT
          stat::INT
          ispare::NTuple{5,INT}
          rspare::NTuple{10,T}

.. _details-structmc64__control:

mc64_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mc64_control{INT}
          f_arrays::INT
          lp::INT
          wp::INT
          sp::INT
          ldiag::INT
          checking::INT

.. _details-structmc64__info:

mc64_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mc64_info{INT}
          flag::INT
          more::INT
          strucrank::INT
          stat::INT

.. _details-structmc68__control:

mc68_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mc68_control{INT}
          f_array_in::INT
          f_array_out::INT
          min_l_workspace::INT
          lp::INT
          wp::INT
          mp::INT
          nemin::INT
          print_level::INT
          row_full_thresh::INT
          row_search::INT

.. _details-structmc68__info:

mc68_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mc68_info{INT}
          flag::INT
          iostat::INT
          stat::INT
          out_range::INT
          duplicate::INT
          n_compressions::INT
          n_zero_eigs::INT
          l_workspace::Clong
          zb01_info::INT
          n_dense_rows::INT

.. _details-structmi20__control:

mi20_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi20_control{T,INT}
          f_arrays::INT
          aggressive::INT
          c_fail::INT
          max_levels::INT
          max_points::INT
          reduction::T
          st_method::INT
          st_parameter::T
          testing::INT
          trunc_parameter::T
          coarse_solver::INT
          coarse_solver_its::INT
          damping::T
          err_tol::T
          levels::INT
          pre_smoothing::INT
          smoother::INT
          post_smoothing::INT
          v_iterations::INT
          print_level::INT
          print::INT
          error::INT
          one_pass_coarsen::INT

.. _details-structmi20__solve__control:

mi20_solve__control structure
-----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi20_solve_control{T,INT}
          abs_tol::T
          breakdown_tol::T
          gmres_restart::INT
          init_guess::Bool
          krylov_solver::INT
          max_its::INT
          preconditioner_side::INT
          rel_tol::T

.. _details-structmi20__info:

mi20_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi20_info{T,INT}
          flag::INT
          clevels::INT
          cpoints::INT
          cnnz::INT
          stat::INT
          getrf_info::INT
          iterations::INT
          residual::T

.. _details-structmi28__control:

mi28_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi28_control{T,INT}
          f_arrays::INT
          alpha::T
          check::Bool
          iorder::INT
          iscale::INT
          lowalpha::T
          maxshift::INT
          rrt::Bool
          shift_factor::T
          shift_factor2::T
          small::T
          tau1::T
          tau2::T
          unit_error::INT
          unit_warning::INT

.. _details-structmi28__info:

mi28_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi28_info{T,INT}
          band_after::INT
          band_before::INT
          dup::INT
          flag::INT
          flag61::INT
          flag64::INT
          flag68::INT
          flag77::INT
          nrestart::INT
          nshift::INT
          oor::INT
          profile_before::T
          profile_after::T
          size_r::Clong
          stat::INT
          alpha::T


detailed documentation
----------------------

HSL package control and info derived types as Julia structures.
See `HSL <https://www.hsl.rl.ac.uk/catalogue/>`_ 
documentation for further details.
