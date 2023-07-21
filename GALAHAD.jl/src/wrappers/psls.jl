mutable struct psls_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    preconditioner::Cint
    semi_bandwidth::Cint
    scaling::Cint
    ordering::Cint
    max_col::Cint
    icfs_vectors::Cint
    mi28_lsize::Cint
    mi28_rsize::Cint
    min_diagonal::Float64
    new_structure::Bool
    get_semi_bandwidth::Bool
    get_norm_residual::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    definite_linear_solver::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    sls_control::sls_control_type
    mi28_control::mi28_control
end

mutable struct psls_time_type
    total::Float32
    assemble::Float32
    analyse::Float32
    factorize::Float32
    solve::Float32
    update::Float32
    clock_total::Float64
    clock_assemble::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
    clock_update::Float64
end

mutable struct psls_inform_type
    status::Cint
    alloc_status::Cint
    analyse_status::Cint
    factorize_status::Cint
    solve_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    preconditioner::Cint
    semi_bandwidth::Cint
    reordered_semi_bandwidth::Cint
    out_of_range::Cint
    duplicates::Cint
    upper::Cint
    missing_diagonals::Cint
    semi_bandwidth_used::Cint
    neg1::Cint
    neg2::Cint
    perturbed::Bool
    fill_in_ratio::Float64
    norm_residual::Float64
    bad_alloc::NTuple{81,Cchar}
    mc61_info::NTuple{10,Cint}
    mc61_rinfo::NTuple{15,Float64}
    time::psls_time_type
    sls_inform::sls_inform_type
    mi28_info::mi28_info
end

function psls_initialize(data, control, status)
    @ccall libgalahad_double.psls_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{psls_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function psls_read_specfile(control, specfile)
    @ccall libgalahad_double.psls_read_specfile(control::Ref{psls_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function psls_import(control, data, status, n, type, ne, row, col, ptr)
    @ccall libgalahad_double.psls_import(control::Ref{psls_control_type},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         type::Ptr{Cchar}, ne::Cint, row::Ptr{Cint},
                                         col::Ptr{Cint}, ptr::Ptr{Cint})::Cvoid
end

function psls_reset_control(control, data, status)
    @ccall libgalahad_double.psls_reset_control(control::Ref{psls_control_type},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function psls_form_preconditioner(data, status, ne, val)
    @ccall libgalahad_double.psls_form_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, ne::Cint,
                                                      val::Ptr{Float64})::Cvoid
end

function psls_form_subset_preconditioner(data, status, ne, val, n_sub, sub)
    @ccall libgalahad_double.psls_form_subset_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                             status::Ptr{Cint}, ne::Cint,
                                                             val::Ptr{Float64},
                                                             n_sub::Cint,
                                                             sub::Ptr{Cint})::Cvoid
end

function psls_update_preconditioner(data, status, ne, val, n_del, del)
    @ccall libgalahad_double.psls_update_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint}, ne::Cint,
                                                        val::Ptr{Float64}, n_del::Cint,
                                                        del::Ptr{Cint})::Cvoid
end

function psls_apply_preconditioner(data, status, n, sol)
    @ccall libgalahad_double.psls_apply_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Cint}, n::Cint,
                                                       sol::Ptr{Float64})::Cvoid
end

function psls_information(data, inform, status)
    @ccall libgalahad_double.psls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{psls_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function psls_terminate(data, control, inform)
    @ccall libgalahad_double.psls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{psls_control_type},
                                            inform::Ref{psls_inform_type})::Cvoid
end
