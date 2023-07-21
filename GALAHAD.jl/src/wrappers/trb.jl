mutable struct trb_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    print_gap::Cint
    maxit::Cint
    alive_unit::Cint
    alive_file::NTuple{31,Cchar}
    more_toraldo::Cint
    non_monotone::Cint
    model::Cint
    norm::Cint
    semi_bandwidth::Cint
    lbfgs_vectors::Cint
    max_dxg::Cint
    icfs_vectors::Cint
    mi28_lsize::Cint
    mi28_rsize::Cint
    advanced_start::Cint
    infinity::Float64
    stop_pg_absolute::Float64
    stop_pg_relative::Float64
    stop_s::Float64
    initial_radius::Float64
    maximum_radius::Float64
    stop_rel_cg::Float64
    eta_successful::Float64
    eta_very_successful::Float64
    eta_too_successful::Float64
    radius_increase::Float64
    radius_reduce::Float64
    radius_reduce_max::Float64
    obj_unbounded::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    hessian_available::Bool
    subproblem_direct::Bool
    retrospective_trust_region::Bool
    renormalize_radius::Bool
    two_norm_tr::Bool
    exact_gcp::Bool
    accurate_bqp::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
    trs_control::trs_control_type
    gltr_control::gltr_control_type
    psls_control::psls_control_type
    lms_control::lms_control_type
    lms_control_prec::lms_control_type
    sha_control::sha_control_type
end

mutable struct trb_time_type
    total::Float32
    preprocess::Float32
    analyse::Float32
    factorize::Float32
    solve::Float32
    clock_total::Float64
    clock_preprocess::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
end

mutable struct trb_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    cg_iter::Cint
    cg_maxit::Cint
    f_eval::Cint
    g_eval::Cint
    h_eval::Cint
    n_free::Cint
    factorization_max::Cint
    factorization_status::Cint
    max_entries_factors::Int64
    factorization_integer::Int64
    factorization_real::Int64
    obj::Float64
    norm_pg::Float64
    radius::Float64
    time::trb_time_type
    trs_inform::trs_inform_type
    gltr_inform::gltr_inform_type
    psls_inform::psls_inform_type
    lms_inform::lms_inform_type
    lms_inform_prec::lms_inform_type
    sha_inform::sha_inform_type
end

function trb_initialize(data, control, status)
    @ccall libgalahad_double.trb_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{trb_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function trb_read_specfile(control, specfile)
    @ccall libgalahad_double.trb_read_specfile(control::Ref{trb_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function trb_import(control, data, status, n, x_l, x_u, H_type, ne, H_row, H_col, H_ptr)
    @ccall libgalahad_double.trb_import(control::Ref{trb_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function trb_reset_control(control, data, status)
    @ccall libgalahad_double.trb_reset_control(control::Ref{trb_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function trb_solve_with_mat(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                            eval_prec)
    @ccall libgalahad_double.trb_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint,
                                                x::Ptr{Float64}, g::Ptr{Float64},
                                                ne::Cint, eval_f::Ptr{Cvoid},
                                                eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_without_mat(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                               eval_shprod, eval_prec)
    @ccall libgalahad_double.trb_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, x::Ptr{Float64},
                                                   g::Ptr{Float64}, eval_f::Ptr{Cvoid},
                                                   eval_g::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_shprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
    @ccall libgalahad_double.trb_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint},
                                                        eval_status::Ptr{Cint}, n::Cint,
                                                        x::Ptr{Float64}, f::Float64,
                                                        g::Ptr{Float64}, ne::Cint,
                                                        H_val::Ptr{Float64},
                                                        u::Ptr{Float64},
                                                        v::Ptr{Float64})::Cvoid
end

function trb_solve_reverse_without_mat(data, status, eval_status, n, x, f, g, u, v,
                                       index_nz_v, nnz_v, index_nz_u, nnz_u)
    @ccall libgalahad_double.trb_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint},
                                                           eval_status::Ptr{Cint}, n::Cint,
                                                           x::Ptr{Float64}, f::Float64,
                                                           g::Ptr{Float64},
                                                           u::Ptr{Float64},
                                                           v::Ptr{Float64},
                                                           index_nz_v::Ptr{Cint},
                                                           nnz_v::Ptr{Cint},
                                                           index_nz_u::Ptr{Cint},
                                                           nnz_u::Cint)::Cvoid
end

function trb_information(data, inform, status)
    @ccall libgalahad_double.trb_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{trb_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function trb_terminate(data, control, inform)
    @ccall libgalahad_double.trb_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{trb_control_type},
                                           inform::Ref{trb_inform_type})::Cvoid
end
