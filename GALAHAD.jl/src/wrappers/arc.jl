mutable struct arc_control_type
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
    stop_g_absolute::Float64
    stop_g_relative::Float64
    stop_s::Float64
    initial_weight::Float64
    minimum_weight::Float64
    reduce_gap::Float64
    tiny_gap::Float64
    large_root::Float64
    eta_successful::Float64
    eta_very_successful::Float64
    eta_too_successful::Float64
    weight_decrease_min::Float64
    weight_decrease::Float64
    weight_increase::Float64
    weight_increase_max::Float64
    obj_unbounded::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    hessian_available::Bool
    subproblem_direct::Bool
    renormalize_weight::Bool
    quadratic_ratio_test::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
    rqs_control::rqs_control_type
    glrt_control::glrt_control_type
    dps_control::dps_control_type
    psls_control::psls_control_type
    lms_control::lms_control_type
    lms_control_prec::lms_control_type
    sha_control::sha_control_type
end

mutable struct arc_time_type
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

mutable struct arc_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    cg_iter::Cint
    f_eval::Cint
    g_eval::Cint
    h_eval::Cint
    factorization_status::Cint
    factorization_max::Cint
    max_entries_factors::Int64
    factorization_integer::Int64
    factorization_real::Int64
    factorization_average::Float64
    obj::Float64
    norm_g::Float64
    weight::Float64
    time::arc_time_type
    rqs_inform::rqs_inform_type
    glrt_inform::glrt_inform_type
    dps_inform::dps_inform_type
    psls_inform::psls_inform_type
    lms_inform::lms_inform_type
    lms_inform_prec::lms_inform_type
    sha_inform::sha_inform_type
end

function arc_initialize(data, control, status)
    @ccall libgalahad_double.arc_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{arc_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function arc_read_specfile(control, specfile)
    @ccall libgalahad_double.arc_read_specfile(control::Ref{arc_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function arc_import(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
    @ccall libgalahad_double.arc_import(control::Ref{arc_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function arc_reset_control(control, data, status)
    @ccall libgalahad_double.arc_reset_control(control::Ref{arc_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function arc_solve_with_mat(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                            eval_prec)
    @ccall libgalahad_double.arc_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint,
                                                x::Ptr{Float64}, g::Ptr{Float64},
                                                ne::Cint, eval_f::Ptr{Cvoid},
                                                eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

function arc_solve_without_mat(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                               eval_prec)
    @ccall libgalahad_double.arc_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, x::Ptr{Float64},
                                                   g::Ptr{Float64}, eval_f::Ptr{Cvoid},
                                                   eval_g::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

function arc_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
    @ccall libgalahad_double.arc_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint},
                                                        eval_status::Ptr{Cint}, n::Cint,
                                                        x::Ptr{Float64}, f::Float64,
                                                        g::Ptr{Float64}, ne::Cint,
                                                        H_val::Ptr{Float64},
                                                        u::Ptr{Float64},
                                                        v::Ptr{Float64})::Cvoid
end

function arc_solve_reverse_without_mat(data, status, eval_status, n, x, f, g, u, v)
    @ccall libgalahad_double.arc_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint},
                                                           eval_status::Ptr{Cint}, n::Cint,
                                                           x::Ptr{Float64}, f::Float64,
                                                           g::Ptr{Float64},
                                                           u::Ptr{Float64},
                                                           v::Ptr{Float64})::Cvoid
end

function arc_information(data, inform, status)
    @ccall libgalahad_double.arc_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{arc_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function arc_terminate(data, control, inform)
    @ccall libgalahad_double.arc_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{arc_control_type},
                                           inform::Ref{arc_inform_type})::Cvoid
end
