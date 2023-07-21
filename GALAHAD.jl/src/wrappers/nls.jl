mutable struct nls_subproblem_control_type
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    print_gap::Cint
    maxit::Cint
    alive_unit::Cint
    alive_file::NTuple{31,Cchar}
    jacobian_available::Cint
    hessian_available::Cint
    model::Cint
    norm::Cint
    non_monotone::Cint
    weight_update_strategy::Cint
    stop_c_absolute::Float64
    stop_c_relative::Float64
    stop_g_absolute::Float64
    stop_g_relative::Float64
    stop_s::Float64
    power::Float64
    initial_weight::Float64
    minimum_weight::Float64
    initial_inner_weight::Float64
    eta_successful::Float64
    eta_very_successful::Float64
    eta_too_successful::Float64
    weight_decrease_min::Float64
    weight_decrease::Float64
    weight_increase::Float64
    weight_increase_max::Float64
    reduce_gap::Float64
    tiny_gap::Float64
    large_root::Float64
    switch_to_newton::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    subproblem_direct::Bool
    renormalize_weight::Bool
    magic_step::Bool
    print_obj::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
    rqs_control::rqs_control_type
    glrt_control::glrt_control_type
    psls_control::psls_control_type
    bsc_control::bsc_control_type
    roots_control::roots_control_type
end

mutable struct nls_control_type
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
    jacobian_available::Cint
    hessian_available::Cint
    model::Cint
    norm::Cint
    non_monotone::Cint
    weight_update_strategy::Cint
    stop_c_absolute::Float64
    stop_c_relative::Float64
    stop_g_absolute::Float64
    stop_g_relative::Float64
    stop_s::Float64
    power::Float64
    initial_weight::Float64
    minimum_weight::Float64
    initial_inner_weight::Float64
    eta_successful::Float64
    eta_very_successful::Float64
    eta_too_successful::Float64
    weight_decrease_min::Float64
    weight_decrease::Float64
    weight_increase::Float64
    weight_increase_max::Float64
    reduce_gap::Float64
    tiny_gap::Float64
    large_root::Float64
    switch_to_newton::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    subproblem_direct::Bool
    renormalize_weight::Bool
    magic_step::Bool
    print_obj::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
    rqs_control::rqs_control_type
    glrt_control::glrt_control_type
    psls_control::psls_control_type
    bsc_control::bsc_control_type
    roots_control::roots_control_type
    subproblem_control::nls_subproblem_control_type
end

mutable struct nls_time_type
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

mutable struct nls_subproblem_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    bad_eval::NTuple{13,Cchar}
    iter::Cint
    cg_iter::Cint
    c_eval::Cint
    j_eval::Cint
    h_eval::Cint
    factorization_max::Cint
    factorization_status::Cint
    max_entries_factors::Int64
    factorization_integer::Int64
    factorization_real::Int64
    factorization_average::Float64
    obj::Float64
    norm_c::Float64
    norm_g::Float64
    weight::Float64
    time::nls_time_type
    rqs_inform::rqs_inform_type
    glrt_inform::glrt_inform_type
    psls_inform::psls_inform_type
    bsc_inform::bsc_inform_type
    roots_inform::roots_inform_type
end

mutable struct nls_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    bad_eval::NTuple{13,Cchar}
    iter::Cint
    cg_iter::Cint
    c_eval::Cint
    j_eval::Cint
    h_eval::Cint
    factorization_max::Cint
    factorization_status::Cint
    max_entries_factors::Int64
    factorization_integer::Int64
    factorization_real::Int64
    factorization_average::Float64
    obj::Float64
    norm_c::Float64
    norm_g::Float64
    weight::Float64
    time::nls_time_type
    subproblem_inform::nls_subproblem_inform_type
    rqs_inform::rqs_inform_type
    glrt_inform::glrt_inform_type
    psls_inform::psls_inform_type
    bsc_inform::bsc_inform_type
    roots_inform::roots_inform_type
end

function nls_initialize(data, control, inform)
    @ccall libgalahad_double.nls_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{nls_control_type},
                                            inform::Ptr{nls_inform_type})::Cvoid
end

function nls_read_specfile(control, specfile)
    @ccall libgalahad_double.nls_read_specfile(control::Ptr{nls_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function nls_import(control, data, status, n, m, J_type, J_ne, J_row, J_col, J_ptr, H_type,
                    H_ne, H_row, H_col, H_ptr, P_type, P_ne, P_row, P_col, P_ptr, w)
    @ccall libgalahad_double.nls_import(control::Ptr{nls_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, J_type::Ptr{Cchar}, J_ne::Cint,
                                        J_row::Ptr{Cint}, J_col::Ptr{Cint},
                                        J_ptr::Ptr{Cint}, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, P_type::Ptr{Cchar}, P_ne::Cint,
                                        P_row::Ptr{Cint}, P_col::Ptr{Cint},
                                        P_ptr::Ptr{Cint}, w::Ptr{Float64})::Cvoid
end

function nls_reset_control(control, data, status)
    @ccall libgalahad_double.nls_reset_control(control::Ptr{nls_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function nls_solve_with_mat(data, userdata, status, n, m, x, c, g, eval_c, j_ne, eval_j,
                            h_ne, eval_h, p_ne, eval_hprods)
    @ccall libgalahad_double.nls_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint, m::Cint,
                                                x::Ptr{Float64}, c::Ptr{Float64},
                                                g::Ptr{Float64}, eval_c::Ptr{Cvoid},
                                                j_ne::Cint, eval_j::Ptr{Cvoid}, h_ne::Cint,
                                                eval_h::Ptr{Cvoid}, p_ne::Cint,
                                                eval_hprods::Ptr{Cvoid})::Cvoid
end

function nls_solve_without_mat(data, userdata, status, n, m, x, c, g, eval_c, eval_jprod,
                               eval_hprod, p_ne, eval_hprods)
    @ccall libgalahad_double.nls_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, m::Cint, x::Ptr{Float64},
                                                   c::Ptr{Float64}, g::Ptr{Float64},
                                                   eval_c::Ptr{Cvoid},
                                                   eval_jprod::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid}, p_ne::Cint,
                                                   eval_hprods::Ptr{Cvoid})::Cvoid
end

function nls_solve_reverse_with_mat(data, status, eval_status, n, m, x, c, g, j_ne, J_val,
                                    y, h_ne, H_val, v, p_ne, P_val)
    @ccall libgalahad_double.nls_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint},
                                                        eval_status::Ptr{Cint}, n::Cint,
                                                        m::Cint, x::Ptr{Float64},
                                                        c::Ptr{Float64}, g::Ptr{Float64},
                                                        j_ne::Cint, J_val::Ptr{Float64},
                                                        y::Ptr{Float64}, h_ne::Cint,
                                                        H_val::Ptr{Float64},
                                                        v::Ptr{Float64}, p_ne::Cint,
                                                        P_val::Ptr{Float64})::Cvoid
end

function nls_solve_reverse_without_mat(data, status, eval_status, n, m, x, c, g, transpose,
                                       u, v, y, p_ne, P_val)
    @ccall libgalahad_double.nls_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint},
                                                           eval_status::Ptr{Cint}, n::Cint,
                                                           m::Cint, x::Ptr{Float64},
                                                           c::Ptr{Float64},
                                                           g::Ptr{Float64},
                                                           transpose::Ptr{Bool},
                                                           u::Ptr{Float64},
                                                           v::Ptr{Float64},
                                                           y::Ptr{Float64}, p_ne::Cint,
                                                           P_val::Ptr{Float64})::Cvoid
end

function nls_information(data, inform, status)
    @ccall libgalahad_double.nls_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{nls_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function nls_terminate(data, control, inform)
    @ccall libgalahad_double.nls_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{nls_control_type},
                                           inform::Ptr{nls_inform_type})::Cvoid
end
