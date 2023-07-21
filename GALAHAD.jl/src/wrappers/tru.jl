mutable struct tru_control_type
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
    stop_g_absolute::Float64
    stop_g_relative::Float64
    stop_s::Float64
    advanced_start::Cint
    initial_radius::Float64
    maximum_radius::Float64
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
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
    trs_control::trs_control_type
    gltr_control::gltr_control_type
    dps_control::dps_control_type
    psls_control::psls_control_type
    lms_control::lms_control_type
    lms_control_prec::lms_control_type
    sec_control::sec_control_type
    sha_control::sha_control_type
end

mutable struct tru_time_type
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

mutable struct tru_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    cg_iter::Cint
    f_eval::Cint
    g_eval::Cint
    h_eval::Cint
    factorization_max::Cint
    factorization_status::Cint
    max_entries_factors::Int64
    factorization_integer::Int64
    factorization_real::Int64
    factorization_average::Float64
    obj::Float64
    norm_g::Float64
    radius::Float64
    time::tru_time_type
    trs_inform::trs_inform_type
    gltr_inform::gltr_inform_type
    dps_inform::dps_inform_type
    psls_inform::psls_inform_type
    lms_inform::lms_inform_type
    lms_inform_prec::lms_inform_type
    sec_inform::sec_inform_type
    sha_inform::sha_inform_type
end

function tru_initialize(data, control, status)
    @ccall libgalahad_double.tru_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{tru_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function tru_read_specfile(control, specfile)
    @ccall libgalahad_double.tru_read_specfile(control::Ptr{tru_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function tru_import(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
    @ccall libgalahad_double.tru_import(control::Ptr{tru_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function tru_reset_control(control, data, status)
    @ccall libgalahad_double.tru_reset_control(control::Ptr{tru_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function tru_solve_with_mat(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                            eval_prec)
    @ccall libgalahad_double.tru_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint,
                                                x::Ptr{Float64}, g::Ptr{Float64},
                                                ne::Cint, eval_f::Ptr{Cvoid},
                                                eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_without_mat(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                               eval_prec)
    @ccall libgalahad_double.tru_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, x::Ptr{Float64},
                                                   g::Ptr{Float64}, eval_f::Ptr{Cvoid},
                                                   eval_g::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
    @ccall libgalahad_double.tru_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint},
                                                        eval_status::Ptr{Cint}, n::Cint,
                                                        x::Ptr{Float64}, f::Float64,
                                                        g::Ptr{Float64}, ne::Cint,
                                                        H_val::Ptr{Float64},
                                                        u::Ptr{Float64},
                                                        v::Ptr{Float64})::Cvoid
end

function tru_solve_reverse_without_mat(data, status, eval_status, n, x, f, g, u, v)
    @ccall libgalahad_double.tru_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint},
                                                           eval_status::Ptr{Cint}, n::Cint,
                                                           x::Ptr{Float64}, f::Float64,
                                                           g::Ptr{Float64},
                                                           u::Ptr{Float64},
                                                           v::Ptr{Float64})::Cvoid
end

function tru_information(data, inform, status)
    @ccall libgalahad_double.tru_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{tru_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function tru_terminate(data, control, inform)
    @ccall libgalahad_double.tru_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{tru_control_type},
                                           inform::Ptr{tru_inform_type})::Cvoid
end
