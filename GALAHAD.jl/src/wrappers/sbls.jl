mutable struct sbls_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    indmin::Cint
    valmin::Cint
    len_ulsmin::Cint
    itref_max::Cint
    maxit_pcg::Cint
    new_a::Cint
    new_h::Cint
    new_c::Cint
    preconditioner::Cint
    semi_bandwidth::Cint
    factorization::Cint
    max_col::Cint
    scaling::Cint
    ordering::Cint
    pivot_tol::Float64
    pivot_tol_for_basis::Float64
    zero_pivot::Float64
    static_tolerance::Float64
    static_level::Float64
    min_diagonal::Float64
    stop_absolute::Float64
    stop_relative::Float64
    remove_dependencies::Bool
    find_basis_by_transpose::Bool
    affine::Bool
    allow_singular::Bool
    perturb_to_make_definite::Bool
    get_norm_residual::Bool
    check_basis::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    symmetric_linear_solver::NTuple{31,Cchar}
    definite_linear_solver::NTuple{31,Cchar}
    unsymmetric_linear_solver::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    sls_control::sls_control_type
    uls_control::uls_control_type
end

mutable struct sbls_time_type
    total::Float64
    form::Float64
    factorize::Float64
    apply::Float64
    clock_total::Float64
    clock_form::Float64
    clock_factorize::Float64
    clock_apply::Float64
end

mutable struct sbls_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    sort_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    preconditioner::Cint
    factorization::Cint
    d_plus::Cint
    rank::Cint
    rank_def::Bool
    perturbed::Bool
    iter_pcg::Cint
    norm_residual::Float64
    alternative::Bool
    time::sbls_time_type
    sls_inform::sls_inform_type
    uls_inform::uls_inform_type
end

function sbls_initialize(data, control, status)
    @ccall libgalahad_double.sbls_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sbls_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function sbls_read_specfile(control, specfile)
    @ccall libgalahad_double.sbls_read_specfile(control::Ptr{sbls_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function sbls_import(control, data, status, n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type,
                     A_ne, A_row, A_col, A_ptr, C_type, C_ne, C_row, C_col, C_ptr)
    @ccall libgalahad_double.sbls_import(control::Ptr{sbls_control_type},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                         H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                         H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint}, C_type::Ptr{Cchar}, C_ne::Cint,
                                         C_row::Ptr{Cint}, C_col::Ptr{Cint},
                                         C_ptr::Ptr{Cint})::Cvoid
end

function sbls_reset_control(control, data, status)
    @ccall libgalahad_double.sbls_reset_control(control::Ptr{sbls_control_type},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function sbls_factorize_matrix(data, status, n, h_ne, H_val, a_ne, A_val, c_ne, C_val, D)
    @ccall libgalahad_double.sbls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   n::Cint, h_ne::Cint,
                                                   H_val::Ptr{Float64}, a_ne::Cint,
                                                   A_val::Ptr{Float64}, c_ne::Cint,
                                                   C_val::Ptr{Float64},
                                                   D::Ptr{Float64})::Cvoid
end

function sbls_solve_system(data, status, n, m, sol)
    @ccall libgalahad_double.sbls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, m::Cint, sol::Ptr{Float64})::Cvoid
end

function sbls_information(data, inform, status)
    @ccall libgalahad_double.sbls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sbls_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function sbls_terminate(data, control, inform)
    @ccall libgalahad_double.sbls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sbls_control_type},
                                            inform::Ptr{sbls_inform_type})::Cvoid
end
