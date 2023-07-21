mutable struct trs_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    problem::Cint
    print_level::Cint
    dense_factorization::Cint
    new_h::Cint
    new_m::Cint
    new_a::Cint
    max_factorizations::Cint
    inverse_itmax::Cint
    taylor_max_degree::Cint
    initial_multiplier::Float64
    lower::Float64
    upper::Float64
    stop_normal::Float64
    stop_absolute_normal::Float64
    stop_hard::Float64
    start_invit_tol::Float64
    start_invitmax_tol::Float64
    equality_problem::Bool
    use_initial_multiplier::Bool
    initialize_approx_eigenvector::Bool
    force_Newton::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    problem_file::NTuple{31,Cchar}
    symmetric_linear_solver::NTuple{31,Cchar}
    definite_linear_solver::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    sls_control::sls_control_type
    ir_control::ir_control_type
end

mutable struct trs_time_type
    total::Float64
    assemble::Float64
    analyse::Float64
    factorize::Float64
    solve::Float64
    clock_total::Float64
    clock_assemble::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
end

mutable struct trs_history_type
    lambda::Float64
    x_norm::Float64
end

mutable struct trs_inform_type
    status::Cint
    alloc_status::Cint
    factorizations::Cint
    max_entries_factors::Int64
    len_history::Cint
    obj::Float64
    x_norm::Float64
    multiplier::Float64
    pole::Float64
    dense_factorization::Bool
    hard_case::Bool
    bad_alloc::NTuple{81,Cchar}
    time::trs_time_type
    history::NTuple{100,trs_history_type}
    sls_inform::sls_inform_type
    ir_inform::ir_inform_type
end

function trs_initialize(data, control, status)
    @ccall libgalahad_double.trs_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{trs_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function trs_read_specfile(control, specfile)
    @ccall libgalahad_double.trs_read_specfile(control::Ref{trs_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function trs_import(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)
    @ccall libgalahad_double.trs_import(control::Ref{trs_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function trs_import_m(data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
    @ccall libgalahad_double.trs_import_m(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          M_type::Ptr{Cchar}, M_ne::Cint, M_row::Ptr{Cint},
                                          M_col::Ptr{Cint}, M_ptr::Ptr{Cint})::Cvoid
end

function trs_import_a(data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.trs_import_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                          A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                          A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

function trs_reset_control(control, data, status)
    @ccall libgalahad_double.trs_reset_control(control::Ref{trs_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function trs_solve_problem(data, status, n, radius, f, c, H_ne, H_val, x, M_ne, M_val, m,
                           A_ne, A_val, y)
    @ccall libgalahad_double.trs_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, radius::Float64, f::Float64,
                                               c::Ptr{Float64}, H_ne::Cint,
                                               H_val::Ptr{Float64}, x::Ptr{Float64},
                                               M_ne::Cint, M_val::Ptr{Float64}, m::Cint,
                                               A_ne::Cint, A_val::Ptr{Float64},
                                               y::Ptr{Float64})::Cvoid
end

function trs_information(data, inform, status)
    @ccall libgalahad_double.trs_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{trs_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function trs_terminate(data, control, inform)
    @ccall libgalahad_double.trs_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{trs_control_type},
                                           inform::Ref{trs_inform_type})::Cvoid
end
