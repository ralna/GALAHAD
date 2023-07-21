mutable struct cro_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    max_schur_complement::Cint
    infinity::Float64
    feasibility_tolerance::Float64
    check_io::Bool
    refine_solution::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    symmetric_linear_solver::NTuple{31,Cchar}
    unsymmetric_linear_solver::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    sls_control::sls_control_type
    sbls_control::sbls_control_type
    uls_control::uls_control_type
    ir_control::ir_control_type
end

mutable struct cro_time_type
    total::Float32
    analyse::Float32
    factorize::Float32
    solve::Float32
    clock_total::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
end

mutable struct cro_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    dependent::Cint
    time::cro_time_type
    sls_inform::sls_inform_type
    sbls_inform::sbls_inform_type
    uls_inform::uls_inform_type
    scu_status::Cint
    scu_inform::scu_inform_type
    ir_inform::ir_inform_type
end

function cro_initialize(data, control, status)
    @ccall libgalahad_double.cro_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{cro_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function cro_read_specfile(control, specfile)
    @ccall libgalahad_double.cro_read_specfile(control::Ref{cro_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function cro_crossover_solution(data, control, inform, n, m, m_equal, h_ne, H_val, H_col,
                                H_ptr, a_ne, A_val, A_col, A_ptr, g, c_l, c_u, x_l, x_u, x,
                                c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.cro_crossover_solution(data::Ptr{Ptr{Cvoid}},
                                                    control::Ref{cro_control_type},
                                                    inform::Ref{cro_inform_type}, n::Cint,
                                                    m::Cint, m_equal::Cint, h_ne::Cint,
                                                    H_val::Ptr{Float64}, H_col::Ptr{Cint},
                                                    H_ptr::Ptr{Cint}, a_ne::Cint,
                                                    A_val::Ptr{Float64}, A_col::Ptr{Cint},
                                                    A_ptr::Ptr{Cint}, g::Ptr{Float64},
                                                    c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                                    x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                    x::Ptr{Float64}, c::Ptr{Float64},
                                                    y::Ptr{Float64}, z::Ptr{Float64},
                                                    x_stat::Ptr{Cint},
                                                    c_stat::Ptr{Cint})::Cvoid
end

function cro_terminate(data, control, inform)
    @ccall libgalahad_double.cro_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{cro_control_type},
                                           inform::Ref{cro_inform_type})::Cvoid
end
