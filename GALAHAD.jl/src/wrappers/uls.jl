mutable struct uls_control_type
    f_indexing::Bool
    error::Cint
    warning::Cint
    out::Cint
    print_level::Cint
    print_level_solver::Cint
    initial_fill_in_factor::Cint
    min_real_factor_size::Cint
    min_integer_factor_size::Cint
    max_factor_size::Int64
    blas_block_size_factorize::Cint
    blas_block_size_solve::Cint
    pivot_control::Cint
    pivot_search_limit::Cint
    minimum_size_for_btf::Cint
    max_iterative_refinements::Cint
    stop_if_singular::Bool
    array_increase_factor::Float64
    switch_to_full_code_density::Float64
    array_decrease_factor::Float64
    relative_pivot_tolerance::Float64
    absolute_pivot_tolerance::Float64
    zero_tolerance::Float64
    acceptable_residual_relative::Float64
    acceptable_residual_absolute::Float64
    prefix::NTuple{31,Cchar}
end

mutable struct uls_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    more_info::Cint
    out_of_range::Int64
    duplicates::Int64
    entries_dropped::Int64
    workspace_factors::Int64
    compresses::Cint
    entries_in_factors::Int64
    rank::Cint
    structural_rank::Cint
    pivot_control::Cint
    iterative_refinements::Cint
    alternative::Bool
    solver::NTuple{21,Cchar}
    gls_ainfo::gls_ainfo
    gls_finfo::gls_finfo
    gls_sinfo::gls_sinfo
    ma48_ainfo::ma48_ainfo
    ma48_finfo::ma48_finfo
    ma48_sinfo::ma48_sinfo
    lapack_error::Cint
end

function uls_initialize(solver, data, control, status)
    @ccall libgalahad_double.uls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{uls_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function uls_read_specfile(control, specfile)
    @ccall libgalahad_double.uls_read_specfile(control::Ptr{uls_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function uls_factorize_matrix(control, data, status, m, n, type, ne, val, row, col, ptr)
    @ccall libgalahad_double.uls_factorize_matrix(control::Ptr{uls_control_type},
                                                  data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  m::Cint, n::Cint, type::Ptr{Cchar},
                                                  ne::Cint, val::Ptr{Float64},
                                                  row::Ptr{Cint}, col::Ptr{Cint},
                                                  ptr::Ptr{Cint})::Cvoid
end

function uls_reset_control(control, data, status)
    @ccall libgalahad_double.uls_reset_control(control::Ptr{uls_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function uls_solve_system(data, status, m, n, sol, trans)
    @ccall libgalahad_double.uls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, sol::Ptr{Float64},
                                              trans::Bool)::Cvoid
end

function uls_information(data, inform, status)
    @ccall libgalahad_double.uls_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{uls_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function uls_terminate(data, control, inform)
    @ccall libgalahad_double.uls_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{uls_control_type},
                                           inform::Ptr{uls_inform_type})::Cvoid
end
