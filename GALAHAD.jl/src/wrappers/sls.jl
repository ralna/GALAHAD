export sls_control_type

mutable struct sls_control_type
    f_indexing::Bool
    error::Cint
    warning::Cint
    out::Cint
    statistics::Cint
    print_level::Cint
    print_level_solver::Cint
    bits::Cint
    block_size_kernel::Cint
    block_size_elimination::Cint
    blas_block_size_factorize::Cint
    blas_block_size_solve::Cint
    node_amalgamation::Cint
    initial_pool_size::Cint
    min_real_factor_size::Cint
    min_integer_factor_size::Cint
    max_real_factor_size::Int64
    max_integer_factor_size::Int64
    max_in_core_store::Int64
    array_increase_factor::Float64
    array_decrease_factor::Float64
    pivot_control::Cint
    ordering::Cint
    full_row_threshold::Cint
    row_search_indefinite::Cint
    scaling::Cint
    scale_maxit::Cint
    scale_thresh::Float64
    relative_pivot_tolerance::Float64
    minimum_pivot_tolerance::Float64
    absolute_pivot_tolerance::Float64
    zero_tolerance::Float64
    zero_pivot_tolerance::Float64
    negative_pivot_tolerance::Float64
    static_pivot_tolerance::Float64
    static_level_switch::Float64
    consistency_tolerance::Float64
    max_iterative_refinements::Cint
    acceptable_residual_relative::Float64
    acceptable_residual_absolute::Float64
    multiple_rhs::Bool
    generate_matrix_file::Bool
    matrix_file_device::Cint
    matrix_file_name::NTuple{31,Cchar}
    out_of_core_directory::NTuple{401,Cchar}
    out_of_core_integer_factor_file::NTuple{401,Cchar}
    out_of_core_real_factor_file::NTuple{401,Cchar}
    out_of_core_real_work_file::NTuple{401,Cchar}
    out_of_core_indefinite_file::NTuple{401,Cchar}
    out_of_core_restart_file::NTuple{501,Cchar}
    prefix::NTuple{31,Cchar}
    sls_control_type() = new()
end

export sls_time_type

mutable struct sls_time_type
    total::Float64
    analyse::Float64
    factorize::Float64
    solve::Float64
    order_external::Float64
    analyse_external::Float64
    factorize_external::Float64
    solve_external::Float64
    clock_total::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
    clock_order_external::Float64
    clock_analyse_external::Float64
    clock_factorize_external::Float64
    clock_solve_external::Float64
    sls_control_type() = new()
end

export sls_inform_type

mutable struct sls_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    more_info::Cint
    entries::Cint
    out_of_range::Cint
    duplicates::Cint
    upper::Cint
    missing_diagonals::Cint
    max_depth_assembly_tree::Cint
    nodes_assembly_tree::Cint
    real_size_desirable::Int64
    integer_size_desirable::Int64
    real_size_necessary::Int64
    integer_size_necessary::Int64
    real_size_factors::Int64
    integer_size_factors::Int64
    entries_in_factors::Int64
    max_task_pool_size::Cint
    max_front_size::Cint
    compresses_real::Cint
    compresses_integer::Cint
    two_by_two_pivots::Cint
    semi_bandwidth::Cint
    delayed_pivots::Cint
    pivot_sign_changes::Cint
    static_pivots::Cint
    first_modified_pivot::Cint
    rank::Cint
    negative_eigenvalues::Cint
    num_zero::Cint
    iterative_refinements::Cint
    flops_assembly::Int64
    flops_elimination::Int64
    flops_blas::Int64
    largest_modified_pivot::Float64
    minimum_scaling_factor::Float64
    maximum_scaling_factor::Float64
    condition_number_1::Float64
    condition_number_2::Float64
    backward_error_1::Float64
    backward_error_2::Float64
    forward_error::Float64
    alternative::Bool
    solver::NTuple{21,Cchar}
    time::sls_time_type
    sils_ainfo::sils_ainfo_type
    sils_finfo::sils_finfo_type
    sils_sinfo::sils_sinfo_type
    ma57_ainfo::ma57_ainfo
    ma57_finfo::ma57_finfo
    ma57_sinfo::ma57_sinfo
    ma77_info::ma77_info
    ma86_info::ma86_info
    ma87_info::ma87_info
    ma97_info::ma97_info
    ssids_inform::spral_ssids_inform
    mc61_info::NTuple{10,Cint}
    mc61_rinfo::NTuple{15,Float64}
    mc64_info::mc64_info
    mc68_info::mc68_info
    mc77_info::NTuple{10,Cint}
    mc77_rinfo::NTuple{10,Float64}
    mumps_error::Cint
    mumps_info::NTuple{80,Cint}
    mumps_rinfo::NTuple{40,Float64}
    pardiso_error::Cint
    pardiso_IPARM::NTuple{64,Cint}
    pardiso_DPARM::NTuple{64,Float64}
    mkl_pardiso_error::Cint
    mkl_pardiso_IPARM::NTuple{64,Cint}
    pastix_info::Cint
    wsmp_error::Cint
    wsmp_iparm::NTuple{64,Cint}
    wsmp_dparm::NTuple{64,Float64}
    mpi_ierr::Cint
    lapack_error::Cint
    sls_inform_type() = new()
end

export sls_initialize

function sls_initialize(solver, data, control, status)
    @ccall libgalahad_double.sls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                            control::Ref{sls_control_type},
                                            status::Ptr{Cint})::Cvoid
end

export sls_read_specfile

function sls_read_specfile(control, specfile)
    @ccall libgalahad_double.sls_read_specfile(control::Ref{sls_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

export sls_analyse_matrix

function sls_analyse_matrix(control, data, status, n, type, ne, row, col, ptr)
    @ccall libgalahad_double.sls_analyse_matrix(control::Ref{sls_control_type},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, type::Ptr{Cchar}, ne::Cint,
                                                row::Ptr{Cint}, col::Ptr{Cint},
                                                ptr::Ptr{Cint})::Cvoid
end

export sls_reset_control

function sls_reset_control(control, data, status)
    @ccall libgalahad_double.sls_reset_control(control::Ref{sls_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export sls_factorize_matrix

function sls_factorize_matrix(data, status, ne, val)
    @ccall libgalahad_double.sls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  ne::Cint, val::Ptr{Float64})::Cvoid
end

export sls_solve_system

function sls_solve_system(data, status, n, sol)
    @ccall libgalahad_double.sls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, sol::Ptr{Float64})::Cvoid
end

export sls_partial_solve_system

function sls_partial_solve_system(part, data, status, n, sol)
    @ccall libgalahad_double.sls_partial_solve_system(part::Ptr{Cchar},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, n::Cint,
                                                      sol::Ptr{Float64})::Cvoid
end

export sls_information

function sls_information(data, inform, status)
    @ccall libgalahad_double.sls_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{sls_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export sls_terminate

function sls_terminate(data, control, inform)
    @ccall libgalahad_double.sls_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{sls_control_type},
                                           inform::Ref{sls_inform_type})::Cvoid
end
