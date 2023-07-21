mutable struct ma48_control
    f_arrays::Cint
    multiplier::Float64
    u::Float64
    switch_::Float64
    drop::Float64
    tolerance::Float64
    cgce::Float64
    lp::Cint
    wp::Cint
    mp::Cint
    ldiag::Cint
    btf::Cint
    struct_::Cint
    maxit::Cint
    factor_blocking::Cint
    solve_blas::Cint
    pivoting::Cint
    diagonal_pivoting::Cint
    fill_in::Cint
    switch_mode::Cint
end

mutable struct ma48_ainfo
    ops::Float64
    flag::Cint
    more::Cint
    lena_analyse::Clong
    lenj_analyse::Clong
    lena_factorize::Clong
    leni_factorize::Clong
    ncmpa::Cint
    rank::Cint
    drop::Clong
    struc_rank::Cint
    oor::Clong
    dup::Clong
    stat::Cint
    lblock::Cint
    sblock::Cint
    tblock::Clong
end

mutable struct ma48_finfo
    ops::Float64
    flag::Cint
    more::Cint
    size_factor::Clong
    lena_factorize::Clong
    leni_factorize::Clong
    drop::Clong
    rank::Cint
    stat::Cint
end

mutable struct ma48_sinfo
    flag::Cint
    more::Cint
    stat::Cint
end

mutable struct ma57_control
    f_arrays::Cint
    multiplier::Float64
    reduce::Float64
    u::Float64
    static_tolerance::Float64
    static_level::Float64
    tolerance::Float64
    convergence::Float64
    consist::Float64
    lp::Cint
    wp::Cint
    mp::Cint
    sp::Cint
    ldiag::Cint
    nemin::Cint
    factorblocking::Cint
    solveblocking::Cint
    la::Cint
    liw::Cint
    maxla::Cint
    maxliw::Cint
    pivoting::Cint
    thresh::Cint
    ordering::Cint
    scaling::Cint
    rank_deficient::Cint
    ispare::NTuple{5,Cint}
    rspare::NTuple{10,Float64}
end

mutable struct ma57_ainfo
    opsa::Float64
    opse::Float64
    flag::Cint
    more::Cint
    nsteps::Cint
    nrltot::Cint
    nirtot::Cint
    nrlnec::Cint
    nirnec::Cint
    nrladu::Cint
    niradu::Cint
    ncmpa::Cint
    ordering::Cint
    oor::Cint
    dup::Cint
    maxfrt::Cint
    stat::Cint
    ispare::NTuple{5,Cint}
    rspare::NTuple{10,Float64}
end

mutable struct ma57_finfo
    opsa::Float64
    opse::Float64
    opsb::Float64
    maxchange::Float64
    smin::Float64
    smax::Float64
    flag::Cint
    more::Cint
    maxfrt::Cint
    nebdu::Cint
    nrlbdu::Cint
    nirbdu::Cint
    nrltot::Cint
    nirtot::Cint
    nrlnec::Cint
    nirnec::Cint
    ncmpbr::Cint
    ncmpbi::Cint
    ntwo::Cint
    neig::Cint
    delay::Cint
    signc::Cint
    static_::Cint
    modstep::Cint
    rank::Cint
    stat::Cint
    ispare::NTuple{5,Cint}
    rspare::NTuple{10,Float64}
end

mutable struct ma57_sinfo
    cond::Float64
    cond2::Float64
    berr::Float64
    berr2::Float64
    error::Float64
    flag::Cint
    stat::Cint
    ispare::NTuple{5,Cint}
    rspare::NTuple{10,Float64}
end

mutable struct ma77_control
    f_arrays::Cint
    print_level::Cint
    unit_diagnostics::Cint
    unit_error::Cint
    unit_warning::Cint
    bits::Cint
    buffer_lpage::NTuple{2,Cint}
    buffer_npage::NTuple{2,Cint}
    file_size::Clong
    maxstore::Clong
    storage::NTuple{3,Clong}
    nemin::Cint
    maxit::Cint
    infnorm::Cint
    thresh::Float64
    nb54::Cint
    action::Cint
    multiplier::Float64
    nb64::Cint
    nbi::Cint
    small::Float64
    static_::Float64
    storage_indef::Clong
    u::Float64
    umin::Float64
    consist_tol::Float64
    ispare::NTuple{5,Cint}
    lspare::NTuple{5,Clong}
    rspare::NTuple{5,Float64}
end

mutable struct ma77_info
    detlog::Float64
    detsign::Cint
    flag::Cint
    iostat::Cint
    matrix_dup::Cint
    matrix_rank::Cint
    matrix_outrange::Cint
    maxdepth::Cint
    maxfront::Cint
    minstore::Clong
    ndelay::Cint
    nfactor::Clong
    nflops::Clong
    niter::Cint
    nsup::Cint
    num_neg::Cint
    num_nothresh::Cint
    num_perturbed::Cint
    ntwo::Cint
    stat::Cint
    index::NTuple{4,Cint}
    nio_read::NTuple{2,Clong}
    nio_write::NTuple{2,Clong}
    nwd_read::NTuple{2,Clong}
    nwd_write::NTuple{2,Clong}
    num_file::NTuple{4,Cint}
    storage::NTuple{4,Clong}
    tree_nodes::Cint
    unit_restart::Cint
    unused::Cint
    usmall::Float64
    ispare::NTuple{5,Cint}
    lspare::NTuple{5,Clong}
    rspare::NTuple{5,Float64}
end

mutable struct ma86_control
    f_arrays::Cint
    diagnostics_level::Cint
    unit_diagnostics::Cint
    unit_error::Cint
    unit_warning::Cint
    nemin::Cint
    nb::Cint
    action::Cint
    nbi::Cint
    pool_size::Cint
    small_::Float64
    static_::Float64
    u::Float64
    umin::Float64
    scaling::Cint
end

mutable struct ma86_info
    detlog::Float64
    detsign::Cint
    flag::Cint
    matrix_rank::Cint
    maxdepth::Cint
    num_delay::Cint
    num_factor::Clong
    num_flops::Clong
    num_neg::Cint
    num_nodes::Cint
    num_nothresh::Cint
    num_perturbed::Cint
    num_two::Cint
    pool_size::Cint
    stat::Cint
    usmall::Float64
end

mutable struct ma87_control
    f_arrays::Cint
    diagnostics_level::Cint
    unit_diagnostics::Cint
    unit_error::Cint
    unit_warning::Cint
    nemin::Cint
    nb::Cint
    pool_size::Cint
    diag_zero_minus::Float64
    diag_zero_plus::Float64
    unused::NTuple{40,Cchar}
end

mutable struct ma87_info
    detlog::Float64
    flag::Cint
    maxdepth::Cint
    num_factor::Clong
    num_flops::Clong
    num_nodes::Cint
    pool_size::Cint
    stat::Cint
    num_zero::Cint
    unused::NTuple{40,Cchar}
end

mutable struct ma97_control
    f_arrays::Cint
    action::Cint
    nemin::Cint
    multiplier::Float64
    ordering::Cint
    print_level::Cint
    scaling::Cint
    small::Float64
    u::Float64
    unit_diagnostics::Cint
    unit_error::Cint
    unit_warning::Cint
    factor_min::Clong
    solve_blas3::Cint
    solve_min::Clong
    solve_mf::Cint
    consist_tol::Float64
    ispare::NTuple{5,Cint}
    rspare::NTuple{10,Float64}
end

mutable struct ma97_info
    flag::Cint
    flag68::Cint
    flag77::Cint
    matrix_dup::Cint
    matrix_rank::Cint
    matrix_outrange::Cint
    matrix_missing_diag::Cint
    maxdepth::Cint
    maxfront::Cint
    num_delay::Cint
    num_factor::Clong
    num_flops::Clong
    num_neg::Cint
    num_sup::Cint
    num_two::Cint
    ordering::Cint
    stat::Cint
    ispare::NTuple{5,Cint}
    rspare::NTuple{10,Float64}
end

mutable struct mc64_control
    f_arrays::Cint
    lp::Cint
    wp::Cint
    sp::Cint
    ldiag::Cint
    checking::Cint
end

mutable struct mc64_info
    flag::Cint
    more::Cint
    strucrank::Cint
    stat::Cint
end

mutable struct mc68_control
    f_array_in::Cint
    f_array_out::Cint
    min_l_workspace::Cint
    lp::Cint
    wp::Cint
    mp::Cint
    nemin::Cint
    print_level::Cint
    row_full_thresh::Cint
    row_search::Cint
end

mutable struct mc68_info
    flag::Cint
    iostat::Cint
    stat::Cint
    out_range::Cint
    duplicate::Cint
    n_compressions::Cint
    n_zero_eigs::Cint
    l_workspace::Clong
    zb01_info::Cint
    n_dense_rows::Cint
end

mutable struct mi20_control
    f_arrays::Cint
    aggressive::Cint
    c_fail::Cint
    max_levels::Cint
    max_points::Cint
    reduction::Float64
    st_method::Cint
    st_parameter::Float64
    testing::Cint
    trunc_parameter::Float64
    coarse_solver::Cint
    coarse_solver_its::Cint
    damping::Float64
    err_tol::Float64
    levels::Cint
    pre_smoothing::Cint
    smoother::Cint
    post_smoothing::Cint
    v_iterations::Cint
    print_level::Cint
    print::Cint
    error::Cint
    one_pass_coarsen::Cint
end

mutable struct mi20_solve_control
    abs_tol::Float64
    breakdown_tol::Float64
    gmres_restart::Cint
    init_guess::Bool
    krylov_solver::Cint
    max_its::Cint
    preconditioner_side::Cint
    rel_tol::Float64
end

mutable struct mi20_info
    flag::Cint
    clevels::Cint
    cpoints::Cint
    cnnz::Cint
    stat::Cint
    getrf_info::Cint
    iterations::Cint
    residual::Float64
end

mutable struct mi28_control
    f_arrays::Cint
    alpha::Float64
    check::Bool
    iorder::Cint
    iscale::Cint
    lowalpha::Float64
    maxshift::Cint
    rrt::Bool
    shift_factor::Float64
    shift_factor2::Float64
    small::Float64
    tau1::Float64
    tau2::Float64
    unit_error::Cint
    unit_warning::Cint
end

mutable struct mi28_info
    band_after::Cint
    band_before::Cint
    dup::Cint
    flag::Cint
    flag61::Cint
    flag64::Cint
    flag68::Cint
    flag77::Cint
    nrestart::Cint
    nshift::Cint
    oor::Cint
    profile_before::Float64
    profile_after::Float64
    size_r::Clong
    stat::Cint
    alpha::Float64
end
