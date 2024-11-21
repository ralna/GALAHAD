#ifdef INTEGER_64
#define ma48_control_r ma48_control_d_64
#define ma48_ainfo_r ma48_ainfo_d_64
#define ma48_finfo_r ma48_finfo_d_64
#define ma48_sinfo_r ma48_sinfo_d_64
#define ma48_initialize_r ma48_initialize_d_64
#define ma48_default_control_r ma48_default_control_d_64
#define ma48_analyse_r ma48_analyse_d_64
#define ma48_get_perm_r ma48_get_perm_d_64
#define ma48_factorize_r ma48_factorize_d_64
#define ma48_solve_r ma48_solve_d_64
#define ma48_finalize_r ma48_finalize_d_64
#define ma48_special_rows_and_cols_r ma48_special_rows_and_cols_d_64
#define ma48_determinant_r ma48_determinant_d_64
#else
#define ma48_control_r ma48_control_d
#define ma48_ainfo_r ma48_ainfo_d
#define ma48_finfo_r ma48_finfo_d
#define ma48_sinfo_r ma48_sinfo_d
#define ma48_initialize_r ma48_initialize_d
#define ma48_default_control_r ma48_default_control_d
#define ma48_analyse_r ma48_analyse_d
#define ma48_get_perm_r ma48_get_perm_d
#define ma48_factorize_r ma48_factorize_d
#define ma48_solve_r ma48_solve_d
#define ma48_finalize_r ma48_finalize_d
#define ma48_special_rows_and_cols_r ma48_special_rows_and_cols_d
#define ma48_determinant_r ma48_determinant_d
#endif

#ifdef INTEGER_64
#define ma57_default_control_r      ma57_default_control_d_64
#define ma57_init_factors_r         ma57_init_factors_d_64
#define ma57_control_r              ma57_control_d_64
#define ma57_ainfo_r                ma57_ainfo_d_64
#define ma57_finfo_r                ma57_finfo_d_64
#define ma57_sinfo_r                ma57_sinfo_d_64
#define ma57_analyse_r              ma57_analyse_d_64
#define ma57_factorize_r            ma57_factorize_d_64
#define ma57_solve_r                ma57_solve_d_64
#define ma57_finalize_r             ma57_finalize_d_64
#define ma57_enquire_perm_r    	  ma57_enquire_perm_d_64
#define ma57_enquire_pivots_r  	  ma57_enquire_pivots_d_64
#define ma57_enquire_d_r       	  ma57_enquire_d_d_64
#define ma57_enquire_perturbation_r ma57_enquire_perturbation_d_64
#define ma57_enquire_scaling_r      ma57_enquire_scaling_d_64
#define ma57_alter_d_r              ma57_alter_d_d_64
#define ma57_part_solve_r           ma57_part_solve_d_64
#define ma57_sparse_lsolve_r        ma57_sparse_lsolve_d_64
#define ma57_fredholm_alternative_r ma57_fredholm_alternative_d_64
#define ma57_lmultiply_r            ma57_lmultiply_d_64
#define ma57_get_factors_r          ma57_get_factors_d_64
#else
#define ma57_default_control_r      ma57_default_control_d
#define ma57_init_factors_r         ma57_init_factors_d
#define ma57_control_r              ma57_control_d
#define ma57_ainfo_r                ma57_ainfo_d
#define ma57_finfo_r                ma57_finfo_d
#define ma57_sinfo_r                ma57_sinfo_d
#define ma57_analyse_r              ma57_analyse_d
#define ma57_factorize_r            ma57_factorize_d
#define ma57_solve_r                ma57_solve_d
#define ma57_finalize_r             ma57_finalize_d
#define ma57_enquire_perm_r    	  ma57_enquire_perm_d
#define ma57_enquire_pivots_r  	  ma57_enquire_pivots_d
#define ma57_enquire_d_r       	  ma57_enquire_d_d
#define ma57_enquire_perturbation_r ma57_enquire_perturbation_d
#define ma57_enquire_scaling_r      ma57_enquire_scaling_d
#define ma57_alter_d_r              ma57_alter_d_d
#define ma57_part_solve_r           ma57_part_solve_d
#define ma57_sparse_lsolve_r        ma57_sparse_lsolve_d
#define ma57_fredholm_alternative_r ma57_fredholm_alternative_d
#define ma57_lmultiply_r            ma57_lmultiply_d
#define ma57_get_factors_r          ma57_get_factors_d
#endif

#ifdef INTEGER_64
#define ma77_control_r ma77_control_d_64
#define ma77_info_r ma77_info_d_64
#define ma77_default_control_r ma77_default_control_d_64
#define ma77_open_nelt_r ma77_open_nelt_d_64
#define ma77_open_r ma77_open_d_64
#define ma77_input_vars_r ma77_input_vars_d_64
#define ma77_input_reals_r ma77_input_reals_d_64
#define ma77_analyse_r ma77_analyse_d_64
#define ma77_factor_r ma77_factor_d_64
#define ma77_factor_solve_r ma77_factor_solve_d_64
#define ma77_solve_r ma77_solve_d_64
#define ma77_resid_r ma77_resid_d_64
#define ma77_scale_r ma77_scale_d_64
#define ma77_enquire_posdef_r ma77_enquire_posdef_d_64
#define ma77_enquire_indef_r ma77_enquire_indef_d_64
#define ma77_alter_r ma77_alter_d_64
#define ma77_restart_r ma77_restart_d_64
#define ma77_finalise_r ma77_finalise_d_64
#define ma77_solve_fredholm_r ma77_solve_fredholm_d_64
#define ma77_lmultiply_r ma77_lmultiply_d_64
#else
#define ma77_control_r ma77_control_d
#define ma77_info_r ma77_info_d
#define ma77_default_control_r ma77_default_control_d
#define ma77_open_nelt_r ma77_open_nelt_d
#define ma77_open_r ma77_open_d
#define ma77_input_vars_r ma77_input_vars_d
#define ma77_input_reals_r ma77_input_reals_d
#define ma77_analyse_r ma77_analyse_d
#define ma77_factor_r ma77_factor_d
#define ma77_factor_solve_r ma77_factor_solve_d
#define ma77_solve_r ma77_solve_d
#define ma77_resid_r ma77_resid_d
#define ma77_scale_r ma77_scale_d
#define ma77_enquire_posdef_r ma77_enquire_posdef_d
#define ma77_enquire_indef_r ma77_enquire_indef_d
#define ma77_alter_r ma77_alter_d
#define ma77_restart_r ma77_restart_d
#define ma77_finalise_r ma77_finalise_d
#define ma77_solve_fredholm_r ma77_solve_fredholm_d
#define ma77_lmultiply_r ma77_lmultiply_d
#endif

#ifdef INTEGER_64
#define ma86_control_r ma86_control_d_64
#define ma86_info_r ma86_info_d_64
#define ma86_default_control_r ma86_default_control_d_64
#define ma86_analyse_r ma86_analyse_d_64
#define ma86_factor_r ma86_factor_d_64
#define ma86_factor_solve_r ma86_factor_solve_d_64
#define ma86_solve_r ma86_solve_d_64
#define ma86_finalise_r ma86_finalise_d_64
#else
#define ma86_control_r ma86_control_d
#define ma86_info_r ma86_info_d
#define ma86_default_control_r ma86_default_control_d
#define ma86_analyse_r ma86_analyse_d
#define ma86_factor_r ma86_factor_d
#define ma86_factor_solve_r ma86_factor_solve_d
#define ma86_solve_r ma86_solve_d
#define ma86_finalise_r ma86_finalise_d
#endif

#ifdef INTEGER_64
#define ma87_control_r ma87_control_d_64
#define ma87_info_r ma87_info_d_64
#define ma87_default_control_r ma87_default_control_d_64
#define ma87_analyse_r ma87_analyse_d_64
#define ma87_factor_r ma87_factor_d_64
#define ma87_factor_solve_r ma87_factor_solve_d_64
#define ma87_solve_r ma87_solve_d_64
#define ma87_sparse_fwd_solve_r ma87_sparse_fwd_solve_d_64
#define ma87_finalise_r ma87_finalise_d_64
#else
#define ma87_control_r ma87_control_d
#define ma87_info_r ma87_info_d
#define ma87_default_control_r ma87_default_control_d
#define ma87_analyse_r ma87_analyse_d
#define ma87_factor_r ma87_factor_d
#define ma87_factor_solve_r ma87_factor_solve_d
#define ma87_solve_r ma87_solve_d
#define ma87_sparse_fwd_solve_r ma87_sparse_fwd_solve_d
#define ma87_finalise_r ma87_finalise_d
#endif

#ifdef INTEGER_64
#define ma97_control_r ma97_control_d_64
#define ma97_info_r ma97_info_d_64
#define ma97_default_control_r ma97_default_control_d_64
#define ma97_analyse_r ma97_analyse_d_64
#define ma97_analyse_coord_r ma97_analyse_coord_d_64
#define ma97_factor_r ma97_factor_d_64
#define ma97_factor_solve_r ma97_factor_solve_d_64
#define ma97_solve_r ma97_solve_d_64
#define ma97_free_akeep_r ma97_free_akeep_d_64
#define ma97_free_fkeep_r ma97_free_fkeep_d_64
#define ma97_finalise_r ma97_finalise_d_64
#define ma97_enquire_posdef_r ma97_enquire_posdef_d_64
#define ma97_enquire_indef_r ma97_enquire_indef_d_64
#define ma97_alter_r ma97_alter_d_64
#define ma97_solve_fredholm_r ma97_solve_fredholm_d_64
#define ma97_lmultiply_r ma97_lmultiply_d_64
#define ma97_sparse_fwd_solve_r ma97_sparse_fwd_solve_d_64
#else
#define ma97_control_r ma97_control_d
#define ma97_info_r ma97_info_d
#define ma97_default_control_r ma97_default_control_d
#define ma97_analyse_r ma97_analyse_d
#define ma97_analyse_coord_r ma97_analyse_coord_d
#define ma97_factor_r ma97_factor_d
#define ma97_factor_solve_r ma97_factor_solve_d
#define ma97_solve_r ma97_solve_d
#define ma97_free_akeep_r ma97_free_akeep_d
#define ma97_free_fkeep_r ma97_free_fkeep_d
#define ma97_finalise_r ma97_finalise_d
#define ma97_enquire_posdef_r ma97_enquire_posdef_d
#define ma97_enquire_indef_r ma97_enquire_indef_d
#define ma97_alter_r ma97_alter_d
#define ma97_solve_fredholm_r ma97_solve_fredholm_d
#define ma97_lmultiply_r ma97_lmultiply_d
#define ma97_sparse_fwd_solve_r ma97_sparse_fwd_solve_d
#endif

#ifdef INTEGER_64
#define mc64_control_r mc64_control_d_64
#define mc64_info_r mc64_info_d_64
#define mc64_default_control_r mc64_default_control_d_64
#define mc64_matching_r mc64_matching_d_64
#else
#define mc64_control_r mc64_control_d
#define mc64_info_r mc64_info_d
#define mc64_default_control_r mc64_default_control_d
#define mc64_matching_r mc64_matching_d
#endif

#ifdef INTEGER_64
#define mi20_default_control_r mi20_default_control_d_64
#define mi20_control_r mi20_control_d_64
#define mi20_default_solve_control_r mi20_default_solve_control_d_64
#define mi20_solve_control_r mi20_solve_control_d_64
#define mi20_info_r mi20_info_d_64
#define mi20_setup_r mi20_setup_d_64
#define mi20_setup_csr_r mi20_setup_csr_d_64
#define mi20_setup_csc_r mi20_setup_csc_d_64
#define mi20_setup_coord_r mi20_setup_coord_d_64
#define mi20_finalize_r mi20_finalize_d_64
#define mi20_precondition_r mi20_precondition_d_64
#define mi20_solve_r mi20_solve_d_64
#else
#define mi20_default_control_r mi20_default_control_d
#define mi20_control_r mi20_control_d
#define mi20_default_solve_control_r mi20_default_solve_control_d
#define mi20_solve_control_r mi20_solve_control_d
#define mi20_info_r mi20_info_d
#define mi20_setup_r mi20_setup_d
#define mi20_setup_csr_r mi20_setup_csr_d
#define mi20_setup_csc_r mi20_setup_csc_d
#define mi20_setup_coord_r mi20_setup_coord_d
#define mi20_finalize_r mi20_finalize_d
#define mi20_precondition_r mi20_precondition_d
#define mi20_solve_r mi20_solve_d
#endif

#ifdef INTEGER_64
#define mi28_control_r mi28_control_d_64
#define mi28_info_r mi28_info_d_64
#define mi28_default_control_r mi28_default_control_d_64
#define mi28_factorize_r mi28_factorize_d_64
#define mi28_precondition_r mi28_precondition_d_64
#define mi28_solve_r mi28_solve_d_64
#define mi28_finalise_r mi28_finalise_d_64
#else
#define mi28_control_r mi28_control_d
#define mi28_info_r mi28_info_d
#define mi28_default_control_r mi28_default_control_d
#define mi28_factorize_r mi28_factorize_d
#define mi28_precondition_r mi28_precondition_d
#define mi28_solve_r mi28_solve_d
#define mi28_finalise_r mi28_finalise_d
#endif
