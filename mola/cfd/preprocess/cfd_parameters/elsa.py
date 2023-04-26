#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

from mola import misc

K_OMEGA_TWO_EQN_MODELS = ['Wilcox2006-klim', 'Wilcox2006-klim-V',
            'Wilcox2006', 'Wilcox2006-V', 'SST-2003', 
            'SST-V2003', 'SST', 'SST-V',  'BSL', 'BSL-V']

K_OMEGA_MODELS = K_OMEGA_TWO_EQN_MODELS + [ 'SST-2003-LM2009',
                 'SST-V2003-LM2009', 'SSG/LRR-RSM-w2012']

AvailableTurbulenceModels = K_OMEGA_MODELS + ['smith', 'SA']


def adapt_to_solver(workflow):

    set_cfdpb(workflow)
    set_model(workflow)
    set_numerics(workflow)

def set_cfdpb(workflow):
    workflow.SolverParameters['cfdpb'] = dict(
        config=f'{workflow.ProblemDimension}d',
        extract_filtering='inactive' # NOTE required with writingmode=2 for NeumannData in coprocess
    )

    if not workflow.tree.isStructured():
        workflow.SolverParameters['cfdpb'].update(
            dict(
                metrics_as_unstruct='active',
                metrics_type='barycenter'
            )
        )
    
def set_model(workflow):

    # _____________________________________________________________________________
    FluidSetup = dict(
        cv               = workflow.Fluid['cv'],
        fluid            = 'pg',
        gamma            = workflow.Fluid['Gamma'],
        phymod           = 'nstur',
        prandtl          = workflow.Fluid['Prandtl'],
        prandtltb        = workflow.Fluid['PrandtlTurbulence'],
        visclaw          = 'sutherland',
        suth_const       = workflow.Fluid['SutherlandConstant'],
        suth_muref       = workflow.Fluid['SutherlandViscosity'],
        suth_tref        = workflow.Fluid['SutherlandTemperature'],

        # Boundary-layer computation parameters
        vortratiolim    = 1e-3,
        shearratiolim   = 2e-2,
        pressratiolim   = 1e-3,
        linearratiolim  = 1e-3,
        delta_compute   = 'first_order_bl',

    )

    # _____________________________________________________________________________
    # Wall distance computation
    if workflow.tree.isStructured():
        WallDistanceSetup = dict(walldistcompute='mininterf_ortho')
    else:
        WallDistanceSetup = dict(walldistcompute='mininterf')

    # _____________________________________________________________________________
    TurbulenceSetup = {

        'SA': dict(
        turbmod        = 'spalart',
        ),

        'Wilcox2006-klim': dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        ),
        
        'Wilcox2006-klim-V': dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        ),

        'Wilcox2006': dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        ),
        
        'Wilcox2006-V': dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        ),

        'SST-2003': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        ),

        'SST-V2003': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',        
        ),

        'SST': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'standard',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        ),

        'SST-V': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'standard',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        ),

        'BSL': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'inactive',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',        
        ),

        'BSL-V': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'inactive',
        k_prod_limiter = 20.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        ),

        'smith': dict(
        turbmod        = 'smith',
        k_prod_compute = 'from_sij',
        # TODO: uncomment the following line ?
        # k_prod_limiter = 20.,
        ),

        'smith-V': dict(
        turbmod        = 'smith',
        k_prod_compute = 'from_vorticity',
        ),

        'SST-2003-LM2009': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        trans_mod      = 'menter',
        ),

        'SST-V2003-LM2009': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
        trans_mod      = 'menter',
        ),

        'SSG/LRR-RSM-w2012': dict(
        turbmod          = 'rsm',
        rsm_name         = 'ssg_lrr_bsl',
        rsm_diffusion    = 'isotropic',
        rsm_bous_limiter = 10.0,
        omega_prolong    = 'linear_extrap',
        ),

    }
    
    # _____________________________________________________________________________
    # Transition Settings
    if workflow.Turbulence['TransitionMode']:

        TransitionModeSetup = dict()
    
    elif workflow.Turbulence['TransitionMode'] == 'NonLocalCriteria-LSTT':

        if 'LM2009' in workflow.Turbulence['Model']:
            raise AttributeError(misc.RED+"Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model"+misc.ENDC)
        
        TransitionModeSetup = dict(
            freqcomptrans     = 1,
            trans_crit        = 'in_ahd_gl_comp',
            trans_max_bubble  = 'inactive',
            ext_turb_lev      = workflow.Turbulence['Level'] * 100,
            intermittency     = 'limited',
            interm_thick_coef = 1.2,
            ext_turb_lev_lim  = 'constant_tu',
            trans_shift       = 1.0,
            firstcomptrans    = 1,
            lastcomptrans     = int(1e9),
            trans_comp_h      = 'h_calc',
            trans_gl_ctrl_h1  = 3.0,
            trans_gl_ctrl_h2  = 3.2,
            trans_gl_ctrl_h3  = 3.6,
            # LSTT specific parameters (see ticket #6501)
            trans_crit_order       = 'first_order',
            trans_crit_extrap      = 'active',
            intermit_region        = 'LSTT', # TODO: Not read in fullCGNS -> https://elsa.onera.fr/issues/8145
            intermittency_form     = 'LSTT19',
            trans_h_crit_ahdgl     = 2.8,
            ahd_n_extract          = 'active',
        )

    elif workflow.Turbulence['TransitionMode'] == 'NonLocalCriteria-Step':
        if 'LM2009' in workflow.Turbulence['Model']:
            raise AttributeError(misc.RED+"Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model"+misc.ENDC)
        TransitionModeSetup = dict(
            freqcomptrans     = 1,
            trans_crit        = 'in_ahd_comp',
            trans_max_bubble  = 'inactive',
            ext_turb_lev      = workflow.Turbulence['Level'] * 100,
            intermittency     = 'limited',
            interm_thick_coef = 1.2,
            ext_turb_lev_lim  = 'constant_tu',
            trans_shift       = 1.0,
            firstcomptrans    = 1,
            lastcomptrans     = int(1e9),
            trans_comp_h      = 'h_calc',
            intermittency_form     = 'LSTT19',
            trans_h_crit_ahdgl     = 2.8,
            ahd_n_extract          = 'active',
        )

    elif workflow.Turbulence['TransitionMode'] == 'Imposed':
        if 'LM2009' in workflow.Turbulence['Model']:
            raise AttributeError(misc.RED+"Modeling incoherency! cannot make imposed transition with Menter-Langtry turbulence model"+misc.ENDC)
        TransitionModeSetup = dict(
            intermittency       = 'full',
            interm_thick_coef   = 1.2,
            intermittency_form  = 'LSTT19',
        )
    
    if workflow.Turbulence['TransitionMode'] and workflow.Turbulence['Model'] in K_OMEGA_MODELS:  
        TransitionModeSetup['prod_omega_red'] = 'active'

    # _____________________________________________________________________________
    workflow.SolverParameters['model'] = dict(
        **FluidSetup,
        **WallDistanceSetup,
        **TurbulenceSetup[workflow.Turbulence['Model']],
        **TransitionModeSetup,
    )


def set_numerics(workflow):

    # _____________________________________________________________________________
    # Convective flux 
    if workflow.Numerics['Scheme'] == 'Jameson':
        SchemeSetup = dict(
        flux               = 'jameson',
        avcoef_k2          = 0.5,
        avcoef_k4          = 0.016,
        avcoef_sigma       = 1.0,
        filter             = 'incr_new+prolong',
        cutoff_dens        = 0.005,
        cutoff_pres        = 0.005,
        cutoff_eint        = 0.005,
        artviscosity       = 'dismrt',
        av_mrt             = 0.3,
        av_border          = 'current', # default elsA is 'dif0null', but JCB, JM, LC use 'current' see https://elsa.onera.fr/issues/10624
        av_formul          = 'current', # default elsA is 'new', but JCB, JM, LC use 'current' see https://elsa.onera.fr/issues/10624
        )
        if workflow.tree.isStructured():
            SchemeSetup.update(dict(
                artviscosity       = 'dismrt',
                av_mrt             = 0.3,
            ))
        else:
            # Martinelli correction not available for unstructured grids
            SchemeSetup['artviscosity'] = 'dissca'
    elif workflow.Numerics['Scheme'] == 'ausm+':
        SchemeSetup = dict(
        flux               = 'ausmplus_pmiles',
        ausm_wiggle        = 'inactive',
        ausmp_diss_cst     = 0.04,
        ausmp_press_vel_cst= 0.04,
        ausm_tref          = workflow.Flow['Temperature'],
        ausm_pref          = workflow.Flow['Pressure'],
        ausm_mref          = workflow.Flow['Mach'],
        limiter            = 'third_order',
        )
    elif workflow.Numerics['Scheme'] == 'Roe':
        SchemeSetup = dict(
        flux               = 'roe',
        limiter            = 'valbada',
        psiroe             = 0.01,
        )
    else:
        raise AttributeError(f'Numerical scheme {workflow.Numerics["Scheme"]} not recognized for the solver elsA')
    
    SchemeSetup['t_harten'] = 0.01

    # Viscous flux 
    if workflow.tree.isStructured():
        SchemeSetup['viscous_fluxes']  = '5p_cor'
    else:
        SchemeSetup['viscous_fluxes']  = '5p_cor2' # adapted to unstructured mesh
        SchemeSetup['implconvectname'] = 'vleer' # only available for unstructured mesh, see https://elsa-e.onera.fr/issues/6492

    # _____________________________________________________________________________
    # CFL 
    if isinstance(workflow.Numerics['CFL'], dict):
        CFLSetup = {
            'cfl_fct': 'f_cfl',
            '.Solver#Function': dict(
                iteri = workflow.Numerics['CFL']['StartIteration'],
                iterf = workflow.Numerics['CFL']['EndIteration'],
                vali  = workflow.Numerics['CFL']['StartValue'],
                valf  = workflow.Numerics['CFL']['EndValue'],
            )
        }
    else:
        CFLSetup = dict(cfl=workflow.Numerics['CFL'])

    # _____________________________________________________________________________
    # Time marching 
    TimeMarchingSetup = dict(
        inititer           = workflow.Numerics['IterationAtInitialState'],
        niter              = workflow.Numerics['NumberOfIterations'],
        ode                = 'backwardeuler',
        implicit           = 'lussorsca',
        ssorcycle          = 4,
        freqcompres        = 1,
    )

    if workflow.Numerics['TimeMarching'] == 'Steady':

        TimeMarchingSetup.update({
            'time_algo'        : 'steady',
            'global_timestep'  : 'inactive',
            'timestep_div'     : 'divided',  # timestep divided by 2 at the boundaries ; should not be used in unsteady simulations
            'residual_type'    : 'explicit_novolum',
            **CFLSetup,
        })

    else:

        TimeMarchingSetup.update(dict(
            timestep           = workflow.Numerics['TimeStep'],
            itime              = workflow.Numerics['TimeAtInitialState'],
            restoreach_cons    = 1e-2,
        ))

        if workflow.Numerics['TimeMarchingOrder'] == 1:

            TimeMarchingSetup['time_algo'] = 'unsteady'

        else:

            TimeMarchingSetup['time_algo']      = 'gear'
            TimeMarchingSetup['gear_iteration'] = 20

   
    # _____________________________________________________________________________
    # Definition of cut-off values for turbulence 
    turbValues = workflow.Flow['ReferenceStateTurbulence'].values()
    if len(turbValues) == 7:  # RSM
        TurbulenceCutOffSetup = dict(
            t_cutvar1 = workflow.Turbulence['TurbulenceCutOffRatio'] * turbValues[0],
            t_cutvar2 = workflow.Turbulence['TurbulenceCutOffRatio'] * turbValues[3],
            t_cutvar3 = workflow.Turbulence['TurbulenceCutOffRatio'] * turbValues[5],
            t_cutvar4 = workflow.Turbulence['TurbulenceCutOffRatio'] * turbValues[6],
        )

    elif len(turbValues) > 4: # unsupported 
        raise ValueError('UNSUPPORTED NUMBER OF TURBULENT FIELDS')
    
    else:
        TurbulenceCutOffSetup = dict()
        for i, value in enumerate(turbValues):
            TurbulenceCutOffSetup[f't_cutvar{i+1}'] = workflow.Turbulence['TurbulenceCutOffRatio'] * value


    # _____________________________________________________________________________
    # Miscellaneous 
    MiscellaneousSetup = dict(
        multigrid        = 'none',
        misc_source_term = 'inactive',
    )

    # TODO Check implementation for BodyForce and Chimera
    if hasattr(workflow, 'BodyForce'):
        MiscellaneousSetup['misc_source_term'] = 'active'

    # Chimera parameters
    if workflow.has_overset_component():
        MiscellaneousSetup.update(dict(
            chm_double_wall      = 'active',
            chm_double_wall_tol  = 2000.,
            chm_orphan_treatment = 'neighbourgsmean',
            chm_impl_interp      = 'none',
            chm_interp_depth     = 2
        ))
        
    # _____________________________________________________________________________
    workflow.SolverParameters['numerics'] = dict(
        **SchemeSetup,
        **TimeMarchingSetup,
        **TurbulenceCutOffSetup,
        **MiscellaneousSetup,
    )

