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

import copy
from mola.logging import mola_logger, MolaException

from treelab import cgns
from mola.cfd.preprocess.cfd_parameters import cfd_parameters

K_OMEGA_TWO_EQN_MODELS = [
    'Wilcox2006-klim',
    'Wilcox2006-klim-V',
    'Wilcox2006',
    'Wilcox2006-V',
    'SST-2003',
    'SST-V2003',
    'SST',
    'SST-V',
    'BSL',
    'BSL-V',
    ]

K_OMEGA_MODELS = K_OMEGA_TWO_EQN_MODELS + [
    'SST-2003-LM2009',
    'SST-V2003-LM2009',
    'SSG/LRR-RSM-w2012',
    ]

AvailableTurbulenceModels = K_OMEGA_MODELS + [
    'smith',
    'SA',
    ]

# NOTE The following dictionary relates NASA names (https://turbmodels.larc.nasa.gov/) 
# for turbulence models to elsa parameters. 
# See https://elsa-doc.onera.fr/MU_tuto/latest/MU-98057/Textes/turbmods.html#nasa-named-turbulence-models-mapping
TURBULENCE_ELSA_KEYS = {

    'SA': dict(
        turbmod        = 'spalart',
    ),

    'Wilcox2006-klim': dict(
        turbmod        = 'komega_kok',
        kok_diff_cor   = 'wilcox2006',
        sst_cor        = 'active',
        sst_version    = 'wilcox2006',
        k_prod_limiter = 20.,
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
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
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
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
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
    ),

    'SST-V2003': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'std_sij',
        k_prod_limiter = 10.,
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',        
    ),

    'SST': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'standard',
        k_prod_limiter = 20.,
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
    ),

    'SST-V': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'active',
        sst_version    = 'standard',
        k_prod_limiter = 20.,
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
    ),

    'BSL': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'inactive',
        k_prod_limiter = 20.,
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
        k_prod_compute = 'from_sij',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',        
    ),

    'BSL-V': dict(
        turbmod        = 'komega_menter',
        sst_cor        = 'inactive',
        k_prod_limiter = 20.,
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
        k_prod_compute = 'from_vorticity',
        zhenglim       = 'inactive',
        omega_prolong  = 'linear_extrap',
    ),

    'smith': dict(
        turbmod        = 'smith',
        k_prod_compute = 'from_sij',
        # TODO: uncomment the following line ?
        # k_prod_limiter = 20.,
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
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
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
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
        k_prod_lim_mut = 'inactive', # https://elsa.onera.fr/issues/12045
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


def apply_to_solver(workflow):

    user_given_parameters = copy.copy(workflow.SolverParameters) 
    set_cfdpb(workflow)
    set_model(workflow)
    set_numerics(workflow)
    cfd_parameters.deep_update(workflow.SolverParameters, user_given_parameters) 

def set_cfdpb(workflow):
    workflow.SolverParameters['cfdpb'] = dict(
        config=f'{workflow.ProblemDimension}d',
        extract_filtering='inactive', # NOTE required with writingmode=2 for NeumannData in coprocess
        cgns_standard = 'active',
    )

    if not workflow.tree.isStructured():
        workflow.SolverParameters['cfdpb'].update(
            dict(
                metrics_as_unstruct='active',
                metrics_type='barycenter'
            )
        )
    
def set_model(workflow):

    workflow.SolverParameters['model'] = get_fluid_setup(workflow.Fluid)

    if workflow.Turbulence['Model'] == 'Euler':
        workflow.SolverParameters['model'].update(
            dict(phymod = 'euler')
        )
    
    else:
        BoundaryLayerParameters = dict(
        vortratiolim    = 1e-3,
        shearratiolim   = 2e-2,
        pressratiolim   = 1e-3,
        linearratiolim  = 1e-3,
        delta_compute   = 'first_order_bl',
        )

        workflow.SolverParameters['model'].update(
            dict(
                phymod = 'nstur',
                prandtltb = workflow.Fluid['PrandtlTurbulent'],
                **BoundaryLayerParameters,
                **get_wall_distance_setup(workflow.tree) ,
                **get_turbulent_setup(workflow.Turbulence),
            )
        )
        

def set_numerics(workflow):

    TurbulenceCutOffSetup = get_turbulence_cutoff_setup(workflow.Turbulence)

    workflow.SolverParameters['numerics'] = dict(
        **get_spatial_fluxes(workflow.Numerics, workflow.tree, workflow.Flow),
        **get_time_marching_setup(workflow.Numerics),
        **TurbulenceCutOffSetup,
        **get_miscellaneous_setup(workflow),
    )

def get_fluid_setup(Fluid):
    FluidSetup = dict(
        cv               = Fluid['cv'],
        fluid            = 'pg',
        gamma            = Fluid['Gamma'],
        prandtl          = Fluid['Prandtl'],
        visclaw          = 'sutherland',
        suth_const       = Fluid['SutherlandConstant'],
        suth_muref       = Fluid['SutherlandViscosity'],
        suth_tref        = Fluid['SutherlandTemperature'],
    ) 
    return FluidSetup

def get_wall_distance_setup(tree):
    if tree.isStructured():
        WallDistanceSetup = dict(walldistcompute='mininterf_ortho')
    else:
        WallDistanceSetup = dict(walldistcompute='mininterf')
    return WallDistanceSetup

def get_turbulent_setup(Turbulence):
    TurbulenceSetup = TURBULENCE_ELSA_KEYS[Turbulence['Model']]
    TurbulenceSetup.update(get_transition_setup(Turbulence))
    return TurbulenceSetup

def get_transition_setup(Turbulence):
    TransitionModeSetup = dict()
    if not 'TransitionMode' in Turbulence: return TransitionModeSetup
    if Turbulence['TransitionMode'] == 'NonLocalCriteria-LSTT':

        if 'LM2009' in Turbulence['Model']:
            raise MolaException('Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model')
        
        TransitionModeSetup = dict(
            freqcomptrans     = 1,
            trans_crit        = 'in_ahd_gl_comp',
            trans_max_bubble  = 'inactive',
            ext_turb_lev      = Turbulence['Level'] * 100,
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

    elif Turbulence['TransitionMode'] == 'NonLocalCriteria-Step':
        if 'LM2009' in Turbulence['Model']:
            raise MolaException('Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model')
        TransitionModeSetup = dict(
            freqcomptrans     = 1,
            trans_crit        = 'in_ahd_comp',
            trans_max_bubble  = 'inactive',
            ext_turb_lev      = Turbulence['Level'] * 100,
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

    elif Turbulence['TransitionMode'] == 'Imposed':
        if 'LM2009' in Turbulence['Model']:
            raise MolaException('Modeling incoherency! cannot make imposed transition with Menter-Langtry turbulence model')
        TransitionModeSetup = dict(
            intermittency       = 'full',
            interm_thick_coef   = 1.2,
            intermittency_form  = 'LSTT19',
        )
    
    if Turbulence['TransitionMode'] and Turbulence['Model'] in K_OMEGA_MODELS:  
        TransitionModeSetup['prod_omega_red'] = 'active'

    return TransitionModeSetup

def get_spatial_fluxes(Numerics, tree, Flow):
    # Convective flux 
    if Numerics['Scheme'] == 'Jameson':
        SchemeSetup = dict(
        flux         = 'jameson',
        avcoef_k2    = 0.5,
        avcoef_k4    = 0.016,
        avcoef_sigma = 1.0,
        av_border    = 'current', # default elsA is 'dif0null', but JCB, JM, LC use 'current' see https://elsa.onera.fr/issues/10624
        av_formul    = 'current', # default elsA is 'new', but JCB, JM, LC use 'current' see https://elsa.onera.fr/issues/10624
        )
        if tree.isStructured():
            SchemeSetup.update(dict(
                artviscosity = 'dismrt',
                av_mrt = 0.3,
            ))
        else:
            # Martinelli correction not available for unstructured grids
            SchemeSetup['artviscosity'] = 'dissca'
    elif Numerics['Scheme'] == 'ausm+':
        SchemeSetup = dict(
        flux               = 'ausmplus_pmiles',
        ausm_wiggle        = 'inactive',
        ausmp_diss_cst     = 0.04,
        ausmp_press_vel_cst= 0.04,
        ausm_tref          = Flow['Temperature'],
        ausm_pref          = Flow['Pressure'],
        ausm_mref          = Flow['Mach'],
        limiter            = 'third_order',
        )
    elif Numerics['Scheme'] == 'Roe':
        SchemeSetup = dict(
        flux    = 'roe',
        limiter = 'valbada',
        psiroe  = 0.01,
        )
    else:
        raise AttributeError(f'Numerical scheme {Numerics["Scheme"]} not recognized for the solver elsA')
    
    SchemeSetup['t_harten'] = 0.01

    # Viscous flux
    SchemeSetup['viscous_fluxes']  = '5p_cor'
    if not tree.isStructured():
        SchemeSetup['viscous_fluxes']  = '5p_cor2'
        SchemeSetup['implconvectname'] = 'vleer' # only available option for unstructured mesh https://elsa-e.onera.fr/issues/6492

    SchemeSetup['extrap_grad_mean'] = 1
    SchemeSetup['extrap_grad_tur'] = 1

    # TODO Put in CHANGELOG: same filtering parameters for all schemes
    FilteringSetup = dict(
        filter = 'incr_new+prolong',
        cutoff_dens = 0.005,
        cutoff_pres = 0.005,
        cutoff_eint = 0.005,
        )
    SchemeSetup.update(FilteringSetup)

    return SchemeSetup

def get_cfl_setup(cfl):
    if isinstance(cfl, dict):
        CFLSetup = {
            'cfl_fct': 'f_cfl',
            '.Solver#Function': dict(
                name  = 'f_cfl', 
                function_type = 'linear',
                iteri = cfl['StartIteration'],
                iterf = cfl['EndIteration'],
                vali  = cfl['StartValue'],
                valf  = cfl['EndValue'],
            )
        }
    else:
        CFLSetup = dict(cfl=cfl)
    return CFLSetup

def get_time_marching_setup(Numerics):
    TimeMarchingSetup = dict(
        inititer           = Numerics['IterationAtInitialState'],
        niter              = Numerics['NumberOfIterations'],
        ode                = 'backwardeuler',
        implicit           = 'lussorsca',
        ssorcycle          = 4,
        freqcompres        = 1,
    )

    if Numerics['TimeMarching'] == 'Steady':

        TimeMarchingSetup.update({
            'time_algo'        : 'steady',
            'global_timestep'  : 'inactive',
            'timestep_div'     : 'divided',  # timestep divided by 2 at the boundaries ; should not be used in unsteady simulations
            'residual_type'    : 'explicit_novolum',
            **get_cfl_setup(Numerics['CFL']),
        })

    else:

        TimeMarchingSetup.update(dict(
            timestep           = Numerics['TimeStep'],
            itime              = Numerics['TimeAtInitialState'],
            restoreach_cons    = 1e-2,
        ))

        if Numerics['TimeMarchingOrder'] == 1:

            TimeMarchingSetup['time_algo'] = 'unsteady'

        else:

            TimeMarchingSetup['time_algo']      = 'gear'
            TimeMarchingSetup['gear_iteration'] = 20

    return TimeMarchingSetup

def get_miscellaneous_setup(workflow):
    MiscellaneousSetup = dict(
        multigrid        = 'none',
        misc_source_term = 'inactive',
        muratiomax = 1.0e20,
    )

    # TODO Check implementation for BodyForceModeling and Chimera
    # if hasattr(workflow, 'BodyForceModeling'):
    #     MiscellaneousSetup['misc_source_term'] = 'active'
    #     tag_zones_with_sourceterm(workflow.tree)

    # Chimera parameters
    if workflow.has_overset_component():
        MiscellaneousSetup.update(dict(
            chm_double_wall      = 'active',
            chm_double_wall_tol  = 2000.,
            chm_orphan_treatment = 'neighbourgsmean',
            chm_impl_interp      = 'none',
            chm_interp_depth     = 2
        ))

    return MiscellaneousSetup

def tag_zones_with_sourceterm(t):
    '''
    Add node xdt_nature='sourceterm' that is mandatory to use body force.
    See https://elsa.onera.fr/issues/11496#note-6
    '''
    zones = t.zones()
    if t.get(Name='FlowSolution#DataSourceTerm'):
        zones = [z for z in zones if z.get(Name='FlowSolution#DataSourceTerm')]

    for zone in zones:
        solverParam = zone.get(Name='.Solver#Param', Depth=1)
        if not solverParam:
            solverParam = cgns.Node(Parent=zone, Name='.Solver#Param', Type='UserDefinedData_t')
        cgns.Node(Parent=solverParam, Name='xdt_nature', Value='sourceterm', Type='DataArray')

def get_turbulence_cutoff_setup(Turbulence):
    # Definition of cut-off values for turbulence 
    turbValues = list(Turbulence['Conservatives'].values())
    if len(turbValues) == 7:  # RSM
        TurbulenceCutOffSetup = dict(
            t_cutvar1 = Turbulence['TurbulenceCutOffRatio'] * turbValues[0],
            t_cutvar2 = Turbulence['TurbulenceCutOffRatio'] * turbValues[3],
            t_cutvar3 = Turbulence['TurbulenceCutOffRatio'] * turbValues[5],
            t_cutvar4 = Turbulence['TurbulenceCutOffRatio'] * turbValues[6],
        )

    elif len(turbValues) > 4: # unsupported 
        raise MolaException('Unsupported number of turbulent fields')
    
    else:
        TurbulenceCutOffSetup = dict()
        for i, value in enumerate(turbValues):
            TurbulenceCutOffSetup[f't_cutvar{i+1}'] = Turbulence['TurbulenceCutOffRatio'] * value

    return TurbulenceCutOffSetup
