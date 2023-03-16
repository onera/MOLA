#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

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

    walldistcompute = 'mininterf_ortho' if workflow.tree.isStructured() else 'mininterf'

    workflow.SolverParameters['model'] = dict(
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

        walldistcompute = walldistcompute
    )

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
        k_prod_limiter = 20.,
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
    
    # Transition Settings
    if workflow.Turbulence['TransitionMode']:

        TransitionModeSetup = dict()
    
    elif workflow.Turbulence['TransitionMode'] == 'NonLocalCriteria-LSTT':

        if 'LM2009' in workflow.Turbulence['Model']:
            raise AttributeError(J.FAIL+"Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model"+J.ENDC)
        
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
            raise AttributeError(J.FAIL+"Modeling incoherency! cannot make Non-local transition criteria with Menter-Langtry turbulence model"+J.ENDC)
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
            raise AttributeError(J.FAIL+"Modeling incoherency! cannot make imposed transition with Menter-Langtry turbulence model"+J.ENDC)
        TransitionModeSetup = dict(
            intermittency       = 'full',
            interm_thick_coef   = 1.2,
            intermittency_form  = 'LSTT19',
        )
    
    if workflow.Turbulence['TransitionMode'] and workflow.Turbulence['Model'] in K_OMEGA_MODELS:  
        TransitionModeSetup['prod_omega_red'] = 'active'

    workflow.SolverParameters['model'].update(TurbulenceSetup[workflow.Turbulence['Model']])
    workflow.SolverParameters['model'].update(TransitionModeSetup)


def set_numerics(workflow):
    #TODO
    workflow.SolverParameters['numerics'] = dict(
    )

