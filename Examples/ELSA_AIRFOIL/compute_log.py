#===============================================================================
# elsA logfile ('elsA-Python' script), an item of the elsA user interface
# automagically written Python logfile ('elsA-Python' script) file for 'compute' script
#  host:spiro-n062-clu (proc:x86_64 os:Linux version:3.10.0-514.26.2.el7.x86_64) on 20210819.113347
#  cwd   = /stck/lbernard/MOLA/Dev/Examples/ELSA_AIRFOIL
# command-line arguments were :
#  
#
# for 'compute' script
#   PYTHONPATH=/home/rboisard/bin/local/x86_64z/Puma_r336_spiro/lib/python2.7/site-packages:/stck/elsa/Tools/spiro/pygridgen_pyproj/lib/python2.7/site-packages:/stck/elsa/Public/v5.0.02/Dist/bin/spiro_mpi/lib/python2.7/site-packages:/stck/elsa/Public/v5.0.02/Dist/bin/spiro_mpi:/stck/elsa/Public/v5.0.02/Dist/lib/py/elsA/Compat:/stck/elsa/Public/v5.0.02/Dist/lib/py/Tools:/stck/elsa/Public/v5.0.02/Dist/lib/py:/home/rboisard/bin/local/x86_64z/Puma_r336_spiro/lib/python2.7/site-packages:/home/lbernard/.local/lib/python2.7/site-packages/:/stck/elsa/Tools/spiro/pygridgen_pyproj/lib/python2.7/site-packages::/home/lbernard/MOLA/Dev:/home/lbernard/MOLA/Dev
# this file is user-editable but overwritten by elsA run
# --- ADAPT as needed, but then please REMOVE above header ---
#===============================================================================

from elsA_user import *

cfd = cfdpb(name='cfd')
Mod = model(name='Mod')
Num = numerics(name='Num')
cfd.set('config', '2d')
cfd.set('extract_filtering', 'inactive')
Mod.set('walldistcompute', 'mininterf_ortho')
Mod.set('suth_muref', 1.78938e-05)
Mod.set('prandtl', 0.72)
Mod.set('k_prod_limiter', 20.0)
Mod.set('prandtltb', 0.9)
Mod.set('sst_version', 'wilcox2006')
Mod.set('sst_cor', 'active')
Mod.set('suth_tref', 288.15)
Mod.set('linearratiolim', 1.0e-03)
Mod.set('pressratiolim', 1.0e-03)
Mod.set('cv', 717.6325)
Mod.set('kok_diff_cor', 'wilcox2006')
Mod.set('delta_compute', 'first_order_bl')
Mod.set('omega_prolong', 'infinit_extrap')
Mod.set('fluid', 'pg')
Mod.set('vortratiolim', 1.0e-03)
Mod.set('suth_const', 110.4)
Mod.set('zhenglim', 'inactive')
Mod.set('phymod', 'nstur')
Mod.set('visclaw', 'sutherland')
Mod.set('turbmod', 'komega_kok')
Mod.set('k_prod_compute', 'from_sij')
Mod.set('gamma', 1.4)
Mod.set('shearratiolim', 0.02)
Num.set('ausm_mref', 0.3)
Num.set('ausmp_diss_cst', 0.04)
Num.set('limiter', 'third_order')
Num.set('ode', 'backwardeuler')
Num.set('ausm_wiggle', 'inactive')
Num.set('t_cutvar1', 0.0027386666)
Num.set('misc_source_term', 'inactive')
Num.set('t_cutvar2', 268.12249)
Num.set('t_harten', 0.01)
Num.set('freqcompres', 1)
Num.set('global_timestep', 'inactive')
Num.set('multigrid', 'none')
Num.set('niter', 50000)
Num.set('ausm_tref', 288.15)
Num.set('time_algo', 'steady')
Num.set('ssorcycle', 4)
Num.set('timestep_div', 'divided')
Num.set('ausm_pref', 1.44903e+04)
Num.set('implicit', 'lussorsca')
Num.set('muratiomax', 1.0e+20)
Num.set('residual_type', 'explimpl')
Num.set('ausmp_press_vel_cst', 0.04)
Num.set('cfl_fct', 'f_cfl')
Num.set('flux', 'ausmplus_pmiles')
Num.set('inititer', 1)
f_cfl = function('linear',name='f_cfl')
f_cfl.set('vali', 1.0)
f_cfl.set('valf', 10.0)
f_cfl.set('iteri', 1)
f_cfl.set('iterf', 1000)
# elsA interface : Message : starting check for <function instance 'f_cfl'>
Num.attach('cfl', function=f_cfl)
