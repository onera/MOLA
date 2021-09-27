#===============================================================================
# elsA logfile ('elsA-Python' script), an item of the elsA user interface
# automagically written Python logfile ('elsA-Python' script) file for 'compute' script
#  host:spiro-n045-clu (proc:x86_64 os:Linux version:3.10.0-514.26.2.el7.x86_64) on 20210924.103637
#  cwd   = /stck/tbontemp/Public/rotor37/SIMU
# command-line arguments were :
#  
#
# for 'compute' script
#   PYTHONPATH=/stck/elsa/Tools/spiro/pygridgen_pyproj/lib/python2.7/site-packages:/stck/elsa/Public/v5.0.02/Dist/bin/spiro_mpi/lib/python2.7/site-packages:/stck/elsa/Public/v5.0.02/Dist/bin/spiro_mpi:/stck/elsa/Public/v5.0.02/Dist/lib/py/elsA/Compat:/stck/elsa/Public/v5.0.02/Dist/lib/py/Tools:/stck/elsa/Public/v5.0.02/Dist/lib/py:/stck/tbontemp/softs:/stck/tbontemp/softs/elsA:/home/lbernard/.local/lib/python2.7/site-packages/:/stck/elsa/Tools/spiro/pygridgen_pyproj/lib/python2.7/site-packages:/stck/tbontemp/softs:/stck/tbontemp/softs/elsA::/home/tbontemp/softs/MOLA/Dev:/home/tbontemp/softs/MOLA/Dev
# this file is user-editable but overwritten by elsA run
# --- ADAPT as needed, but then please REMOVE above header ---
#===============================================================================

from elsA_user import *

cfd = cfdpb(name='cfd')
Mod = model(name='Mod')
Num = numerics(name='Num')
cfd.set('config', '3d')
cfd.set('extract_filtering', 'inactive')
Mod.set('suth_tref', 288.15)
Mod.set('delta_compute', 'first_order_bl')
Mod.set('suth_muref', 1.78938e-05)
Mod.set('linearratiolim', 1.0e-03)
Mod.set('pressratiolim', 1.0e-03)
Mod.set('prandtl', 0.72)
Mod.set('walldistcompute', 'mininterf_ortho')
Mod.set('fluid', 'pg')
Mod.set('vortratiolim', 1.0e-03)
Mod.set('k_prod_compute', 'from_sij')
Mod.set('prandtltb', 0.9)
Mod.set('turbmod', 'smith')
Mod.set('suth_const', 110.4)
Mod.set('visclaw', 'sutherland')
Mod.set('phymod', 'nstur')
Mod.set('cv', 717.6325)
Mod.set('gamma', 1.4)
Mod.set('shearratiolim', 0.02)
Num.set('viscous_fluxes', '5p_cor')
Num.set('limiter', 'valbada')
Num.set('ode', 'backwardeuler')
Num.set('t_cutvar1', 43.354293)
Num.set('misc_source_term', 'inactive')
Num.set('t_cutvar2', 5.0040178e-07)
Num.set('t_harten', 0.01)
Num.set('freqcompres', 1)
Num.set('global_timestep', 'inactive')
Num.set('multigrid', 'none')
Num.set('niter', 1000000)
Num.set('psiroe', 0.01)
Num.set('time_algo', 'steady')
Num.set('ssorcycle', 4)
Num.set('timestep_div', 'divided')
Num.set('implicit', 'lussorsca')
Num.set('muratiomax', 1.0e+20)
Num.set('residual_type', 'explimpl')
Num.set('cfl_fct', 'f_cfl')
Num.set('flux', 'roe')
Num.set('inititer', 1)
Num.set('harten_type', 2)
f_cfl = function('linear',name='f_cfl')
f_cfl.set('vali', 1.0)
f_cfl.set('valf', 3.0)
f_cfl.set('iteri', 1)
f_cfl.set('iterf', 1000)
# elsA interface : Message : starting check for <function instance 'f_cfl'>
Num.attach('cfl', function=f_cfl)
