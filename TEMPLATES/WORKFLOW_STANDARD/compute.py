import Converter.PyTree as C
import Converter.Internal as I
import Converter.Mpi as Cmpi
import Post.PyTree as P
import elsA_user
import elsAxdt
elsAxdt.trace(4)

Cfdpb = elsA_user.cfdpb(name='cfd')
Cfdpb.set('config', '3d')

Mod   = elsA_user.model(name='Mod')
Mod.set('cv', 717.6325000000002)
Mod.set('delta_compute', 'first_order_bl')
Mod.set('fluid', 'pg')
Mod.set('gamma', 1.4)
Mod.set('k_prod_compute', 'from_sij')
Mod.set('k_prod_limiter', 20.0)
Mod.set('kok_diff_cor', 'wilcox2006')
Mod.set('linearratiolim', 0.001)
Mod.set('omega_prolong', 'linear_extrap')
Mod.set('phymod', 'nstur')
Mod.set('prandtl', 0.72)
Mod.set('prandtltb', 0.9)
Mod.set('pressratiolim', 0.001)
Mod.set('shearratiolim', 0.02)
Mod.set('sst_cor', 'active')
Mod.set('sst_version', 'standard')
Mod.set('suth_const', 110.4)
Mod.set('suth_muref', 1.78938e-05)
Mod.set('suth_tref', 288.15)
Mod.set('turbmod', 'komega_kok')
Mod.set('visclaw', 'sutherland')
Mod.set('vortratiolim', 0.001)
Mod.set('walldistcompute', 'mininterf_ortho')
Mod.set('zhenglim', 'inactive')

Num   = elsA_user.numerics(name='Num')
Num.set('artviscosity', 'dismrt')
Num.set('av_mrt', 0.3)
Num.set('avcoef_k2', 0.5)
Num.set('avcoef_k4', 0.016)
Num.set('avcoef_sigma', 1.0)
Num.set('cfl_fct', 'f_cfl')
Num.set('chm_conn_fprefix', 'OVERSET/overset')
Num.set('chm_conn_io', 'read')
Num.set('chm_double_wall', 'active')
Num.set('chm_double_wall_tol', 2000.0)
Num.set('chm_impl_interp', 'none')
Num.set('chm_interp_depth', 2)
Num.set('chm_interpcoef_frozen', 'active')
Num.set('chm_orphan_treatment', 'neighbourgsmean')
Num.set('chm_ovlp_minimize', 'inactive')
Num.set('chm_ovlp_thickness', 2)
Num.set('chm_preproc_method', 'mask_based')
Num.set('cutoff_dens', 0.005)
Num.set('cutoff_eint', 0.005)
Num.set('cutoff_pres', 0.005)
Num.set('filter', 'incr_new+prolong')
Num.set('flux', 'jameson')
Num.set('freqcompres', 1)
Num.set('global_timestep', 'inactive')
Num.set('harten_type', 2)
Num.set('implicit', 'lussorsca')
Num.set('inititer', 1)
Num.set('misc_source_term', 'inactive')
Num.set('multigrid', 'none')
Num.set('muratiomax', 1e+20)
Num.set('niter', 10)
Num.set('ode', 'backwardeuler')
Num.set('residual_type', 'explicit_novolum')
Num.set('ssorcycle', 4)
Num.set('t_cutvar1', 0.00028874033106374115)
Num.set('t_cutvar2', 192.06896539126376)
Num.set('t_harten', 0.01)
Num.set('time_algo', 'steady')
Num.set('timestep_div', 'divided')
Num.set('viscous_fluxes', '5p_cor')
f_cfl=elsA_user.function('linear',name='f_cfl')
f_cfl.set('iterf', 1000)
f_cfl.set('iteri', 1)
f_cfl.set('valf', 10.0)
f_cfl.set('vali', 1.0)
Num.attach('cfl', function=f_cfl)


e=elsAxdt.XdtCGNS('main.cgns')
e.action=elsAxdt.COMPUTE
e.mode=elsAxdt.READ_ALL
# e.mode |= elsAxdt.CGNS_CHIMERACOEFF
e.compute()
e.save('fields.cgns')

t = C.convertFile2PyTree('fields.cgns')
I._renameNode(t, 'cellnf', 'cellN')
I._renameNode(t, 'FlowSolution#EndOfRun', I.__FlowSolutionCenters__)
FScoords = I.getNodeFromName(t, 'FlowSolution#EndOfRun#Coords')
if FScoords:
    I._renameNode(t,'FlowSolution#EndOfRun#Coords','GridCoordinates')
    for GridCoordsNode in I.getNodesFromName3(t, 'GridCoordinates'):
        GridLocationNode = I.getNodeFromType1(GridCoordsNode, 'GridLocation_t')
        if I.getValue(GridLocationNode) != 'Vertex':
            zone = I.getParentOfNode(t, GridCoordsNode)
            ERRMSG = ('Extracted coordinates of zone '
                      '%s must be located in Vertex')%I.getName(zone)
            raise ValueError(ERRMSG)
        I.rmNode(t, GridLocationNode)
        I.setType(GridCoordsNode, 'GridCoordinates_t')


C.convertPyTree2File(t,'fields.cgns')


slice = P.isoSurfMC(t,'CoordinateY',0.4)
C.convertPyTree2File(slice,'obtainedOverset.cgns')
