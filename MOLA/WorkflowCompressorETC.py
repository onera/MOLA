'''
MOLA - WorkflowCompressorETC.py

Collection of functions that should be imported inside WorkflowCompressor.py

It has been detached from WorkflowCompressor.py to group all the functions with
a dependency to the ETC module.

File history:
28/10/2021 - T. Bontemps - Creation
'''

import numpy as np

import Converter.PyTree    as C
import Converter.Internal  as I

from . import InternalShortcuts as J

# IMPORT etc module
from etc.globborder.globborder_dict import globborder_dict
import etc.transform.__future__  as trf

@J.mute_stdout
def setBC_stage_mxpl(t, left, right, method='globborder_dict'):

    if method == 'globborder_dict':
        t = trf.defineBCStageFromBC(t, left)
        t = trf.defineBCStageFromBC(t, right)
        t, stage = trf.newStageMxPlFromFamily(t, left, right)

    elif method == 'poswin':
        t = trf.defineBCStageFromBC(t, left)
        t = trf.defineBCStageFromBC(t, right)

        gbdu = computeGlobborderPoswin(t, left)
        # print("newStageMxPlFromFamily(up): gbdu = {}".format(gbdu))
        ups = []
        for bc in C.getFamilyBCs(t, left):
          bcpath = I.getPath(t, bc)
          bcu = trf.BCStageMxPlUp(t, bc)
          globborder = bcu.glob_border(left, opposite=right)
          globborder.i_poswin   = gbdu[bcpath]['i_poswin']
          globborder.j_poswin   = gbdu[bcpath]['j_poswin']
          globborder.glob_dir_i = gbdu[bcpath]['glob_dir_i']
          globborder.glob_dir_j = gbdu[bcpath]['glob_dir_j']
          ups.append(bcu)

        # Downstream BCs declaration
        gbdd = computeGlobborderPoswin(t, right)
        # print("newStageMxPlFromFamily(down): gbdd = {}".format(gbdd))
        downs = []
        for bc in C.getFamilyBCs(t, right):
          bcpath = I.getPath(t, bc)
          bcd = trf.BCStageMxPlDown(t, bc)
          globborder = bcd.glob_border(right, opposite=left)
          globborder.i_poswin   = gbdd[bcpath]['i_poswin']
          globborder.j_poswin   = gbdd[bcpath]['j_poswin']
          globborder.glob_dir_i = gbdd[bcpath]['glob_dir_i']
          globborder.glob_dir_j = gbdd[bcpath]['glob_dir_j']
          downs.append(bcd)

        # StageMxpl declaration
        stage = trf.BCStageMxPl(t, up=ups, down=downs)
    else:
        raise Exception

    stage.jtype = 'nomatch_rad_line'
    stage.create()

    setRotorStatorFamilyBC(t, left, right)

@J.mute_stdout
def setBC_stage_mxpl_hyb(t, left, right, nbband=100, c=None):

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageMxPlHybFromFamily(t, left, right)
    stage.jtype = 'nomatch_rad_line'
    stage.hray_tolerance = 1e-16
    for stg in stage.down:
        filename = "state_radius_{}_{}.plt".format(right, nbband)
        radius = stg.repartition(mxpl_dirtype='axial', filename=filename, fileformat="bin_tp")
        radius.compute(t, nbband=nbband, c=c)
        radius.write()
    for stg in stage.up:
        filename = "state_radius_{}_{}.plt".format(left, nbband)
        radius = stg.repartition(mxpl_dirtype='axial', filename=filename, fileformat="bin_tp")
        radius.compute(t, nbband=nbband, c=c)
        radius.write()
    stage.create()

    setRotorStatorFamilyBC(t, left, right)

@J.mute_stdout
def setBC_stage_red(t, left, right, stage_ref_time):

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageRedFromFamily(t, left, right, stage_ref_time=stage_ref_time)
    stage.create()

    setRotorStatorFamilyBC(t, left, right)

@J.mute_stdout
def setBC_stage_red_hyb(t, left, right, stage_ref_time, nbband=100, c=None):

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageRedFromFamily(t, left, right, stage_ref_time=stage_ref_time)
    stage.hray_tolerance = 1e-16
    for stg in stage.down:
        filename = "state_radius_{}_{}.plt".format(right, nbband)
        radius = stg.repartition(mxpl_dirtype='axial', filename=filename, fileformat="bin_tp")
        radius.compute(t, nbband=nbband, c=c)
        radius.write()
    for stg in stage.up:
        filename = "state_radius_{}_{}.plt".format(left, nbband)
        radius = stg.repartition(mxpl_dirtype='axial', filename=filename, fileformat="bin_tp")
        radius.compute(t, nbband=nbband, c=c)
        radius.write()
    stage.create()

    for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
        I._rmNodesByType(gc, 'FamilyBC_t')

@J.mute_stdout
def setBC_outradeq(t, FamilyName, valve_type=0, valve_ref_pres=None,
    valve_ref_mflow=None, valve_relax=0.1, ReferenceValues=None,
    TurboConfiguration=None, method='globborder_dict'):

    if valve_ref_pres is None:
        try:
            valve_ref_pres = ReferenceValues['Pressure']
        except:
            MSG = 'valve_ref_pres or ReferenceValues must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)
    if valve_type != 0 and valve_ref_mflow is None:
        try:
            bc = C.getFamilyBCs(t, FamilyName)[0]
            zone = I.getParentFromType(t, bc, 'Zone_t')
            row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
            rowParams = TurboConfiguration['Rows'][row]
            fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
            valve_ref_mflow = ReferenceValues['MassFlow'] / fluxcoeff
        except:
            MSG = 'Either valve_ref_mflow or both ReferenceValues and TurboConfiguration must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, FamilyName):
        I._rmNodesByName(bc, '.Solver#BC')
    # Create Family BC
    family_node = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(family_node, '.Solver#BC')
    I.newFamilyBC(value='BCOutflowSubsonic', parent=family_node)

    # Outflow (globborder+outradeq, valve 4)
    if method == 'globborder_dict':
        gbd = globborder_dict(t, FamilyName, config="axial")
    elif method == 'poswin':
        gbd = computeGlobborderPoswin(t, FamilyName)
    else:
        raise Exception
    for bcn in  C.getFamilyBCs(t, FamilyName):
        bcpath = I.getPath(t, bcn)
        bc = trf.BCOutRadEq(t, bcn)
        bc.indpiv   = 1
        bc.dirorder = -1
        # Valve laws:
        # <bc>.valve_law(valve_type, pref, Qref, valve_relax=relax, valve_file=None, valve_file_freq=1) # v4.2.01 pour valve_file*
        # valvelaws = [(1, 'SlopePsQ'),     # p(it+1) = p(it) + relax*( pref * (Q(it)/Qref) -p(it)) # relax = sans dim. # isoPs/Q
        #              (2, 'QTarget'),      # p(it+1) = p(it) + relax*pref * (Q(it)/Qref-1)         # relax = sans dim. # debit cible
        #              (3, 'QLinear'),      # p(it+1) = pref + relax*(Q(it)-Qref)                  # relax = Pascal    # lin en debit
        #              (4, 'QHyperbolic'),  # p(it+1) = pref + relax*(Q(it)/Qref)**2               # relax = Pascal    # comp. exp.
        #              (5, 'SlopePiQ')]     # p(it+1) = p(it) + relax*( pref * (Q(it)/Qref) -pi(it)) # relax = sans dim. # isoPi/Q
        # for law 5, pref = reference total pressure
        if valve_type == 0:
            bc.prespiv = valve_ref_pres
        else:
            valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget', 3: 'QLinear', 4: 'QHyperbolic'}
            bc.valve_law(valve_law_dict[valve_type], valve_ref_pres, valve_ref_mflow, valve_relax=valve_relax)
        globborder = bc.glob_border(current=FamilyName)
        globborder.i_poswin        = gbd[bcpath]['i_poswin']
        globborder.j_poswin        = gbd[bcpath]['j_poswin']
        globborder.glob_dir_i      = gbd[bcpath]['glob_dir_i']
        globborder.glob_dir_j      = gbd[bcpath]['glob_dir_j']
        globborder.azi_orientation = gbd[bcpath]['azi_orientation']
        globborder.h_orientation   = gbd[bcpath]['h_orientation']
        bc.create()

@J.mute_stdout
def setBC_outradeqhyb(t, FamilyName, valve_type, valve_ref_pres,
    valve_ref_mflow, valve_relax=0.1, nbband=100, c=None):

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, FamilyName):
        I._rmNodesByName(bc, '.Solver#BC')
    # Create Family BC
    family_node = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(family_node, '.Solver#BC')
    I.newFamilyBC(value='BCOutflowSubsonic', parent=family_node)

    bc = trf.BCOutRadEqHyb(t, I.getNodeFromNameAndType(t, FamilyName, 'Family_t'))
    bc.glob_border()
    bc.indpiv   = 1
    valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget', 3: 'QLinear', 4: 'QHyperbolic'}
    bc.valve_law(valve_law_dict[valve_type], valve_ref_pres, valve_ref_mflow, valve_relax=valve_relax)
    bc.dirorder = -1
    radius_filename = "state_radius_{}_{}.plt".format(FamilyName, nbband)
    radius = bc.repartition(filename=radius_filename, fileformat="bin_tp")
    radius.compute(t, nbband=nbband, c=c)
    radius.write()
    bc.create()

def setRotorStatorFamilyBC(t, left, right):
    for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
        I._rmNodesByType(gc, 'FamilyBC_t')

    leftFamily = I.getNodeFromNameAndType(t, left, 'Family_t')
    rightFamily = I.getNodeFromNameAndType(t, right, 'Family_t')
    I.newFamilyBC(value='BCOutflow', parent=leftFamily)
    I.newFamilyBC(value='BCInflow', parent=rightFamily)

def computeGlobborderPoswin(tree, win):
    from turbo.poswin import computePosWin
    gbd = computePosWin(tree, win)
    for path, obj in gbd.items():
        gbd.pop(path)
        bc = I.getNodeFromPath(tree, path)
        gdi, gdj = getGlobDir(tree, bc)
        gbd['CGNSTree/'+path] = dict(glob_dir_i=gdi, glob_dir_j=gdj,
                                    i_poswin=obj.i, j_poswin=obj.j,
                                    azi_orientation=gdi, h_orientation=gdj)
    return gbd

def getGlobDir(tree, bc):
    # Remember: glob_dir_i is the opposite of theta, which is positive when it goes from Y to Z
    # Remember: glob_dir_j is as the radius, which is positive when it goes from hub to shroud

    # Check if the BC is in i, j or k constant: need pointrage of BC
    ptRi = I.getValue(I.getNodeFromName(bc, 'PointRange'))[0]
    ptRj = I.getValue(I.getNodeFromName(bc, 'PointRange'))[1]
    ptRk = I.getValue(I.getNodeFromName(bc, 'PointRange'))[2]
    x, y, z = J.getxyz(I.getParentFromType(tree, bc, 'Zone_t'))
    y0 = y[0, 0, 0]
    z0 = z[0, 0, 0]

    if ptRi[0] == ptRi[1]:
        dir1 = 2  # j
        dir2 = 3  # k
        y1 = y[0,-1, 0]
        z1 = z[0,-1, 0]
        y2 = y[0, 0,-1]
        z2 = y[0, 0,-1]

    elif ptRj[0] == ptRj[1]:
        dir1 = 1  # i
        dir2 = 3  # k
        y1 = y[-1, 0, 0]
        z1 = z[-1, 0, 0]
        y2 = y[ 0, 0,-1]
        z2 = y[ 0, 0,-1]

    elif ptRk[0] == ptRk[1]:
        dir1 = 1  # i
        dir2 = 2  # j
        y1 = y[-1, 0, 0]
        z1 = z[-1, 0, 0]
        y2 = y[ 0,-1, 0]
        z2 = y[ 0,-1, 0]

    rad0 = np.sqrt(y0**2+z0**2)
    rad1 = np.sqrt(y1**2+z1**2)
    rad2 = np.sqrt(y2**2+z2**2)
    tet0 = np.arctan2(z0,y0)
    tet1 = np.arctan2(z1,y1)
    tet2 = np.arctan2(z2,y2)

    ang1 = np.arctan2(rad1-rad0, rad1*tet1-rad0*tet0)
    ang2 = np.arctan2(rad2-rad0, rad2*tet2-rad0*tet0)

    if abs(np.sin(ang2)) < abs(np.sin(ang1)):
        # dir2 is more vertical than dir1
        # => globDirJ = +/- dir2
        if np.cos(ang1) > 0:
            # dir1 points towards theta>0
            globDirI = -dir1
        else:
            # dir1 points towards thetaw0
            globDirI = dir1
        if np.sin(ang2) > 0:
            # dir2 points towards r>0
            globDirJ = dir2
        else:
            # dir2 points towards r<0
            globDirJ = -dir2
    else:
        # dir1 is more vertical than dir2
        # => globDirJ = +/- dir1
        if np.cos(ang2) > 0:
            # dir2 points towards theta>0
            globDirI = -dir2
        else:
            # dir2 points towards thetaw0
            globDirI = dir2
        if np.sin(ang1) > 0:
            # dir1 points towards r>0
            globDirJ = dir1
        else:
            # dir1 points towards r<0
            globDirJ = -dir1

    print('  * glob_dir_i = %s\n  * glob_dir_j = %s'%(globDirI, globDirJ))
    assert globDirI != globDirJ
    return globDirI, globDirJ
