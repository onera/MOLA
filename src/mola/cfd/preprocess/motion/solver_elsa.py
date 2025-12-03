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

import numpy as np

from mola.cfd.preprocess.motion import motion
from mola.logging import mola_logger, MolaException
import mola.pytree.InternalShortcuts as J
from mola.cfd.preprocess.mesh.families import getFamilyBCTypeFromFamilyBCName
from mola.cfd.preprocess.mesh.overset import hasAnyOversetMotion

from treelab import cgns
import Converter.Internal as I

def apply_to_solver(workflow):
    '''
    Set Motion for each families for the solver elsA.

    The **workflow** must have a **Motion** attribute like this:

    .. code-block:: python
        Motion = dict(
            Rotor = dict(
                RotationSpeed = [1000., 0., 0.],
                RotationAxisOrigin = [0., 0., 0.],
                TranslationSpeed = [0., 0., 0.]
            )
        )

    Parameters
    ----------

        workflow : Workflow object

    '''

    if hasAnyOversetMotion(workflow.RawMeshComponents):
        workflow.tree = addOversetMotion(workflow.tree, workflow.Motion)

    else:

        if motion.all_families_are_fixed(workflow):
            return
        
        for family, MotionOnFamily in workflow.Motion.items():
            # NOTE The node .Solver#Motion must be defined even for fixed zones, 
            # if at least one zone is moving. Otherwise, elsA rises an error like in 
            # the issue https://elsa-e.onera.fr/issues/11050 :
            #   User Error : Block motion parameter must be defined consistently over all the blocks

            famNode = workflow.tree.get(Name=family, Type='Family', Depth=2)
            if famNode is None:
                available_families= [n.name() for n in workflow.tree.group(Type='Family_t',Depth=2)]
                raise MolaException(f'did not find family "{family}", available families: {str(available_families)}')

            assert_rotation_axis_is_correct(MotionOnFamily)

            mola_logger.debug(f'set motion on {family}: {MotionOnFamily}')

            famNode.setParameters('.Solver#Motion',
                                    motion='mobile',
                                    **translate_motion_to_elsa(MotionOnFamily)
                                    )
 
def assert_rotation_axis_is_correct(Motion):
    # For elsA, the rotation must be around one axis only
    onlyOneRotationComponent = \
        (Motion['RotationSpeed'][0] == Motion['RotationSpeed'][1] == 0) \
    or (Motion['RotationSpeed'][0] == Motion['RotationSpeed'][2] == 0) \
    or (Motion['RotationSpeed'][1] == Motion['RotationSpeed'][2] == 0)
    
    assert onlyOneRotationComponent, 'For elsA, the rotation must be around one axis only'    

def translate_motion_to_elsa(Motion, remove_null_motions=True):
    
    
    RotationAxis = np.array(Motion['RotationSpeed'])
    assert RotationAxis[1] == RotationAxis[2] == 0
    RotationSpeed = RotationAxis[0]
    # RotationSpeed = np.sqrt(RotationAxis.dot(RotationAxis)) # not working, the sign is always positive!
    if RotationSpeed != 0:
        RotationAxis = np.absolute(RotationAxis / RotationSpeed)
    else:
        RotationAxis = [1., 0., 0.]


    TranslationVector = np.array(Motion.get('TranslationSpeed',[0.0,0.0,0.0]))
    TranslationSpeed = np.sqrt(TranslationVector.dot(TranslationVector))
    if TranslationSpeed != 0:
        TranslationVector /= TranslationSpeed
    else:
        TranslationVector = [1., 0., 0.]

    RotationAxisOrigin = Motion.get('RotationAxisOrigin',[0,0,0])

    motion_elsa = dict()
    # if not remove_null_motions or RotationSpeed != 0.:
    motion_elsa.update(dict(
        omega        = RotationSpeed,
        axis_pnt_x   = RotationAxisOrigin[0], 
        axis_pnt_y   = RotationAxisOrigin[1], 
        axis_pnt_z   = RotationAxisOrigin[2],
        axis_vct_x   = RotationAxis[0], 
        axis_vct_y   = RotationAxis[1], 
        axis_vct_z   = RotationAxis[2], 
    ))
    if not remove_null_motions or TranslationSpeed != 0.:
        motion_elsa.update(dict(
            transl_vct_x = TranslationVector[0],
            transl_vct_y = TranslationVector[1],
            transl_vct_z = TranslationVector[2],
            transl_speed = TranslationSpeed, 
        ))

    return motion_elsa
        
def addOversetMotion(t, OversetMotion):
    if not OversetMotion: return
    bases = I.getBases(t)
    bases_names = [b[0] for b in bases]
    NewOversetMotion = dict()
    for k in OversetMotion:
        base_found = bool([b for b in bases if b[0]==k])
        if base_found: continue
        base_candidates = [b for b in bases if b[0].startswith(k)]
        never_found = True
        for i, b in enumerate(base_candidates):
            try:
                base_found = [b for b in base_candidates if b[0]==k+'_%d'%(i+1)][0]
                never_found = False
            except IndexError:
                continue
            NewOversetMotion[base_found[0]] = OversetMotion[k]
        if never_found:
            msg=('tried to set motion to component %s or inherited, but never found.'
                 '\nAvailable component names are: %s')%(k,str(bases_names))
            raise ValueError(J.FAIL+msg+J.ENDC)
    OversetMotion.update(NewOversetMotion)

    for base in bases:
        motion_keys = dict( motion=1, omega=0.0, transl_speed=0.0,
                            axis_ang_1=1, axis_ang_2=1 )

        try:             OversetMotionData = OversetMotion[base[0]]
        except KeyError: OversetMotionData = dict(RPM=0.0)


        FamilyMotionName = 'MOTION_'+base[0]
        for z in I.getZones(base):
            I.createUniqueChild(z,'FamilyName','FamilyName_t',
                                    value=FamilyMotionName)
        family = I.createChild(base, FamilyMotionName, 'Family_t')
        I.createChild(family,'FamilyBC','FamilyBC_t',value='UserDefined')

        rc, ra, td = _getMotionDataFromMeshInfo(base)
        motion_keys['function_name']=FamilyMotionName
        motion_keys['omega']=OversetMotionData['RPM']*np.pi/30
        motion_keys['axis_pnt_x']=rc[0]
        motion_keys['axis_pnt_y']=rc[1]
        motion_keys['axis_pnt_z']=rc[2]
        motion_keys['axis_vct_x']=ra[0]
        motion_keys['axis_vct_y']=ra[1]
        motion_keys['axis_vct_z']=ra[2]
        motion_keys['transl_vct_x']=td[0]
        motion_keys['transl_vct_y']=td[1]
        motion_keys['transl_vct_z']=td[2]
        
        _setMobileCoefAtBCsExceptOverlap(base, mobile_coef=-1.0)

        J.set(family,'.Solver#Motion', **motion_keys)

        MeshInfo = J.get(base,'.MOLA#InputMesh') 
        try: is_duplicated = bool(MeshInfo['DuplicatedFrom'] != base[0])
        except KeyError: is_duplicated = False

        
        phase = 0.0

        if is_duplicated:
            blade_id = int(base[0].split('_')[-1])
            blade_nb = MeshInfo['OversetMotion']['NumberOfBlades']
            try:
                RH=MeshInfo['OversetMotion']['RequestedFrame']['RightHandRuleRotation']
                sign = 1 if RH else -1
            except KeyError:
                sign = 1
            psi0_b = (blade_id-1)*sign*(360.0/float(blade_nb)) + phase
        else:
            psi0_b = phase

        try: bd = MeshInfo['OversetMotion']['RequestedFrame']['BladeDirection']
        except KeyError: bd = [1,0,0]
        bd = np.array(bd,dtype=float)

        default_rotor_motion = dict(type='rotor_motion',
            initial_angles=[0.,psi0_b],
            alp0=0.,
            alp_pnt=[0.,0.,0.],
            alp_vct=[0.,1.,0.],
            rot_pnt=[float(rc[0]),float(rc[1]),float(rc[2])],
            rot_vct=[float(ra[0]),float(ra[1]),float(ra[2])],
            rot_omg=motion_keys['omega'],
            span_vct=bd,
            pre_lag_pnt=[0.,0.,0.],
            pre_lag_vct=[0.,0.,1.],
            pre_lag_ang=0.,
            pre_con_pnt=[0.,0.,0.],
            pre_con_vct=[0.,1.,0.],
            pre_con_ang=0.,
            del_pnt=[0.,0.,0.],
            del_vct=[0.,0.,1.],
            del0=0.,
            bet_pnt=[0.,0.,0.],
            bet_vct=[0.,1.,0.],
            bet0=0.,
            tet_pnt=[0.,0.,0.],
            tet_vct=[1.,0.,0.],
            tet0=0.)        
        

        try:
            function_motion_type = OversetMotionData['Function']['type']
        except KeyError:
            function_motion_type = 'rotor_motion'
            try:
                OversetMotionData['Function']['type'] = 'rotor_motion'
            except KeyError:
                OversetMotionData['Function'] = dict(type='rotor_motion')

        if function_motion_type == 'rotor_motion':
            default_rotor_motion.update(OversetMotionData['Function'])
            J.set(family,'.MOLA#Motion',**default_rotor_motion)
        else:
            J.set(family,'.MOLA#Motion',**OversetMotionData['Function'])
        
        MOLA_Motion = I.getNodeFromName1(family,'.MOLA#Motion')
        I.setValue(MOLA_Motion, FamilyMotionName)

    return cgns.castNode(t)

def _getMotionDataFromMeshInfo(base):
    defaultRotationCenter = np.array([0.,0.,0.],dtype=float,order='F')
    defaultRotationAxis = np.array([0.,0.,1.],dtype=float,order='F')
    defaultTranslationDirection = np.array([1.,0.,0.],dtype=float,order='F')
    default = defaultRotationCenter, defaultRotationAxis, defaultTranslationDirection

    MeshInfo = J.get(base,'.MOLA#InputMesh')
    if not MeshInfo:
        raise ValueError(f'base {base[0]} must have .MOLA#InputMesh node')
    
    if not 'OversetMotion' in MeshInfo:
        print(f'base {base[0]} does not have Motion attribute. Assigning default.')
        return default


    if not 'RequestedFrame' in MeshInfo['OversetMotion']:
        print(J.WARN+f'no requested frame in {base[0]}, using InitialFrame data'+J.ENDC)
    MotionData = MeshInfo['OversetMotion']['InitialFrame']
    
    RotationCenter = np.array(MotionData['RotationCenter'],dtype=float)
    RotationAxis = np.array(MotionData['RotationAxis'],dtype=float)

    try: TranslationDirection = MotionData['TranslationDirection']
    except KeyError: TranslationDirection = defaultTranslationDirection

    return RotationCenter, RotationAxis, TranslationDirection

def _setMobileCoefAtBCsExceptOverlap(t, mobile_coef=-1.0):
    for base in I.getBases(t):
        for family in I.getNodesFromType1(base,'Family_t'):
            BCType = getFamilyBCTypeFromFamilyBCName(base, family[0])
            if BCType and BCType != 'BCOverlap':
                SolverBC = I.getNodeFromName1(family,'.Solver#BC')
                if not SolverBC:
                    J.set(family, '.Solver#BC', mobile_coef=mobile_coef)
                else:
                    I.createUniqueChild(SolverBC,'mobile_coef',
                                        'DataArray_t', value=mobile_coef)
