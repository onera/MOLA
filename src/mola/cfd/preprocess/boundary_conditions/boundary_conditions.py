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
import os
import copy
import numpy as np
from treelab import cgns

from mola.dependency_injector.retriever import load_source
from mola.logging import mola_logger, MolaException, MolaUserError, redirect_streams_to_null
from mola.cfd.postprocess.interpolation import migrateFields

def apply(workflow, selected_boundaries_conditions=None):
    '''
    Set all boundary conditions for **workflow**.
    It transforms the tree attribute of the **workflow**.

    Parameters
    ----------
    workflow : Workflow object

    selected_boundaries_conditions : :py:class:`list` of :py:class:`dict`, optional
        Boudaries to apply. 
        If not given, the attribute `BoundaryConditions` of the **workflow** is used.
        Otherwise, it is possible to give a filtered list.
    '''

    _adapt_bc_to_euler(workflow)

    if selected_boundaries_conditions is None:
        selected_boundaries_conditions = workflow.BoundaryConditions
    # Deep copy to prevent modification on the Workflow attribute BoundaryConditions
    selected_boundaries_conditions = copy.deepcopy(selected_boundaries_conditions)

    # workflow.tree = _add_skeleton_in_tree(workflow.tree)  # NOTE tests for MPI preprocess, not working yet

    for bc in selected_boundaries_conditions:

        _check_family_exists(workflow.tree, bc['Family'])
        
        bc_type = bc.pop('Type')
        if bc_type == 'InterfaceBetweenWorkflows':
            continue

        if bc_type == 'OutflowRadialEquilibrium':
            OutflowRadialEquilibrium_interface(workflow, bc)

        if 'LinkedFamily' in bc:
            mola_logger.info(f'  > {bc_type} between families {bc["Family"]} and {bc["LinkedFamily"]}', rank=0)
        else:
            mola_logger.info(f'  > {bc_type} on family {bc["Family"]}', rank=0)
        
        _call_solver_specific_bc_preparation_function(workflow, bc_type, **bc)

    # _force_unique_names_for_gc(workflow.tree) # NOTE tests for MPI preprocess, not working yet

    add_missing_PointRange_in_BCDataSet(workflow)
    fix_FaceCenter_in_BCDataSet(workflow.tree)

def _adapt_bc_to_euler(workflow):
    if workflow.Turbulence['Model'] == 'Euler':
        for bc in workflow.BoundaryConditions:
            if bc['Type'] in ['WallViscous']:
                mola_logger.user_warning(
                    f"Inconsistency between BC {bc['Family']} of type {bc['Type']} and the Euler model.\n"
                    "-> Type is automatically changed into WallInviscid."
                    )
                bc['Type'] = 'WallInviscid'
            
            elif bc['Type'] == ['Wall']:
                bc['Type'] = 'WallInviscid'


def _call_solver_specific_bc_preparation_function(workflow, bc_type, **kwargs):
    current_path = os.path.dirname(os.path.realpath(__file__))
    solverModule = load_source('solverModule', os.path.join(current_path, f'solver_{workflow.Solver}.py'))

    bc_dispatcher = workflow.get_bc_dispatcher()
    solverSpecificFunctionName = bc_dispatcher.get_name_used_by_solver(bc_type)
    
    try:
        solverSpecificFunction = getattr(solverModule, solverSpecificFunctionName)
    except AttributeError:
        raise MolaException(f'The function {solverSpecificFunctionName} does not exist for the solver {workflow.Solver}.')
    else:
        solverSpecificFunction(workflow, **kwargs)

def _check_family_exists(tree, family_name):
    if not tree.get(Name=family_name, Type='Family', Depth=2):
        raise MolaException(f'Cannot apply a boundary condition on family {family_name}: This family does not exist in the mesh.')

def apply_function_to_BCDataSet(workflow, Family: str, functions_to_apply: dict):
    '''
    Apply a function to all face centers in the BC attached to **Family**

    Parameters
    ----------
    workflow : Workflow object

    Family: str
        Name of the Family attached to the given boundary condition

    function_to_apply: fun
        Function to apply to all face centers of BC. The arguments of the function must be variables names
        present in the tree. 

    Return
    ------
    ???

    Example
    -------
    To define the wall velocity at the hub, a function could be defined: 

    .. code-block::python

        def hub_function(CoordinateX):
            omega = np.zeros(CoordinateX.shape, dtype=float)
            omega[(x1<=CoordinateX) & (CoordinateX<=x2)] = 500.
            return dict(Motion = omega * np.array(RotationAxis))

        apply_function_to_BCDataSet(workflow, 'Hub', hub_function)
    '''
    import Converter.PyTree as C
    import Converter.Internal as I

    bc_dict = dict()

    for base in workflow.tree.bases():
        bc_list = C.extractBCOfName(base, f'FamilySpecified:{Family}')
        bc_list = C.node2Center(bc_list)

        for bc in bc_list:

            VarDictToImpose = dict()
            for variable_name, function_to_apply in functions_to_apply.items():
                # args_names is the tuple of the names of arguments of function_to_apply
                args_names = function_to_apply.__code__.co_varnames[:function_to_apply.__code__.co_argcount]
                kwargs = dict()
                for arg_name in args_names:
                    # nodes = bc.group(Name=arg_name, Type='DataArray')
                    nodes = I.getNodesFromNameAndType(bc, arg_name, 'DataArray_t')
                    if len(nodes) == 0:
                        raise Exception(f'{arg_name} is not found in {bc.name()}')
                    elif len(nodes) == 1:
                        node = nodes[0]
                    else:
                        raise Exception(f'Several nodes with name {arg_name} are found in {bc.name()}')

                    kwargs[arg_name] = I.getValue(node)

                result = function_to_apply(**kwargs)
                VarDictToImpose[variable_name] = np.asfortranarray(result).ravel(order='K')

            # Get BC path in the main tree
            zname, wname = bc[0].split('\\')
            bc_path = f'CGNSTree/{base[0]}/{zname}/ZoneBC/{wname}'

            bc_dict[bc_path] = VarDictToImpose

    return bc_dict      

def get_bc_nodes_from_family(t, Family):
    bcs = []
    all_bcs = t.group(Type='BC')
    for bc in all_bcs:
        if bc.get('FamilyName').value() == Family:
            bcs.append(bc)
    return bcs

def get_fields_from_file(t, FamilyName, filename, var2interp):

    def _get_original_bc_node(t, w):
        zname = w.get(Name='.parentZone').value()
        bcname = w.get(Name='.originalBC').value()            
        znode = t.get(Name=zname, Type='Zone')
        bcnode = znode.get(Name=bcname, Type='BC')
        return bcnode
 
    input_data_from_file = dict()
    donor_tree = cgns.load(filename)

    from mola.cfd.postprocess import extract_bc
    inlet_BC_nodes = extract_bc(t, Family=FamilyName)

    inlet_BC_nodes.findAndRemoveNodes(Type='FlowSolution')
    inlet_BC_nodes.findAndRemoveNodes(Type='*FamilyName')
    donor_tree.findAndRemoveNodes(Type='*FamilyName')

    migrateFields(donor_tree, inlet_BC_nodes)  
    inlet_BC_nodes = cgns.castNode(inlet_BC_nodes)

    for w in inlet_BC_nodes.zones():

        ImposedVariables = dict()
        for var in var2interp:
            # search data in every FlowSolution at CellCenter
            for FS in w.group(Type='FlowSolution'):
                GridLocation_node = FS.get(Name='GridLocation', Depth=1)
                if not GridLocation_node or GridLocation_node.value() != 'CellCenter':
                    continue
                varNode = FS.get(Name=var, Type='DataArray', Depth=1)
                if varNode:
                    ImposedVariables[var] = np.asfortranarray(varNode.value())
                    break
            if not var in ImposedVariables:
                raise TypeError(f'variable {var} not found in {filename}')
            
        bcnode = _get_original_bc_node(t, w)
        input_data_from_file[bcnode.path()] = ImposedVariables
    
    return input_data_from_file

def recompute_turbulence_variables(workflow, **kwargs):

    if 'TurbulenceLevel' in kwargs or 'Viscosity_EddyMolecularRatio' in kwargs:   
        mola_logger.info('  recomputing turbulent variables for this BC...', rank=0)       

        workflow_copy = copy.copy(workflow)
        for name, value in kwargs.items():
            if name in workflow_copy.Fluid:
                workflow_copy.Fluid[name] = value
            elif name in workflow_copy.Flow:
                workflow_copy.Flow[name] = value
            elif name in workflow_copy.Turbulence:
                workflow_copy.Turbulence[name] = value
            elif name in workflow_copy.ApplicationContext:
                workflow_copy.ApplicationContext[name] = value
            else:
                raise MolaException(f'Variable {name} cannot be updated neither in Fluid, Flow, Turbulence or ApplicationContext attributes.')

        FlowGen = workflow.Flow['Generator']
        FlowGen.Turbulence.update(workflow_copy.Turbulence)
        FlowGen.Turbulence.update(workflow_copy.Turbulence)
        FlowGen.Turbulence.update(workflow_copy.Turbulence)
        FlowGen.Turbulence.update(workflow_copy.Turbulence)
        FlowGen.generate()
        Turbulence = FlowGen.Turbulence

        del workflow_copy

    else:
        Turbulence = workflow.Turbulence
    
    return Turbulence

def get_turbulent_primitives(workflow, **kwargs):
    '''
    Get the primitive (without the Density factor) turbulent variables (names and values) 
    to inject in an inflow boundary condition.

    For RSM models, see issue https://elsa.onera.fr/issues/5136 for the naming convention.

    Parameters
    ----------
    workflow, bc

    Returns
    -------
    dict
        Imposed turbulent variables
    '''
    if 'TurbulenceLevel' in kwargs or 'Viscosity_EddyMolecularRatio' in kwargs:   
        recompute_turbulence_variables(workflow, **kwargs)
    else:
        Turbulence = workflow.Turbulence
        
    turbDict = get_turbulent_primitives_from_conservatives(Turbulence, workflow.Flow['Density'], **kwargs)
        
    return turbDict

def get_turbulent_primitives_from_conservatives(Turbulence, Density, **kwargs):
    turbDict = dict()
    for name, value in Turbulence['Conservatives'].items():
        # If the 'conservative' value is given in kwargs
        value = kwargs.get(name, value)

        if name.endswith('Density'):
            name = name.replace('Density', '')
            value /= Density
        elif name == 'ReynoldsStressDissipationScale':
            name = 'TurbulentDissipationRate'
            value /= Density
        elif name.startswith('ReynoldsStress'):
            name = name.replace('ReynoldsStress', 'VelocityCorrelation')
            value /= Density
        turbDict[name] = value

        # If the 'primitive' value is given in kwargs
        turbDict[name] = kwargs.get(name, value)
    return turbDict

def add_missing_PointRange_in_BCDataSet(workflow):
    from maia.io.fix_tree import add_missing_pr_in_bcdataset
    with redirect_streams_to_null(): 
        # no stdout to prevent the "Error" message, because the function is used 
        # here to add PointRange nodes and not to check if they are present
        add_missing_pr_in_bcdataset(workflow.tree)
    workflow.tree = cgns.castNode(workflow.tree)

def fix_FaceCenter_in_BCDataSet(t):
    import maia.pytree as PT

    for zone in PT.get_all_Zone_t(t):
        if PT.get_value(PT.get_node_from_label(zone, 'ZoneType_t')) == 'Structured':
            for node in PT.get_nodes_from_label(zone, 'BCDataSet_t'):
                if PT.Subset.GridLocation(node) == 'FaceCenter':
                    axis = PT.Subset.normal_axis(node)
                    PT.update_child(node, 'GridLocation', value='IJK'[axis] + 'FaceCenter')

def _instantiate_bc_dispatcher(workflow):
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_path, f'boundary_conditions_dispatcher_{workflow.Solver}.py')
    solverModule = load_source('solverModule', file_path)
    dispatcher = getattr(solverModule, 
                            f"BoundaryConditionsDispatcher{workflow.Solver.capitalize()}")
    workflow._bc_dispatcher = dispatcher()

    return workflow._bc_dispatcher

def get_fluxcoeff_on_bc(workflow, Family):
    from mpi4py.MPI import COMM_WORLD as comm
    try:
        bc = get_bc_nodes_from_family(workflow.tree, Family)[0]
    except:
        bc = None
    nodes_on_all_ranks = comm.allgather(bc)
    bc = [node for node in nodes_on_all_ranks if node is not None][0]
    if bc is None:
        raise MolaException(f'Cannot find a BC associated to Family {Family}')
    
    zone = bc.getParent(Type='Zone_t')
    row = zone.get(Type='FamilyName', Depth=1).value()

    try:
        rowParams = workflow.ApplicationContext['Rows'][row]
    except:
        raise MolaException('Worklow must have an attribute ApplicationContext with a dict named "Rows" inside.')
    fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
            
    return fluxcoeff

def OutflowRadialEquilibrium_interface(workflow, bcparams):
    
    # Check minimal information is given
    possible_arguments = ['PressureAtHub', 'PressureAtShroud', 'PressureAtSpecifiedHeight', 'MassFlow', 'ValveLaw']
    if sum(1 for arg in possible_arguments if arg in bcparams) != 1:
        raise MolaUserError((
            'For BC of Type "OutflowRadialEquilibrium", exactly one of the following '
            f'arguments must be provided: {possible_arguments}'
        ))

    # Check that both PressureAtSpecifiedHeight and Height are specified together
    if 'PressureAtSpecifiedHeight' in bcparams and not 'Height' in bcparams:
        raise MolaUserError((
            'For BC of Type "OutflowRadialEquilibrium", if "PressureAtSpecifiedHeight", '
            'then "Height" must also be specified.'
        ))
    
    if 'ValveLaw' in bcparams:
        possible_valve_types = ['Linear', 'Quadratic']
        if not isinstance(bcparams['ValveLaw'], dict):
            raise MolaUserError('For BC of Type "OutflowRadialEquilibrium", parameter ValveLaw must be a dict')
        if not 'Type' in bcparams['ValveLaw'] or bcparams['ValveLaw']['Type'] not in possible_valve_types:
            raise MolaUserError(f'For BC of Type "OutflowRadialEquilibrium", ValveLaw["Type"] must be defined out of {possible_valve_types}')
    
        if bcparams['ValveLaw']['Type'] == 'Linear':
            bcparams['ValveLaw'].setdefault('RelaxationCoefficient', 0.1)
            bcparams['ValveLaw'].setdefault('PressureRef', workflow.Flow['Pressure'])
            bcparams['ValveLaw'].setdefault('MassFlowRef', workflow.Flow['MassFlow'])

        elif bcparams['ValveLaw']['Type'] == 'Quadratic':
            bcparams['ValveLaw'].setdefault('PressureRef', 0.75 * workflow.Flow['PressureStagnation'])
            bcparams['ValveLaw'].setdefault('MassFlowRef', workflow.Flow['MassFlow'])

            if not 'ValveCoefficient' in bcparams['ValveLaw'] \
                or not isinstance(bcparams['ValveLaw']['ValveCoefficient'], float):
                raise MolaUserError((
                    'For BC of Type "OutflowRadialEquilibrium" with ValveLaw of '
                    f'Type={bcparams["ValveLaw"]["Type"]}, parameter ValveCoefficient must be defined and must be a float.'
                ))

def _add_skeleton_in_tree(tree):
    from mpi4py import MPI
    import Converter.Mpi as Cmpi
    MPI.COMM_WORLD
    
    skeleton = Cmpi.convert2SkeletonTree(tree)
    skeleton = Cmpi.allgatherTree(skeleton)
    skeleton = cgns.castNode(skeleton)

    tree = cgns.merge([tree, skeleton])
    return tree

def _force_unique_names_for_gc(tree):

    def _get_new_name(name, all_gc_names):
        for i in range(100):
            new_name = f'{name}.{i}'
            if new_name not in all_gc_names:
                return new_name
        raise MolaException(f'Could not find a new name for {gc.path()}')

    def _set_new_gc_name(gc, new_name):
        gc.setName(new_name)

        # Change also donor gc name
        try:
            donor_gc_name = gc.get(Name='GridConnectivityDonorName').value()
        except: 
            # Check that this gc is a rotor/stator interface
            assert gc.get(Type='GridConnectivityType').value() == 'Abutting', gc.path()
            return
        
        donor_zone_name = gc.value()
        donor_zone = tree.get(Type='Zone', Depth=2, Name=donor_zone_name)
        zgc = donor_zone.get(Type='ZoneGridConnectivity', Depth=1)
        donor_gc = zgc.get(Name=donor_gc_name, Depth=1)
        donor_gc.get(Name='GridConnectivityDonorName').setValue(gc.name())

    all_gc = tree.group(Type='GridConnectivity') + tree.group(Type='GridConnectivity1to1')
    all_gc_names = set([gc.name() for gc in all_gc])
    for gc in all_gc:
        name = gc.name()
        if '.' not in name or name in all_gc_names:
            # Pick a new name based on the current one, and that is not already in all_gc_names
            new_name = _get_new_name(name, all_gc_names)
            # update the set of all gc names
            all_gc_names.discard(name)
            all_gc_names.add(new_name)
            # 
            _set_new_gc_name(gc, new_name)
