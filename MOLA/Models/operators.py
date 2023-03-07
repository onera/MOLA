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

#!/bin/bash/python
#Encoding:UTF8
"""
Submodule for the automatic computation of models-related variables
"""
#__________Generic modules_____________________________________________#
import sys

#__________ONERA_______________________________________________________#
# Cassiopee
import Converter.PyTree as C
import Converter.Internal as I



debug_operators=False

verbose={0:0}


def set_verbose(level,scale=0):
  """
  Sets the verbosity level. Messages are only printed to stdout if the verbosity
  level is greater than the verbosity need of the printv function.

  Parameters
  ----------

    scale : :py:class:`str` or :py:class:`int`
      category of the messages whose verbose level must be set

    level : int
      verbosity level henceforth
  """
  verbose[scale]=level


def printv(string,v=0,s=0,error=False):
  """
  Prints the character string **string** if the verbosity level allows it,
  or, if **error** is :py:obj:`True`, to standard error.

  Parameters
  ----------

    string : str
      to be printed. Do not forget to end lines with if you do not want your
      stdout to be messy

    v : int
      verbosity level beyond which string must be printed

    s : scale
      category of messages you want printed

    error : bool
      if :py:obj:`True`, print to stderr, whether the verbosity level allows it or not

  """
  if not error:
    if v<=verbose[s]:
      sys.stdout.write(string)
      sys.stdout.flush()
  else:
    sys.stderr.write(string)
    sys.stderr.flush()


def printTreeV(tree,v=0):
  """
  Prints a Python tree if the verbose level allows it.

  Parameters
  ----------

    tree : PyTree

    v : int
      verbosity level beyond which tree must be printed
  """
  if v<=verbose[0]:
    I.printTree(tree)















class operator(object):
  """
  Class of objects that automatically performs operations using given models and a dataset.


  """
  def __init__(self,models): #,adimensionnement=False):
    """
    Constructor of an operator object instance that automatically performs operations using given models and a dataset.

    Parameters
    ----------
      models : list
        :py:class:`list` of model objects that the operator must use to perform
        the computations. Any object in this list must inherit from :py:class:`MOLA.Models.base.model`

    """
    self.models=models
    self.n_models=len(self.models)

    self.errors=''

    self.initializeOperationsTree()
    self.data=dict()
    self.computed_variables=list()

  def append_error(self,error_string):
    self.errors='{0}{1}'.format(self.errors,error_string)


  def compute(self,variable_name,replace=False):
    """
    Computes the value of the variable asked for by name **variable_name** if the
    models available allow it.

    In case of a conflict with an existing variable in the dataset, replaces it
    if **replace** is :py:obj:`True`, and does nothing otherwise.
    The chain of operations to perform is automatically determined using available models.
    In order to compute currently inaccessible variables, more models can be supplied using addModels,
    or more functions to one of the available models, using model.supplyFunctions.

    Parameters
    ----------

      variable_name : str
        name of the variable to be computed (must lead to at least
        one available operation of at least one available model)

      replace : bool
        what to do if the variable already exists in the working dataset

    .. seealso:: get_data
    """

    def _get_node_model(node):
      return node[1][0]

    def _get_node_operation_index(node):
      return node[1][1]

    def _get_node_data(node):
      return node[1][3]

    def _set_all_same_nodes_data(node,data):
      for other_same_node in I.getNodesFromNameAndType(self.operations_tree,node[0],'UserDefinedData_t'):
        other_same_node[1].append(data)

    def _compute(node):
      arguments=dict()
      result=None
      if not node[2]:
        result=self.get_data(node[0])
        _set_all_same_nodes_data(node, result)
      else:
        if not node[0] in self.computed_variables:
          variable_name=node[0]
          i_model=_get_node_model(node)
          i_operation=_get_node_operation_index(node)
          for child_node in node[2]:
            necessary_variable_name=child_node[0]
            _compute(child_node)
            arguments.update({necessary_variable_name:_get_node_data(child_node)})

          result=self.models[i_model].carryOutOperation(variable_name, i_operation, arguments)
          printv('{0} = {1}\n'.format(node[0],result),v=9)
          _set_all_same_nodes_data(node, result)
        else:
          result=_get_node_data(node)

      self.computed_variables.append(node[0])
      # Empty node children : their values can be memory-consuming and are no longer useful
      node[2]=list()


    self.initializeOperationsTree()
    if self.is_available(variable_name):
      if not replace:
        return
      else:
        printv("Les données correspondant à la variable demandée existent déjà. Elles seront effacées puis recalculée si possible.\n",v=0)
        self.eraseData(variable_name)

    if self.is_computable(variable_name):
      printv("Calcul de la variable {0} par l'arbre suivant :\n".format(variable_name),v=2)
      printTreeV(self.operations_tree,v=2)
      base=I.getBases(self.operations_tree)[0]
      variable_node=I.getChildren(base)[0]
      self.initializeComputedVariables()
      _compute(variable_node)
      self.set_data(variable_name,_get_node_data(variable_node))

  def eraseAllData(self):
    """
    Erases all the available data from which the current operator can compute other variables.
    """
    self.data=dict()

  def eraseData(self,variable_name):
    """
    Erases the data given by name <variable_name> from the available data
    from which the current operator can compute other variables.

    Parameters
    ----------
      variable_name : str
        name of the data to erase
    """
    self.data.pop(variable_name)


  def get_data(self,variable_name):
    """
    Returns the value associated to the name **variable_name** in the current dataset.

    Parameters
    ----------

      variable_name : str
        name of the variable to get

    Returns
    -------

      value : :py:class:`float` of :py:class:`np.ndarray`
        value associated to **variable_name**
    """
    if not variable_name in self.data.keys():
      raise Exception("Attention : aucune donnée ne correspond au nom de variable demandé. Fin du processus.")
    return self.data[variable_name]

  def get_dataset(self,variables_names):
    """
    Returns the values associated to one or several names.

    Parameters
    ----------
      variables_names : :py:class:`list` of :py:class:`str`
        names of the variables to get

    Returns
    -------
      values : dict
        values associated to **variables_names**, indexed on **variables_names**
    """
    dataset=dict()
    for variable_name in variables_names:
      dataset.update({variable_name:self.get_data(variable_name)})
    return dataset


  def is_computable(self,variable_name):
    """
    Checks whether a variable asked for by name **variable_name** is computable
    using the operations available to the current operator through its models.

    This method builds the operations tree self.operations_tree (yes, truly), which contains
    the paths through several operations that the compute method should use to obtain the result.
    Note that the operator has no control over the path chosen. In a near future, it could be
    advantageous to compute all possible paths and use the shortest one.
    Furthermore, the same operation can currently be done several times in the same tree if the same
    variable is needed for different computations. In order to improve the execution time and memory
    consumption of this module, it would be interesting to commit the result of a computation
    to memory if it must be reused, instead of recomputing.

    Parameters
    ----------
      variable_name : str
        name of the variable whose computability is checked

    Returns
    -------
      is_computable : bool
        whether an operations tree can lead to the variable asked for or not.
        more importantly, the method builds the internal attribute self.operations_tree (see above).
    """

    def _define_computabilityInTree(variable_node,computability,error_string='', v=0):
      if v<=verbose[0]:
        self.append_error(error_string)
      variable_node[1].append(computability)

    def _get_node_computability(node):
      return node[1][2]

    def _is_computable(variable_node):
      printTreeV(self.operations_tree,v=6)
      path_variable_node=I.getPath(self.operations_tree,variable_node)
      parent_nodes_names=path_variable_node.split('/')
      all_same_nodes=I.getNodesFromNameAndType(self.operations_tree,variable_node[0],'UserDefinedData_t')
      if variable_node[0] in parent_nodes_names[:-1]:
        # Si la variable demandée doit servir à se calculer elle-même, l'opération courante n'est pas réalisable
        _define_computabilityInTree(variable_node, False, "La variable {0} est déjà dans la chaîne de calcul et ne peut pas servir à se calculer elle-même.\n", v=4)
      else:
        if len(all_same_nodes)>1:
          # On a déjà essayé de trouver un chemin de calcul pour cette variable, inutile de recommencer, il suffit de copier le résultat précédent
          variable_node[:]=all_same_nodes[0][:]
          # variable_node[2]=all_same_nodes[0][2]
          return
        else:
          # Sinon, on boucle sur les modèles connus et les opérations associées pour déterminer la suite de l'tree de calcul
          operation_exists=False
          for i_model in range(self.n_models):
            variable_node[1][0]=i_model
            variable_node[1][1]=0
            model=self.models[i_model]
            operations_properties=model.get_operations_properties(variable_node[0])
            if operations_properties:
              operation_exists=True
              while variable_node[1][1]<len(operations_properties):
                local_computability=True
                for subvariable_name in operations_properties[variable_node[1][1]]:
                  subvariable_node=I.newUserDefinedData(name=subvariable_name,parent=variable_node)

                  subvariable_node[1]=[0,0]
                  if not self.is_available(subvariable_name):
                    _is_computable(subvariable_node)
                    local_computability*=subvariable_node[1][2]
                  else:
                    _define_computabilityInTree(subvariable_node, True)

                  if not local_computability:
                    # Mise à jour de l'arbre-fils
                    variable_node[1][1]+=1
                    variable_node[2]=[]
                    break
                if local_computability:
                  _define_computabilityInTree(variable_node, True)
                  return

          if not operation_exists:
            _define_computabilityInTree(variable_node, False, "La variable demandée par la chaîne de calcul {0} ne peut être calculée, parce qu'aucune opération définie ne permet son obtention.\n".format(variable_node[0]),v=1)
            return

        _define_computabilityInTree(variable_node, False, "Aucune chaîne de calcul ne permet de calculer la variable {0}, bien que des opérations pour le faire existent. Les données ou les opérations disponibles sont insuffisantes.\n".format(variable_node[0]))
        return

    # Initialisation de récursivité de _is_computable
    self.errors=''
    base=I.getBases(self.operations_tree)[0]
    variable_node=I.newUserDefinedData(name=variable_name,parent=base)
    variable_node[1]=[0,0]
    _is_computable(variable_node)
    local_is_computable=variable_node[1][2]
    if not local_is_computable:
      printv(self.errors,error=True)
    else:
      self.errors=''
    return local_is_computable

  def is_available(self,variable_name):
    """
    Checks whether any data exists in the current dataset under the name <variable_name>.

    Parameters
    ----------

      variable_name : str
        name of the variable to check

    Returns
    -------

      availability : bool
        whether there exists a valueassociated to the name **variable_name**
    """
    availability=False
    if variable_name in self.data.keys():
      availability=True
    return availability

  def set_data(self,variable_name,value):
    """
    Defines the value associated to the name **variable_name** as being **value**

    Parameters
    ----------

      variable_name : str
        name of the variable to define

      value : :py:class:`float` or :py:class:`np.ndarray`
        value to assign
    """
    self.data.update({variable_name:value})

  def set_dataset(self,dataset):
    """
    Defines the values associated to several variables names.

    Parameters
    ----------

      dataset : dict
        named values to add to the current dataset.

    .. note::
      The dataset passed through this method is supplied to the current dataset, but it does not replace it :
      if a value is newly defined, the new value will thereafter be taken into account,
      but all old valued that are not newly defined will remain.
      Use eraseData or eraseAllData to make sure that values not overridden are not used anymore.
    """
    for variable_name in dataset.keys():
      self.set_data(variable_name, dataset[variable_name])

  def initializeOperationsTree(self):
    """
    Initializes the operations tree to an empty tree.
    """
    self.operations_tree=C.newPyTree(['Base'])

  def initializeComputedVariables(self):
    '''
    as name indicates
    '''
    self.computed_variables=list()




















class PyTree_operator(operator):
  """
  Class of objects that automatically performs operations on a python tree using given models and a dataset.

  See also
  --------
    operator.__init__
  """
  def __init__(self,models):
    """
    Constructor of a PyTree_operator object instance that automatically performs operations on a python tree using given models and a dataset.

    Parameters
    ----------
      models : list(.base.model), :py:class:`list` of model objects that the operator must use to perform the computations.
               any object in this list must inherit from .base.model.

    Returns
    -------
      self : operator object which can compute variables using models
    """
    super(PyTree_operator,self).__init__(models)
    self.data_localization='CellCenter'
    self.current_containers=None

  def compute(self,variable_name,replace=False):
    """
    Computes the value of the variable asked for by name <variable_name> in any zone of the working tree if possible.
    Calls the compute method inherited from the operator class.

    Parameters
    ----------
      variable_name : string, name of the variable to be computed (must lead to at least
                      one available operation of at least one available model).
      replace       : bool, what to do if the variable already exists in the working dataset

    See also
    --------
      get_data
    """
    # FlowSolution_nodes_name=self.get_FlowSolution_nodes_name()
    for zone in I.getNodesFromType(self.tree,'Zone_t'):
      self.set_currentContainers(zone)
      super(PyTree_operator, self).compute(variable_name,replace=replace)

  def eraseAllData(self):
    """
    Erases all the available data from the current FlowSolutionNode.
    """
    self.current_containers[2]=list()

  def eraseData(self,variable_name):
    """
    Erases the data given by name <variable_name> from the current FlowSolution node.

    Parameters
    ----------
      variable_name : string, name of the data to erase
    """
    I._rmNodesByNameAndType(self.current_containers,variable_name,'DataArray_t')

  def get_tree(self):
    """
    Returns the current working tree.

    Returns
    -------
      tree : PyTree, current working tree
    """
    return self.tree

  def get_data(self,variable_name):
    """
    Returns the data registered in the current FlowSolution node under the name <variable_name>.

    Parameter
    ---------

      variable_name : str
        name of the variable node which value is returned

    Returns
    -------

      value : np.ndarray
        value found in the DataArray_t node named after <variable_name>
    """
    return I.getNodeFromNameAndType(self.current_containers,variable_name,'DataArray_t')[1]

  def renameVariable(self,initial_name,new_name):
    """
    Rename a variable given by name <initial_name> into <new_name>.

    Parameters
    ----------
      initial_name : string, current name of the variable to rename
      new_name     : string, name of the variable after the call of this method
    """
    for FS in I.getNodesFromNameAndType(self.tree,self.get_FlowSolution_nodes_name(),'FlowSolution_t'):
      I._renameNode(FS,initial_name,new_name)

  def set_tree(self,tree):
    """
    Sets the working tree.

    Parameters
    ----------

      tree : PyTree
        tree on which this PyTree_operator will perform henceforth
    """
    self.tree=tree

  def set_currentContainers(self,zone):
    if self.data_localization=='CellCenter':
      self.current_containers=[I.getNodeFromNameAndType(zone,I.__FlowSolutionCenters__,'FlowSolution_t')]
    elif self.data_localization=='Vertex':
      self.current_containers=[
        I.getNodeFromNameAndType(zone,I.__FlowSolutionNodes__,'FlowSolution_t'),
        I.getNodeFromNameAndType(zone,I.__GridCoordinates__,'GridCoordinates_t')
      ]
    self.main_container=self.current_containers[0]

  def set_data(self,variable_name,value):
    """
    Defines the value associated to the name **variable_name** as being **value**
    in the current FlowSolution node.

    .. attention:: No check is performed on the dimension of the array given.

    Parameters
    ----------

      variable_name : str
        name of the variable to define

      value : :py:class:`float` or :py:class:`np.ndarray`
        value to assign
    """
    noeuds_variable=I.getNodesFromNameAndType(self.current_containers,variable_name,'DataArray_t')
    if noeuds_variable:
      printv("Attention : les données associées à la variable {0} existent déjà sur la zone {1}, elles vont être remplacées.".format(variable_name,self.zone_courante[0]))
      I._rmNodesByNameAndType(self.current_containers,variable_name,'DataArray_t')
    I.newDataArray(name=variable_name,value=value,parent=self.main_container)

  def set_dataLocalization(self,data_localization):
    """
    Defines which data with which to carry out future computations using this
    PyTree_operator object, nodes or centers.

    Parameters
    ----------

      data_localization : str
        new data localization where the computations must be performed.
        Only the values ``'Vertex'`` and ``'CellCenter'`` are valid.
    """
    if not self.data_localization in ['CellCenter','Vertex']:
      raise Exception("Le type de données fourni n'est pas valide. Les données peuvent seulement être définies aux centres cellules 'CellCenter' où aux noeuds du maillage 'Vertex' (comprenant les coordonnées)")
    self.data_localization=data_localization


  def is_available(self,variable_name):
    """
    Checks whether any data exists in the current FlowSolution node under the name **variable_name**.

    Parameters
    ----------

      variable_name : str
        name of the variable to check

    Returns
    -------

      availability  : bool
        whether there exists a DataArray_t node with the name **variable_name**
    """
    variable_nodes=I.getNodesFromNameAndType(self.current_containers,variable_name,'DataArray_t')
    availability=True
    if not variable_nodes:
      availability=False
    return availability




#__________End of file operators.py____________________________________#
