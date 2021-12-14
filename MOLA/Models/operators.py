#!/bin/bash/python
#Encoding:UTF8
"""
Author : Michel Bouchard
Contact : michel.bouchard@onera.fr, michel-j.bouchard@intradef.gouv.fr
Name : Models.operators.py
Description : Submodule for the automatic computation of models-related variables
History :
Date       | version | git rev. | Comment
___________|_________|__________|_______________________________________#
2021.09.01 | v2.0.00 |          | Architecture correction sparing an initial code analysis,
           |         |          | and algorithm correction for cases when several models able
           |         |          | to compute the same variable are passed to the operator
2021.06.01 | v1.0.00 |          | Creation
           |         |          |

"""
#__________Generic modules_____________________________________________#
import sys

#__________ONERA_______________________________________________________#
# Cassiopee
import Converter.PyTree as C
import Converter.Internal as I



debug_operators=False

verbose=[0]


def set_verbose(v):
  verbose[0]=v


def printv(string,v=0,error=False):
  """
  Prints the character string <string> if the verbosity level allows it,
  or, if <error> is True, to standard error.

  Arguments
  ---------
    string : string, to be printed. Do not forget to end lines with  if you do not want your stdout to be messy :)
    v     : int, verbosity level beyond which string must be printed
    error : bool, if True, print to stderr, whether the verbosity level allows it or not

  """
  if not error:
    if v<=verbose[0]:
      sys.stdout.write(string)
      sys.stdout.flush()
  else:
    sys.stderr.write(string)
    sys.stderr.flush()


def printTreeV(tree,v=0):
  """
  Prints a Python tree if the verbose level allows it.

  Arguments
  ---------
    tree : Python tree
    v    : verbosity level beyond which tree must be printed

  """
  if v<=verbose[0]:
    I.printTree(tree)















class operator(object):
  """
  Class of objects that automatically performs operations using given models and a dataset.

  See also
  --------
    operator.__init__
  """
  def __init__(self,models): #,adimensionnement=False):
    """
    Constructor of an operator object instance that automatically performs operations using given models and a dataset.

    Parameters
    ----------
      models : list(.base.model), :py:class:`list` of model objects that the operator must use to perform the computations.
               any object in this list must inherit from .base.model.

    Returns
    -------
      self : operator object which can compute variables using models
    """
    self.models=models
    self.n_models=len(self.models)

    self.errors=''

    self.initializeOperationsTree()
    self.data=dict()

  def append_error(self,error_string):
    self.errors='{0}{1}'.format(self.errors,error_string)


  def compute(self,variable_name,replace=False):
    """
    Computes the value of the variable asked for by name <variable_name> if the models available allow it.
    In case of a conflict with an existing variable in the dataset, replaces it if <replace> is True,
    and does nothing otherwise.
    The chain of operations to perform is automatically determined using available models.
    In order to compute currently inaccessible variables, more models can be supplied using addModels,
    or more functions to one of the available models, using model.supplyFunctions.

    Arguments
    ---------
      variable_name : string, name of the variable to be computed (must lead to at least
                      one available operation of at least one available model).
      replace       : bool, what to do if the variable already exists in the working dataset

    See also
    --------
      get_data
    """

    def _compute(node):
      arguments=dict()
      if not node[2]:
        return self.get_data(node[0])
      else:
        variable_name=node[0]
        i_model=node[1][0]
        i_operation=node[1][1]
        for child_node in node[2]:
          necessary_variable_name=child_node[0]
          arguments.update({necessary_variable_name:_compute(child_node)})
        result=self.models[i_model].carryOutOperation(variable_name, i_operation, arguments)
        printv('{0} = {1}\n'.format(node[0],result),v=4)
        return result

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
      self.set_data(variable_name,_compute(variable_node))

  def eraseAllData(self):
    """
    Erases all the available data from which the current operator can compute other variables.
    """
    self.data=dict()

  def eraseData(self,variable_name):
    """
    Erases the data given by name <variable_name> from the available data
    from which the current operator can compute other variables.

    Arguments
    ---------
      variable_name : string, name of the data to erase
    """
    self.data.pop(variable_name)

  def get_data(self,variable_name):
    """
    Returns the value associated to the name <variable_name> in the current dataset.

    Arguments
    ---------
      variable_name : string, name of the variable to get

    Return
    ------
      value         : float of np.ndarray, value associated to <variable_name>
    """
    if not variable_name in self.data.keys():
      raise Exception("Attention : aucune donnée ne correspond au nom de variable demandé. Fin du processus.")
    return self.data[variable_name]

  def get_dataset(self,variables_names):
    """
    Returns the values associated to one or several names.

    Arguments
    ---------
      variables_names : list(string), names of the variables to get

    Return
    ------
      values          : dictionary of floats of np.ndarrays, values associated to <variables_names>,
                        indexed on <variables_names>
    """
    dataset=dict()
    for variable_name in variables_names:
      dataset.update({variable_name:self.get_data(variable_name)})
    return dataset


  def is_computable(self,variable_name):
    """
    Checks whether a variable asked for by name <variable_name> is computable using the operations
    available to the current operator through its models.

    This method builds the operations tree self.operations_tree (yes, truly), which contains
    the paths through several operations that the compute method should use to obtain the result.
    Note that the operator has no control over the path chosen. In a near future, it could be
    advantageous to compute all possible paths and use the shortest one.
    Furthermore, the same operation can currently be done several times in the same tree if the same
    variable is needed for different computations. In order to improve the execution time and memory
    consumption of this module, it would be interesting to commit the result of a computation
    to memory if it must be reused, instead of recomputing.

    Arguments
    ---------
      variable_name : string, name of the variable whose computability is checked

    Returns
    -------
      is_computable : bool, whether an operations tree can lead to the variable asked for or not.
                      more importantly, the method builds the internal attribute self.operations_tree (see above).
    """

    def _define_computabilityInTree(variable_node,computability,error_string='', v=0):
      if v<=verbose[0]:
        self.append_error(error_string)
      variable_node[1].append(computability)

    def _is_computable(variable_node):
      printTreeV(self.operations_tree,v=6)
      path_variable_node=I.getPath(self.operations_tree,variable_node)
      parent_nodes_names=path_variable_node.split('/')
      if variable_node[0] in parent_nodes_names[:-1]:
        # Si la variable demandée doit servir à se calculer elle-même, l'opération courante n'est pas réalisable
        _define_computabilityInTree(variable_node, False, "La variable {0} est déjà dans la chaîne de calcul et ne peut pas servir à se calculer elle-même.\n", v=4)
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

    Argument
    --------
      variable_name : string, name of the variable to check

    Returns
    -------
      availability  : bool, whether there exists a valueassociated to the name <variable_name>.
    """
    availability=False
    if variable_name in self.data.keys():
      availability=True
    return availability

  def set_data(self,variable_name,value):
    """
    Defines the value associated to the name <variable_name> as being <value>.

    Argument
    --------
      variable_name : string, name of the variable to define
      value         : float or np.ndarray, value to assign
    """
    self.data.update({variable_name:value})

  def set_dataset(self,dataset):
    """
    Defines the values associated to several variables names.
    Argument
    --------
      dataset : dict, named values to add to the current dataset.
    Note
    ----
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
    self.FS_courant=None

  def compute(self,variable_name,replace=False):
    """
    Computes the value of the variable asked for by name <variable_name> in any zone of the working tree if possible.
    Calls the compute method inherited from the operator class.

    Arguments
    ---------
      variable_name : string, name of the variable to be computed (must lead to at least
                      one available operation of at least one available model).
      replace       : bool, what to do if the variable already exists in the working dataset

    See also
    --------
      get_data
    """
    FlowSolution_nodes_name=self.get_FlowSolution_nodes_name()
    for zone in I.getNodesFromType(self.tree,'Zone_t'):
      self.zone_courante=zone
      for FS in I.getNodesFromNameAndType(self.zone_courante,FlowSolution_nodes_name,'FlowSolution_t'):
        self.FS_courant=FS
        super(PyTree_operator, self).compute(variable_name,replace=replace)

  def eraseAllData(self):
    """
    Erases all the available data from the current FlowSolutionNode.
    """
    self.FS_courant[2]=list()

  def eraseData(self,variable_name):
    """
    Erases the data given by name <variable_name> from the current FlowSolution node.

    Arguments
    ---------
      variable_name : string, name of the data to erase
    """
    I._rmNodesByNameAndType(self.FS_courant,variable_name,'DataArray_t')

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

    Argument
    --------
      variable_name : string, name of the variable node which value is returned

    Returns
    -------
      value         : np.ndarray, value found in the DataArray_t node named after <variable_name>
    """
    return I.getNodeFromNameAndType(self.FS_courant,variable_name,'DataArray_t')[1]

  def get_FlowSolution_nodes_name(self):
    """
    Returns the FlowSolution nodes names that must be considered for any computation using the current working tree.
    Computations at cell centers and node can be performed by calling the set_dataLocalization method and modifying
    the corresponding variables of the Cassiopee Converter.Internal module.
    Returns
    -------
      FlowSolution_nodes_names : string, name of the FlowSolution containers to use for the computations with this PyTree_operator
    """
    if self.data_localization=='CellCenter':
      FlowSolution_nodes_name=I.__FlowSolutionCenters__
    elif self.data_localization=='Vertex':
      FlowSolution_nodes_name=I.__FlowSolutionNodes__
    else:
      FlowSolution_nodes_name=None
      raise Exception("Le type de données fourni n'est pas valide. Les données peuvent seulement être définies aux centres cellules 'CellCenter' où aux noeuds du maillage 'Vertex'")
    return FlowSolution_nodes_name

  def renameVariable(self,initial_name,new_name):
    """
    Rename a variable given by name <initial_name> into <new_name>.

    Arguments
    ---------
      initial_name : string, current name of the variable to rename
      new_name     : string, name of the variable after the call of this method
    """
    for FS in I.getNodesFromNameAndType(self.tree,self.get_FlowSolution_nodes_name(),'FlowSolution_t'):
      I._renameNode(FS,initial_name,new_name)

  def set_tree(self,tree):
    """
    Sets the working tree.

    Argument
    --------
      tree : PyTree, tree on which this PyTree_operator will perform henceforth
    """
    self.tree=tree

  def set_data(self,variable_name,value):
    """
    Defines the value associated to the name <variable_name> as being <value> in the current FlowSolution node.
    .. No check is performed on the dimension of the array given.

    Argument
    --------
      variable_name : string, name of the variable to define
      value         : float or np.ndarray, value to assign
    """
    noeuds_variable=I.getNodesFromNameAndType(self.FS_courant,variable_name,'DataArray_t')
    if noeuds_variable:
      printv("Attention : les données associées à la variable {0} existent déjà sur la zone {1}, elles vont être remplacées.".format(variable_name,self.zone_courante[0]))
      I._rmNodesByNameAndType(self.FS_courant,variable_name,'DataArray_t')
    I.newDataArray(name=variable_name,value=value,parent=self.FS_courant)

  def set_dataLocalization(self,data_localization):
    """
    Defines which data with which to carry out future computations using this PyTree_operator object, nodes or centers.

    Argument
    --------
      data_localization : string, new data localization where the computations must be performed.
                          Only the values 'Vertex' and 'CellCenter' are valid.
    """
    self.data_localization=data_localization

  def set_FlowSolution_nodes_name(self,FlowSolution_nodes_name):
    """
    Modifies the names of the FlowSolution containers on which this PyTree_operator object acts.
    .. Warning : this method modifies the values of attributes Converter.Internal.__FlowSolutionCenters__
    and Converter.Internal.__FlowSolutionNodes__, depending on the current data localization.

    Argument
    --------
      FlowSolution_nodes_name : string, new name of the containers in which to look to carry out computations

    See also
    --------
      set_dataLocalization

    """
    if self.data_localization=='CellCenter':
      I.__FlowSolutionCenters__=FlowSolution_nodes_name
    elif self.data_localization=='Vertex':
      I.__FlowSolutionNodes__=FlowSolution_nodes_name

  def is_available(self,variable_name):
    """
    Checks whether any data exists in the current FlowSolution node under the name <variable_name>.

    Argument
    --------
      variable_name : string, name of the variable to check

    Returns
    -------
      availability  : bool, whether there exists a DataArray_t node with the name <variable_name>.
    """
    variable_nodes=I.getNodesFromNameAndType(self.FS_courant,variable_name,'DataArray_t')
    availability=True
    if not variable_nodes:
      availability=False
    return availability




#__________End of file operators.py____________________________________#
