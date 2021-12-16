"""
Author : Michel Bouchard
Contact : michel.bouchard@onera.fr, michel-j.bouchard@intradef.gouv.fr
Name : Models.base.py
Description : Submodule that defines the core methods and architecture of a model object
History :
Date       | version | git rev. | Comment
___________|_________|__________|_______________________________________#
2021.12.16 | v2.1.00 |          | Possibility to introduce external operations in the computing chain
2021.12.14 | v2.0.00 | ea8e5b3  | Translation with integration into MOLA
2021.09.01 | v1.2.00 |          | Simplification in the typography of operations by providing
           |         |          | the list of necessary arguments for a given computation instead
           |         |          | of performing a code analysis of the arguments names. More subtle and suple :) 
2021.06.01 | v1.1.00 |          | Modification of the listing method for available operations (methods) :
           |         |          | lists instead of dictionaries to prevent redundance in the given information
2021.06.01 | v1.0.00 |          | Creation
           |         |          |

"""
debug_base=False










#__________Generic modules_____________________________________________#
import inspect as ip
import sys









verbose={0:0}


def set_verbose(level,scale=0):
  """
  Sets the verbosity level. Messages are only printed to stdout if the verbosity level is greater than
  the verbosity need of the printv function.

  Argument
  --------
    scale : string or int, category of the messages whose verbose level must be set
    level : int, verbosity level henceforth
  """
  verbose[scale]=level


def printv(string,v=0,s=0,error=False):
  """
  Prints the character string <string> if the verbosity level allows it,
  or, if <error> is True, to standard error.

  Arguments
  ---------
    string : string, to be printed. Do not forget to end lines with  if you do not want your stdout to be messy :)
    v      : int, verbosity level beyond which string must be printed
    s      : scale, category of messages you want printed
    error  : bool, if True, print to stderr, whether the verbosity level allows it or not

  """
  if not error:
    if v<=verbose[s]:
      sys.stdout.write(string)
      sys.stdout.flush()
  else:
    sys.stderr.write(string)
    sys.stderr.flush()










class model(object):
  """
  Describes any model apt to act on some variables with the use of parameters.
  This class and any that inherits from it must be compliant with the following rule : it must be able to handle
  both float and np.ndarray objects. Be wary when writing conditions in your models !
  The arguments names, when applicable, must follow the CGNS norm :
  <http://cgns.github.io/>
  When the name of a variable is not defined in the CGNS standard, it must be shown as such in the class description as follows :
  turbulenceIntensity : intensity of turbulence velocity fluctuations, in percent
  The convention adopted here is to write these variables without the leading upper case letter.

  Note
  ----
    A model can be used on its own to access specific operations, but any future development must allow use through
    an operator object.

  See also
  --------
    .operators.operator

  """
  
  def __init__(self):
    self.operations=dict()

  def get_operations_properties(self,variable_name):
    """
    Returns th list of lists of the arguments necessary to compute the value of <variable_name>
    through the current model's operations.

    Arguments
    --------
      variable_name : string, name of the variable that the user might want to compute using this model

    Returns
    -------
      operations_properties : list(list(string)), arguments needed to compute <variable_name>,
      grouped by operation which allows it
    """
    operations_properties=[]
    if variable_name in self.operations.keys():
      for operation in self.operations[variable_name]:
        operations_properties.append(operation['arguments'])
    return operations_properties

  def carryOutOperation(self,variable_name,numero_operation,arguments):
    """
    Performs the operation listed in the available operations, selected with its index <numero_operation>,
    to obtain the value of <variable_name> using the input variables in <arguments>.

    Arguments
    ---------
      variable_name    : string, name of the variable to compute
      numero_operation : int, index of the operation to carry out in the possible operations for the computation of <variable_name>
      arguments        : dict(float or np.ndarray), dictionary of variables need to carry out the operation, given by name

    Returns
    -------
      value : float or np.ndarray, value associated with <variable_name> after the computation
    """
    real_arguments_names=ip.getargspec(self.operations[variable_name][numero_operation]['operation'])[0] #[1:]
    if real_arguments_names[0]=='self':
      real_arguments_names.pop(0)

    real_arguments=dict()
    for numero_variable in range(len(self.operations[variable_name][numero_operation]['arguments'])):
      real_arguments.update(
        {
          real_arguments_names[numero_variable]:arguments[self.operations[variable_name][numero_operation]['arguments'][numero_variable]]
        }
      )
    return self.operations[variable_name][numero_operation]['operation'](**real_arguments)

  def supplyOperations(self,operations):
    """
    Adds several operations to the already available ones.

    Argument
    --------
      operations : dict(string=list(func)), dictionary of the operations to add.
                   Keys of the dictionary are the variables names that the supplemented operations can compute.

    Note
    ----
      .. As of yet (2021.12.14), only functions that are methods of the current class can be used that way.
         Any other function can be added, but I think that this use-case should create malfunctioning.
         Opening the computation to external functions is the object of a development.
         For now, please create your own class through class-inheritance if you want to benefit
         from the operations given by a defined model as well as add your own. MB.
    """
    for variable_name in operations.keys():
      if not variable_name in self.operations.keys():
        self.operations.update({variable_name:operations[variable_name]})
      else:
        for operation in operations[variable_name]:
          self.operations[variable_name].insert(0,operation)

  def supplyExternalOperations(self,variable_name,arguments,operation):
    """
    Adds an external operation to the already available ones.

    Argument
    --------
      variable_name : string, variable that can be computed using the provided <operation>
      arguments     : list(string), name of the arguments necessary for the computation of <variable_name>
      operation     : func, operation to add that allows the computation of <variable_name> using the arguments named <arguments>
    """
    if not variable_name in self.operations.keys():
      self.operations.update(
        {
          variable_name:[
            dict(
              arguments=arguments,
              operation=operation,
            )
          ]
        }
      )
    else:
      self.operations[variable_name].insert(0,dict(
        arguments=arguments,
        operation=operation,
      ))


  #__________Universal useful functions________________________________#

  def identity(self,variable):
    """
    Returns the <variable> value itself. Can be used to rename a variable.

    Argument
    --------
      variable : float or np.ndarray, value with which to compute

    Returns
    -------
      result : float or np.ndarray, equal to <variable>

    Note
    ----
      A reference to <value> is returned. Ulterior modifications of it will affect the original <variable>
    """
    return variable

  def opposite(self,variable):
    """
    Returns the opposite value of <variable>.

    Argument
    --------
      variable : float or np.ndarray, value with which to compute

    Returns
    -------
      result : float or np.ndarray, opposite of <variable>
    """
    return -variable

  def inverse(self,variable):
    """
    Returns the inverse value of <variable>. No protection against zero-division is given.

    Argument
    --------
      variable : float or np.ndarray, value with which to compute

    Returns
    -------
      result : float or np.ndarray, inverse of <variable>
    """
    return 1./variable

#__________End of file base.py_________________________________________#