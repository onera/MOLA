#!/bin/bash/python
#Encoding:UTF8
"""
Author : Michel Bouchard
Contact : michel.bouchard@onera.fr, michel-j.bouchard@intradef.gouv.fr
Name : Models.thermodynamics.py
Description : Submodule that defines some often-used thermodynamics models.
History :
Date       | version | git rev. | Comment
___________|_________|__________|_______________________________________#
2021.12.16 | v2.1.00 |          | Energy-, Enthalpy- and Entropy- operations added
2021.12.14 | v2.0.00 | ea8e5b3  | Translation with integration into MOLA
2021.09.01 | v1.2.00 |          | Simplification in the typography of operations by providing
           |         |          | the list of necessary arguments for a given computation instead
           |         |          | of performing a code analysis of the arguments names. More subtle and suple :) 
2021.06.01 | v1.1.00 |          | Modification of the listing method for available operations (methods) :
           |         |          | lists instead of dictionaries to prevent redundance in the given information
2021.06.01 | v1.0.00 |          | Creation
           |         |          |

"""
debug_thermodynamics=False

#__________Generic modules_____________________________________________#
import numpy as np

#__________This module_________________________________________________#
from .base import model




















class thermodynamics(model):
  """
  Introduces some thermodynamics constants and several operations that are often used in thermodynamics computations.

  Constants
  ---------
    Boltzmann constant :math:`k=1.380649e-23\ J\cdot K^{-1}`, official CODATA value
    Avogadro constant :math:`6.02214076e23\ mol^{-1}`, official value fixed since May 20th, 2019

  """
  def __init__(self):
    self.k_Boltzmann     = 1.380649e-23                     
    self.N_Avogadro      = 6.02214076e23                    
    # self.R_perfect_gazes = self.k_Boltzmann*self.N_Avogadro 

    self.factor_matrix_norm=1. # Pour récupérer la norm de modélisation, utiliser factor_matrix_norm=2.

    self.operations=dict()

  # def get_operations(self):
  #   return self.operations

  # def set_proprietes(self,dicProps):
  #   for nom_prop in dicProps.keys():
  #     self.__setattr__(nom_prop,dicProps[nom_prop])

  def set_factor_matrix_norm(self,factor_matrix_norm):
    """
    Sets the value of the Frobenius matrix norm factor :math:`\alpha`,
    used for :math:`\Vert A\Vert=\sqrt{\alpha \sumA_{ij}^2}`
    By default, the value of this factor is 1. Do not forget to modify it for turbulence models,
    for instance, where the value 2 is often used.

    Argument
    --------
      factor_matrix_norm : float, value given to the Frobenius norm factor
    """
    self.factor_matrix_norm=factor_matrix_norm

  #__________Fonctions universelles et utiles en thermo_________________________________________#

  def conservative_from_Density_primitive(self,Density,primitive):
    """
    Computes the conservative variable associated to a primitive using the given density value.
    """
    conservative = Density*primitive
    return conservative

  def Density_from_conservative_primitive(self,conservative,primitive):
    """
    Computes the primitive variable associated to a conservative using the given density value.
    """
    Density = conservative/primitive
    return Density

  def primitive_from_conservative_Density(self,conservative,Density):
    """
    Computes density from the values of both a conservative variable and the associated primitive one.
    """
    primitive = conservative/Density
    return primitive

  def norm_from_component(self,componentX,componentY,componentZ):
    """
    Computes the Frobenius norm of a vector from its components in a three-dimensional space

    Arguments
    ---------
      componentX : float or np.ndarray, first component of the vector on the canonic base
      componentY : float or np.ndarray, second component of the vector on the canonic base
      componentZ : float or np.ndarray, third component of the vector on the canonic base

    Returns
    -------
      norm : float or np.ndarray, norm of the vector described by the given components

    Note
    ----
      The Frobenius norm is computed, without application of the Frobenius factor. Therefore, :math:`\Vert v\Vert=\sqrt{\sum v_{i}^2}`
    """
    norm = np.sqrt(np.power(componentX,2)+np.power(componentY,2)+np.power(componentZ,2))
    return norm

  def matrixNorm_from_components(self,componentXX,componentXY,componentXZ,componentYX,componentYY,componentYZ,componentZX,componentZY,componentZZ):
    """
    Computes the Frobenius norm of a matrix from its components in a three-dimensional space, applying a predefined factor

    Arguments
    ---------
      componentXX : float or np.ndarray, first component of the first vector of the matrix on the canonic base
      componentXY : float or np.ndarray, second component of the first vector of the matrix on the canonic base
      componentXZ : float or np.ndarray, third component of the first vector of the matrix on the canonic base
      componentYX : float or np.ndarray, first component of the second vector of the matrix on the canonic base
      componentYY : float or np.ndarray, second component of the second vector of the matrix on the canonic base
      componentYZ : float or np.ndarray, third component of the second vector of the matrix on the canonic base
      componentZX : float or np.ndarray, first component of the third vector of the matrix on the canonic base
      componentZY : float or np.ndarray, second component of the third vector of the matrix on the canonic base
      componentZZ : float or np.ndarray, third component of the third vector of the matrix on the canonic base

    Returns
    -------
      norm : float or np.ndarray, norm of the matrix described by the given components

    Note
    ----
      The Frobenius norm is computed, with application of the Frobenius factor. Therefore, :math:`\Vert A\Vert=\sqrt{\alpha \sum v_{i}^2}`.
      About the factor :math:`\alpha`, see set_factor_matrix_norm
    """
    norm = np.sqrt(self.factor_matrix_norm*
      ( np.power(componentXX,2) + np.power(componentXY,2) + np.power(componentXZ,2) +
        np.power(componentYX,2) + np.power(componentYY,2) + np.power(componentYZ,2) +
        np.power(componentZX,2) + np.power(componentZY,2) + np.power(componentZZ,2) )
    )
    return norm




















class continuous_medium(thermodynamics):
  """
  Describes the continuous medium model, with introduction of extensive
  and intensive variables concept, such as Density, Energy, etc...

  The following variables are introduced, not following the CGNS norm:
    strainRateXY : component of the strain rate tensor along the first direction,
                   with derivatives along the second :math:`S_{xy}=\frac{1}{2}\left(\frac{\partial u_x}{\partial y}+frac{\partial u_y}{\partial x}\right)`
                   other commponents of the same tensor are likewise defined
    gradMagnitude_Velocity : Frobenius norm of the velocity gradient tensor


  """
  def __init__(self):
    super(continuous_medium, self).__init__()
    self.supplyOperations(
      dict(
        Density=[
          {
            'arguments':['LengthScale','Reynolds','VelocityMagnitude','ViscosityMolecular'],
            'operation':self.Density_from_LengthScale_Reynolds_VelocityMagnitude_ViscosityMolecular,
          },
          {
            'arguments':['MomentumX','VelocityX'],
            'operation':self.Density_from_conservative_primitive,
          },
          {
            'arguments':['MomentumY','VelocityY'],
            'operation':self.Density_from_conservative_primitive,
          },
          {
            'arguments':['MomentumZ','VelocityZ'],
            'operation':self.Density_from_conservative_primitive,
          },
          {
            'arguments':['EnergyStagnationDensity','EnergyStagnation'],
            'operation':self.Density_from_conservative_primitive,
          },
        ],
        Energy=[
          {
            'arguments':['Density','Enthalpy','Pressure'],
            'operation':self.Energy_from_Density_Enthalpy_Pressure,
          },
          {
            'arguments':['EnergyStagnation','VelocityMagnitude'],
            'operation':self.Energy_from_EnergyStagnation_VelocityMagnitude,
          },
        ],
        Enthalpy=[
          {
            'arguments':['Energy', 'Pressure'],
            'operation':self.Enthalpy_from_Density_Energy_Pressure,
          },
          {
            'arguments':['EnthalpyStagnation', 'VelocityMagnitude'],
            'operation':self.Enthalpy_from_EnthalpyStagnation_VelocityMagnitude,
          },
        ],
        EnergyStagnation=[
          {
            'arguments':['EnergyStagnationDensity','Density'],
            'operation':self.primitive_from_conservative_Density,
          }
        ],
        EnergyStagnationDensity=[
          {
            'arguments':['Density','EnergyStagnation'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        gradMagnitude_Velocity=[
          {
            'arguments':['gradCoordinateX_VelocityX','gradCoordinateY_VelocityX','gradCoordinateZ_VelocityX',
                         'gradCoordinateX_VelocityY','gradCoordinateY_VelocityY','gradCoordinateZ_VelocityY',
                         'gradCoordinateX_VelocityZ','gradCoordinateY_VelocityZ','gradCoordinateZ_VelocityZ'],
            'operation':self.matrixNorm_from_components,
          },
        ],
        Mach=[
          {
            'arguments':['VelocityMagnitude','VelocitySound'],
            'operation':self.Mach_from_VelocityMagnitude_VelocitySound,
          },
        ],
        MomentumX=[
          {
            'arguments':['Density','VelocityX'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        MomentumY=[
          {
            'arguments':['Density','VelocityY'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        MomentumZ=[
          {
            'arguments':['Density','VelocityZ'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        Reynolds=[
          {
            'arguments':['Density','LengthScale','VelocityMagnitude','ViscosityMolecular'],
            'operation':self.Reynolds_from_Density_LengthScale_VelocityMagnitude_ViscosityMolecular,
          },
        ],
        strainRateMagnitude=[
          {
            'arguments':['strainRateXX','strainRateXY','strainRateXZ','strainRateYY','strainRateYZ','strainRateZZ'],
            'operation':self.strainRateMagnitude_from_strainRateComponents,
          },
        ],
        strainRateXX=[
          {
            'arguments':['gradCoordinateX_VelocityX'],
            'operation':self.strainRateXX_from_gradCoordinateX_VelocityX,
          },
        ],
        strainRateXY=[
          {
            'arguments':['gradCoordinateX_VelocityY','gradCoordinateY_VelocityX'],
            'operation':self.strainRateXY_from_gradCoordinateX_VelocityY_gradCoordinateY_VelocityX,
          },
        ],
        strainRateXZ=[
          {
            'arguments':['gradCoordinateX_VelocityZ','gradCoordinateZ_VelocityX'],
            'operation':self.strainRateXZ_from_gradCoordinateX_VelocityZ_gradCoordinateZ_VelocityX,
          },
        ],
        strainRateYY=[
          {
            'arguments':['gradCoordinateY_VelocityY'],
            'operation':self.strainRateYY_from_gradCoordinateY_VelocityY,
          },
        ],
        strainRateYZ=[
          {
            'arguments':['gradCoordinateY_VelocityZ','gradCoordinateZ_VelocityY'],
            'operation':self.strainRateYZ_from_gradCoordinateY_VelocityZ_gradCoordinateZ_VelocityY,
          },
        ],
        strainRateZZ=[
          {
            'arguments':['gradCoordinateZ_VelocityZ'],
            'operation':self.strainRateZZ_from_gradCoordinateZ_VelocityZ,
          },
        ],
        VelocityMagnitude=[
          {
            'arguments':['Mach','VelocitySound'],
            'operation':self.VelocityMagnitude_from_Mach_VelocitySound,
          },
          {
            'arguments':['VelocityX','VelocityY','VelocityZ'],
            'operation':self.norm_from_component,
          },
        ],
        VelocitySound=[
          {
            'arguments':['Mach','VelocityMagnitude',],
            'operation':self.VelocitySound_from_Mach_VelocityMagnitude,
          },
        ],
        VelocityX=[
          {
            'arguments':['MomentumX','Density',],
            'operation':self.primitive_from_conservative_Density,
          },
        ],
        VelocityY=[
          {
            'arguments':['MomentumY','Density'],
            'operation':self.primitive_from_conservative_Density,
          },
        ],
        VelocityZ=[
          {
            'arguments':['MomentumZ','Density'],
            'operation':self.primitive_from_conservative_Density,
          },
        ],
        VorticityMagnitude=[
          {
            'arguments':['VorticityX','VorticityY','VorticityZ'],
            'operation':self.VorticityMagnitude_from_vorticityComponents,
          },
          {
            'arguments':['Energy', 'EnergyStagnation'],
            'operation':self.VelocityMagnitude_from_Energy_EnergyStagnation,
          },
          {
            'arguments':['Enthalpy','EnthalpyStagnation'],
            'operation':self.VelocityMagnitude_from_Enthalpy_EnthalpyStagnation,
          },
        ],
        VorticityX=[
          {
            'arguments':['gradCoordinateY_VelocityZ','gradCoordinateZ_VelocityY'],
            'operation':self.VorticityX_from_gradCoordinateY_VelocityZ_gradCoordinateZ_VelocityY,
          },
        ],
        VorticityY=[
          {
            'arguments':['gradCoordinateZ_VelocityX','gradCoordinateX_VelocityZ'],
            'operation':self.VorticityY_from_gradCoordinateZ_VelocityX_gradCoordinateX_VelocityZ,
          },
        ],
        VorticityZ=[
          {
            'arguments':['gradCoordinateX_VelocityY','gradCoordinateY_VelocityX'],
            'operation':self.VorticityZ_from_gradCoordinateX_VelocityY_gradCoordinateY_VelocityX,
          },
        ],
      )
    ) 

  def Density_from_LengthScale_Reynolds_VelocityMagnitude_ViscosityMolecular(self,LengthScale,Reynolds,VelocityMagnitude,ViscosityMolecular):
    Density = Reynolds/VelocityMagnitude/LengthScale*ViscosityMolecular
    return Density

  def Energy_from_EnergyStagnation_VelocityMagnitude(self,EnergyStagnation,VelocityMagnitude):
    Energy=EnergyStagnation-.5*np.power(VelocityMagnitude,2)
    return Energy

  def Energy_from_Density_Enthalpy_Pressure(self,Density,Enthalpy,Pressure):
    Energy=Enthalpy-Pressure/Density
    return Energy

  def EnergyStagnation_from_Energy_VelocityMagnitude(self,Energy,VelocityMagnitude):
    EnergyStagnation=Energy+.5*np.power(VelocityMagnitude,2)
    return EnergyStagnation

  def Enthalpy_from_Density_Energy_Pressure(Density,Energy,Pressure):
    Enthalpy=Energy+Pressure/Density
    return Enthalpy

  def Enthalpy_from_EnthalpyStagnation_VelocityMagnitude(self,EnthalpyStagnation,VelocityMagnitude):
    Enthalpy=EnthalpyStagnation-.5*np.power(VelocityMagnitude,2)
    return Enthalpy

  def EnthalpyStagnation_from_Enthalpy_VelocityMagnitude(self,Enthalpy,VelocityMagnitude):
    EnthalpyStagnation=Enthalpy+.5*np.power(VelocityMagnitude,2)
    return EnthalpyStagnation

  def Mach_from_VelocityMagnitude_VelocitySound(self,VelocityMagnitude,VelocitySound):
    Mach = VelocityMagnitude/VelocitySound
    return Mach

  def strainRateMagnitude_from_strainRateComponents(self,strainRateXX,strainRateXY,strainRateXZ,strainRateYY,strainRateYZ,strainRateZZ):
    return self.matrixNorm_from_components(strainRateXX, strainRateXY, strainRateXZ,
                                           strainRateXY, strainRateYY, strainRateYZ,
                                           strainRateXZ, strainRateYZ, strainRateZZ)

  def strainRateXX_from_gradCoordinateX_VelocityX(self,gradCoordinateX_VelocityX):
    return gradCoordinateX_VelocityX

  def strainRateXY_from_gradCoordinateX_VelocityY_gradCoordinateY_VelocityX(self,gradCoordinateX_VelocityY,gradCoordinateY_VelocityX):
    strainRateXY = 1./2.*(gradCoordinateY_VelocityX+gradCoordinateX_VelocityY)
    return strainRateXY

  def strainRateXZ_from_gradCoordinateX_VelocityZ_gradCoordinateZ_VelocityX(self,gradCoordinateX_VelocityZ,gradCoordinateZ_VelocityX):
    strainRateXZ = 1./2.*(gradCoordinateZ_VelocityX+gradCoordinateX_VelocityZ)
    return strainRateXZ

  def strainRateYY_from_gradCoordinateY_VelocityY(self,gradCoordinateY_VelocityY):
    return gradCoordinateY_VelocityY

  def strainRateYZ_from_gradCoordinateY_VelocityZ_gradCoordinateZ_VelocityY(self,gradCoordinateY_VelocityZ,gradCoordinateZ_VelocityY):
    strainRateYZ = 1./2.*(gradCoordinateZ_VelocityY+gradCoordinateY_VelocityZ)
    return strainRateYZ

  def strainRateZZ_from_gradCoordinateZ_VelocityZ(self,gradCoordinateZ_VelocityZ):
    return gradCoordinateZ_VelocityZ

  def Reynolds_from_Density_LengthScale_VelocityMagnitude_ViscosityMolecular(self,Density,LengthScale,VelocityMagnitude,ViscosityMolecular):
    Reynolds = Density*VelocityMagnitude*LengthScale/ViscosityMolecular
    return Reynolds

  def VelocityMagnitude_from_Energy_EnergyStagnation(self,Energy,EnergyStagnation):
    VelocityMagnitude=np.sqrt(2.*(EnergyStagnation-Energy))
    return VelocityMagnitude

  def VelocityMagnitude_from_Enthalpy_EnthalpyStagnation(self,Enthalpy,EnthalpyStagnation):
    VelocityMagnitude=np.sqrt(2.*(EnthalpyStagnation-Enthalpy))
    return VelocityMagnitude

  def VelocityMagnitude_from_Mach_VelocitySound(self,Mach,VelocitySound):
    VelocityMagnitude = Mach*VelocitySound
    return VelocityMagnitude

  def VelocitySound_from_Mach_VelocityMagnitude(self,Mach,VelocityMagnitude):
    VelocitySound = VelocityMagnitude/Mach
    return VelocitySound

  def VorticityMagnitude_from_vorticityComponents(self,VorticityX,VorticityY,VorticityZ):
    null_component=np.zeros_like(VorticityX,dtype=np.float64)
    return self.matrixNorm_from_components(null_component, VorticityZ    , VorticityY    ,
                                           VorticityZ    , null_component, VorticityX    ,
                                           VorticityY    , VorticityX    , null_component)

  def VorticityX_from_gradCoordinateY_VelocityZ_gradCoordinateZ_VelocityY(self,gradCoordinateY_VelocityZ,gradCoordinateZ_VelocityY):
    VorticityX = 1./2.*(gradCoordinateZ_VelocityY-gradCoordinateY_VelocityZ)
    return VorticityX

  def VorticityY_from_gradCoordinateZ_VelocityX_gradCoordinateX_VelocityZ(self,gradCoordinateZ_VelocityX,gradCoordinateX_VelocityZ):
    VorticityY = 1./2.*(gradCoordinateX_VelocityZ-gradCoordinateZ_VelocityX)
    return VorticityY

  def VorticityZ_from_gradCoordinateX_VelocityY_gradCoordinateY_VelocityX(self,gradCoordinateX_VelocityY,gradCoordinateY_VelocityX):
    VorticityZ = 1./2.*(gradCoordinateY_VelocityX-gradCoordinateX_VelocityY)
    return VorticityZ





















class Sutherland_1893(continuous_medium):
  """
  Describes a fluid which obeys to the Sutherland law.
  Default values for the Sutherland constants are :

  :math:`mu_{\mathrm{ref}}=1.716e-5\ K`
  :math:`T_{\mathrm{ref}}=273.15\ kg\cdot m^{-1}\cdot s^{-1}`
  :math:`S=110.4\ K`
  They must be provided in this order in the list given to the constructor, or set one by one.
  """
  def __init__(self,Cs_Sutherland=[1.716e-5,273.15,110.4]):
    super(Sutherland_1893,self).__init__()
    self.mu_Sutherland = Cs_Sutherland[0]
    self.T_Sutherland  = Cs_Sutherland[1]
    self.S_Sutherland  = Cs_Sutherland[2]

    self.supplyOperations(
      dict(
        ViscosityMolecular=[
          {
            'arguments':['Temperature'],
            'operation':self.ViscosityMolecular_from_Temperature,
          },
        ],
      )
    )

  def get_SutherlandConstants(self):
    """
    Returns
    -------
      constants : list(float), the current values of the Sutherland constants,
                  in the following order :math:`\left[mu_{\mathrm{ref}},T_{\mathrm{ref}},S \right]`
    """
    return [self.mu_Sutherland,self.T_Sutherland,self.S_Sutherland]


  def get_muSutherland(self):
    return self.mu_Sutherland


  def get_SSutherland(self):
    return self.S_Sutherland


  def get_TSutherland(self):
    return self.T_Sutherland


  def ViscosityMolecular_from_Temperature(self,Temperature):
    """
    Computes the molecular dynamic viscosity of the Sutherland fluid from its static temperature.
    """
    ViscosityMolecular = self.mu_Sutherland*np.power(Temperature/self.T_Sutherland,3./2.)*(self.T_Sutherland+self.S_Sutherland)/(Temperature+self.S_Sutherland)
    return ViscosityMolecular




















class perfect_gaz(continuous_medium):
  """
  Describes the perfect gaz model and associated variables and operations.

  Constants
  ---------
    See Sutherland_1893
    Perfect gazes constant :math:`[J.mol-1.K-1]`
    Reduced perfect gaz constant :math:`[J.kg-1.K-1]`
    Molar mass :math:`[kg.mol-1]`
  """
  def __init__(self,r_perfect_gaz=287.058):
    super(perfect_gaz, self).__init__()
    self.r_perfect_gaz   = r_perfect_gaz             
    self.R_perfect_gazes = self.k_Boltzmann*self.N_Avogadro 

    self.supplyOperations(
      dict(
        Density=[
          {
            'arguments':['Pressure','Temperature'],
            'operation':self.Density_from_Pressure_Temperature,
          },
        ],
        DensityStagnation=[
          {
            'arguments':['PressureStagnation','TemperatureStagnation'],
            'operation':self.Density_from_Pressure_Temperature,
          },
        ],
        Pressure=[
          {
            'arguments':['Density','Temperature'],
            'operation':self.Pressure_from_Density_Temperature,
          },
        ],
        PressureStagnation=[
          {
            'arguments':['DensityStagnation','TemperatureStagnation'],
            'operation':self.Pressure_from_Density_Temperature,
          },
        ],
        Temperature=[
          {
            'arguments':['Density','Pressure'],
            'operation':self.Temperature_from_Density_Pressure,
          },
        ],
        TemperatureStagnation=[
          {
            'arguments':['DensityStagnation','PressureStagnation'],
            'operation':self.Temperature_from_Density_Pressure,
          },
        ],
      )
    )

  def get_r_perfect_gaz(self):
    """
    Returns
    -------
      value : float, reduced perfect gaz constant of the current gaz
    """
    return self.r_perfect_gaz

  def get_M(self):
    """
    Returns
    -------
      value : float, molar mass of the current perfect gaz
    """
    return self.R_perfect_gazes/self.r_perfect_gaz

  def Density_from_Pressure_Temperature(self,Pressure,Temperature):
    Density = Pressure/self.r_perfect_gaz/Temperature
    return Density

  def Pressure_from_Density_Temperature(self,Density,Temperature):
    Pressure = Density*self.r_perfect_gaz*Temperature
    return Pressure

  def Temperature_from_Density_Pressure(self,Density,Pressure):
    Temperature = Pressure/self.r_perfect_gaz/Density
    return Temperature










class polytropic_perfect_gaz(perfect_gaz):
  """
  A clarifier : Quelle est l'influence de gamma_Laplace ? En écoulement isentropique on a la loi de Laplace,
  mais un gaz parfait polytropique (i.e. calorifiquement parfait) vérifie déjà Cp/Cv=gamma_Laplace, non ?
  En fait, un gaz parfait polytropique n'est pas forcément calorifiquement parfait. A corriger.
  En attendant, utiliser la classe isentropic_polytropic_perfect_gaz pour un gaz parfait, calorifiquement parfait, polytropique, en écoulement isentropique.
  """
  def __init__(self,r_perfect_gaz=287.058,gamma_Laplace=1.4):
    super(polytropic_perfect_gaz, self).__init__(r_perfect_gaz=r_perfect_gaz)
    self.gamma_Laplace=gamma_Laplace

    self.Temperature_ref_Entropy=273.15
    self.Density_ref_Entropy=1.
    self.Pressure_ref_Entropy=self.Pressure_from_Density_Temperature(Density=self.Density_ref_Entropy, Temperature=self.Temperature_ref_Entropy)

    self.supplyOperations(
      dict(
        Energy=[
          {
            'arguments':['Temperature'],
            'operation':self.Energy_from_Temperature,
          },
        ],
        Enthalpy=[
          {
            'arguments':['Temperature'],
            'operation':self.Enthalpy_from_Temperature,
          },
        ],
        Entropy=[
          {
            'arguments':['Density', 'Temperature'],
            'operation':self.Entropy_from_Density_Temperature,
          },
        ],
        Temperature=[
          {
            'arguments':['Energy'],
            'operation':self.Temperature_from_Energy,
          },
          {
            'arguments':['Enthalpy'],
            'operation':self.Temperature_from_Enthalpy,
          },
        ],
        TemperatureStagnation=[
          {
            'arguments':['EnergyStagnation'],
            'operation':self.TemperatureStagnation_from_EnergyStagnation,
          },
          {
            'arguments':['EnthalpyStagnation'],
            'operation':self.TemperatureStagnation_from_EnthalpyStagnation,
          },
        ],
      )
    )

  def get_Cv(self):
    """
    Returns
    -------
      Cp : float, heat coefficient at constant volume of the current gaz
    """
    return self.r_perfect_gaz/(self.gamma_Laplace-1.)

  def get_Cp(self):
    """
    Returns
    -------
      Cp : float, heat coefficient at constant pressure of the current gaz
    """
    return self.gamma_Laplace*self.r_perfect_gaz/(self.gamma_Laplace-1.)

  def get_gamma_Laplace(self):
    """
    Returns
    -------
      :math:`\gamma` : float, heat coefficients ratio and Laplace exponent of the current gaz
    """
    return self.gamma_Laplace

  def Energy_from_Temperature(self,Temperature):
    Energy=self.r_perfect_gaz/(self.gamma_Laplace-1.)*Temperature
    return Energy

  def Enthalpy_from_Temperature(self,Temperature):
    Enthalpy=self.gamma_Laplace*self.r_perfect_gaz/(self.gamma_Laplace-1.)*Temperature
    return Enthalpy

  def Entropy_from_Density_Temperature(self,Density,Temperature):
    Entropy=self.r_perfect_gaz/(self.gamma_Laplace-1.)*np.log(Temperature/self.Temperature_ref_Entropy)-self.r_perfect_gaz*np.log(Density/self.Density_ref_Entropy)
    return Entropy

  def Temperature_from_Energy(self,Energy):
    Temperature=(self.gamma_Laplace-1.)/self.r_perfect_gaz*Energy
    return Temperature

  def Temperature_from_Enthalpy(self,Energy):
    Temperature=(self.gamma_Laplace-1.)/self.gamma_Laplace/self.r_perfect_gaz*Enthalpy
    return Temperature

  def TemperatureStagnation_from_EnergyStagnation(self,EnergyStagnation):
    TemperatureStagnation=(self.gamma_Laplace-1.)/self.r_perfect_gaz*EnergyStagnation
    return TemperatureStagnation

  def TemperatureStagnation_from_EnthalpyStagnation(self,EnthalpyStagnation):
    TemperatureStagnation=(self.gamma_Laplace-1.)/self.gamma_Laplace/self.r_perfect_gaz*EnthalpyStagnation
    return TemperatureStagnation












class isentropic_polytropic_perfect_gaz(polytropic_perfect_gaz):
  """
  Describes a perfect gaz with constant :math:`\frac{C_p}{C_v}=\gamma_Laplace`,
  which happens to follow a polytropic behavior when an isentropic flow is considered.

  Constant
  --------
    See perfect_gaz
    :math:`\gamma` : Laplace constant of the polytropic gaz. Default value is :math:`1.4`.

  Note
  ----
    The flow considered with this model is always isentropic. A future development should
    introduce an intermediary class which describes the fllow obtained when that assumption fails.
  """
  def __init__(self,gamma_Laplace=1.4,r_perfect_gaz=287.058):
    super(isentropic_polytropic_perfect_gaz, self).__init__(r_perfect_gaz=r_perfect_gaz,gamma_Laplace=gamma_Laplace)
      
    self.supplyOperations(
      dict(
        DensityStagnation=[
          {
            'arguments':['Density','Mach'],
            'operation':self.DensityStagnation_from_Density_Mach,
          },
        ],
        EnergyStagnation=[
          {
            'arguments':['Density','Pressure','VelocityMagnitude'],
            'operation':self.EnergyStagnation_from_Density_Pressure_VelocityMagnitude,
          },
          # {
          #   'arguments':['TemperatureStagnation'],
          #   'operation':self.EnergyStagnation_from_TemperatureStagnation,
          # },
        ],
        EnthalpyStagnation=[
          {
            'arguments':['TemperatureStagnation'],
            'operation':self.EnthalpyStagnation_from_TemperatureStagnation,
          },
        ],
        Mach=[
          {
            'arguments':['Pressure','PressureStagnation'],
            'operation':self.Mach_from_Pressure_PressureStagnation,
          },
        ],
        Pressure=[
          {
            'arguments':['Mach','PressureStagnation'],
            'operation':self.Pressure_from_Mach_PressureStagnation,
          },
        ],
        PressureStagnation=[
          {
            'arguments':['Mach','Pressure'],
            'operation':self.PressureStagnation_from_Mach_Pressure,
          },
        ],
        Temperature=[
          {
            'arguments':['Mach','TemperatureStagnation'],
            'operation':self.Temperature_from_Mach_TemperatureStagnation,
          },
          {
            'arguments':['VelocitySound'],
            'operation':self.Temperature_from_VelocitySound,
          },
        ],
        TemperatureStagnation=[
          # {
          #   'arguments':['EnergyStagnation'],
          #   'operation':self.TemperatureStagnation_from_EnergyStagnation,
          # },
          {
            'arguments':['Mach','Temperature'],
            'operation':self.TemperatureStagnation_from_Mach_Temperature,
          },
        ],
        VelocitySound=[
          {
            'arguments':['Density','Pressure'],
            'operation':self.VelocitySound_from_Density_Pressure,
          },
          {
            'arguments':['Temperature'],
            'operation':self.VelocitySound_from_Temperature,
          },
        ],
      )
    )

  def EnergyStagnation_from_Density_Pressure_VelocityMagnitude(self,Density,Pressure,VelocityMagnitude):
    EnergyStagnation = (Pressure/((self.gamma_Laplace-1.)*Density)+0.5*np.power(VelocityMagnitude,2))
    return EnergyStagnation

  def EnergyStagnation_from_TemperatureStagnation(self,TemperatureStagnation):
    EnergyStagnation = self.r_perfect_gaz/(self.gamma_Laplace-1)*TemperatureStagnation
    return EnergyStagnation

  def DensityStagnation_from_Density_Mach(self,Density,Mach):
    DensityStagnation = Density*np.power(1+(self.gamma_Laplace-1.)/2.*np.power(Mach,2),1./(self.gamma_Laplace-1.))
    return DensityStagnation

  def EnthalpyStagnation_from_TemperatureStagnation(self,TemperatureStagnation):
    EnthalpyStagnation = self.gamma_Laplace/(self.gamma_Laplace-1.)*self.r_perfect_gaz*TemperatureStagnation
    return EnthalpyStagnation

  def Mach_from_Pressure_PressureStagnation(self,Pressure,PressureStagnation):
    Mach = np.sqrt(2./(self.gamma_Laplace-1.)*(np.power(PressureStagnation/Pressure,(self.gamma_Laplace-1)/self.gamma_Laplace)-1))
    return Mach
  
  def Pressure_from_Mach_PressureStagnation(self,Mach,PressureStagnation):
    Pressure = PressureStagnation / np.power(1.+0.5*(self.gamma_Laplace-1.)*np.power(Mach,2),self.gamma_Laplace/(self.gamma_Laplace-1.))
    return Pressure

  def PressureStagnation_from_Mach_Pressure(self,Mach,Pressure):
    PressureStagnation = Pressure * np.power(1.+0.5*(self.gamma_Laplace-1.)*np.power(Mach,2),self.gamma_Laplace/(self.gamma_Laplace-1.))
    return PressureStagnation

  def Temperature_from_Mach_TemperatureStagnation(self,Mach,TemperatureStagnation):
    Temperature = TemperatureStagnation / (1.+0.5*(self.gamma_Laplace-1.)*np.power(Mach,2))
    return Temperature

  def Temperature_from_VelocitySound(self,VelocitySound):
    Temperature = 1./self.gamma_Laplace/self.r_perfect_gaz*np.power(VelocitySound,2)
    return Temperature

  def TemperatureStagnation_from_Mach_Temperature(self,Mach,Temperature):
    TemperatureStagnation = Temperature * (1.+0.5*(self.gamma_Laplace-1.)*np.power(Mach,2))
    return TemperatureStagnation

  def VelocitySound_from_Density_Pressure(self,Density,Pressure):
    VelocitySound = np.sqrt(self.gamma_Laplace*Pressure/Density)
    return VelocitySound

  def VelocitySound_from_Temperature(self,Temperature):
    VelocitySound = np.sqrt(self.gamma_Laplace*self.r_perfect_gaz*Temperature)
    return VelocitySound



#__________End of file thermodynamics.py_______________________________#
