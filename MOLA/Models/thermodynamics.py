#!/bin/bash/python
#Encoding:UTF8
"""
Author : Michel Bouchard
Contact : michel.bouchard@onera.fr, michel-j.bouchard@intradef.gouv.fr
Name : Models.thermodynamics.py
Description : Submodule that defines some often-used thermodynamics models.
              For the moment, the models are not independent. For instance, the user cannot
              have a perfect gaz that does not follow Sutherland's law.
              A parser is on its way to correct this and build lower level models.
History :
Date       | version | git rev. | Comment
___________|_________|__________|_______________________________________#
2021.12.14 | v2.0.00 |          | Translation with integration into MOLA
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
            'noms_arguments':['LengthScale','Reynolds','VelocityMagnitude','ViscosityMolecular'],
            'fonction':self.Density_from_LengthScale_Reynolds_VelocityMagnitude_ViscosityMolecular,
          },
          {
            'noms_arguments':['MomentumX','VelocityX'],
            'fonction':self.Density_from_conservative_primitive,
          },
          {
            'noms_arguments':['MomentumY','VelocityY'],
            'fonction':self.Density_from_conservative_primitive,
          },
          {
            'noms_arguments':['MomentumZ','VelocityZ'],
            'fonction':self.Density_from_conservative_primitive,
          },
          {
            'noms_arguments':['EnergyStagnationDensity','EnergyStagnation'],
            'fonction':self.Density_from_conservative_primitive,
          },
        ],
        EnergyStagnation=[
          {
            'noms_arguments':['EnergyStagnationDensity','Density'],
            'fonction':self.primitive_from_conservative_Density,
          }
        ],
        EnergyStagnationDensity=[
          {
            'noms_arguments':['Density','EnergyStagnation'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        gradMagnitude_Velocity=[
          {
            'noms_arguments':['gradCoordinateX_VelocityX','gradCoordinateY_VelocityX','gradCoordinateZ_VelocityX',
                              'gradCoordinateX_VelocityY','gradCoordinateY_VelocityY','gradCoordinateZ_VelocityY',
                              'gradCoordinateX_VelocityZ','gradCoordinateY_VelocityZ','gradCoordinateZ_VelocityZ'],
            'fonction':self.matrixNorm_from_components,
          },
        ],
        Mach=[
          {
            'noms_arguments':['VelocityMagnitude','VelocitySound'],
            'fonction':self.Mach_from_VelocityMagnitude_VelocitySound,
          },
        ],
        MomentumX=[
          {
            'noms_arguments':['Density','VelocityX'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        MomentumY=[
          {
            'noms_arguments':['Density','VelocityY'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        MomentumZ=[
          {
            'noms_arguments':['Density','VelocityZ'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        Reynolds=[
          {
            'noms_arguments':['Density','LengthScale','VelocityMagnitude','ViscosityMolecular'],
            'fonction':self.Reynolds_from_Density_LengthScale_VelocityMagnitude_ViscosityMolecular,
          },
        ],
        strainRateMagnitude=[
          {
            'noms_arguments':['strainRateXX','strainRateXY','strainRateXZ','strainRateYY','strainRateYZ','strainRateZZ'],
            'fonction':self.strainRateMagnitude_from_strainRateComponents,
          },
        ],
        strainRateXX=[
          {
            'noms_arguments':['gradCoordinateX_VelocityX'],
            'fonction':self.strainRateXX_from_gradCoordinateX_VelocityX,
          },
        ],
        strainRateXY=[
          {
            'noms_arguments':['gradCoordinateX_VelocityY','gradCoordinateY_VelocityX'],
            'fonction':self.strainRateXY_from_gradCoordinateX_VelocityY_gradCoordinateY_VelocityX,
          },
        ],
        strainRateXZ=[
          {
            'noms_arguments':['gradCoordinateX_VelocityZ','gradCoordinateZ_VelocityX'],
            'fonction':self.strainRateXZ_from_gradCoordinateX_VelocityZ_gradCoordinateZ_VelocityX,
          },
        ],
        strainRateYY=[
          {
            'noms_arguments':['gradCoordinateY_VelocityY'],
            'fonction':self.strainRateYY_from_gradCoordinateY_VelocityY,
          },
        ],
        strainRateYZ=[
          {
            'noms_arguments':['gradCoordinateY_VelocityZ','gradCoordinateZ_VelocityY'],
            'fonction':self.strainRateYZ_from_gradCoordinateY_VelocityZ_gradCoordinateZ_VelocityY,
          },
        ],
        strainRateZZ=[
          {
            'noms_arguments':['gradCoordinateZ_VelocityZ'],
            'fonction':self.strainRateZZ_from_gradCoordinateZ_VelocityZ,
          },
        ],
        VelocityMagnitude=[
          {
            'noms_arguments':['Mach','VelocitySound'],
            'fonction':self.VelocityMagnitude_from_Mach_VelocitySound,
          },
          {
            'noms_arguments':['VelocityX','VelocityY','VelocityZ'],
            'fonction':self.norm_from_component,
          },
        ],
        VelocitySound=[
          {
            'noms_arguments':['Mach','VelocityMagnitude',],
            'fonction':self.VelocitySound_from_Mach_VelocityMagnitude,
          },
        ],
        VelocityX=[
          {
            'noms_arguments':['MomentumX','Density',],
            'fonction':self.primitive_from_conservative_Density,
          },
        ],
        VelocityY=[
          {
            'noms_arguments':['MomentumY','Density'],
            'fonction':self.primitive_from_conservative_Density,
          },
        ],
        VelocityZ=[
          {
            'noms_arguments':['MomentumZ','Density'],
            'fonction':self.primitive_from_conservative_Density,
          },
        ],
        VorticityMagnitude=[
          {
            'noms_arguments':['VorticityX','VorticityY','VorticityZ'],
            'fonction':self.VorticityMagnitude_from_vorticityComponents,
          },
        ],
        VorticityX=[
          {
            'noms_arguments':['gradCoordinateY_VelocityZ','gradCoordinateZ_VelocityY'],
            'fonction':self.VorticityX_from_gradCoordinateY_VelocityZ_gradCoordinateZ_VelocityY,
          },
        ],
        VorticityY=[
          {
            'noms_arguments':['gradCoordinateZ_VelocityX','gradCoordinateX_VelocityZ'],
            'fonction':self.VorticityY_from_gradCoordinateZ_VelocityX_gradCoordinateX_VelocityZ,
          },
        ],
        VorticityZ=[
          {
            'noms_arguments':['gradCoordinateX_VelocityY','gradCoordinateY_VelocityX'],
            'fonction':self.VorticityZ_from_gradCoordinateX_VelocityY_gradCoordinateY_VelocityX,
          },
        ],
      )
    ) 

  def Density_from_LengthScale_Reynolds_VelocityMagnitude_ViscosityMolecular(self,LengthScale,Reynolds,VelocityMagnitude,ViscosityMolecular):
    Density = Reynolds/VelocityMagnitude/LengthScale*ViscosityMolecular
    return Density

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

  def VelocityMagnitude_from_Mach_VelocitySound(self,Mach,VelocitySound):
    VelocityMagnitude = Mach*VelocitySound
    return VelocityMagnitude

  def VelocitySound_from_Mach_VelocityMagnitude(self,Mach,VelocityMagnitude):
    VelocitySound = VelocityMagnitude/Mach
    return VelocitySound

  def VorticityMagnitude_from_vorticityComponents(self,VorticityX,VorticityY,VorticityZ):
    composanteNulle=np.zeros_like(VorticityX,dtype=np.float64)
    return self.matrixNorm_from_components(composanteNulle, VorticityZ     , VorticityY     ,
                                                  VorticityZ     , composanteNulle, VorticityX     ,
                                                  VorticityY     , VorticityX     , composanteNulle)

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
            'noms_arguments':['Temperature'],
            'fonction':self.ViscosityMolecular_from_Temperature,
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




















class perfect_gaz(Sutherland_1893):
  """
  Describes the perfect gaz model and associated variables and operations.

  Constants
  ---------
    See Sutherland_1893
    Perfect gazes constant :math:`[J.mol-1.K-1]`
    Reduced perfect gaz constant :math:`[J.kg-1.K-1]`
    Molar mass :math:`[kg.mol-1]`
  """
  def __init__(self,r_perfect_gaz=287.058,Cs_Sutherland=[1.716e-5,273.15,110.4]):
    super(perfect_gaz, self).__init__(Cs_Sutherland=Cs_Sutherland)
    self.r_perfect_gaz   = r_perfect_gaz             
    self.R_perfect_gazes = self.k_Boltzmann*self.N_Avogadro 

    super(perfect_gaz,self).__init__(Cs_Sutherland)

    self.supplyOperations(
      dict(
        Density=[
          {
            'noms_arguments':['Pressure','Temperature'],
            'fonction':self.Density_from_Pressure_Temperature,
          },
        ],
        DensityStagnation=[
          {
            'noms_arguments':['PressureStagnation','TemperatureStagnation'],
            'fonction':self.Density_from_Pressure_Temperature,
          },
        ],
        Pressure=[
          {
            'noms_arguments':['Density','Temperature'],
            'fonction':self.Pressure_from_Density_Temperature,
          },
        ],
        PressureStagnation=[
          {
            'noms_arguments':['DensityStagnation','TemperatureStagnation'],
            'fonction':self.Pressure_from_Density_Temperature,
          },
        ],
        Temperature=[
          {
            'noms_arguments':['Density','Pressure'],
            'fonction':self.Temperature_from_Density_Pressure,
          },
        ],
        TemperatureStagnation=[
          {
            'noms_arguments':['DensityStagnation','PressureStagnation'],
            'fonction':self.Temperature_from_Density_Pressure,
          },
        ],
      )
    )

  def get_r(self):
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



  
















class polytropic_pg(perfect_gaz):
  """
  A clarifier : Quelle est l'influence de gamma_Laplace ? En écoulement isentropique on a la loi de Laplace,
  mais un gaz parfait polytropique (i.e. calorifiquement parfait) vérifie déjà Cp/Cv=gamma_Laplace, non ?
  En fait, un gaz parfait polytropique n'est pas forcément calorifiquement parfait. A corriger.
  En attendant, utiliser la classe isentropic_ppg pour un gaz parfait, calorifiquement parfait, polytropique, en écoulement isentropique.
  """
  def __init__(self,r_perfect_gaz=287.058,Cs_Sutherland=[1.716e-5,273.15,110.4]):
    super(polytropic_pg, self).__init__(r_perfect_gaz,Cs_Sutherland)




















class isentropic_ppg(polytropic_pg):
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
  def __init__(self,gamma_Laplace=1.4,r_perfect_gaz=287.058,Cs_Sutherland=[1.716e-5,273.15,110.4]):
    super(isentropic_ppg, self).__init__(r_perfect_gaz,Cs_Sutherland)
    self.gamma_Laplace = gamma_Laplace
      
    self.supplyOperations(
      dict(
        DensityStagnation=[
          {
            'noms_arguments':['Density','Mach'],
            'fonction':self.DensityStagnation_from_Density_Mach,
          },
        ],
        EnergyStagnation=[
          {
            'noms_arguments':['Density','Pressure','VelocityMagnitude'],
            'fonction':self.EnergyStagnation_from_Density_Pressure_VelocityMagnitude,
          },
          # {
          #   'noms_arguments':['TemperatureStagnation'],
          #   'fonction':self.EnergyStagnation_from_TemperatureStagnation,
          # },
        ],
        EnthalpyStagnation=[
          {
            'noms_arguments':['TemperatureStagnation'],
            'fonction':self.EnthalpyStagnation_from_TemperatureStagnation,
          },
        ],
        Mach=[
          {
            'noms_arguments':['Pressure','PressureStagnation'],
            'fonction':self.Mach_from_Pressure_PressureStagnation,
          },
        ],
        Pressure=[
          {
            'noms_arguments':['Mach','PressureStagnation'],
            'fonction':self.Pressure_from_Mach_PressureStagnation,
          },
        ],
        PressureStagnation=[
          {
            'noms_arguments':['Mach','Pressure'],
            'fonction':self.PressureStagnation_from_Mach_Pressure,
          },
        ],
        Temperature=[
          {
            'noms_arguments':['Mach','TemperatureStagnation'],
            'fonction':self.Temperature_from_Mach_TemperatureStagnation,
          },
          {
            'noms_arguments':['VelocitySound'],
            'fonction':self.Temperature_from_VelocitySound,
          },
        ],
        TemperatureStagnation=[
          # {
          #   'noms_arguments':['EnergyStagnation'],
          #   'fonction':self.TemperatureStagnation_from_EnergyStagnation,
          # },
          {
            'noms_arguments':['Mach','Temperature'],
            'fonction':self.TemperatureStagnation_from_Mach_Temperature,
          },
        ],
        VelocitySound=[
          {
            'noms_arguments':['Density','Pressure'],
            'fonction':self.VelocitySound_from_Density_Pressure,
          },
          {
            'noms_arguments':['Temperature'],
            'fonction':self.VelocitySound_from_Temperature,
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

  def get_gamma_Laplace(self):
    """
    Returns
    -------
      :math:`\gamma` : float, heat coefficients ratio and Laplace exponent of the current gaz
    """
    return self.gamma_Laplace

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

  # def TemperatureStagnation_from_EnergyStagnation(self,EnergyStagnation):
  #   """
  #   Etant données la masse volumique, la pression statique et la vitesse du gaz parfait, calcule son énergie interne
  #   pas sûr de ça
  #   """
  #   TemperatureStagnation = (self.gamma_Laplace-1)/self.r_perfect_gaz*EnergyStagnation
  #   return TemperatureStagnation

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
