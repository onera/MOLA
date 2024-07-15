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

#!/bin/bash/python
#Encoding:UTF8
"""
Submodule that defines some often-used turbulence models
"""

debug_turbulence=False


#__________Generic modules_____________________________________________#
import numpy as np

#__________This module_________________________________________________#
from .base import printv
from .thermodynamics import thermodynamics










class turbulence(thermodynamics):
  """
  Defines some constants and attributes necessary for the computations of a turbulence model
  """
  def __init__(self):
    super(turbulence, self).__init__()
    self.kappa_Von_Karman = 0.41

    # Threshold for some operations
    self.computation_threshold=1.e-16
    self.iterative_threshold=1.e-12
    self.iterations_max_number=50

  def set_iterative_threshold(self,iterative_threshold):
    '''
    as name indicates
    '''
    self.iterative_threshold=iterative_threshold

  def set_iterations_max_number(self,iterations_max_number):
    '''
    as name indicates
    '''
    self.iterations_max_number=iterations_max_number

  def set_computation_threshold(self,computation_threshold):
    '''
    as name indicates
    '''
    self.computation_threshold=computation_threshold











class Boussinesq_1877(turbulence):
  """
  Model which computes with the Boussinsesq eddy-viscosity hypothesis
  """
  def __init__(self):
    super(Boussinesq_1877,self).__init__()
    self.factor_matrix_norm=2.
    self.supplyOperations(
      dict(
        Viscosity_EddyMolecularRatio=[
          {
            'arguments':['ViscosityEddy','ViscosityMolecular'],
            'operation':self.Viscosity_EddyMolecularRatio_from_ViscosityEddy_ViscosityMolecular,
          },
        ],
        ViscosityEddy=[
          {
            'arguments':['Viscosity_EddyMolecularRatio','ViscosityMolecular'],
            'operation':self.ViscosityEddy_from_Viscosity_EddyMolecularRatio_ViscosityMolecular,
          },
        ],
        ViscosityMolecular=[
          {
            'arguments':['Viscosity_EddyMolecularRatio','ViscosityEddy'],
            'operation':self.ViscosityMolecular_from_Viscosity_EddyMolecularRatio_ViscosityEddy,
          },
        ],
      )
    )

  def Viscosity_EddyMolecularRatio_from_ViscosityEddy_ViscosityMolecular(self,ViscosityEddy,ViscosityMolecular):
    '''
    as name indicates
    '''
    Viscosity_EddyMolecularRatio=ViscosityEddy/ViscosityMolecular
    return Viscosity_EddyMolecularRatio

  def ViscosityEddy_from_Viscosity_EddyMolecularRatio_ViscosityMolecular(self,Viscosity_EddyMolecularRatio,ViscosityMolecular):
    '''
    as name indicates
    '''
    ViscosityEddy = Viscosity_EddyMolecularRatio*ViscosityMolecular
    return ViscosityEddy

  def ViscosityMolecular_from_Viscosity_EddyMolecularRatio_ViscosityEddy(self,Viscosity_EddyMolecularRatio,ViscosityEddy):
    '''
    as name indicates
    '''
    ViscosityMolecular = ViscosityEddy/Viscosity_EddyMolecularRatio
    return ViscosityMolecular




















class Spalart_Allmaras_1992(Boussinesq_1877):
  """
  Spalart-Allmaras model with 1 transport equation (1992)
  doi:10.2514/6.1992-439
  """
  def __init__(self,cb1=0.1355,cb2=0.622,sig=2./3.,cv1=7.1,cw2=.3,cw3=2.):
    super(Spalart_Allmaras_1992,self).__init__()
    self.cb1   = cb1
    self.cb2   = cb2
    self.sigma = sig
    self.cv1   = cv1
    self.cw1   = cb1/self.kappa_Von_Karman**2+(1.+cb2)/sig
    self.cw2   = cw2
    self.cw3   = cw3

    self.supplyOperations(
      dict(
        Density=[
          {
            'arguments':['TurbulentSANuTildeDensity','TurbulentSANuTilde'],
            'operation':self.Density_from_conservative_primitive,
          },
        ],
        chi=[
          {
            'arguments':['TurbulentSANuTildeDensity','ViscosityMolecular'],
            'operation':self.chi_from_TurbulentSANuTildeDensity_ViscosityMolecular,
          },
          {
            'arguments':['Viscosity_EddyMolecularRatio'],
            'operation':self.chi_from_Viscosity_EddyMolecularRatio,
          },
        ],
        fv1=[
          {
            'arguments':['chi'],
            'operation':self.fv1_from_chi,
          },
        ],
        TurbulentSANuTilde=[
          {
            'arguments':['TurbulentSANuTildeDensity','Density'],
            'operation':self.primitive_from_conservative_Density,
          },
        ],
        TurbulentSANuTildeDensity=[
          {
            'arguments':['Density','TurbulentSANuTilde'],
            'operation':self.conservative_from_Density_primitive,
          },
          {
            'arguments':['chi','ViscosityMolecular'],
            'operation':self.TurbulentSANuTildeDensity_from_chi_ViscosityMolecular,
          },
        ],
        Viscosity_EddyMolecularRatio=[
          {
            'arguments':['chi'],
            'operation':self.Viscosity_EddyMolecularRatio_from_chi,
          },
        ],
      )
    )

  # def Density_from_TurbulentSANuTilde_TurbulentSANuTildeDensity(self,TurbulentSANuTilde,TurbulentSANuTildeDensity):
  #   return self.Density_from_conservative_primitive(conservative=TurbulentSANuTildeDensity, primitive=TurbulentSANuTilde)

  def dfv1dchi_from_chi(self,chi):
    '''
    as name indicates
    '''
    return 3.*np.power(chi,2)*(1.-np.power(chi,3)/(np.power(self.cv1,3)+np.power(chi,3)))/(np.power(self.cv1,3)+np.power(chi,3))

  def dViscosity_EddyMolecularRatiodchi_from_chi(self,chi):
    '''
    as name indicates
    '''
    return chi*self.dfv1dchi_from_chi(chi=chi)+self.fv1_from_chi(chi)

  def chi_from_TurbulentSANuTildeDensity_ViscosityMolecular(self,TurbulentSANuTildeDensity,ViscosityMolecular):
    '''
    as name indicates
    '''
    return TurbulentSANuTildeDensity/ViscosityMolecular

  def chi_from_Viscosity_EddyMolecularRatio(self,Viscosity_EddyMolecularRatio):
    '''
    as name indicates
    '''

    chi_m=np.fmax(Viscosity_EddyMolecularRatio,self.computation_threshold)
    chi=np.power(chi_m*np.power(self.cv1,3),1./4.) # Initialisation par en bas
    for i_Newton in range(self.iterations_max_number):
      delta_chi=-(self.Viscosity_EddyMolecularRatio_from_chi(chi)-Viscosity_EddyMolecularRatio)/self.dViscosity_EddyMolecularRatiodchi_from_chi(chi)
      chi+=delta_chi
      if np.all(np.abs(delta_chi/chi)<self.iterative_threshold):
        return np.fmax(chi,self.computation_threshold)-chi_m+np.fmax(chi_m-self.computation_threshold,0.)

    printv("Méthode de Newton non convergée pour la détermination de nutilde/nu dans le modèle de Spalart-Allmaras. Résultat peu fiable.",error=True)
    return np.fmax(chi,self.computation_threshold)-chi_m+np.fmax(chi_m-self.computation_threshold,0.)

  def fv1_from_chi(self,chi):
    '''
    as name indicates
    '''
    return np.power(chi,3)/(np.power(self.cv1,3)+np.power(chi,3))

  def TurbulentSANuTilde_from_Density_TurbulentSANuTildeDensity(self,Density,TurbulentSANuTildeDensity):
    '''
    as name indicates
    '''
    return self.primitive_from_conservative_Density(conservative=TurbulentSANuTildeDensity, Density=Density)

  def TurbulentSANuTildeDensity_from_chi_ViscosityMolecular(self,chi,ViscosityMolecular):
    '''
    as name indicates
    '''
    TurbulentSANuTildeDensity = chi*ViscosityMolecular
    return TurbulentSANuTildeDensity

  def TurbulentSANuTildeDensity_from_Density_TurbulentSANuTilde(self,Density,TurbulentSANuTilde):
    '''
    as name indicates
    '''
    return self.conservative_from_Density_primitive(Density=Density, primitive=TurbulentSANuTilde)

  def TurbulentSANuTildeDensity_from_chi_VicosityMolecular(self,chi,ViscosityMolecular):
    '''
    as name indicates
    '''
    TurbulentSANuTildeDensity = chi * ViscosityMolecular
    return TurbulentSANuTildeDensity

  def Viscosity_EddyMolecularRatio_from_chi(self,chi):
    '''
    as name indicates
    '''
    return chi*self.fv1_from_chi(chi=chi)




















class Deck_2012(Spalart_Allmaras_1992):
  """
  Hybrid RANS-LES model by Deck (2012), also called ZDES mode 2 (deprecated, see Deck_Renard_2020)
  doi:10.1007/s00162-011-0240-z
  """
  def __init__(self,C1=8.,C2=3.,cb1=0.1355,cb2=0.622,sig=2./3.,cv1=7.1,cw2=.3,cw3=2.):
    super(Deck_2012,self).__init__(cb1,cb2,sig,cv1,cw2,cw3)
    self.C1=C1
    self.C2=C2
    self.supplyOperations(
      dict(
        logInternalLayersSensor=[
          {
            'arguments':['Density','gradMagnitude_Velocity','TurbulentDistance','TurbulentSANuTilde','ViscosityMolecular'],
            'operation':self.logInternalLayersSensor_from_Density_gradMagnitude_Velocity_TurbulentDistance_TurbulentSANuTilde_ViscosityMolecular,
          },
        ],
        logInternalLayersShieldFunction=[
          {
            'arguments':['logInternalLayersSensor'],
            'operation':self.logInternalLayersShieldFunction_from_logInternalLayersSensor,
          },
        ],
      )
    )

  def logInternalLayersSensor_from_Density_gradMagnitude_Velocity_TurbulentDistance_TurbulentSANuTilde_ViscosityMolecular(self,Density,gradMagnitude_Velocity,TurbulentDistance,TurbulentSANuTilde,ViscosityMolecular):
    '''
    as name indicates
    '''
    logInternalLayersSensor = (ViscosityMolecular/Density+TurbulentSANuTilde)/gradMagnitude_Velocity/np.power(self.kappa_Von_Karman*TurbulentDistance,2)
    return logInternalLayersSensor

  def logInternalLayersShieldFunction_from_logInternalLayersSensor(self,logInternalLayersSensor):
    '''
    as name indicates
    '''
    logInternalLayersShieldFunction = 1.-np.tanh(np.power(self.C1*logInternalLayersSensor,self.C2))
    return logInternalLayersShieldFunction




















class Deck_Renard_2020(Deck_2012):
  """
  Hybrid RANS-LES model by Deck and Renard (2020), also called ZDES mode 2 (with enhanced protection)
  doi:10.1016/j.jcp.2019.108970
  """
  def __init__(self,C1=8.,C2=3.,C3=25.,C4=0.03,beta=2.5,zeta=0.8,cb1=0.1355,cb2=0.622,sig=2./3.,cv1=7.1,cw2=.3,cw3=2.):
    super(Deck_Renard_2020,self).__init__(C1,C2,cb1,cb2,sig,cv1,cw2,cw3)
    self.C3=C3
    self.C4=C4
    self.beta=beta
    self.zeta=zeta

    self.supplyOperations(
      dict(
        separationSensor=[
          {
            'arguments':['gradCoordinateN_VorticityMagnitude','gradMagnitude_Velocity','TurbulentSANuTilde'],
            'operation':self.separationSensor_from_gradCoordinateN_VorticityMagnitude_gradMagnitude_Velocity_TurbulentSANuTilde,
          },
        ],
        shieldInhibitionFunction=[
          {
            'arguments':['separationSensor'],
            'operation':self.shieldInhibitionFunction_from_separationSensor,
          },
        ],
        shieldInhibitionFunctionLim=[
          {
            'arguments':['gradMagnitude_Velocity','shieldInhibitionFunction','VorticityMagnitude'],
            'operation':self.shieldInhibitionFunctionLim_from_gradMagnitude_Velocity_shieldInhibitionFunction_VorticityMagnitude,
          },
        ],
        shieldAlpha=[
          {
            'arguments':['separationSensor'],
            'operation':self.shieldAlpha_from_separationSensor,
          },
        ],
        wakeLayerSensor=[
          {
            'arguments':['gradCoordinateN_TurbulentSANuTilde','gradMagnitude_Velocity','TurbulentDistance'],
            'operation':self.wakeLayerSensor_from_gradCoordinateN_TurbulentSANuTilde_gradMagnitude_Velocity_TurbulentDistance,
          },
        ],
        wakeLayerShieldFunction=[
          {
            'arguments':['wakeLayerSensor'],
            'operation':self.wakeLayerShieldFunction_from_wakeLayerSensor,
          },
        ],
        wakeLayerStrongShieldFunction=[
          {
            'arguments':['logInternalLayersSensor','wakeLayerShieldFunction'],
            'operation':self.wakeLayerStrongShieldFunction_from_logInternalLayersSensor_wakeLayerShieldFunction,
          },
        ],
        shieldFunction=[
          {
            'arguments':['logInternalLayersShieldFunction','shieldInhibitionFunctionLim','wakeLayerShieldFunction'],
            'operation':self.shieldFunction_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerShieldFunction,
          },
        ],
        strongShieldFunction=[
          {
            'arguments':['logInternalLayersShieldFunction','shieldInhibitionFunctionLim','wakeLayerStrongShieldFunction'],
            'operation':self.strongShieldFunction_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerStrongShieldFunction,
          },
        ],
      )
    )

  def separationSensor_from_gradCoordinateN_VorticityMagnitude_gradMagnitude_Velocity_TurbulentSANuTilde(self,gradCoordinateN_VorticityMagnitude,gradMagnitude_Velocity,TurbulentSANuTilde):
    """
    Computes the value of the separation sensor :math:`{G}_\Omega` from :math:`\\tilde{\\nu}`, the matrix norm of the velocity gradient tensor
    and the wall-normal vorticity gradient.
    """
    separationSensor = gradCoordinateN_VorticityMagnitude*np.sqrt(TurbulentSANuTilde/np.power(gradMagnitude_Velocity,3))
    return separationSensor

  def shieldAlpha_from_separationSensor(self,separationSensor):
    """
    Computes the values of :math:`\\alpha` which enters the computation of the inhibition function using :math:`{G}_\Omega`.
    For the sake of readbility, this function is not accessible through an operator object, only through direct call.
    """
    shieldAlpha = 7.-6./self.C4*separationSensor
    return shieldAlpha

  def shieldFunction_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerShieldFunction(self,logInternalLayersShieldFunction,shieldInhibitionFunctionLim,wakeLayerShieldFunction):
    """
    Computes the shielding function without the strengthening induced by the :math:`\\beta` factor
    """
    shieldFunction = logInternalLayersShieldFunction * (1.-shieldInhibitionFunctionLim*(1.-wakeLayerShieldFunction))
    return shieldFunction

  def shieldInhibitionFunction_from_separationSensor(self,separationSensor):
    """
    Computes the values of the unlimited inhibition function :math:`f_R` from the separation sensor :math:`{G}_\Omega`
    """
    shieldAlpha = self.shieldAlpha_from_separationSensor(separationSensor)
    shieldInhibitionFunction = (separationSensor<=self.C4)\
                             + (separationSensor> self.C4)\
                               * ( (separationSensor<=4./3.*self.C4)\
                                   * 1./(1.+np.exp(-np.fmax(np.fmin(6.*shieldAlpha/(1.-np.power(shieldAlpha,2)),-np.log(self.computation_threshold)),np.log(self.computation_threshold))))\
                                 )
    return shieldInhibitionFunction

  def shieldInhibitionFunctionLim_from_gradMagnitude_Velocity_shieldInhibitionFunction_VorticityMagnitude(self,gradMagnitude_Velocity,shieldInhibitionFunction,VorticityMagnitude):
    """
    Computes the values of the inhibition function :math:`f_R` from the separation sensor :math:`{G}_\Omega`, limited with regard to vorticity
    """
    shieldInhibitionFunctionLim = (np.sqrt(self.factor_matrix_norm)*VorticityMagnitude< self.zeta*gradMagnitude_Velocity)\
                                + (np.sqrt(self.factor_matrix_norm)*VorticityMagnitude>=self.zeta*gradMagnitude_Velocity)\
                                  * shieldInhibitionFunction
    return shieldInhibitionFunctionLim


  def strongShieldFunction_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerStrongShieldFunction(self,logInternalLayersShieldFunction,shieldInhibitionFunctionLim,wakeLayerStrongShieldFunction):
    """
    Computes the shielding function with the application of the :math:`\\beta` factor which strengthens protection in the middle f the boundary layer
    in the case of strong adverse pressure gradients
    """
    strongShieldFunction = logInternalLayersShieldFunction * (1.-shieldInhibitionFunctionLim*(1.-wakeLayerStrongShieldFunction))
    return strongShieldFunction

  def wakeLayerSensor_from_gradCoordinateN_TurbulentSANuTilde_gradMagnitude_Velocity_TurbulentDistance(self,gradCoordinateN_TurbulentSANuTilde,gradMagnitude_Velocity,TurbulentDistance):
    """
    Computes the values of the sensor for the external region of the boundary layer
    """
    wakeLayerSensor = self.C3*np.fmax(0.,-gradCoordinateN_TurbulentSANuTilde)/gradMagnitude_Velocity/TurbulentDistance/self.kappa_Von_Karman
    return wakeLayerSensor

  def wakeLayerShieldFunction_from_wakeLayerSensor(self,wakeLayerSensor):
    """
    Computes thye subshielding function :math:`f_d\left({G}_{\\tilde{\\nu}}\\right)` which covers the wake layer of the boundary layer
    using the external region sensor :math:`{G}_{\\tilde{\\nu}}`
    """
    wakeLayerShieldFunction = self.logInternalLayersShieldFunction_from_logInternalLayersSensor(wakeLayerSensor)
    return wakeLayerShieldFunction

  def wakeLayerStrongShieldFunction_from_logInternalLayersSensor_wakeLayerShieldFunction(self,logInternalLayersSensor,wakeLayerShieldFunction):
    """
    Computes the full reinforcing shielding function compared to the 2012 version of the model :math:`f_{P,2}`, including the strengthening of the shielding
    in the transition region between log layer and wake layer.
    """
    wakeLayerStrongShieldFunction = wakeLayerShieldFunction*self.logInternalLayersShieldFunction_from_logInternalLayersSensor(self.beta*logInternalLayersSensor)/np.fmax(self.logInternalLayersShieldFunction_from_logInternalLayersSensor(logInternalLayersSensor),1./self.computation_threshold)
    return wakeLayerStrongShieldFunction




















class Kolmogorov_1942(Boussinesq_1877):
  """
  Model by Kolmogorov (1942), first :math:`k-\omega` model.
  """
  def __init__(self):
    super(Kolmogorov_1942,self).__init__()
    self.supplyOperations(
      dict(
        Density=[
          {
            'arguments':['TurbulentEnergyKineticDensity','TurbulentEnergyKinetic'],
            'operation':self.Density_from_conservative_primitive,
          },
          {
            'arguments':['TurbulentDissipationRateDensity','TurbulentDissipationRate'],
            'operation':self.Density_from_conservative_primitive,
          },
        ],
        TurbulentEnergyKinetic=[
          {
            'arguments':['TurbulentEnergyKineticDensity','Density'],
            'operation':self.primitive_from_conservative_Density,
          },
          {
            'arguments':['TurbulenceIntensity','VelocityMagnitude'],
            'operation':self.TurbulentEnergyKinetic_from_TurbulenceIntensity_velocityModulus,
          },
        ],
        TurbulentEnergyKineticDensity=[
          {
            'arguments':['Density','TurbulentEnergyKinetic'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        TurbulentDissipationRate=[
          {
            'arguments':['Density','TurbulentDissipationRateDensity'],
            'operation':self.conservative_from_Density_primitive,
          },
          {
            'arguments':['Density','TurbulentEnergyKinetic','ViscosityEddy'],
            'operation':self.TurbulentDissipationRate_from_Density_TurbulentEnergyKinetic_ViscosityEddy,
          },
        ],
        TurbulentDissipationRateDensity=[
          {
            'arguments':['Density','TurbulentDissipationRate'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        ViscosityEddy=[
          {
            'arguments':['Density','TurbulentEnergyKinetic','TurbulentDissipationRate'],
            'operation':self.ViscosityEddy_from_Density_TurbulentEnergyKinetic_TurbulentDissipationRate,
          },
        ],
      )
    )

  def TurbulentEnergyKinetic_from_TurbulenceIntensity_velocityModulus(self,TurbulenceIntensity,VelocityMagnitude):
    '''
    as name indicates
    '''
    TurbulentEnergyKinetic = 3./2. * np.power(1./100.*TurbulenceIntensity*VelocityMagnitude,2)
    return TurbulentEnergyKinetic

  def TurbulentDissipationRate_from_Density_TurbulentEnergyKinetic_ViscosityEddy(self,Density,TurbulentEnergyKinetic,ViscosityEddy):
    '''
    as name indicates
    '''
    TurbulentDissipationRate = Density*TurbulentEnergyKinetic/ViscosityEddy
    return TurbulentDissipationRate

  def ViscosityEddy_from_Density_TurbulentEnergyKinetic_TurbulentDissipationRate(self,Density,TurbulentEnergyKinetic,TurbulentDissipationRate):
    '''
    as name indicates
    '''
    ViscosityEddy = Density*TurbulentEnergyKinetic/TurbulentDissipationRate
    return ViscosityEddy



















class Menter_Langtry_2009(Kolmogorov_1942):
  """
  Transition model by Menter, Langtry et al. (2006,2009,2015).
  doi:10.1115/1.2184352
  doi:10.2514/1.42362
  doi:10.2514/6.2015-2474
  """
  def __init__(self,version='Menter_2006'):
    super(Menter_Langtry_2009,self).__init__()
    self.version=version
    self.supplyOperations(
      dict(
        Density=[
          {
            'arguments':['IntermittencyDensity','Intermittency'],
            'operation':self.Density_from_conservative_primitive,
          },
          {
            'arguments':['MomentumThicknessReynoldsDensity','MomentumThicknessReynolds'],
            'operation':self.Density_from_conservative_primitive,
          },
        ],
        Intermittency=[
          {
            'arguments':['IntermittencyDensity','Density'],
            'operation':self.primitive_from_conservative_Density,
          },
        ],
        IntermittencyDensity=[
          {
            'arguments':['Density','Intermittency'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        MomentumThicknessReynolds=[
          {
            'arguments':['MomentumThicknessReynoldsDensity','Density'],
            'operation':self.primitive_from_conservative_Density,
          },
        ],
        momentumThicknessReynoldsT=[
          {
            'arguments':['TurbulenceIntensity'],
            'operation':self.momentumThicknessReynoldsT_from_TurbulenceIntensity,
          },
        ],
        MomentumThicknessReynoldsDensity=[
          {
            'arguments':['Density','MomentumThicknessReynolds'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        vorticityReynolds=[
          {
            'arguments':['Density','TurbulentDistance','ViscosityMolecular','VorticityMagnitude'],
            'operation':self.vorticityReynolds_from_Density_TurbulentDistance_ViscosityMolecular_VorticityMagnitude,
          },
        ],
      )
    )

  def Density_from_Intermittency_IntermittencyDensity(self,Intermittency,IntermittencyDensity):
    '''
    as name indicates
    '''
    return self.Density_from_conservative_primitive(conservative=IntermittencyDensity, primitive=Intermittency)

  def Density_from_MomentumThicknessReynolds_MomentumThicknessReynoldsDensity(self,MomentumThicknessReynolds,MomentumThicknessReynoldsDensity):
    '''
    as name indicates
    '''
    return self.Density_from_conservative_primitive(conservative=MomentumThicknessReynoldsDensity, primitive=MomentumThicknessReynolds)

  def Intermittency_from_Density_IntermittencyDensity(self,Density,IntermittencyDensity):
    '''
    as name indicates
    '''
    return self.primitive_from_conservative_Density(Density=Density, conservative=IntermittencyDensity)

  def IntermittencyDensity_from_Density_Intermittency(self,Density,Intermittency):
    '''
    as name indicates
    '''
    return self.conservative_from_Density_primitive(Density=Density, primitive=Intermittency)

  def MomentumThicknessReynolds_from_Density_MomentumThicknessReynoldsDensity(self,Density,MomentumThicknessReynoldsDensity):
    '''
    as name indicates
    '''
    return self.primitive_from_conservative_Density(Density=Density, conservative=MomentumThicknessReynoldsDensity)

  def MomentumThicknessReynoldsDensity_from_Density_MomentumThicknessReynolds(self,Density,MomentumThicknessReynolds):
    '''
    as name indicates
    '''
    return self.conservative_from_Density_primitive(Density=Density, primitive=MomentumThicknessReynolds)

  def momentumThicknessReynoldsT_from_TurbulenceIntensity(self,TurbulenceIntensity):
    """
    Computes the correlated value of :math:`Re_{\\theta,t}`. For the time being, only one version is available, that of Menter et al. (2006).
    Future work can be dedicated to the introduction of the Mayle (1991) and Abu-Ghannam & Shaw (1980) correlations, among others,
    as well as introducing the effects of the pressure gradient, which for the moment, is assumed null.
    """
    MomentumThicknessReynolds=None
    if self.version=='Menter_2006':
      MomentumThicknessReynolds = (TurbulenceIntensity<1.3)\
                                  * (1173.51-589.428*TurbulenceIntensity + 0.2196/np.power(TurbulenceIntensity,2))\
                                + (TurbulenceIntensity>=1.3)\
                                  * 331.5*np.power(np.fmax(TurbulenceIntensity,1.3)-0.5658,-0.671)
    return MomentumThicknessReynolds

  def vorticityReynolds_from_Density_TurbulentDistance_ViscosityMolecular_VorticityMagnitude(self,Density,TurbulentDistance,ViscosityMolecular,VorticityMagnitude):
    '''
    as name indicates
    '''
    vorticityReynolds=Density*VorticityMagnitude*np.power(TurbulentDistance,2)/ViscosityMolecular
    return vorticityReynolds




















class Smith_1994(Boussinesq_1877):
  """
  :math:`k-l` model of Smith (1994).

  *Please forget it and forgive me for commiting it once more to memory. MB*
  """
  def __init__(self,B1=18.,E2=1.2,c1=25.5,c2=2.):
    super(Smith_1994,self).__init__()
    self.B1=B1
    self.E2=E2
    self.c1=c1
    self.c2=c2

    self.supplyOperations(
      dict(
        chi=[
          {
            'arguments':['Density','TurbulentEnergyKinetic','TurbulentLengthScale','ViscosityMolecular'],
            'operation':self.chi_from_Density_TurbulentKineticEnergy_TurbulentLengthScale_ViscosityMolecular,
          },
          {
            'arguments':['Density','TurbulentDistance','TurbulentEnergyKinetic','ViscosityMolecular','Viscosity_EddyMolecularRatio'],
            'operation':self.chi_from_Density_TurbulentDistance_TurbulentEnergyKinetic_ViscosityMolecular_Viscosity_EddyMolecularRatio,
          },
        ],
        Density=[
          {
            'arguments':['TurbulentEnergyKineticDensity','TurbulentEnergyKinetic'],
            'operation':self.Density_from_conservative_primitive,
          },
          {
            'arguments':['TurbulentLengthScaleDensity','TurbulentLengthScaleRate'],
            'operation':self.Density_from_conservative_primitive,
          },
        ],
        df1dTurbulentLengthScale=[
          {
            'arguments':['TurbulentDistance','TurbulentLengthScale'],
            'operation':self.df1dTurbulentLengthScale_from_TurbulentDistance_TurbulentLengthScale,
          },
        ],
        dfmudchi=[
          {
            'arguments':['chi','Density','TurbulentDistance','TurbulentEnergyKinetic','TurbulentLengthScale','ViscosityMolecular'],
            'operation':self.dfmudchi_from_chi_Density_TurbulentDistance_TurbulentEnergyKinetic_TurbulentLengthScale_ViscosityMolecular,
          },
        ],
        dTurbulentLengthScaledchi=[
          {
            'arguments':['Density','chi','TurbulentEnergyKinetic','ViscosityMolecular'],
            'operation':self.dTurbulentLengthScaledchi_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular,
          },
        ],
        dViscosity_EddyMolecularRatiodchi=[
          {
            'arguments':['chi','fmu','dfmudchi'],
            'operation':self.dViscosity_EddyMolecularRatiodchi_from_chi_fmu_dfmudchi,
          },
        ],
        f1=[
          {
            'arguments':['TurbulentDistance','TurbulentLengthScale'],
            'operation':self.f1_from_TurbulentDistance_TurbulentLengthScale,
          },
        ],
        fmu=[
          {
            'arguments':['chi','f1'],
            'operation':self.fmu_from_chi_f1,
          },
        ],
        TurbulentEnergyKinetic=[
          {
            'arguments':['TurbulentEnergyKineticDensity','Density'],
            'operation':self.primitive_from_conservative_Density,
          },
          {
            'arguments':['TurbulenceIntensity','VelocityMagnitude'],
            'operation':self.TurbulentEnergyKinetic_from_TurbulenceIntensity_velocityModulus,
          },
        ],
        TurbulentEnergyKineticDensity=[
          {
            'arguments':['Density','TurbulentEnergyKinetic'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        TurbulentLengthScale=[
          {
            'arguments':['TurbulentLengthScaleDensity','Density'],
            'operation':self.primitive_from_conservative_Density,
          },
          {
            'arguments':['Density','chi','TurbulentEnergyKinetic','ViscosityMolecular'],
            'operation':self.TurbulentLengthScale_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular,
          },
        ],
        TurbulentLengthScaleDensity=[
          {
            'arguments':['Density','TurbulentLengthScale'],
            'operation':self.conservative_from_Density_primitive,
          },
        ],
        Viscosity_EddyMolecularRatio=[
          {
            'arguments':['chi','fmu'],
            'operation':self.Viscosity_EddyMolecularRatio_from_chi_fmu,
          },
        ],
      )
    )

  def TurbulentEnergyKinetic_from_TurbulenceIntensity_velocityModulus(self,TurbulenceIntensity,VelocityMagnitude):
    '''
    as name indicates
    '''
    TurbulentEnergyKinetic = 3./2. * np.power(1./100.*TurbulenceIntensity*VelocityMagnitude,2)
    return TurbulentEnergyKinetic


  def Viscosity_EddyMolecularRatio_from_chi_fmu(self,chi,fmu):
    '''
    as name indicates
    '''
    Viscosity_EddyMolecularRatio=chi*fmu
    return Viscosity_EddyMolecularRatio


  def chi_from_Density_TurbulentKineticEnergy_TurbulentLengthScale_ViscosityMolecular(self,Density,TurbulentEnergyKinetic,TurbulentLengthScale,ViscosityMolecular):
    '''
    as name indicates
    '''
    chi=Density*np.sqrt(2*TurbulentEnergyKinetic)*TurbulentLengthScale/ViscosityMolecular/np.power(self.B1,1./3.)
    return chi


  def fmu_from_chi_f1(self,chi,f1):
    '''
    as name indicates
    '''
    fmu=np.power((np.power(self.c1,4)*f1+np.power(self.c2,2)*np.power(chi,2)+np.power(chi,4))/(np.power(self.c1,4)+np.power(self.c2,2)*np.power(chi,2)+np.power(chi,4)),1./4.)
    return fmu


  def f1_from_TurbulentDistance_TurbulentLengthScale(self,TurbulentDistance,TurbulentLengthScale):
    '''
    as name indicates
    '''
    f1=np.exp(-50*np.power(TurbulentLengthScale/self.kappa_Von_Karman/TurbulentDistance,2.))
    return f1


  def df1dTurbulentLengthScale_from_TurbulentDistance_TurbulentLengthScale(self,TurbulentDistance,TurbulentLengthScale):
    '''
    as name indicates
    '''
    df1dTurbulentLengthScale=-100.*TurbulentLengthScale/np.power(self.kappa_Von_Karman*TurbulentDistance,2.)\
                            * self.f1_from_TurbulentDistance_TurbulentLengthScale(TurbulentDistance, TurbulentLengthScale)
    return df1dTurbulentLengthScale


  def dTurbulentLengthScaledchi_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular(self,Density,chi,TurbulentEnergyKinetic,ViscosityMolecular):
    '''
    as name indicates
    '''
    dTurbulentLengthScaledchi=ViscosityMolecular*np.power(self.B1,1./3.)/Density/np.sqrt(2.*TurbulentEnergyKinetic)
    return dTurbulentLengthScaledchi


  def TurbulentLengthScale_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular(self,Density,chi,TurbulentEnergyKinetic,ViscosityMolecular):
    '''
    as name indicates
    '''
    TurbulentLengthScale=chi*ViscosityMolecular*np.power(self.B1,1./3.)/Density/np.sqrt(2*TurbulentEnergyKinetic)
    return TurbulentLengthScale


  def dViscosity_EddyMolecularRatiodchi_from_chi_fmu_dfmudchi(self,chi,fmu,dfmudchi):
    '''
    as name indicates
    '''
    dViscosity_EddyMolecularRatiodchi=fmu+chi*dfmudchi
    return dViscosity_EddyMolecularRatiodchi


  def dfmudchi_from_chi_Density_TurbulentDistance_TurbulentEnergyKinetic_TurbulentLengthScale_ViscosityMolecular(self,Density,chi,TurbulentDistance,TurbulentEnergyKinetic,TurbulentLengthScale,ViscosityMolecular):
    '''
    as name indicates
    '''
    f1=self.f1_from_TurbulentDistance_TurbulentLengthScale(TurbulentDistance, TurbulentLengthScale)
    df1dTurbulentLengthScale=self.df1dTurbulentLengthScale_from_TurbulentDistance_TurbulentLengthScale(TurbulentDistance,TurbulentLengthScale)
    dTurbulentLengthScaledchi=self.dTurbulentLengthScaledchi_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular(Density, chi, TurbulentEnergyKinetic, ViscosityMolecular)
    dfmudchi=1./4.*(
              np.power(self.c1,4.)*df1dTurbulentLengthScale*dTurbulentLengthScaledchi/(np.power(self.c1,4.)+np.power(self.c2,2.)*np.power(chi,2.)+np.power(chi,4.))\
            - (np.power(self.c1,4.)
                * (self.f1_from_TurbulentDistance_TurbulentLengthScale(TurbulentDistance, TurbulentLengthScale)-1.)\
                * (2.*np.power(self.c2,2.)*chi+4.*np.power(chi,3.))\
                / np.power(
                    np.power(self.c1,4.)+np.power(self.c2,2.)*np.power(chi,2.)+np.power(chi,4.),
                    2.
                  )
              )
            )\
            *np.power(
              self.fmu_from_chi_f1(chi, f1),
              -3./4.
            )
    return dfmudchi


  def chi_from_Density_TurbulentDistance_TurbulentEnergyKinetic_ViscosityMolecular_Viscosity_EddyMolecularRatio(self,Density,TurbulentDistance,TurbulentEnergyKinetic,ViscosityMolecular,Viscosity_EddyMolecularRatio):
    '''
    as name indicates
    '''
    Viscosity_EddyMolecularRatio_m=np.fmax(Viscosity_EddyMolecularRatio,self.computation_threshold)
    chi_m=Viscosity_EddyMolecularRatio_m # Initialisation par en bas
    chi=np.copy(chi_m)

    for i_Newton in range(self.iterations_max_number):
      TurbulentLengthScale=self.TurbulentLengthScale_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular(Density, chi, TurbulentEnergyKinetic, ViscosityMolecular)
      f1=self.f1_from_TurbulentDistance_TurbulentLengthScale(TurbulentDistance, TurbulentLengthScale)
      fmu=self.fmu_from_chi_f1(chi, f1)
      dfmudchi=self.dfmudchi_from_chi_Density_TurbulentDistance_TurbulentEnergyKinetic_TurbulentLengthScale_ViscosityMolecular(chi, Density, TurbulentDistance, TurbulentEnergyKinetic, TurbulentLengthScale, ViscosityMolecular)
      delta_chi=-(self.Viscosity_EddyMolecularRatio_from_chi_fmu(chi,fmu)-Viscosity_EddyMolecularRatio_m)/self.dViscosity_EddyMolecularRatiodchi_from_chi_fmu_dfmudchi(chi,fmu,dfmudchi)
      chi+=delta_chi
      if np.all(np.abs(delta_chi/chi)<self.iterative_threshold):
        return np.fmax(chi,self.computation_threshold)-chi_m+np.fmax(chi_m-self.computation_threshold,0.)
    printv("Attention : la méthode de Newton pour le calcul de chi dans le modèle k-l de Smith n'est pas convergée. Les valeurs retournés peuvent être approximatives.\n",error=True)
    return np.fmax(chi,self.computation_threshold)-chi_m+np.fmax(chi_m-self.computation_threshold,0.)

















#__________End of file turbulence.py___________________________________#
