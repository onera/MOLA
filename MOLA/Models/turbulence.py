#!/bin/bash/python
#Encoding:UTF8
"""
Author : Michel Bouchard
Contact : michel.bouchard@onera.fr, michel-j.bouchard@intradef.gouv.fr
Name : Models.turbulence.py
Description : Submodule that defines some often-used turbulence models
History :
Date       | version | git rev. | Comment
___________|_________|__________|_______________________________________#
2021.12.14 | v3.0.00 |          | Translation with integration into MOLA
2021.09.09 | v2.1.00 |          | Addition of the Smith (1994) :math:`k-l` model
2021.09.01 | v1.3.00 |          | Simplification in the typography of operations by providing
           |         |          | the list of necessary arguments for a given computation instead
           |         |          | of performing a code analysis of the arguments names. More subtle and suple :) 
2021.06.16 | v1.2.00 |          | Completion of the ZDES mode 2 (2020) model by Deck and Renard
2021.06.01 | v1.1.00 |          | Modification of the listing method for available operations (methods) :
           |         |          | lists instead of dictionaries to prevent redundance in the given information
2021.06.01 | v1.0.00 |          | Creation
           |         |          |

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

    # Threshold for some operations (iterative)
    self.computation_threshold=1.e-16
    self.iterative_threshold=1.e-12
    self.iterations_max_number=50

  def set_iterative_threshold(self,iterative_threshold):
    self.iterative_threshold=iterative_threshold

  def set_iterations_max_number(self,iterations_max_number):
    self.iterations_max_number=iterations_max_number

  def set_computation_threshold(self,computation_threshold):
    self.computation_threshold=computation_threshold











class Boussinesq_1877(turbulence):
  """
  Modèle introduisant les variables propres à l'hypothèse de viscosité de
  turbulence de Joseph Boussinesq, et les opérations associées
  """
  def __init__(self):
    super(Boussinesq_1877,self).__init__()
    self.facteur_norme_matricielle=2. # Pour récupérer la norme de Frobenius, utiliser facteur_norme_matricielle=1.
    self.supplyOperations(
      dict(
        Viscosity_EddyMolecularRatio=[
          {
            'noms_arguments':['ViscosityEddy','ViscosityMolecular'],
            'fonction':self.Viscosity_EddyMolecularRatio_from_ViscosityEddy_ViscosityMolecular,
          },
        ],
        ViscosityEddy=[
          {
            'noms_arguments':['Viscosity_EddyMolecularRatio','ViscosityMolecular'],
            'fonction':self.ViscosityEddy_from_Viscosity_EddyMolecularRatio_ViscosityMolecular,
          },
        ],
        ViscosityMolecular=[
          {
            'noms_arguments':['Viscosity_EddyMolecularRatio','ViscosityEddy'],
            'fonction':self.ViscosityMolecular_from_Viscosity_EddyMolecularRatio_ViscosityEddy,
          },
        ],
      )
    )

  def Viscosity_EddyMolecularRatio_from_ViscosityEddy_ViscosityMolecular(self,ViscosityEddy,ViscosityMolecular):
    Viscosity_EddyMolecularRatio=ViscosityEddy/ViscosityMolecular
    return Viscosity_EddyMolecularRatio

  def ViscosityEddy_from_Viscosity_EddyMolecularRatio_ViscosityMolecular(self,Viscosity_EddyMolecularRatio,ViscosityMolecular):
    ViscosityEddy = Viscosity_EddyMolecularRatio*ViscosityMolecular
    return ViscosityEddy

  def ViscosityMolecular_from_Viscosity_EddyMolecularRatio_ViscosityEddy(self,Viscosity_EddyMolecularRatio,ViscosityEddy):
    ViscosityMolecular = ViscosityEddy/Viscosity_EddyMolecularRatio
    return ViscosityMolecular




















class Spalart_Allmaras_1992(Boussinesq_1877):
  """
  Modèle de Spalart-Allmaras (1992) à une équation de transport
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
            'noms_arguments':['TurbulentSANuTildeDensity','TurbulentSANuTilde'],
            'fonction':self.Density_from_conservative_primitive,
          },
        ],
        chi=[
          {
            'noms_arguments':['TurbulentSANuTildeDensity','ViscosityMolecular'],
            'fonction':self.chi_from_TurbulentSANuTildeDensity_ViscosityMolecular,
          },
          {
            'noms_arguments':['Viscosity_EddyMolecularRatio'],
            'fonction':self.chi_from_Viscosity_EddyMolecularRatio,
          },
        ],
        fv1=[
          {
            'noms_arguments':['chi'],
            'fonction':self.fv1_from_chi,
          },
        ],
        TurbulentSANuTilde=[
          {
            'noms_arguments':['TurbulentSANuTildeDensity','Density'],
            'fonction':self.primitive_from_conservative_Density,
          },
        ],
        TurbulentSANuTildeDensity=[
          {
            'noms_arguments':['Density','TurbulentSANuTilde'],
            'fonction':self.conservative_from_Density_primitive,
          },
          {
            'noms_arguments':['chi','ViscosityMolecular'],
            'fonction':self.TurbulentSANuTildeDensity_from_chi_ViscosityMolecular,
          },
        ],
        Viscosity_EddyMolecularRatio=[
          {
            'noms_arguments':['chi'],
            'fonction':self.Viscosity_EddyMolecularRatio_from_chi,
          },
        ],
      )
    )

  # def Density_from_TurbulentSANuTilde_TurbulentSANuTildeDensity(self,TurbulentSANuTilde,TurbulentSANuTildeDensity):
  #   return self.Density_from_conservative_primitive(conservative=TurbulentSANuTildeDensity, primitive=TurbulentSANuTilde)

  def dfv1dchi_from_chi(self,chi):
    return 3.*np.power(chi,2)*(1.-np.power(chi,3)/(np.power(self.cv1,3)+np.power(chi,3)))/(np.power(self.cv1,3)+np.power(chi,3))


  def dViscosity_EddyMolecularRatiodchi_from_chi(self,chi):
    return chi*self.dfv1dchi_from_chi(chi=chi)+self.fv1_from_chi(chi)

  def chi_from_TurbulentSANuTildeDensity_ViscosityMolecular(self,TurbulentSANuTildeDensity,ViscosityMolecular):
    return TurbulentSANuTildeDensity/ViscosityMolecular

  def chi_from_Viscosity_EddyMolecularRatio(self,Viscosity_EddyMolecularRatio):

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
    return np.power(chi,3)/(np.power(self.cv1,3)+np.power(chi,3))

  def TurbulentSANuTilde_from_Density_TurbulentSANuTildeDensity(self,Density,TurbulentSANuTildeDensity):
    return self.primitive_from_conservative_Density(conservative=TurbulentSANuTildeDensity, Density=Density)

  def TurbulentSANuTildeDensity_from_chi_ViscosityMolecular(self,chi,ViscosityMolecular):
    TurbulentSANuTildeDensity = chi*ViscosityMolecular
    return TurbulentSANuTildeDensity

  def TurbulentSANuTildeDensity_from_Density_TurbulentSANuTilde(self,Density,TurbulentSANuTilde):
    return self.conservative_from_Density_primitive(Density=Density, primitive=TurbulentSANuTilde)

  def TurbulentSANuTildeDensity_from_chi_VicosityMolecular(self,chi,ViscosityMolecular):
    TurbulentSANuTildeDensity = chi * ViscosityMolecular
    return TurbulentSANuTildeDensity

  def Viscosity_EddyMolecularRatio_from_chi(self,chi):
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
            'noms_arguments':['Density','gradMagnitude_Velocity','TurbulentDistance','TurbulentSANuTilde','ViscosityMolecular'],
            'fonction':self.logInternalLayersSensor_from_Density_gradMagnitude_Velocity_TurbulentDistance_TurbulentSANuTilde_ViscosityMolecular,
          },
        ],
        logInternalLayersShieldFunction=[
          {
            'noms_arguments':['logInternalLayersSensor'],
            'fonction':self.logInternalLayersShieldFunction_from_logInternalLayersSensor,
          },
        ],
      )
    )

  def logInternalLayersSensor_from_Density_gradMagnitude_Velocity_TurbulentDistance_TurbulentSANuTilde_ViscosityMolecular(self,Density,gradMagnitude_Velocity,TurbulentDistance,TurbulentSANuTilde,ViscosityMolecular):
    logInternalLayersSensor = (ViscosityMolecular/Density+TurbulentSANuTilde)/gradMagnitude_Velocity/np.power(self.kappa_Von_Karman*TurbulentDistance,2)
    return logInternalLayersSensor

  def logInternalLayersShieldFunction_from_logInternalLayersSensor(self,logInternalLayersSensor):
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
            'noms_arguments':['gradCoordinateN_VorticityMagnitude','gradMagnitude_Velocity','TurbulentSANuTilde'],
            'fonction':self.separationSensor_from_gradCoordinateN_VorticityMagnitude_gradMagnitude_Velocity_TurbulentSANuTilde,
          },
        ],
        shieldInhibitionFunction=[
          {
            'noms_arguments':['separationSensor'],
            'fonction':self.shieldInhibitionFunction_from_separationSensor,
          },
        ],
        shieldInhibitionFunctionLim=[
          {
            'noms_arguments':['gradMagnitude_Velocity','shieldInhibitionFunction','VorticityMagnitude'],
            'fonction':self.shieldInhibitionFunctionLim_from_gradMagnitude_Velocity_shieldInhibitionFunction_VorticityMagnitude,
          },
        ],
        shieldAlpha=[
          {
            'noms_arguments':['separationSensor'],
            'fonction':self.shieldAlpha_from_separationSensor,
          },
        ],
        wakeLayerSensor=[
          {
            'noms_arguments':['gradCoordinateN_TurbulentSANuTilde','gradMagnitude_Velocity','TurbulentDistance'],
            'fonction':self.wakeLayerSensor_from_gradCoordinateN_TurbulentSANuTilde_gradMagnitude_Velocity_TurbulentDistance,
          },
        ],
        wakeLayerShieldFunction=[
          {
            'noms_arguments':['wakeLayerSensor'],
            'fonction':self.wakeLayerShieldFunction_from_wakeLayerSensor,
          },
        ],
        wakeLayerStrongShieldFunction=[
          {
            'noms_arguments':['logInternalLayersSensor','wakeLayerShieldFunction'],
            'fonction':self.wakeLayerStrongShieldFunction_from_logInternalLayersSensor_wakeLayerShieldFunction,
          },
        ],
        shieldFunction=[
          {
            'noms_arguments':['logInternalLayersShieldFunction','shieldInhibitionFunctionLim','wakeLayerShieldFunction'],
            'fonction':self.shieldFunction_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerShieldFunction,
          },
        ],
        strongShieldFunction=[
          {
            'noms_arguments':['logInternalLayersShieldFunction','shieldInhibitionFunctionLim','wakeLayerStrongShieldFunction'],
            'fonction':self.strongShieldFunction_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerStrongShieldFunction,
          },
        ],
      )
    )

  def separationSensor_from_gradCoordinateN_VorticityMagnitude_gradMagnitude_Velocity_TurbulentSANuTilde(self,gradCoordinateN_VorticityMagnitude,gradMagnitude_Velocity,TurbulentSANuTilde):
    """
    Calcule les valeurs du senseur de décollement G_Omega en fonction de nutilde, de la norme matricielle
    de Frobenius du gradient de vitesse et de la composante normale à la paroi du gradient de vorticité
    """
    separationSensor = np.sqrt(facteur_norme_matricielle)*gradCoordinateN_VorticityMagnitude*np.sqrt(TurbulentSANuTilde/np.power(gradMagnitude_Velocity,3))
    return separationSensor

  def shieldAlpha_from_separationSensor(self,separationSensor):
    """
    Calcule les valeurs de la variable alpha entrant dans le calcul de la fonction d'inhibition de protection
    en fonction des valeurs du senseur de décollement G_Omega
    """
    shieldAlpha = 7.-6./self.C4*separationSensor
    return shieldAlpha

  def shieldFunction_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerShieldFunction(self,logInternalLayersShieldFunction,shieldInhibitionFunctionLim,wakeLayerShieldFunction):
    """
    Calcule la fonction de protection sans le renforcement introduit par le paramètre beta de la sous-couche intermédiaire entre fd et fP2.
    """
    shieldFunction = logInternalLayersShieldFunction * (1.-shieldInhibitionFunctionLim*(1.-wakeLayerShieldFunction))
    return shieldFunction

  def shieldInhibitionFunction_from_separationSensor(self,separationSensor):
    """
    Calcule les valeurs de la fonction d'inhibition de protection fR (non limitée)
    en fonction des valeurs du senseur de décollement G_Omega
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
    Calcule les valeurs de la fonction d'inhibition de protection fR_lim
    en fonction des valeurs du senseur de décollement G_Omega, de la vorticité
    et de la norme de Frobenius du gradient de vitesse
    """
    shieldInhibitionFunctionLim = (np.sqrt(facteur_norme_matricielle)*VorticityMagnitude< self.zeta*gradMagnitude_Velocity)\
                                + (np.sqrt(facteur_norme_matricielle)*VorticityMagnitude>=self.zeta*gradMagnitude_Velocity)\
                                  * shieldInhibitionFunction
    return shieldInhibitionFunctionLim


  def strongShieldFunction_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerStrongShieldFunction(self,logInternalLayersShieldFunction,shieldInhibitionFunctionLim,wakeLayerStrongShieldFunction):
    """
    Calcule la fonction de protection en incluant le renforcement introduit par le paramètre beta de la sous-couche intermédiaire entre fd et fP2.
    """
    strongShieldFunction = logInternalLayersShieldFunction * (1.-shieldInhibitionFunctionLim*(1.-wakeLayerStrongShieldFunction))
    return strongShieldFunction

  def wakeLayerSensor_from_gradCoordinateN_TurbulentSANuTilde_gradMagnitude_Velocity_TurbulentDistance(self,gradCoordinateN_TurbulentSANuTilde,gradMagnitude_Velocity,TurbulentDistance):
    """
    Calcule les valeurs du senseur de la sous-couche externe de sillage de la couche limite G_nutilde en fonction
    de la distance paroi, du gradient de nutilde dans la direction normale à la paroi et de la norme de Frobenius
    du tenseur gradient de vitesse.
    """
    wakeLayerSensor = self.C3*np.fmax(0.,-gradCoordinateN_TurbulentSANuTilde)/gradMagnitude_Velocity/TurbulentDistance/self.kappa_Von_Karman
    return wakeLayerSensor

  def wakeLayerShieldFunction_from_wakeLayerSensor(self,wakeLayerSensor):
    """
    Calcule la fonction de protection pour la frontière de couche limite, f_d(G_nutilde)
    """
    wakeLayerShieldFunction = self.logInternalLayersShieldFunction_from_logInternalLayersSensor(wakeLayerSensor)
    return wakeLayerShieldFunction

  def wakeLayerStrongShieldFunction_from_logInternalLayersSensor_wakeLayerShieldFunction(self,logInternalLayersSensor,wakeLayerShieldFunction):
    """
    Calcule la fonction de protection de renforcement complète f_{P,2}, incluant la couverture de la zone de transition log-sillage
    """
    wakeLayerStrongShieldFunction = wakeLayerShieldFunction*self.logInternalLayersShieldFunction_from_logInternalLayersSensor(self.beta*logInternalLayersSensor)/np.fmax(self.logInternalLayersShieldFunction_from_logInternalLayersSensor(logInternalLayersSensor),1./self.computation_threshold)
    return wakeLayerStrongShieldFunction



                
















class Kolmogorov_1942(Boussinesq_1877):
  """
  Modèle k-omega de Kolmogorov (1942)
  """
  def __init__(self):
    super(Kolmogorov_1942,self).__init__()
    self.supplyOperations(
      dict(
        Density=[
          {
            'noms_arguments':['TurbulentEnergyKineticDensity','TurbulentEnergyKinetic'],
            'fonction':self.Density_from_conservative_primitive,
          },
          {
            'noms_arguments':['TurbulentDissipationRateDensity','TurbulentDissipationRate'],
            'fonction':self.Density_from_conservative_primitive,
          },
        ],
        TurbulentEnergyKinetic=[
          {
            'noms_arguments':['TurbulentEnergyKineticDensity','Density'],
            'fonction':self.primitive_from_conservative_Density,
          },
          {
            'noms_arguments':['TurbulenceIntensity','VelocityMagnitude'],
            'fonction':self.TurbulentEnergyKinetic_from_TurbulenceIntensity_velocityModulus,
          },
        ],
        TurbulentEnergyKineticDensity=[
          {
            'noms_arguments':['Density','TurbulentEnergyKinetic'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        TurbulentDissipationRate=[
          {
            'noms_arguments':['Density','TurbulentDissipationRateDensity'],
            'fonction':self.conservative_from_Density_primitive,
          },
          {
            'noms_arguments':['Density','TurbulentEnergyKinetic','ViscosityEddy'],
            'fonction':self.TurbulentDissipationRate_from_Density_TurbulentEnergyKinetic_ViscosityEddy,
          },
        ],
        TurbulentDissipationRateDensity=[
          {
            'noms_arguments':['Density','TurbulentDissipationRate'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        ViscosityEddy=[
          {
            'noms_arguments':['Density','TurbulentEnergyKinetic','TurbulentDissipationRate'],
            'fonction':self.ViscosityEddy_from_Density_TurbulentEnergyKinetic_TurbulentDissipationRate,
          },
        ],
      )
    )

  def TurbulentEnergyKinetic_from_TurbulenceIntensity_velocityModulus(self,TurbulenceIntensity,VelocityMagnitude):
    TurbulentEnergyKinetic = 3./2. * np.power(1./100.*TurbulenceIntensity*VelocityMagnitude,2)
    return TurbulentEnergyKinetic

  def TurbulentDissipationRate_from_Density_TurbulentEnergyKinetic_ViscosityEddy(self,Density,TurbulentEnergyKinetic,ViscosityEddy):
    TurbulentDissipationRate = Density*TurbulentEnergyKinetic/ViscosityEddy
    return TurbulentDissipationRate

  def ViscosityEddy_from_Density_TurbulentEnergyKinetic_TurbulentDissipationRate(self,Density,TurbulentEnergyKinetic,TurbulentDissipationRate):
    ViscosityEddy = Density*TurbulentEnergyKinetic/TurbulentDissipationRate
    return ViscosityEddy



















class Menter_Langtry_2009(Kolmogorov_1942):
  def __init__(self,version='Langtry_2006'):
    super(Menter_Langtry_2009,self).__init__()
    self.version=version
    self.supplyOperations(
      dict(
        Density=[
          {
            'noms_arguments':['IntermittencyDensity','Intermittency'],
            'fonction':self.Density_from_conservative_primitive,
          },
          {
            'noms_arguments':['MomentumThicknessReynoldsDensity','MomentumThicknessReynolds'],
            'fonction':self.Density_from_conservative_primitive,
          },
        ],
        Intermittency=[
          {
            'noms_arguments':['IntermittencyDensity','Density'],
            'fonction':self.primitive_from_conservative_Density,
          },
        ],
        IntermittencyDensity=[
          {
            'noms_arguments':['Density','Intermittency'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        MomentumThicknessReynolds=[
          {
            'noms_arguments':['MomentumThicknessReynoldsDensity','Density'],
            'fonction':self.primitive_from_conservative_Density,
          },
          {
            'noms_arguments':['TurbulenceIntensity'],
            'fonction':self.MomentumThicknessReynolds_from_TurbulenceIntensity,
          },
        ],
        MomentumThicknessReynoldsDensity=[
          {
            'noms_arguments':['Density','MomentumThicknessReynolds'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        vorticityReynolds=[
          {
            'noms_arguments':['Density','TurbulentDistance','ViscosityMolecular','VorticityMagnitude'],
            'fonction':self.vorticityReynolds_from_Density_TurbulentDistance_ViscosityMolecular_VorticityMagnitude,
          },
        ],
      )
    )

  def Density_from_Intermittency_IntermittencyDensity(self,Intermittency,IntermittencyDensity):
    return self.Density_from_conservative_primitive(conservative=IntermittencyDensity, primitive=Intermittency)

  def Density_from_MomentumThicknessReynolds_MomentumThicknessReynoldsDensity(self,MomentumThicknessReynolds,MomentumThicknessReynoldsDensity):
    return self.Density_from_conservative_primitive(conservative=MomentumThicknessReynoldsDensity, primitive=MomentumThicknessReynolds)
  
  def Intermittency_from_Density_IntermittencyDensity(self,Density,IntermittencyDensity):
    return self.primitive_from_conservative_Density(Density=Density, conservative=IntermittencyDensity)

  def IntermittencyDensity_from_Density_Intermittency(self,Density,Intermittency):
    return self.conservative_from_Density_primitive(Density=Density, primitive=Intermittency)

  def MomentumThicknessReynolds_from_Density_MomentumThicknessReynoldsDensity(self,Density,MomentumThicknessReynoldsDensity):
    return self.primitive_from_conservative_Density(Density=Density, conservative=MomentumThicknessReynoldsDensity)

  def MomentumThicknessReynoldsDensity_from_Density_MomentumThicknessReynolds(self,Density,MomentumThicknessReynolds):
    return self.conservative_from_Density_primitive(Density=Density, primitive=MomentumThicknessReynolds)

  def MomentumThicknessReynolds_from_TurbulenceIntensity(self,TurbulenceIntensity):
    """
    Calcul de Re_{theta,t} dans le modèle de Menter-Langtry, où il ne dépend que du taux de turbulence. Pour le moment, c'est la seule option disponible.
    """
    MomentumThicknessReynolds=None
    if self.version=='Langtry_2006':
      MomentumThicknessReynolds = (TurbulenceIntensity<1.3)\
                                  * (1173.51-589.428*TurbulenceIntensity + 0.2196/np.power(TurbulenceIntensity,2))\
                                + (TurbulenceIntensity>=1.3)\
                                  * 331.5*np.power(np.fmax(TurbulenceIntensity,1.3)-0.5658,-0.671)
    return MomentumThicknessReynolds

  def vorticityReynolds_from_Density_TurbulentDistance_ViscosityMolecular_VorticityMagnitude(self,Density,TurbulentDistance,ViscosityMolecular,VorticityMagnitude):
    vorticityReynolds=Density*VorticityMagnitude*np.power(TurbulentDistance,2)/ViscosityMolecular
    return vorticityReynolds




















class Smith_1994(Boussinesq_1877):
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
            'noms_arguments':['Density','TurbulentEnergyKinetic','TurbulentLengthScale','ViscosityMolecular'],
            'fonction':self.chi_from_Density_TurbulentKineticEnergy_TurbulentLengthScale_ViscosityMolecular,
          },
          {
            'noms_arguments':['Density','TurbulentDistance','TurbulentEnergyKinetic','ViscosityMolecular','Viscosity_EddyMolecularRatio'],
            'fonction':self.chi_from_Density_TurbulentDistance_TurbulentEnergyKinetic_ViscosityMolecular_Viscosity_EddyMolecularRatio,
          },
        ],
        Density=[
          {
            'noms_arguments':['TurbulentEnergyKineticDensity','TurbulentEnergyKinetic'],
            'fonction':self.Density_from_conservative_primitive,
          },
          {
            'noms_arguments':['TurbulentLengthScaleDensity','TurbulentLengthScaleRate'],
            'fonction':self.Density_from_conservative_primitive,
          },
        ],
        df1dTurbulentLengthScale=[
          {
            'noms_arguments':['TurbulentDistance','TurbulentLengthScale'],
            'fonction':self.df1dTurbulentLengthScale_from_TurbulentDistance_TurbulentLengthScale,
          },
        ],
        dfmudchi=[
          {
            'noms_arguments':['chi','Density','TurbulentDistance','TurbulentEnergyKinetic','TurbulentLengthScale','ViscosityMolecular'],
            'fonction':self.dfmudchi_from_chi_Density_TurbulentDistance_TurbulentEnergyKinetic_TurbulentLengthScale_ViscosityMolecular,
          },
        ],
        dTurbulentLengthScaledchi=[
          {
            'noms_arguments':['Density','chi','TurbulentEnergyKinetic','ViscosityMolecular'],
            'fonction':self.dTurbulentLengthScaledchi_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular,
          },
        ],
        dViscosity_EddyMolecularRatiodchi=[
          {
            'noms_arguments':['chi','fmu','dfmudchi'],
            'fonction':self.dViscosity_EddyMolecularRatiodchi_from_chi_fmu_dfmudchi,
          },
        ],
        f1=[
          {
            'noms_arguments':['TurbulentDistance','TurbulentLengthScale'],
            'fonction':self.f1_from_TurbulentDistance_TurbulentLengthScale,
          },
        ],
        fmu=[
          {
            'noms_arguments':['chi','f1'],
            'fonction':self.fmu_from_chi_f1,
          },
        ],
        TurbulentEnergyKinetic=[
          {
            'noms_arguments':['TurbulentEnergyKineticDensity','Density'],
            'fonction':self.primitive_from_conservative_Density,
          },
          {
            'noms_arguments':['TurbulenceIntensity','VelocityMagnitude'],
            'fonction':self.TurbulentEnergyKinetic_from_TurbulenceIntensity_velocityModulus,
          },
        ],
        TurbulentEnergyKineticDensity=[
          {
            'noms_arguments':['Density','TurbulentEnergyKinetic'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        TurbulentLengthScale=[
          {
            'noms_arguments':['TurbulentLengthScaleDensity','Density'],
            'fonction':self.primitive_from_conservative_Density,
          },
          {
            'noms_arguments':['Density','chi','TurbulentEnergyKinetic','ViscosityMolecular'],
            'fonction':self.TurbulentLengthScale_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular,
          },
        ],
        TurbulentLengthScaleDensity=[
          {
            'noms_arguments':['Density','TurbulentLengthScale'],
            'fonction':self.conservative_from_Density_primitive,
          },
        ],
        Viscosity_EddyMolecularRatio=[
          {
            'noms_arguments':['chi','fmu'],
            'fonction':self.Viscosity_EddyMolecularRatio_from_chi_fmu,
          },
        ],
      )
    )

  def TurbulentEnergyKinetic_from_TurbulenceIntensity_velocityModulus(self,TurbulenceIntensity,VelocityMagnitude):
    TurbulentEnergyKinetic = 3./2. * np.power(1./100.*TurbulenceIntensity*VelocityMagnitude,2)
    return TurbulentEnergyKinetic


  def Viscosity_EddyMolecularRatio_from_chi_fmu(self,chi,fmu):
    Viscosity_EddyMolecularRatio=chi*fmu
    return Viscosity_EddyMolecularRatio


  def chi_from_Density_TurbulentKineticEnergy_TurbulentLengthScale_ViscosityMolecular(self,Density,TurbulentEnergyKinetic,TurbulentLengthScale,ViscosityMolecular):
    chi=Density*np.sqrt(2*TurbulentEnergyKinetic)*TurbulentLengthScale/ViscosityMolecular/np.power(self.B1,1./3.)
    return chi


  def fmu_from_chi_f1(self,chi,f1):
    fmu=np.power((np.power(self.c1,4)*f1+np.power(self.c2,2)*np.power(chi,2)+np.power(chi,4))/(np.power(self.c1,4)+np.power(self.c2,2)*np.power(chi,2)+np.power(chi,4)),1./4.)
    return fmu


  def f1_from_TurbulentDistance_TurbulentLengthScale(self,TurbulentDistance,TurbulentLengthScale):
    f1=np.exp(-50*np.power(TurbulentLengthScale/self.kappa_Von_Karman/TurbulentDistance,2.))
    return f1


  def df1dTurbulentLengthScale_from_TurbulentDistance_TurbulentLengthScale(self,TurbulentDistance,TurbulentLengthScale):
    df1dTurbulentLengthScale=-100.*TurbulentLengthScale/np.power(self.kappa_Von_Karman*TurbulentDistance,2.)\
                            * self.f1_from_TurbulentDistance_TurbulentLengthScale(TurbulentDistance, TurbulentLengthScale)
    return df1dTurbulentLengthScale


  def dTurbulentLengthScaledchi_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular(self,Density,chi,TurbulentEnergyKinetic,ViscosityMolecular):
    dTurbulentLengthScaledchi=ViscosityMolecular*np.power(self.B1,1./3.)/Density/np.sqrt(2.*TurbulentEnergyKinetic)
    return dTurbulentLengthScaledchi


  def TurbulentLengthScale_from_Density_chi_TurbulentEnergyKinetic_ViscosityMolecular(self,Density,chi,TurbulentEnergyKinetic,ViscosityMolecular):
    TurbulentLengthScale=chi*ViscosityMolecular*np.power(self.B1,1./3.)/Density/np.sqrt(2*TurbulentEnergyKinetic)
    return TurbulentLengthScale
  

  def dViscosity_EddyMolecularRatiodchi_from_chi_fmu_dfmudchi(self,chi,fmu,dfmudchi):
    dViscosity_EddyMolecularRatiodchi=fmu+chi*dfmudchi
    return dViscosity_EddyMolecularRatiodchi


  def dfmudchi_from_chi_Density_TurbulentDistance_TurbulentEnergyKinetic_TurbulentLengthScale_ViscosityMolecular(self,Density,chi,TurbulentDistance,TurbulentEnergyKinetic,TurbulentLengthScale,ViscosityMolecular):
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





















#======================================================================#
# Interpréteur de clés pour la                                         #
# création de modèles prédéfinis                                       #
#======================================================================#

existing_models={
  'Boussinesq_1877'       : Boussinesq_1877,
  'Spalart-Allmaras_1992' : Spalart_Allmaras_1992,
  'Deck-Renard_2020'      : Deck_Renard_2020,
  'Menter_1994'           : Kolmogorov_1942,
  'Menter_1994_BSL'       : Kolmogorov_1942,
  'Menter_1994_SST'       : Kolmogorov_1942,
  'Wilcox_1988'           : Kolmogorov_1942,
  'Kok_2005'              : Kolmogorov_1942,
  'Wilcox_2006'           : Kolmogorov_1942,
  'Menter-Langtry_2006'   : Menter_Langtry_2009,
  'Menter-Langtry_2009'   : Menter_Langtry_2009,
  'Menter-Langtry_2015'   : Menter_Langtry_2009,
  'Smith_1994'            : Smith_1994,
}


def build_models(cles):
  modeles=[]
  for cle in cles:
    modeles.append(existing_models[cle]())
  return modeles




#__________Fin de fichier turbulence.py________________________________#