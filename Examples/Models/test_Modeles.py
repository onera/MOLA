#!/bin/bash/python
#Encoding:UTF8
"""
Author : Michel Bouchard
Contact : michel.bouchard@onera.fr, michel-j.bouchard@intradef.gouv.fr
Name : test_Modeles.py
Description : Example of uses for the Models submodule
History :
Date       | version | git rev. | Comment
___________|_________|__________|_______________________________________#
2021.12.16 | v1.0.00 |          | Creation from non-regression files
           |         | ea8e5b3  |

"""
import numpy as np
import sys

import Models.operators as Mo

# Pour plus d'information, passer la verbosité à 2, puis 5, 6 et 10
Mo.set_verbose(0)




sys.stdout.write("\n")
sys.stdout.write("#======================================================================#\n")
sys.stdout.write("# Première application : en boîte noire, je veux calculer les          #\n")
sys.stdout.write("# caractéristiques d'un écoulement isentropique de gaz parfait         #\n")
sys.stdout.write("# polytropique dont je précise Mach, Reynolds, Longueur de référence   #\n")
sys.stdout.write("# et température d'arrêt                                               #\n")
sys.stdout.write("#======================================================================#\n")
from Models.thermodynamics import isentropic_polytropic_perfect_gaz, Sutherland_1893

gaz1=[isentropic_polytropic_perfect_gaz(gamma_Laplace=1.4,r_perfect_gaz=287.058),Sutherland_1893(Cs_Sutherland=[1.716e-5,273.15,110.4])]
calculette=Mo.operator(gaz1)
calculette.set_dataset(
  dict(
    Reynolds=100000.,
    Mach=.65,
    TemperatureStagnation=299.001,
    LengthScale=0.07998,
  )
)
calculette.compute('PressureStagnation')
calculette.compute('Pressure')
print("Valeurs nécessaires pour optenir Mach et Reynolds étant donnée la température génératrice et en supposant un écoulement isentropique de gaz parfait polytropique:")
print(calculette.get_dataset(['Pressure','PressureStagnation']))


sys.stdout.write("\n\n\n\n")
sys.stdout.write("#======================================================================#\n")
sys.stdout.write("# Deuxième application : toujours en boîte noire, je veux calculer les #\n")
sys.stdout.write("# caractéristiques d'un écoulement isentropique de gaz parfait dont je #\n")
sys.stdout.write("# précise la vitesse axiale, la masse volumique, le nombre de Mach et  #\n")
sys.stdout.write("# une température génératrice                                          #\n")
sys.stdout.write("#======================================================================#\n")

# Pour un effet de Reynolds par rapport à une référence (beurk, mais bon, ça venait d'une demande à propos des scripts d'A. Minot. Sans commentaire sauf celui-ci :D) :
Re_cible=100000.
Re_ref=250000.

gaz2=[isentropic_polytropic_perfect_gaz(gamma_Laplace=1.4,r_perfect_gaz=287.058),Sutherland_1893(Cs_Sutherland=[1.716e-5*Re_ref/Re_cible,273.15,110.4])]
calculette=Mo.operator(gaz2)
calculette.set_dataset(
  dict(
    VelocityX=10.,
    Density=1.043,
    Mach=.65,
    TemperatureStagnation=299.001,
  )
)
calculette.compute('PressureStagnation')
calculette.compute('MomentumX')
print(calculette.get_dataset(['MomentumX','PressureStagnation']))
# On peut effacer des variables pour modifier des choses en cours de route (non recommandé, mais disponible)
calculette.eraseData('TemperatureStagnation')
calculette.set_data('Temperature',356.)
calculette.set_data('Mach',.68)
calculette.compute('PressureStagnation',replace=True)
print(calculette.get_dataset(['PressureStagnation']))






sys.stdout.write("\n\n\n\n")
sys.stdout.write("#======================================================================#\n")
sys.stdout.write("# Troisième application : toujours en boîte noire, je veux             #\n")
sys.stdout.write("# post-traiter un CGNS contenant certaines variables et pas d'autres,  #\n")
sys.stdout.write("# en utilisant le modèle de Spalart-Allmaras                           #\n")
sys.stdout.write("# J'en profite pour préciser que les classes décrivant les opérations  #\n")
sys.stdout.write("# des modèles de turbulence ne sont pas complètes, loin de là. Ce sera #\n")
sys.stdout.write("# aux gens de coder les nouvelles classes en fonctions des besoins du  #\n")
sys.stdout.write("# moment.                                                              #\n")
sys.stdout.write("#======================================================================#\n")


# Cassiopée
import Converter.PyTree as C
import Converter.Internal as I

# Opérateur et modèles
from Models.parse import build_models


gaz=[isentropic_polytropic_perfect_gaz(gamma_Laplace=1.4,r_perfect_gaz=287.058),Sutherland_1893(Cs_Sutherland=[1.716e-5*Re_ref/Re_cible,273.15,110.4])]

modeles_turb=build_models({'Spalart-Allmaras_1992':dict()})
modeles=gaz+modeles_turb
calculette=Mo.PyTree_operator(modeles)
arbre=C.convertFile2PyTree('Median_ite_0059100.hdf')
calculette.set_tree(arbre)
calculette.compute('VelocityX')
calculette.compute('VelocityY')
calculette.compute('VelocityZ')
calculette.compute('fv1')
# Si la variable est déjà présente, elle n'est pas recalculée
calculette.compute('Viscosity_EddyMolecularRatio')
C.convertPyTree2File(arbre,'sortie_cas_3.hdf')



sys.stdout.write("\n\n\n\n")
sys.stdout.write("#======================================================================#\n")
sys.stdout.write("# Quatrième application (pour développeur plus avancé) : je souhaite   #\n")
sys.stdout.write("# introduire de nouvelles opérations associées à un ou plusieurs       #\n")
sys.stdout.write("# modèles et les utiliser pour post-traiter un cas ou faire du tracé   #\n")
sys.stdout.write("# de champs                                                            #\n")
sys.stdout.write("#======================================================================#\n")


from Models.turbulence import Deck_Renard_2020
from Models.base import model


# Premier modèle additionnel, pour construire les composantes normales et
# tangentielles de la vitesse par rapport à la paroi,
# étant données les directions dans lesquelles se trouve la paroi en tout point du maillage,
# via les deux angles wallLatitude et wallLongitude (voir le CGNS)
class modele_paroi(model):
  def __init__(self):
    super(modele_paroi,self).__init__()
    self.supplyOperations(
      dict(
        wallTangentialVelocity=[
          {
            'arguments':['VelocityX','VelocityY','WallLongitude'],
            'operation':self.wallTangentialVelocity_from_VelocityX_VelocityY_WallLongitude,
          },
        ],
        wallNormalVelocity=[
          {
            'arguments':['VelocityX','VelocityY','WallLatitude','WallLongitude'],
            'operation':self.wallNormalVelocity_from_VelocityX_VelocityY_WallLatitude_WallLongitude,
          },
        ],
        groundCoordinateX=[
          {
            'arguments':['CoordinateX'],
            'operation':self.identity,
          }
        ],
      )
    )

  def wallTangentialVelocity_from_VelocityX_VelocityY_WallLongitude(self,VelocityX,VelocityY,WallLongitude):
    wallTangentialVelocity = VelocityY*np.cos(WallLongitude)-VelocityX*np.sin(WallLongitude)
    return wallTangentialVelocity

  def wallNormalVelocity_from_VelocityX_VelocityY_WallLatitude_WallLongitude(self,VelocityX,VelocityY,WallLatitude,WallLongitude):
    wallNormalVelocity = VelocityX*np.cos(WallLongitude)*np.cos(WallLatitude)+VelocityY*np.sin(WallLongitude)*np.cos(WallLatitude)
    return wallNormalVelocity



# Deuxième modèle additionnel, surcouche sur la ZDES 2020 : on peut ajouter des opérations possibles
# par héritage de classe pour bénéficier de leurs attributs ou donner de la cohérence au code
class analyse_ZDES_mode_2(Deck_Renard_2020):
  def __init__(self):
    super(analyse_ZDES_mode_2,self).__init__()
    self.supplyOperations(
      dict(
        usefulEnhancedProtection=[
          {
            'arguments':['logInternalLayersShieldFunction','shieldInhibitionFunctionLim','wakeLayerShieldFunction'],
            'operation':self.usefulEnhancedProtection_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerShieldFunction,
          },
        ],
        usefulStrongEnhancedProtection=[
          {
            'arguments':['logInternalLayersShieldFunction','shieldInhibitionFunctionLim','wakeLayerStrongShieldFunction'],
            'operation':self.usefulStrongEnhancedProtection_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerStrongShieldFunction,
          },
        ],
      )
    )

  def usefulEnhancedProtection_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerShieldFunction(self,logInternalLayersShieldFunction,shieldInhibitionFunctionLim,wakeLayerShieldFunction):
    # Peu importe ce que ça fait
    usefulEnhancedProtection = logInternalLayersShieldFunction*(1.-shieldInhibitionFunctionLim)*(1.-wakeLayerShieldFunction)
    return usefulEnhancedProtection

  def usefulStrongEnhancedProtection_from_logInternalLayersShieldFunction_shieldInhibitionFunctionLim_wakeLayerStrongShieldFunction(self,logInternalLayersShieldFunction,shieldInhibitionFunctionLim,wakeLayerStrongShieldFunction):
    usefulStrongEnhancedProtection = logInternalLayersShieldFunction*(1.-shieldInhibitionFunctionLim)*(1.-wakeLayerStrongShieldFunction)
    return usefulStrongEnhancedProtection



# Exécution proprement dite
modeles=gaz+[modele_paroi(),analyse_ZDES_mode_2()] # Les objets modèles ne sont pas affecté par le fait de compute avec, on peut donc réutiliser la même instance de la classe isentropic_polytropic_perfect_gaz que pour l'exemple précédent
calculette=Mo.PyTree_operator(modeles)
arbre=C.convertFile2PyTree('Median_ite_0059100.hdf')
# print("Arbre avant opérations :")
# I.printTree(arbre)
calculette.set_tree(arbre)
calculette.compute('wallTangentialVelocity')
calculette.compute('wallNormalVelocity')
# Si on travaille avec les coordonnées
calculette.set_dataLocalization('Vertex')
calculette.compute('groundCoordinateX')
# Et s'il manque des opérations dans la chaîne...
calculette.set_dataLocalization('CellCenter')
calculette.compute('usefulEnhancedProtection')
# print("Arbre après opérations.")
# I.printTree(arbre)
C.convertPyTree2File(arbre,'sortie_cas_4.hdf')
















sys.stdout.write("\n\n\n\n")
sys.stdout.write("#======================================================================#\n")
sys.stdout.write("# Cinquième application : je souhaite changer de modèle de turbulence  #\n")
sys.stdout.write("# avant de relancer mon calcul, et prendre en compte une opération     #\n")
sys.stdout.write("# supplémentaire extérieure aux classes modèles                        #\n")
sys.stdout.write("#======================================================================#\n")


modeles=build_models(
  {
    'Sutherland_1893':dict(),
    'isentropic_polytropic_perfect_gaz':dict(),
    'Smith_1994':dict(),
    'Spalart-Allmaras_1992':dict(),
  },
)
modeles[2].set_iterations_max_number(100)
modeles[3].set_iterations_max_number(100)

# Fonction unitaire à ajouter au calcul
incidence=np.radians(16.)
def groundVelocityX_fromVelocityComponents(VelocityX,VelocityY):
  groundVelocityX=np.cos(incidence)*VelocityX+np.sin(incidence)*VelocityY
  return groundVelocityX

modeles[0].supplyExternalOperations(variable_name='groundVelocityX',arguments=['VelocityX','VelocityY'],operation=groundVelocityX_fromVelocityComponents)


prisme=Mo.operator(modeles)

prisme.set_dataset(
  dict(
    Viscosity_EddyMolecularRatio=12.5,
    TurbulenceIntensity=1.,
    Density=1.03,
    MomentumX=55.,
    MomentumY=7.5,
    MomentumZ=0.,
    Temperature=277.,
    TurbulentDistance=1.e-3,
  )
)
variables_a_calculer=['Density','TurbulentEnergyKinetic','ViscosityMolecular','chi','TurbulentLengthScale','TurbulentSANuTilde','groundVelocityX']
for variable_a_compute in variables_a_calculer:
  prisme.compute(variable_a_compute)
  print(prisme.get_dataset([variable_a_compute]))



variables_a_calculer=['Viscosity_EddyMolecularRatio']
for variable_a_compute in variables_a_calculer:
  prisme.eraseData(variable_a_compute)
  prisme.compute(variable_a_compute)
  print(prisme.get_dataset([variable_a_compute]))




#__________Fin du fichier test_Modeles.py______________________________#
