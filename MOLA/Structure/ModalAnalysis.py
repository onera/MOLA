'''
MOLA - ModalAnalysis.py 

This module defines some handy shortcuts of the Cassiopee's
Converter.Internal module.

Furthermore it provides functions that interact with Code_Aster.

First creation:
31/05/2021 - M. Balmaseda
'''
FAIL  = '\033[91m'
GREEN = '\033[92m' 
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'

try:
    #Code Aster:  
    from code_aster.Cata.Commands import *
    from code_aster.Cata.Syntax import _F, CO
except:
    print(WARN + 'Code_Aster modules are not loaded!!' + ENDC)

# System modules
import numpy as np
# MOLA modules
import Converter.Internal as I
import Converter.PyTree as C
import MOLA.InternalShortcuts as J

import MOLA.Structure.ShortCuts as SJ




def Freq_Phi(t, RPM, MODE):
    
    DictStructParam = J.get(t, '.StructuralParameters')

    freq = MODE.LIST_VARI_ACCES()['FREQ']
    
    tabmod_T = CREA_TABLE(RESU=_F(RESULTAT= MODE,
                          NOM_CHAM='DEPL',
                          NOM_CMP= ('DX','DY','DZ'),
                          TOUT='OUI',),
                          TYPE_TABLE='TABLE',
                          TITRE='Table_Modes',);

    # Tableau complet des modes, coordonnees modales,... :
    
    depl_mod = tabmod_T.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']
     
    # Liste des valeurs de la table:
    
    dep_md_X = depl_mod.values()['DX'][:]  
    dep_md_Y = depl_mod.values()['DY'][:]
    dep_md_Z = depl_mod.values()['DZ'][:]

    # Create a matrix with the base and save it into the .AssembledMatrices node:
    #PHI = SJ.BuildMatrixFromCoordinatesXYZ(DictStructParam, CoordinateX, CoordinateY, CoordinateZ)
    


    # Extract the nodes coordinates:
    NodeZones = []
    for NMode in range(DictStructParam['ROMProperties']['NModes'][0]):
        
        CoordX = dep_md_X[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
        CoordY = dep_md_Y[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
        CoordZ = dep_md_Z[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
        
        NewZone,_ = SJ.CreateNewSolutionFromNdArray(t, 
                                       FieldDataArray= [CoordX, CoordY , CoordZ], 
                                       ZoneName = str(np.round(RPM, 2))+'Mode'+str(NMode), 
                                       FieldNames = ['ModeX', 'ModeY', 'ModeZ'], 
                                       Depl = True)

        I.createChild(NewZone, 'Freq', 'DataArray_t', freq[NMode])

        NodeZones.append(NewZone)

    
    I.addChild(I.getNodeFromName(t, 'ModalBases'), NodeZones)
    
    DETRUIRE (CONCEPT = _F (NOM = (tabmod_T),
                            ), 
              INFO = 1,
              )

    return t

    
def CalcLNM(t, RPM, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')

    MODESR = CALC_MODES(MATR_RIGI = kwargs['Komeg2'],
                        MATR_MASS = kwargs['MASS1'],
                        OPTION = 'PLUS_PETITE',
                        CALC_FREQ = _F(NMAX_FREQ = DictStructParam['ROMProperties']['NModes'],),
                        VERI_MODE = _F(SEUIL = 1.E-3, STOP_ERREUR = 'NON'),
                        )
    MODE   = NORM_MODE(MODE = MODESR, 
                       NORME = 'TRAN',)

    t = Freq_Phi(t, RPM, MODE)
    

    DETRUIRE (CONCEPT = _F (NOM = (MODESR, MODE),
                            ), 
              INFO = 1,
              )

    return t
