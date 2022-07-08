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

from .. import InternalShortcuts as J

from . import ShortCuts as SJ
from . import Models as SM




def Freq_Phi(t, RPM, MODE):

    DictStructParam = J.get(t, '.StructuralParameters')

    freq = MODE.LIST_VARI_ACCES()['FREQ']
    freq = np.array(freq[:DictStructParam['ROMProperties']['NModes'][0]])

    PHImatrix = np.zeros((DictStructParam['MeshProperties']['Nddl'][0],DictStructParam['ROMProperties']['NModes'][0]))

    #ModZones = []
    for Mode in range(DictStructParam['ROMProperties']['NModes'][0]):
        try:
            tabmod_T = CREA_TABLE(RESU=_F(RESULTAT= MODE,
                                  NOM_CHAM='DEPL',
                                  NOM_CMP= ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                  NUME_ORDRE=Mode+1,
                                  TOUT = 'OUI'),
                                  TYPE_TABLE='TABLE',
                                  TITRE='Table_Modes',);
        except:
            tabmod_T = CREA_TABLE(RESU=_F(RESULTAT= MODE,
                                  NOM_CHAM='DEPL',
                                  NOM_CMP= ('DX','DY','DZ'),
                                  NUME_ORDRE=Mode+1,
                                  TOUT = 'OUI'),
                                  TYPE_TABLE='TABLE',
                                  TITRE='Table_Modes',);
            print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

        ModZone = SJ.CreateNewSolutionFromAsterTable(t, FieldDataTable= tabmod_T,
                                                        ZoneName = 'Mode%s_'%Mode+str(np.round(RPM)),
                                                        FieldName = 'Mode',
                                                        )

        PHImatrix[:,Mode] = SM.VectFromAsterTable2Full(t, tabmod_T)

        DETRUIRE (CONCEPT = _F (NOM = (tabmod_T),
                            ),
              INFO = 1,
              )



        #ModZones.append(ModZone)

        try:
          I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
        except:
          t = I.merge([t, C.newPyTree(['ModalBases', []])])
          I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
        I._addChild(I.getNodeFromName(t, 'ModalBases'), I.createNode('Freq_%sRPM'%np.round(RPM,2), 'DataArray_t', value = freq))





    t = SJ.AddFOMVars2Tree(t, RPM, Vars = [PHImatrix], VarsName = ['PHI'], Type = '.AssembledMatrices')


    C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.cgns', 'bin_adf')
    #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.tp', 'bin_tp')

#    print(len(ModZones))
#    XX
#    VectUsOmega = VectFromAsterTable2Full(t, tabmod_T)
#
#    t = SJ.AddFOMVars2Tree(t, RPM, Vars = [VectUsOmega],
#                                   VarsName = ['Us'],
#                                   Type = '.AssembledVectors',
#                                   )
#

    # Tableau complet des modes, coordonnees modales,... :
#    print(tabmod_T.EXTR_TABLE().values().keys())
#    XXX

#    depl_mod = tabmod_T.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']
#
#    # Liste des valeurs de la table:
#    depl_list = []
#    depl_list_Names = ['ModeX', 'ModeY', 'ModeZ']
#    depl_list.append(depl_mod.values()['DX'][:])
#    depl_list.append(depl_mod.values()['DY'][:])
#    depl_list.append(depl_mod.values()['DZ'][:])
#    #dep_md_X = depl_mod.values()['DX'][:]
#    #dep_md_Y = depl_mod.values()['DY'][:]
#    #dep_md_Z = depl_mod.values()['DZ'][:]
#
#    elif DictStructParam['MeshProperties']['ddlElem'][0] == 6:
#
#        tabmod_T = CREA_TABLE(RESU=_F(RESULTAT= MODE,
#                              NOM_CHAM='DEPL',
#                              NOM_CMP= ('DX','DY','DZ','DRX','DRY','DRZ'),
#                              TOUT='OUI',),
#                              TYPE_TABLE='TABLE',
#                              TITRE='Table_Modes',);
#
#        # Tableau complet des modes, coordonnees modales,... :
#
#        depl_mod = tabmod_T.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ','DRX','DRY','DRZ']
#
#        # Liste des valeurs de la table:
#        depl_list = []
#        depl_list.append(depl_mod.values()['DX'][:])
#        depl_list.append(depl_mod.values()['DY'][:])
#        depl_list.append(depl_mod.values()['DZ'][:])
#        depl_list.append(depl_mod.values()['DRX'][:])
#        depl_list.append(depl_mod.values()['DRY'][:])
#        depl_list.append(depl_mod.values()['DRZ'][:])
#        depl_list_Names = ['ModeX', 'ModeY', 'ModeZ','ModeThetaX','ModeThetaY','ModeThetaZ']
#
#    # Create a matrix with the base and save it into the .AssembledMatrices node:


#    t, PHI = SJ.BuildMatrixFromComponents(t, 'PHI', depl_list)
#
#    # Extract the nodes coordinates:
#    NodeZones = []
#    for NMode in range(DictStructParam['ROMProperties']['NModes'][0]):
#
#        Coord_list = []
#        for pos in range(len(depl_list)):
#
#            Coord_list.append(PHI[pos::len(depl_list), NMode])
#
#        #CoordX = dep_md_X[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
#        #CoordY = dep_md_Y[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
#        #CoordZ = dep_md_Z[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
#
#        NewZone,_ = SJ.CreateNewSolutionFromNdArray(t,
#                                       FieldDataArray= Coord_list,
#                                       ZoneName = str(np.round(RPM, 2))+'Mode'+str(NMode),
#                                       FieldNames = depl_list_Names,
#                                       Depl = True)
#
#        I.createChild(NewZone, 'Freq', 'DataArray_t', freq[NMode])
#
#        NodeZones.append(NewZone)
#
#
#    I.addChild(I.getNodeFromName(t, 'ModalBases'), NodeZones)
#
#    DETRUIRE (CONCEPT = _F (NOM = (tabmod_T),
#                            ),
#              INFO = 1,
#              )

    return t


def CalcLNM(t, RPM, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')

    if not DictStructParam['ROMProperties']['RigidMotion']:

        MODESR = CALC_MODES(MATR_RIGI = kwargs['Komeg2'],
                            MATR_MASS = kwargs['MASS1'],
                            OPTION = 'PLUS_PETITE',
                            CALC_FREQ = _F(NMAX_FREQ = DictStructParam['ROMProperties']['NModes'],),
                            VERI_MODE = _F(SEUIL = 1.E-3, STOP_ERREUR = 'NON'),
                            )
    elif DictStructParam['ROMProperties']['RigidMotion']:

        MODESR = CALC_MODES(MATR_RIGI = kwargs['Komeg2'],
                            MATR_MASS = kwargs['MASS1'],
                            OPTION = 'BANDE',
                            CALC_FREQ = _F(FREQ = (-1,250.),),
                            VERI_MODE = _F(SEUIL = 1.E-3, STOP_ERREUR = 'NON'),
                            #SOLVEUR_MODAL = _F(METHODE = 'TRI_DIAG',
                            #                   MODE_RIGIDE = 'OUI',),
                          )


    #MODESR = CALC_MODES(affe_modes)
    MODE   = NORM_MODE(MODE = MODESR,
                       NORME = 'TRAN_ROTA',)

    t = Freq_Phi(t, RPM, MODE)


    DETRUIRE (CONCEPT = _F (NOM = (MODESR, MODE),
                            ),
              INFO = 1,
              )

    return t
