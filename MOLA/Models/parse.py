#!/bin/bash/python
#Encoding:UTF8
"""
Author : Michel Bouchard
Contact : michel.bouchard@onera.fr, michel-j.bouchard@intradef.gouv.fr
Name : Models.parse.py
Description : Submodule that contains a register of all predefined available models and their parameters
History :
Date       | version | git rev. | Comment
___________|_________|__________|_______________________________________#
2021.12.14 | v1.0.00 |          | Creation for MOLA
           |         | ea8e5b3  |

"""
debug_parse=False


from . import turbulence as Mtu
from . import thermodynamics as Mth






existing_models={
# Thermodynamics
  'continuous_medium'                 : Mth.continuous_medium,
  'perfect_gaz'                       : Mth.perfect_gaz,
  'Sutherland_1893'                   : Mth.Sutherland_1893,
  'polytropic_perfect_gaz'            : Mth.polytropic_perfect_gaz,
  'isentropic_polytropic_perfect_gaz' : Mth.isentropic_polytropic_perfect_gaz,

# Turbulence
  'Boussinesq_1877'       : Mtu.Boussinesq_1877,
  'Spalart-Allmaras_1992' : Mtu.Spalart_Allmaras_1992,
  'Deck-Renard_2020'      : Mtu.Deck_Renard_2020,
  'Deck_2012'             : Mtu.Deck_2012,
  # 'Menter_1994'           : Mtu.Kolmogorov_1942,
  # 'Menter_1994_BSL'       : Mtu.Kolmogorov_1942,
  # 'Menter_1994_SST'       : Mtu.Kolmogorov_1942,
  # 'Wilcox_1988'           : Mtu.Kolmogorov_1942,
  # 'Kok_2005'              : Mtu.Kolmogorov_1942,
  # 'Wilcox_2006'           : Mtu.Kolmogorov_1942,
  'Menter-Langtry_2009'   : Mtu.Menter_Langtry_2009,
  'Smith_1994'            : Mtu.Smith_1994,
}

predefined_parameters:{
  'isentropic_air':{
    'Sutherland_1983':dict(
      Cs_Sutherland=[1.716e-5,273.15,110.4],
    ),
    'isentropic_polytropic_perfect_gaz':dict(
      r_perfect_gaz=287.058,
      gamma_Laplace=1.4,
    ),
  },
  'air':{
    'Sutherland_1983':dict(
      Cs_Sutherland=[1.716e-5,273.15,110.4],
    ),
    'perfect_gaz':dict(
      r_perfect_gaz=287.058,
    ),
  },
}

def build_model(model_type,model_parameters=dict()):
  """
  Builds a referenced model with given parameters

  Arguments
  ---------
    model_type : string, reference of the model in the <existing_models>
    model_parameters : dict(), parameters to be passed to the model constructor

  Returns
  -------
    model : .base.model object created from given type and parameters
  """
  return existing_models[model_type](**model_parameters)

def build_predefinedModels(macromodel_name):
  """
  Builds a referenced list of models with predefined parameters

  Arguments
  ---------
    macromodel_name : string, reference of the macromodel in the <predefined_macromodels> dictionary

  Returns
  -------
    model: list(.base.model) created from predefined types and parameters
  """
  models=list()
  predefined_macromodel=predefined_macromodels.get(macromodel_name)
  if predefined_macromodel:
    for (model_type,model_parameters) in predefined_macromodel.items():
      models.append(build_model(model_type,model_parameters))
  return models

def build_models(models_descp=dict()):
  """
  Builds several models according to the keys and values of the <models_descp> dictionary.

  Argument
  --------
    models_descp : dict(string=<parameter>), dictionary of the models to build.

  Returns
  -------
    models : list(.base.model), list of the created models using the parameters in <models_descp>.
  """
  models=list()
  for (model_type,model_parameters) in models_descp.items():
    models.append(build_model(model_type,model_parameters))
  return models


#__________End of file parse.py________________________________________#
