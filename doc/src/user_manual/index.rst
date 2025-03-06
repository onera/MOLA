###########
User manual
###########

For most of the functions provided by Miles, two levels of API are available.
The **higher-level API** aims at enabling *one-liner* functions calls, mimicking 
the behavior of the 
`elsA tool chain (etc) <https://elsa-doc.onera.fr/restricted/MU_tuto/Doc_v5.2.01/MU_Annexe/etc/index.html#main>`_, 
using an integrated atomic processing and relying on default values to reduce the 
amount of user input in order to keep pre-processing as simple as possible.

The **lower-level API** is intended for advanced uses and enables finer option 
tuning, the easier injection of user code in the pre-processing at the cost of a 
more verbose code. The low-level API is based on an object-oriented approach 
while the higher-level API wraps these objects in a functional fashion.

.. toctree::
   :maxdepth: 2

   environment
   workflow/introduction
   workflow/workflows_by_application/index
   workflow/inputs
   workflow/manager
   commands