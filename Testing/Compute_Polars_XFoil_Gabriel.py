#export PYTHONPATH=$PYTHONPATH:/home/lbernard/MOLA/v1.9/

import os
import MOLA.XFoil as XFoil
import Converter.PyTree as CP
import Converter.Internal as IP
import math
import numpy as Nu

# XFoil.XFOIL_EXEC = '/stck/greboul/CODES/Xfoil/bin/xfoil_VISIO'
#XFoil.XFOIL_EXEC = os.path.join('D:','lbernard','XFoil','xfoil.exe')#'D:\lbernard\XFoil'

doXfoil=True
polars2BeRead='Polars.cgns' 
airfoilFilename='NACA0012'
ainf=340.294
muinf=1.81206e-5
rhoinf=1.225
RPM=6000
Rmin=0.1#0.00625
Rmax=0.4#0.125
MachMin=Rmin*RPM*math.pi/30./ainf
MachMax=Rmax*RPM*math.pi/30./ainf
MachDelta=0.05
MachS= Nu.arange(MachMin,MachMax+MachDelta,MachDelta)[:2]
ChordMean=0.025
ReS=Nu.array((rhoinf*ChordMean*ainf)/muinf)*MachS
AlphaS=[-6.,-4.,-2.,0.,2.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,18.,21.]
# AlphaS=[0.,2.,4.]

if doXfoil:
	PolarsDict = XFoil.computePolars(airfoilFilename, ReS, MachS, AlphaS,
									Ncr=9,PyZonePolarKind='Struct_AoA_Mach',
									itermax=60,	 LogStdOutErr=False,
									lowMachForced=0.3, storeFoilVariables=True,
									rediscretizeFoil=False,
									removeTmpXFoilFiles=True)
	PolarsCGNS = XFoil.convertXFoilDict2PyZonePolar(PolarsDict, airfoilFilename)
	CP.convertPyTree2File(PolarsCGNS,  polars2BeRead)
	CP.convertPyTree2File(PolarsCGNS, 'Polars.tp')


PolarsCGNS=CP.convertFile2PyTree(polars2BeRead)
BigAngleOfAttackCd=Nu.zeros(18)
BigAngleOfAttackCl=Nu.zeros(18)
BigAngleOfAttackCm=Nu.zeros(18)
BigAngleOfAttackCd=PolarsCGNS[2][1][2][0][2][3][2][0][1]
BigAngleOfAttackCl=PolarsCGNS[2][1][2][0][2][3][2][2][1]
BigAngleOfAttackCm=PolarsCGNS[2][1][2][0][2][3][2][1][1]
AoA = IP.getNodeFromName(PolarsCGNS, 'AoA')[1][:,0]
Cl  = IP.getNodeFromName(PolarsCGNS,  'Cl')[1][:,:]
Cd  = IP.getNodeFromName(PolarsCGNS,  'Cd')[1][:,:]
Cm  = IP.getNodeFromName(PolarsCGNS,  'Cm')[1][:,:]
Na,Nm=IP.getZoneDim(IP.getNodeFromType(PolarsCGNS, 'Zone_t'))[1:3]
MachNumb=Nu.zeros(Nm)
MachNumb=PolarsCGNS[2][1][2][0][2][2][2][1][1]

print 'AoA range: ',AoA
print 'Mach range: ',MachNumb
file = open('HOST_Profil_'+airfoilFilename,'w')
file.write('   36     '+airfoilFilename+'\n')
file.write('1 Cl \n')
file.write(str(Na)+'  '+str(Nm)+'\n')
for i in xrange (Na) :
	data="%10.5f" %(AoA[i])
	file.write(data)
file.write('\n')
for j in xrange (Nm) :
	data="%10.5f" %(MachNumb[j])
	file.write(data)
file.write('\n')
for i in xrange (Na):
	for j in xrange(Nm):
		data="%10.5f" %(Cl[i][j])
		file.write(data)
	file.write('\n')
file.write('9 \n')
file.write('  30.00000  40.00000  60.00000  80.00000 100.00000 120.00000 140.00000 160.00000 180.00000 \n')
for i in xrange (9):
	data="%10.5f" %(BigAngleOfAttackCl[8-i])
	file.write(data)
file.write('\n')
file.write('9 \n')
file.write('-180.00000-160.00000-140.00000-120.00000-100.00000 -80.00000 -60.00000 -40.00000 -30.00000 \n')
for i in xrange (9):
	data="%10.5f" %(BigAngleOfAttackCl[17-i])
	file.write(data)
file.write('\n')
file.write('1 Cd \n')
file.write(str(Na)+'  '+str(Nm)+'\n')
for i in xrange (Na) :
	data="%10.5f" %(AoA[i])
	file.write(data)
file.write('\n')
for j in xrange (Nm) :
	data="%10.5f" %(MachNumb[j])
	file.write(data)
file.write('\n')
for i in xrange (Na):
	for j in xrange(Nm):
		data="%10.5f" %(Cd[i][j])
		file.write(data)
	file.write('\n')
file.write('9 \n')
file.write('  30.00000  40.00000  60.00000  80.00000 100.00000 120.00000 140.00000 160.00000 180.00000 \n')
for i in xrange (9):
	data="%10.5f" %(BigAngleOfAttackCd[8-i])
	file.write(data)
file.write('\n')
file.write('9 \n')
file.write('-180.00000-160.00000-140.00000-120.00000-100.00000 -80.00000 -60.00000 -40.00000 -30.00000 \n')
for i in xrange (9):
	data="%10.5f" %(BigAngleOfAttackCd[17-i])
	file.write(data)
file.write('\n')
file.write('1 Cm \n')
file.write(str(Na)+'  '+str(Nm)+'\n')
for i in xrange (Na) :
	data="%10.5f" %(AoA[i])
	file.write(data)
file.write('\n')
for j in xrange (Nm) :
	data="%10.5f" %(MachNumb[j])
	file.write(data)
file.write('\n')
for i in xrange (Na):
	for j in xrange(Nm):
		data="%10.5f" %(Cm[i][j])
		file.write(data)
	file.write('\n')
file.write('9 \n')
file.write('  30.00000  40.00000  60.00000  80.00000 100.00000 120.00000 140.00000 160.00000 180.00000 \n')
for i in xrange (9):
	data="%10.5f" %(BigAngleOfAttackCm[8-i])
	file.write(data)
file.write('\n')
file.write('9 \n')
file.write('-180.00000-160.00000-140.00000-120.00000-100.00000 -80.00000 -60.00000 -40.00000 -30.00000 \n')
for i in xrange (9):
	data="%10.5f" %(BigAngleOfAttackCm[17-i])
	file.write(data)
file.write('\n')
file.write('COEFFICIENT (C*L/NU)I0 (OU BIEN REYNOLDS/MACH) ............    '+str(ReS[0]/MachS[0]))



file.close()
