import os
import pytest
import numpy as np

from treelab import cgns

class CylinderMeshBuilder():

    def __init__(self, n_pts_dir = 14, n_sectors = 12, rmin=0.1, rmax=1.0,
            zones_family_name = 'Propeller',hub_family_name = 'HUB',
            blade_family_name = 'BLADE', farfield_family_name = 'FARFIELD',
            interface_boundary = ''):
        
        assert n_pts_dir > 13 # otherwise RSD_L2_rh == 0 and elsa stops at it=1
        
        self.n_pts_dir = n_pts_dir
        self.n_sectors = n_sectors
        self.rmin = rmin
        self.rmax = rmax
        self.zones_family_name = zones_family_name
        self.hub_family_name = hub_family_name
        self.blade_family_name = blade_family_name
        self.farfield_family_name = farfield_family_name

        self.imin = np.array([[1,1],[1,self.n_pts_dir],[1,self.n_pts_dir]], dtype=np.int32, order='F')
        self.imax = np.array([[self.n_pts_dir,self.n_pts_dir],[1,self.n_pts_dir],[1,self.n_pts_dir]], dtype=np.int32, order='F')
        self.jmin = np.array([[1,self.n_pts_dir],[1,1],[1,self.n_pts_dir]], dtype=np.int32, order='F')
        self.jmax = np.array([[1,self.n_pts_dir],[self.n_pts_dir,self.n_pts_dir],[1,self.n_pts_dir]], dtype=np.int32, order='F')
        self.kmin = np.array([[1,self.n_pts_dir],[1,self.n_pts_dir],[1,1]], dtype=np.int32, order='F')
        self.kmax = np.array([[1,self.n_pts_dir],[1,self.n_pts_dir],[self.n_pts_dir,self.n_pts_dir]], dtype=np.int32, order='F')

        self.interface_boundary = interface_boundary
        self.deduce_farfield_boundaries()

    def deduce_farfield_boundaries(self):
        
        if not self.interface_boundary:
            self.farfield_boundaries = ['imin','imax','jmax']

        elif self.interface_boundary == 'imin':
            self.farfield_boundaries = ['imax','jmax']

        elif self.interface_boundary == 'imax':
            self.farfield_boundaries = ['imin','jmax']

        else: 
            raise AttributeError("interface_boundary must be imin, imax, or empty str")

    def get_tree(self):
        self.generate_block()
        self.tag_blade()
        self.tag_hub()
        self.tag_farfield()
        self.tag_interface()
        self.connect()
        self.tag_zone()

        assert isinstance(self.tree, cgns.Tree)
        return self.tree

    def generate_block(self):
        half_angle = 180.0 / self.n_sectors    

        x, y, z = np.meshgrid( np.linspace(0, 1, self.n_pts_dir),
                            np.linspace(self.rmin, self.rmax, self.n_pts_dir),
                            np.linspace(0, 1, self.n_pts_dir), indexing='ij')

        azimut = np.linspace(-half_angle,+half_angle,self.n_pts_dir)
        radius = y*1.0
        azimut_rad = np.deg2rad(azimut)

        y[:] = -radius * np.sin(azimut_rad)
        z[:] =  radius * np.cos(azimut_rad)

        block = cgns.newZoneFromArrays( 'blk-'+self.zones_family_name, ['x','y','z'], [ x,  y,  z ])

        self.tree = cgns.Tree(Base=block)
        self.zone = self.tree.zones()[0]
        self.base = self.tree.bases()[0]
    
    def tag_blade(self):
        name = self.blade_family_name 
        zone_bc = self.add_zone_bc_if_absent()
        
        point_range_blade = np.array([[1,7],[1,1],[1,self.n_pts_dir]],
                                     dtype=np.int32, order='F')
        self.add_bc_to(zone_bc, name, point_range_blade)
        self.add_family_to_base(name)

    def add_zone_bc_if_absent(self):
        zone_bc = self.zone.get(Type='ZoneBC_t', Depth=1)
        if not zone_bc:
            zone_bc = cgns.Node(Name='ZoneBC', Type='ZoneBC_t')
            zone_bc.attachTo(self.zone)

        return zone_bc
    
    def add_bc_to(self, zone_bc, family_tag_name : str, point_range : np.ndarray):
        bc = cgns.Node(Name=family_tag_name,Type='BC_t',Value='FamilySpecified')
        bc.attachTo(zone_bc, override_sibling_by_name=False)

        point_range_node = cgns.Node(Name='PointRange', Type='IndexRange_t', Value=point_range)
        point_range_node.attachTo(bc)

        family_name_node = cgns.Node(Name='FamilyName', Type='FamilyName_t', Value=family_tag_name)
        family_name_node.attachTo(bc)

    def add_family_to_base(self, name):
        family_node = cgns.Node(Name=name, Type='Family_t')
        family_node.attachTo(self.base)

        family_bc_node = cgns.Node(Name='FamilyBC', Type='FamilyBC_t', Value='BCWall')
        family_bc_node.attachTo(family_node)

    def tag_hub(self):
        name = self.hub_family_name
        zone_bc = self.add_zone_bc_if_absent()
        
        point_range_hub = np.array([[7,self.n_pts_dir],[1,1],[1,self.n_pts_dir]],
                                     dtype=np.int32, order='F')
        self.add_bc_to(zone_bc, name, point_range_hub)
        self.add_family_to_base(name)

    def tag_farfield(self):
        name = self.farfield_family_name
        zone_bc = self.add_zone_bc_if_absent()
        
        for bnd in self.farfield_boundaries:
            point_range = getattr(self,bnd)
            self.add_bc_to(zone_bc, name, point_range)
        
        self.add_family_to_base(name)
    
    def tag_interface(self):
        if not self.interface_boundary: return
        name = self.zones_family_name+"_INTERFACE"
        zone_bc = self.add_zone_bc_if_absent()
        
        point_range = getattr(self,self.interface_boundary)
        self.add_bc_to(zone_bc, name, point_range)
        
        self.add_family_to_base(name)

    def connect(self):
        self.add_grid_connectivity_1to1_at('kmin')
        self.add_grid_connectivity_1to1_at('kmax')

    def add_grid_connectivity_1to1_at(self, location : str):
        zone_gc = self.add_zone_grid_connectivity_if_absent()
        

        if location == 'kmin':
            point_range = self.kmin
            point_range_donor = self.kmax
        elif location == 'kmax':
            point_range = self.kmax
            point_range_donor = self.kmin
        else:
            raise NotImplementedError(f"location={location}")

        point_range_node = cgns.Node(Name='PointRange', Type='IndexRange_t', Value=point_range)
        point_range_donor_node = cgns.Node(Name='PointRangeDonor', Type='IndexRange_t', Value=point_range_donor)
        transform_node = cgns.Node(Name='Transform', Type='IndexRange_t', Value=np.array([1,2,3],dtype=np.int32))
        transform_node[3] = '"int[IndexDimension]"'

        gc1to1 = cgns.Node(Name='match_'+location, Value=self.zone.name(), Type='GridConnectivity1to1_t')
        gc1to1.attachTo(zone_gc)

        for n in (point_range_node, point_range_donor_node, transform_node):
            n.attachTo(gc1to1)

        gcp = self.build_grid_connectivity_property()
        gcp.attachTo(gc1to1)


    def add_zone_grid_connectivity_if_absent(self):
        zone_gc = self.zone.get(Type='ZoneGridConnectivity_t', Depth=1)
        if not zone_gc:
            zone_gc = cgns.Node(Name='ZoneGridConnectivity', Type='ZoneGridConnectivity_t')
            zone_gc.attachTo(self.zone)

        return zone_gc
    
    def build_grid_connectivity_property(self):
        gcp = cgns.Node(Name='GridConnectivityProperty', Type='GridConnectivityProperty_t')
        periodic = cgns.Node(Name='Periodic', Type='Periodic_t')
        rot_center = cgns.Node(Name='RotationCenter', Value=np.array([0,0,0],dtype=np.float64))
        rot_angle = cgns.Node(Name='RotationAngle', Value=np.array([np.deg2rad(-360/self.n_sectors),0,0],dtype=np.float64))
        units = cgns.Node(Name='DimensionalUnits', Type='DimensionalUnits_t',
                          Value=['Kilogram','Meter','Second','Kelvin','Radian'])
        translation = cgns.Node(Name='Translation', Value=np.array([0,0,0],dtype=np.float64))

        gcp.addChild(periodic)
        periodic.addChild(rot_center)
        periodic.addChild(rot_angle)
        rot_angle.addChild(units)
        periodic.addChild(translation)

        return gcp
    
    def tag_zone(self):
        family_at_base = cgns.Node(Name=self.zones_family_name, Type='Family_t')
        family_at_base.attachTo(self.base)
        family_at_zone = cgns.Node(Name='FamilyName', Value=self.zones_family_name, Type='FamilyName_t')
        family_at_zone.attachTo(self.zone)

@pytest.fixture
def workflow_sector_params():
    mesh_builder = CylinderMeshBuilder()
    mesh = mesh_builder.get_tree()

    params = dict(
        RawMeshComponents=[
            dict(
                Name='Base',
                Source=mesh,
            )
        ],

        Flow=dict(
            Density = 1.0,
            Temperature = 300.0,
            Velocity = 10.,
        ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Numerics = dict(
            NumberOfIterations=4,
            MinimumNumberOfIterations=3,
            CFL=1.0,
        ),

        BoundaryConditions=[
            dict(Family='BLADE',   Type='WallViscous'),
            dict(Family='HUB',    Type='WallInviscid'),
            dict(Family='FARFIELD', Type='Farfield'),
        ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory='.',
            Scheduler = 'local',
            ),
        )

    return params


def write_mesh_and_read_it_again(workflow_sector_params, dir_path):
    t : cgns.Tree = workflow_sector_params["RawMeshComponents"][0]["Source"]
    mesh_path = os.path.join(str(dir_path),'mesh.cgns')
    t.save(mesh_path)
    workflow_sector_params["RawMeshComponents"][0]["Source"] = mesh_path
