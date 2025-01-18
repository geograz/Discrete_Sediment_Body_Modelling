# -*- coding: utf-8 -*-
"""
Source code to the publication
"Stochastic 3D modelling of discrete sediment bodies for geotechnical applications"
by G.H. Erharter, F. Tschuchnigg, G. Poscher in the journal
Applied Computing and Geosciences
DOI: https://doi.org/10.1016/j.acags.2021.100066

Coding: G.H. Erharter
"""

import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
import trimesh as tri
from typing import Iterable


class SedimentaryOverflowError(Exception):
    ''' custom error if the volume of sedimentary bodies is too big '''
    pass


class modifier:
    ''' class with functions that modify the surrogate point clouds'
    properties '''

    def __init__(self):
        pass

    def correct_direction(self, direction: int, dip: int) -> int:
        ''' as the orientation of each sediment body is subject to some
        randomnes, invalid orientations might occur that should be corrected'''
        if direction > 180:
            direction = direction - 180
        elif direction < 0:
            direction = direction + 180
        
        if dip > 90:
            dip = 90 - (dip - 90)
        elif dip < 0:
            dip = 0 + dip * -1
        
        return direction, dip

    def rotate_pc(self, DIRECTION: int, DIR_STD: int, DIP: int, DIP_STD: int,
                  point_cloud: np.array) -> np.array:
        ''' function rotates a surrogate point cloud according to the
        preferrred orientation and standard deviation '''
        # apply deviation of main direction
        if DIRECTION == 'random':
            DIRECTION = np.random.uniform(0, 180)
        else:
            DIRECTION = np.random.normal(loc=DIRECTION,
                                         scale=DIR_STD, size=1)[0]

        if DIP == 'random':
            DIP = np.random.uniform(0, 90)
        else:
            DIP = np.random.normal(loc=DIP, scale=DIP_STD, size=1)[0]

        # correct direction if > 180° or < 0°
        DIRECTION, DIP = self.correct_direction(DIRECTION, DIP)

        # first rotate point cloud accoridng to dip around X-axis
        dip_radians = np.radians(-DIP)
        dip_rotation_axis = np.array([0, 1, 0])
        dip_rotation_vector = dip_radians * dip_rotation_axis
        rotation = Rotation.from_rotvec(dip_rotation_vector)
        point_cloud = rotation.apply(point_cloud)

        # secondly rotate around vertical center axis of point cloud
        DIRECTION = -(90 + DIRECTION)
        dir_radians = np.radians(DIRECTION)
        dir_rotation_axis = np.array([0, 0, 1])
        dir_rotation_vector = dir_radians * dir_rotation_axis
        rotation = Rotation.from_rotvec(dir_rotation_vector)
        point_cloud = rotation.apply(point_cloud)

        return point_cloud


class generator(modifier):
    ''' class of functions that generates the surrogate point clouds '''

    def __init__(self, SCALE: Iterable, POINT_COUNT: int, DIRECTION: int,
                 DIR_STD: int, DIP: int, DIP_STD: int):
        self.SCALE = SCALE
        self.POINT_COUNT = POINT_COUNT
        self.DIRECTION = DIRECTION
        self.DIR_STD = DIR_STD
        self.DIP = DIP
        self.DIP_STD = DIP_STD

    def generate_centers(self, lower_left_corner: Iterable,
                         upper_right_corner: Iterable,
                         n_bodies: int) -> np.array:
        ''' function generates the centers of the surrogate point clouds '''
        centers_x = np.random.uniform(lower_left_corner[0],
                                      upper_right_corner[0], n_bodies)
        centers_y = np.random.uniform(lower_left_corner[1],
                                      upper_right_corner[1], n_bodies)
        centers_z = np.random.uniform(lower_left_corner[2],
                                      upper_right_corner[2], n_bodies)

        centers = np.vstack((centers_x, centers_y, centers_z)).T
        return centers

    def generate_pc(self, center: Iterable) -> np.array:
        ''' function generates a surrogate point cloud '''
        x = np.random.normal(loc=center[0], scale=self.SCALE[0],
                             size=self.POINT_COUNT)
        y = np.random.normal(loc=center[1], scale=self.SCALE[1],
                             size=self.POINT_COUNT)
        z = np.random.normal(loc=center[2], scale=self.SCALE[2],
                             size=self.POINT_COUNT)
        xyz = np.vstack((x, y, z)).T
        return xyz

    def generate_clouds(self, centers: np.array, savepath: str) -> np.array:
        ''' function calls other functions that generate the final surrogate
        point clouds with labels that indicate each point cloud's index and
        saves them '''
        clouds = []
        idxs = []

        for i, center in enumerate(centers):
            # create pc at position 0,0,0
            cloud = self.generate_pc([0, 0, 0])
            # rotate point cloud into preferred direction of elongation
            cloud = self.rotate_pc(self.DIRECTION, self.DIR_STD,
                                   self.DIP, self.DIP_STD, cloud)
            cloud += center  # shift pc to new position
            clouds.append(cloud)  # store clouds in list
            idxs.append(np.full(len(cloud), i))  # store indexes in list

        clouds = np.concatenate(clouds)
        idxs = np.concatenate(idxs)

        clouds_labelled = np.hstack((clouds,
                                     np.reshape(idxs, (idxs.shape[0], 1))))
        np.savetxt(savepath, clouds_labelled)

        return clouds


class meshtools:
    ''' class contains functions that generate and edit a mesh around the
    surrogate point clouds '''

    def __init__(self, ALPHA: int, MIN_VOL: int, max_vol: int):
        self.ALPHA = ALPHA
        self.MIN_VOL = MIN_VOL
        self.max_vol = max_vol

    def pc_to_o3dmesh(self, data: np.array) -> o3d.geometry.TriangleMesh:
        ''' function creates an alpha shape - mesh around a given surrogate
        point cloud '''
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(data)
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)

        # See Edelsbrunner and Muecke, “Three-Dimensional Alpha Shapes”, 1994.
        shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=point_cloud,
                                                                              alpha=self.ALPHA, tetra_mesh=tetra_mesh, pt_map=pt_map)

        shape.compute_triangle_normals()
        shape.compute_vertex_normals()
        shape.orient_triangles()
        shape.triangle_normals = o3d.utility.Vector3dVector(np.array(shape.triangle_normals)*-1)
        return shape

    def o3d_to_trimesh(self, mesh: o3d.geometry.TriangleMesh) -> tri.Trimesh:
        ''' convert open3d mesh to trimesh-mesh for volume computation and
        fixing of mesh face orientation '''
        # load temporary mesh and check if normals ar oriented normally
        mesh = tri.load(r'meshes\temp_hull.stl')
        os.remove(r'meshes\temp_hull.stl')
        V = mesh.volume

        # V < 0 ... means that faces were flipped before
        if V < 0:
            tri.repair.fix_inversion(mesh)  # inverse mesh face orientation

        return mesh

    def bbox(self, width: int, height: int, depth: int,
             translation: Iterable) -> None:
        ''' function saves bounding box around centers of surrogate point
        clouds and sediment - body meshes '''
        bbox = o3d.geometry.TriangleMesh.create_box(width=width,
                                                    height=height,
                                                    depth=depth)
        bbox.compute_vertex_normals()
        bbox.translate(translation)
        o3d.io.write_triangle_mesh(r'meshes\bbox.stl', bbox)

    def save_mesh(self, o3d_mesh: o3d.geometry.TriangleMesh,
                  LLC_ABS: Iterable) -> (tri.Trimesh, float):
        ''' function saves meshes of sediment bodies and computes volumes '''
        o3d_mesh = o3d_mesh.translate(LLC_ABS, relative=True)
        o3d.io.write_triangle_mesh(r'meshes\temp_hull.stl', o3d_mesh)

        mesh = self.o3d_to_trimesh(o3d_mesh)
        V = mesh.volume
        return mesh, V

    def check_mesh(self, mesh, V):
        ''' function checks if mesh geometry is valid and volume is OK '''
        if mesh.is_volume is True and V > self.MIN_VOL and V < self.max_vol:
            return True
        else:
            return False


##############################################################################
# main code
if __name__ == "__main__":

    ##########################################################################
    # fixed variables and (hyper)parameters

    # sediment body hyperparameters
    A_B_C = [10, 5, 3]  # x, y, z scale with respect to the center [m]
    ALPHA = 120  # primary direction of bodies between 0 & 180°, also: 'random'
    BETA = 10  # standard deviation of direction orientation
    GAMMA = 0  # average dip of the sediment bodies
    DELTA = 0  # standard deviation of the average dip of the sediment bodies
    MIN_VOL = 50  # [m³]
    TARGET_FRACTION = 0.10  # target % of volume that consists of sediments
    MAX_FRACTION = 0.35  # maximum % of volume that consists of sediment bodies

    # volume hyperparameters
    N_b = 700  # initial number of surrogate point clouds
    PLUS = 100  # number of bodies that are added in each iteration
    LLC_ABS = (0, 0, 0)  # Left Lower Corner of volume [metric coordinates]
    LLC = [0, 0, 0]  # relative left lower corner of bounding box
    L, W, H = 220, 220, 150  # length width and height of bounding box

    # cluster hyperparameters
    N_p = 500  # number of points of each surrogate point cloud
    MIN_CL_COUNT = 10  # minimum number of generated clusters
    EPS = 5.7  # DBSCAN main parameter / maximum distance between two samples for one to be considered as in the neighborhood of the other
    MIN_SAMPLES = 100  # minimum number of datapoints in one cluster

    # mesh parameters
    ALPHA_c = 5  # ALPHA_c determines "tightness" of  meshes around the clusters
    N_it = 15  # number of iterations for surface smoothing after Taubin (1995)
    N_ia = 5  # number of iterations for simple average surface smoothing

    SEED = 1  # all models for paper were made with SEED = 1

    ##########################################################################
    # computed variables
    ruc = [LLC[0]+L, LLC[1]+W, LLC[2]+H]  # right upper corner
    tot_space = L * W * H  # [m³]
    max_vol = tot_space * TARGET_FRACTION  # [m³]

    ##########################################################################
    # instanciation

    gen = generator(A_B_C, N_p, ALPHA, BETA, GAMMA, DELTA)
    mod = modifier()
    mtls = meshtools(ALPHA_c, MIN_VOL, max_vol)

    ##########################################################################
    # main sediment body generation

    # make folders if necessary
    if not os.path.exists('meshes'):
        os.makedirs('meshes')
    if not os.path.exists('clouds'):
        os.makedirs('clouds')

    # remove old meshes
    for file in os.listdir('meshes'):
        os.remove(fr'meshes\{file}')

    # save boundinx box if desired
    mtls.bbox(width=L, height=W, depth=H, translation=LLC_ABS)

    # fix the seed for reproducibility
    np.random.seed(SEED)

    # initial fraction of volume of bodies to total volume
    perc_volumes = 0

    # loop starts with N_b surrogate point clouds and adds PLUS point clouds
    # after every iteration that doesn't fulfill the minimum vol. fraction
    # -> simple automatic parameter optimization
    while perc_volumes < TARGET_FRACTION:

        # point cloud generation
        centers = gen.generate_centers(LLC, ruc, N_b)
        clouds = gen.generate_clouds(centers, r'clouds\0_clouds_labelled.txt')

        # unsupervised point cloud filtering
        print(f'{N_b} clouds generated')
        # cluster analysis of clouds
        clusterer = DBSCAN(eps=EPS, n_jobs=-1, min_samples=MIN_SAMPLES)
        clusterer.fit(clouds)
        labels = clusterer.labels_

        clouds_clustered = np.hstack((clouds, np.reshape(labels,
                                                         (labels.shape[0], 1))))
        np.savetxt(r'clouds\1_clouds_clustered.txt', clouds_clustered)

        n_clusters = len(np.unique(labels))  # count number of clusters
        print(f'cluster analysis finished. {n_clusters} cluster')

        # mesh generation from clustered pcs
        if n_clusters > MIN_CL_COUNT:

            o3d_meshes = []
            volumes = []
            meshes = []

            for cluster in np.unique(labels):
                # only use clusters and not noise (label = -1)
                if cluster > -1:

                    xyz_ = clouds[np.where(labels == cluster)[0]]

                    o3d_mesh = mtls.pc_to_o3dmesh(xyz_)
                    # filter / smooth mesh
                    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=N_it)
                    o3d_mesh.compute_vertex_normals()

                    if len(meshes) == 0:  # save first temporary mesh
                        mesh, V = mtls.save_mesh(o3d_mesh, LLC_ABS)
                        if mtls.check_mesh(mesh, V) is True:
                            o3d_meshes.append(o3d_mesh)
                            volumes.append(V)
                            meshes.append(mesh)
                    else:
                        # check for mesh intersections with other existing meshes
                        check = [o3d_mesh.is_intersecting(other) for other in o3d_meshes]
                        if True in check:
                            # reduce size of mesh then try again
                            o3d_mesh = o3d_mesh.filter_smooth_simple(number_of_iterations=N_ia)
                            o3d_mesh.compute_vertex_normals()
                            check = [o3d_mesh.is_intersecting(other) for other in o3d_meshes]
                            if True in check:
                                pass  # if still intersecting pass
                            else:
                                print('smaller mesh fits')
                                mesh, V = mtls.save_mesh(o3d_mesh, LLC_ABS)
                                if mtls.check_mesh(mesh, V) is True:
                                    o3d_meshes.append(o3d_mesh)
                                    volumes.append(V)
                                    meshes.append(mesh)
                        else:
                            mesh, V = mtls.save_mesh(o3d_mesh, LLC_ABS)
                            if mtls.check_mesh(mesh, V) is True:
                                o3d_meshes.append(o3d_mesh)
                                volumes.append(V)
                                meshes.append(mesh)

            tot_vol = sum(volumes)
            perc_volumes = tot_vol / tot_space
            print(f'{round(perc_volumes*100, 2)}% of {TARGET_FRACTION*100}%\n')

            # break

            if perc_volumes > MAX_FRACTION:
                raise SedimentaryOverflowError('too much bodies created')
            elif perc_volumes < TARGET_FRACTION:
                N_b += PLUS

    for i, mesh in enumerate(meshes):
        # mesh.export(fr'meshes\{i}_.obj')
        mesh.export(fr'meshes\{i}_.stl')
