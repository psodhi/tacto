import trimesh
import pyrender
import numpy as np
from PIL import Image
from urdfpy import URDF

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(
                "/home/paloma/code/lib_ws/pyrender/pyrender/shaders/mesh.vert", "/home/paloma/code/lib_ws/pyrender/pyrender/shaders/mesh.frag", defines=defines)
        return self.program

camera_pose = np.array(
    [[ 1,  0,  0,  0],
     [ 0,  0, -1, -4],
     [ 0,  1,  0,  0],
     [ 0,  0,  0,  1]]
)

scene = pyrender.Scene(bg_color=(0, 0, 0))


# load object from file
urdf_file = "objects/sphere_small.urdf"
obj = URDF.load(urdf_file)
visual = obj.links[0].visuals[0]
obj_mesh = visual.geometry.meshes[0]

# load obj primitive 
# obj_mesh = trimesh.primitives.Capsule()

scene.add(pyrender.Mesh.from_trimesh(obj_mesh, smooth = False))
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, znear = 0.5, zfar = 40)
scene.add(camera, pose=camera_pose)

renderer = pyrender.OffscreenRenderer(512, 512)
#renderer._renderer._program_cache = CustomShaderCache()

normals, depth = renderer.render(scene)
world_space_normals = normals / 255 * 2 - 1

image_normal = Image.fromarray(normals, 'RGB')
image_normal.show()

image_depth = Image.fromarray(depth, 'P')
image_depth.show()
