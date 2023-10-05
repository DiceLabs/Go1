import pywavefront
from pyglet.gl import *
from pyglet.window import Window, mouse, key

# Load OBJ file and MTL file
obj_file = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/floor2.obj'
mtl_file = '/home/kunal2204/projects/fieldAI/github_files/FieldAI_Kunal/tests/floor2.mtl'
meshes = pywavefront.Wavefront(obj_file, create_materials=True, strict=True)
# meshes.parse_mtl(mtl_file)

meshes = pywavefront.Wavefront(obj_file, create_materials=True, strict=True)

# Set up Pyglet window
window = Window(width=800, height=600, resizable=True)

# Initialize OpenGL
glClearColor(0.5, 0.5, 0.5, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)

# Set up material properties
glMaterialfv(GL_FRONT, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
glMaterialfv(GL_FRONT, GL_SHININESS, 100.0)

# Set up light properties
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
glLightfv(GL_LIGHT0, GL_POSITION, (1.0, 1.0, 1.0, 0.0))

@window.event
def on_draw():
    window.clear()
    glLoadIdentity()
    glTranslatef(0, 0, -5)
    meshes.draw()

@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    return True

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if buttons & mouse.LEFT:
        glRotatef(dy, 1, 0, 0)
        glRotatef(dx, 0, 1, 0)
    elif buttons & mouse.RIGHT:
        glTranslatef(dx / 10.0, -dy / 10.0, 0)

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        window.close()

pyglet.app.run()