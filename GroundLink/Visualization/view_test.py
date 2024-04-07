from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
import numpy as np

output = "../Data/moshpp/"
# path = "../dance.fbx"
path = "../guy.pk"
modeldata = np.load(path, allow_pickle=True)

if __name__ == '__main__':
    v = Viewer()
    v.scene.add(SMPLSequence.export_to_npz(modeldata, output))
    v.run()

