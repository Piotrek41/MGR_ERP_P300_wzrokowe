import os.path
import matplotlib.image as mpimg


def get_brain_pictures():
    this_folder = os.path.dirname(__file__)
    back = mpimg.imread(os.path.join(this_folder, 'brain_back.png'))
    side = mpimg.imread(os.path.join(this_folder, 'brain_side.png'))
    top = mpimg.imread(os.path.join(this_folder, 'brain_top.png'))
    return back, side, top

