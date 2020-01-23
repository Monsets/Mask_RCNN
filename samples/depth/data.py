import numpy as np
import sys
import os
from mrcnn import utils
from io import BytesIO
from PIL import Image


def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution * 4 / 3)), preserve_range=True, mode='reflect', anti_aliasing=True)


ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)


#################################################################
# MaskRCNN Dataset
#################################################################

class NyuDataset(utils.Dataset):
    def load_nyu(self, dataset_dir):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        self.dataset_dir = dataset_dir

        with open(os.path.join(dataset_dir, 'new_names.txt'), 'r') as f:
            for i, l in enumerate(f.readlines()):
                self.add_class('nyu', i + 1, l)

        self.maxDepth = 1000.0
        images = [f for f in os.listdir(os.path.join(dataset_dir, 'images')) \
                  if os.path.isfile(os.path.join(dataset_dir, 'images', f))]
        for i, image in enumerate(images):
            self.add_image('nyu', image_id=i, path=os.path.join(dataset_dir, 'images', image),
                           depth_path=os.path.join(dataset_dir, 'depths', image),
                           label_path=os.path.join(dataset_dir, 'labels', image),
                           width=640, height=480)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.

        instance_masks = []
        class_ids = []
        image = np.asarray(Image.open(self.image_info[image_id]['label_path'])).reshape(480, 640, 1)
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for cl in np.unique(image):
            if cl == 0:
                continue
            class_id = self.map_source_class_id(
                "nyu.{}".format(cl))
            if class_id:
                m = (image == cl).astype(int)

                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def load_depth_map(self, image_id):
        depth = np.clip(np.asarray(Image.open(self.image_info[image_id]['depth_path'])) \
                        .reshape(480, 640, 1) / 255 * self.maxDepth, 0, self.maxDepth)

        return nyu_resize(np.array(depth), 240)

