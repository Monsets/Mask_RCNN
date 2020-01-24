import os
import sys

import keras
import keras.layers as KL
import keras.model as KM

ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)

from mrcnn.model import MaskRCNN
from data import data_generator

class ModifiedMaskRCNN(MaskRCNN):
    def __init__(self, mode, config, model_dir):
        super().__init__(mode, config, model_dir)

        self.base_model = self.keras_model
        self.keras_model = self.build(mode, config)

    def build(self, mode, config):
        # Depth network part
        # Starting point for decoder
        if mode == 'training':
            input_depth = KL.Input(shape=[240, 320, 1], name='input_depth')
        else:
            print("Not implemented")
            exit(1)
        base_model_output_shape = self.base_model.get_layer('bn5c_branch2c').shape

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
            up_i = KL.Concatenate(name=name + '_concat')(
                [up_i, self.base_model.get_layer(concat_with).output])  # Skip connection
            up_i = KL.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(up_i)
            up_i = KL.LeakyReLU(alpha=0.2)(up_i)
            up_i = KL.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(up_i)
            up_i = KL.LeakyReLU(alpha=0.2)(up_i)
            return up_i

        decode_filters = int(base_model_output_shape[-1])
        # Decoder Layers
        decoder = KL.Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                         name='conv2')(self.base_model.get_layer('bn5c_branch2c').output)

        decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='bn4f_branch2c')
        decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='bn3d_branch2c')
        decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='bn2c_branch2c')
        decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='conv1')

        # Extract depths (final layer)
        conv3 = KL.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='output_depth')(decoder)

        # getting input and output layers of base model
        inputs = [l for l in self.base_model.layers if 'input' in l.name]
        outputs = [l for l in self.base_model.layers if 'output' in l.name]

        if mode == 'training':
            #Depth loss
            depth_loss = KL.Lambda(lambda x: depth_loss_function(*x), name="depth_loss")(
                [input_depth, conv3])

            inputs.append(input_depth)
            outputs.append(conv3, depth_loss)
        else:
            print("Not implemented")
            exit(1)

        model = KM.Model(inputs, outputs, name='depth_mask_rcnn')

        return model

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Trainf
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)