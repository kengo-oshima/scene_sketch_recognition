from keras import layers, models, activations, optimizers

class ModelGenerator():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.model_name = None

    def vgg16(self):
        from keras.applications import VGG16

        self.model_name = "VGG16"

        input_tensor = layers.Input(shape=self.input_shape)
        top_model = models.Sequential()
        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_tensor=input_tensor)
        flat = layers.Flatten()(conv_base.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        dense1 = layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(drop1)
        drop2 = layers.Dropout(0.5)(dense1)
        dense2 = layers.Dense(250, activation="softmax")(drop2)
        self.model = models.Model(inputs=conv_base.input, outputs=dense2)

        set_trainable = False
        for layer in self.model.layers:
            if layer.name == 'input_1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
            print("{}:{}".format(layer.name, layer.trainable))

    def resnet50(self):
        from keras.applications import ResNet50

        self.model_name = "ResNet50"

        input_tensor = layers.Input(shape=self.input_shape)
        conv_base = ResNet50(weights='imagenet',
                             include_top=False,
                             input_tensor=input_tensor)
        flat = layers.Flatten()(conv_base.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        dense1 = layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(drop1)
        drop2 = layers.Dropout(0.5)(dense1)
        dense2 = layers.Dense(10, activation="softmax")(drop2)
        self.model = models.Model(inputs=conv_base.input, outputs=dense2)

        set_trainable = False
        for layer in self.model.layers:
            if layer.name == 'flatten_1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
            print("{}:{}".format(layer.name, layer.trainable))

    def inception_resnet_v2(self):
        from keras.applications import InceptionResNetV2

        self.model_name = "InceptionResNetV2"

        input_tensor = layers.Input(shape=self.input_shape)
        top_model = models.Sequential()
        conv_base = InceptionResNetV2(weights='imagenet',
                                      include_top=False,
                                      input_tensor=input_tensor)
        top_model.add(layers.Flatten())
        top_model.add(layers.Dropout(0.5))
        top_model.add(layers.Dense(1024, activation='relu'))
        top_model.add(layers.Dropout(0.5))
        top_model.add(layers.Dense(10, activation="softmax"))
        self.model = models.Model(inputs=conv_base.input, outputs=top_model(conv_base.output))

        set_trainable = False
        for layer in self.model.layers:
            if layer.name == 'sequential_1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

            print("{}:{}".format(layer.name, layer.trainable))

    def inception_v3(self):
        from keras.applications import InceptionV3

        self.model_name = "InceptionV3"

        input_tensor = layers.Input(shape=self.input_shape)
        top_model = models.Sequential()
        conv_base = InceptionV3(weights='imagenet',
                                include_top=False,
                                input_tensor=input_tensor)
        top_model.add(layers.Flatten())
        top_model.add(layers.Dropout(0.5))
        top_model.add(layers.Dense(4096, activation='relu'))
        top_model.add(layers.Dropout(0.5))
        top_model.add(layers.Dense(10, activation="softmax"))
        self.model = models.Model(inputs=conv_base.input, outputs=top_model(conv_base.output))

        set_trainable = False
        for layer in self.model.layers:
            if layer.name == 'sequential_1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

            print("{}:{}".format(layer.name, layer.trainable))

    def sketch_a_net(self):
        self.model_name = "Sketch-a-Net"

        self.model = models.Sequential()
        self.model.add(
            layers.Conv2D(64, 15, strides=3, padding='valid', activation='relu', input_shape=self.input_shape,
                          kernel_initializer='he_normal'))
        self.model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid'))
        self.model.add(
            layers.Conv2D(128, 5, strides=1, padding='valid', activation='relu', kernel_initializer='he_normal'))
        self.model.add(layers.AveragePooling2D(3, 2, padding='valid'))
        self.model.add(
            layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
        self.model.add(
            layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
        self.model.add(
            layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
        self.model.add(layers.AveragePooling2D(3, 2, padding='valid'))
        self.model.add(
            layers.Conv2D(512, 7, strides=1, padding='valid', activation='relu', kernel_initializer='he_normal'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(
            layers.Conv2D(512, 1, strides=1, padding='valid', activation='relu', kernel_initializer='he_normal'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(10, activation='softmax'))

    def my_model1(self):
        self.model_name = 'my_model1'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(2048, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model2(self):

        self.model_name = "my_model2"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        conv2_1 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti1_2)
        acti2_1 = layers.Activation('relu')(conv2_1)
        conv2_2 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti2_1)
        acti2_2 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(conv2_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti2_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(1024, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(4096, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model3(self):

        self.model_name = "my_model3"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(64, 15, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(128, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        norm1_3 = layers.BatchNormalization()(conv1_3)
        acti1_3 = layers.Activation('relu')(norm1_3)
        conv1_4 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        norm1_4 = layers.BatchNormalization()(conv1_4)
        acti1_4 = layers.Activation('relu')(norm1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        norm1_5 = layers.BatchNormalization()(conv1_5)
        acti1_5 = layers.Activation('relu')(norm1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 7, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        norm1_6 = layers.BatchNormalization()(conv1_6)
        acti1_6 = layers.Activation('relu')(norm1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)

        conv2_1 = layers.Conv2D(64, 5, strides=3, padding='same', kernel_initializer='he_normal')(input1)
        norm2_1 = layers.BatchNormalization()(conv2_1)
        acti2_1 = layers.Activation('relu')(norm2_1)
        pool2_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti2_1)
        conv2_2 = layers.Conv2D(128, 3, strides=2, padding='same', kernel_initializer='he_normal')(pool2_1)
        acti2_2 = layers.Activation('relu')(conv2_2)
        conv2_3 = layers.Conv2D(128, 3, strides=2, padding='same', kernel_initializer='he_normal')(acti2_2)
        norm2_2 = layers.BatchNormalization()(conv2_3)
        acti2_3 = layers.Activation('relu')(norm2_2)
        conv2_4 = layers.Conv2D(128, 3, strides=2, padding='same', kernel_initializer='he_normal')(acti2_3)
        norm2_4 = layers.BatchNormalization()(conv2_4)
        acti2_4 = layers.Activation('relu')(norm2_4)
        pool2_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti2_4)
        drop2_1 = layers.Dropout(0.5)(pool2_2)

        conc1 = layers.concatenate([drop1_1, drop2_1])

        flat1 = layers.Flatten()(conc1)
        dens1 = layers.Dense(1024)(flat1)
        norm1 = layers.BatchNormalization()(dens1)
        acti1 = layers.Activation('relu')(norm1)
        dens2 = layers.Dense(250, activation='softmax')(acti1)

        self.model = models.Model(inputs=input1, outputs=dens2)

    def my_sketch_a_net(self):

        self.model_name = "my_sketch_a_net"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(64, 15, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        acti1_1 = layers.Activation('relu')(conv1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(128, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        acti1_2 = layers.Activation('relu')(conv1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 7, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(512, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(10, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_sketch_a_net2(self):

        self.model_name = "my_sketch_a_net2"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.SeparableConv2D(64, 15, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.SeparableConv2D(128, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        acti1_2 = layers.Activation('relu')(conv1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.SeparableConv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        norm1_2 = layers.BatchNormalization()(conv1_3)
        acti1_3 = layers.Activation('relu')(norm1_2)
        conv1_4 = layers.SeparableConv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.SeparableConv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.SeparableConv2D(512, 7, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.SeparableConv2D(512, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model4(self):

        self.model_name = "my_model4"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(64, 15, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(128, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        norm1_3 = layers.BatchNormalization()(conv1_3)
        acti1_3 = layers.Activation('relu')(norm1_3)
        conv1_4 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        norm1_4 = layers.BatchNormalization()(conv1_4)
        acti1_4 = layers.Activation('relu')(norm1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        norm1_5 = layers.BatchNormalization()(conv1_5)
        acti1_5 = layers.Activation('relu')(norm1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        flat1_1 = layers.Flatten()(pool1_3)

        conv2_1 = layers.Conv2D(64, 5, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm2_1 = layers.BatchNormalization()(conv2_1)
        acti2_1 = layers.Activation('relu')(norm2_1)
        pool2_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti2_1)
        conv2_2 = layers.Conv2D(128, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool2_1)
        acti2_2 = layers.Activation('relu')(conv2_2)
        conv2_3 = layers.Conv2D(128, 3, strides=2, padding='same', kernel_initializer='he_normal')(acti2_2)
        norm2_2 = layers.BatchNormalization()(conv2_3)
        acti2_3 = layers.Activation('relu')(norm2_2)
        conv2_4 = layers.Conv2D(128, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti2_3)
        norm2_4 = layers.BatchNormalization()(conv2_4)
        acti2_4 = layers.Activation('relu')(norm2_4)
        pool2_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti2_4)
        flat2_1 = layers.Flatten()(pool2_2)

        conc1 = layers.concatenate([flat1_1, flat2_1])

        dens1 = layers.Dense(1024)(conc1)
        norm1 = layers.BatchNormalization()(dens1)
        acti1 = layers.Activation('relu')(norm1)
        dens2 = layers.Dense(250, activation='softmax')(acti1)

        self.model = models.Model(inputs=input1, outputs=dens2)

    # モデル読み込み
    def my_model5(self):

        self.model_name = "my_model5"
        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(64, 15, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(128, 13, strides=2, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        conv2_1 = layers.Conv2D(256, 9, strides=1, padding='same', kernel_initializer='he_normal')(acti1_2)
        acti2_1 = layers.Activation('relu')(conv2_1)
        conv2_2 = layers.Conv2D(512, 7, strides=1, padding='same', kernel_initializer='he_normal')(acti2_1)
        acti2_2 = layers.Conv2D(1024, 5, strides=1, padding='same', kernel_initializer='he_normal')(conv2_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti2_2)
        conv1_3 = layers.Conv2D(2048, 3, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_7 = layers.Conv2D(4096, 2, strides=1, padding='valid', kernel_initializer='he_normal')(acti1_3)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model7(self):

        self.model_name = "my_model7"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        conv2_1 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti1_2)
        acti2_1 = layers.Activation('relu')(conv2_1)
        conv2_2 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti2_1)
        acti2_2 = layers.Activation('relu')(conv2_2)
        conv2_3 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti2_2)
        acti2_2 = layers.Activation('relu')(conv2_3)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti2_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(1024, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(4096, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model8(self):

        self.model_name = "my_model8"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        conv2_1 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti1_2)
        acti2_1 = layers.Activation('relu')(conv2_1)
        conv2_2 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti2_1)
        acti2_2 = layers.Activation('relu')(conv2_2)
        conv2_3 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti2_2)
        acti2_2 = layers.Activation('relu')(conv2_3)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti2_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(2048, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model9(self):

        self.model_name = "my_model9"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        conv2_1 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti1_2)
        acti2_1 = layers.Activation('relu')(conv2_1)
        conv2_2 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti2_1)
        acti2_2 = layers.Activation('relu')(conv2_2)
        conv2_3 = layers.Conv2D(256, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti2_2)
        acti2_2 = layers.Activation('relu')(conv2_3)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti2_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_3)
        conv1_6 = layers.Conv2D(1024, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(4096, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model10(self):
        self.model_name = 'my_model10'
        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.SeparableConv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.SeparableConv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.SeparableConv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.SeparableConv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.SeparableConv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.SeparableConv2D(512, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.SeparableConv2D(2048, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_scene_net(self):
        self.model_name = 'my_scene_net'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(1024, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(4096, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_scene_net2(self):
        self.model_name = 'my_scene_net2'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        acti1_1 = layers.Activation('relu')(conv1_1)
        norm1_1 = layers.BatchNormalization()(acti1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(norm1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        acti1_2 = layers.Activation('relu')(conv1_2)
        norm1_2 = layers.BatchNormalization()(acti1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(norm1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(1024, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(4096, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_scene_net3(self):
        self.model_name = 'my_scene_net3'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        acti1_1 = layers.Activation('relu')(conv1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        acti1_2 = layers.Activation('relu')(conv1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(1024, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(4096, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def create_model(self, model_path, model_name):
        self.model_name = model_name

        model = models.load_model(model_path)

        flatten = model.layers[-2].output
        print(flatten.name)

        dens2 = layers.Dense(10, activation='softmax', name='output')(flatten)

        self.model = models.Model(inputs=model.input, outputs=dens2)

        set_trainable = False
        for layer in self.model.layers:
            if 'aaaaaaaaaaa' in layer.name:
                layer.trainable = False
            else:
                layer.trainable = True
            print("{}:{}".format(layer.name, layer.trainable))

    def r_sketch_a_Net(self):
        self.model_name = "r_sketch_a_Net"

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(64, 15, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        acti1_1 = layers.Activation('relu')(conv1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(128, 5, strides=1, padding='same', kernel_initializer='he_normal')(pool1_1)
        norm1_1 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_1)
        conv1_3 = layers.Conv2D(128, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti1_2)
        norm1_2 = layers.BatchNormalization()(conv1_3)
        acti1_3 = layers.Activation('relu')(norm1_2)
        conv1_4 = layers.Conv2D(128, 5, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        add1 = layers.add([conv1_2, conv1_4])
        acti2_5 = layers.Activation('relu')(add1)
        pool2_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti2_5)

        conv2_1 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool2_2)
        norm2_1 = layers.BatchNormalization()(conv2_1)
        acti2_3 = layers.Activation('relu')(norm2_1)
        conv2_4 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti2_3)
        norm2_2 = layers.BatchNormalization()(conv2_4)
        acti2_4 = layers.Activation('relu')(norm2_2)
        conv2_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti2_4)
        add2 = layers.add([conv2_1, conv2_5])

        acti1_5 = layers.Activation('relu')(add2)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 7, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(512, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model6(self, model_path1, model_path2):
        self.model_name = 'my_model6'
        # input1 = layers.Input(self.input_shape)
        # input2 = layers.Input(193, 193, 3)
        model1 = models.load_model(model_path1)
        model2 = models.load_model(model_path2)
        model1_output = model1.layers[-1].output
        model2_output = model2.layers[-1].output
        conc = layers.concatenate([model1_output, model2_output])
        dens = layers.Dense(10, activation='relu')(conc)

        self.model = models.Model(inputs=[model1.layers[0], model2.layers[0]], outputs=dens)

        self.model.summary()

    def my_model11(self):
        self.model_name = 'my_model11'

        input1 = layers.Input(self.input_shape)
        drop1 = layers.Dropout(0.7)(input1)
        dens1 = layers.Dense(2048, kernel_initializer='he_normal')(drop1)
        norm1 = layers.BatchNormalization()(dens1)
        acti1 = layers.Activation('relu')(norm1)
        drop2 = layers.Dropout(0.5)(acti1)
        dens2 = layers.Dense(10, activation='softmax')(drop2)

        self.model = models.Model(inputs=input1, outputs=dens2)

        self.model.summary()

    def my_mdoel12(self):
        self.model_name = 'my_model12'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(2048, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)

        from keras.applications import VGG16

        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_tensor=input1)
        flat = layers.Flatten()(conv_base.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        dense1 = layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(drop1)
        drop2 = layers.Dropout(0.5)(dense1)
        conc1 = layers.Concatenate()([drop1_2, drop2])

        out = layers.Dense(250, activation='softmax')(conc1)

        self.model = models.Model(inputs=input1, outputs=out)

        set_trainable = False
        for layer in self.model.layers:
            if 'block' in layer.name:
                layer.trainable = False
            else:
                layer.trainable = True
            print("{}:{}".format(layer.name, layer.trainable))

    def my_model13(self):
        self.model_name = 'my_model13'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(2048, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)

        from keras.applications import VGG16

        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_tensor=input1)
        flat = layers.Flatten()(conv_base.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        dense1 = layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(drop1)
        conc1 = layers.Concatenate()([flat1_1, dense1])
        drop2 = layers.Dropout(0.3)(conc1)

        out = layers.Dense(250, activation='softmax')(drop2)

        self.model = models.Model(inputs=input1, outputs=out)

        set_trainable = False
        for layer in self.model.layers:
            if 'block' in layer.name:
                layer.trainable = False
            else:
                layer.trainable = True
            print("{}:{}".format(layer.name, layer.trainable))

    def my_model14(self):
        self.model_name = 'my_model14'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(1024, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(4098, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)

        from keras.applications import VGG16

        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_tensor=input1)
        flat = layers.Flatten()(conv_base.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        dense1 = layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(drop1)
        conc1 = layers.Concatenate()([flat1_1, dense1])
        drop2 = layers.Dropout(0.3)(conc1)
        dens2_2 = layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(drop2)
        drop3 = layers.Dropout(0.3)(dens2_2)

        out = layers.Dense(250, activation='softmax')(drop3)

        self.model = models.Model(inputs=input1, outputs=out)

        set_trainable = False
        for layer in self.model.layers:
            if 'block' in layer.name:
                layer.trainable = False
            else:
                layer.trainable = True
            print("{}:{}".format(layer.name, layer.trainable))

    def my_model15(self):
        self.model_name = 'my_model15'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        norm1_3 = layers.BatchNormalization()(conv1_3)
        acti1_3 = layers.Activation('relu')(norm1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        norm1_4 = layers.BatchNormalization()(conv1_4)
        acti1_4 = layers.Activation('relu')(norm1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        norm1_5 = layers.BatchNormalization()(conv1_5)
        acti1_5 = layers.Activation('relu')(norm1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(1024, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        norm1_6 = layers.BatchNormalization()(conv1_6)
        acti1_6 = layers.Activation('relu')(norm1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(4096, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        norm1_7 = layers.BatchNormalization()(conv1_7)
        acti1_7 = layers.Activation('relu')(norm1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)
        dens1_1 = layers.Dense(250, activation='softmax')(drop1_2)

        self.model = models.Model(inputs=input1, outputs=dens1_1)

    def my_model16(self):
        self.model_name = 'my_model16'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('relu')(norm1_1)
        pool1_1 = layers.AveragePooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('relu')(norm1_2)
        pool1_2 = layers.AveragePooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('relu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('relu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('relu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('relu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(2048, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('relu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)

        from keras.applications import VGG16

        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_tensor=input1)
        flat = layers.Flatten()(conv_base.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        dense1 = layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(drop1)
        drop2 = layers.Dropout(0.5)(dense1)
        conc1 = layers.Concatenate()([drop1_2, drop2])

        out = layers.Dense(250, activation='softmax')(conc1)

        self.model = models.Model(inputs=input1, outputs=out)

        set_trainable = False
        for layer in self.model.layers:
            if 'block' in layer.name:
                layer.trainable = False
            else:
                layer.trainable = True
            print("{}:{}".format(layer.name, layer.trainable))

    def my_model17(self):
        self.model_name = 'my_model17'

        input1 = layers.Input(self.input_shape)
        conv1_1 = layers.Conv2D(96, 7, strides=3, padding='valid', kernel_initializer='he_normal')(input1)
        norm1_1 = layers.BatchNormalization()(conv1_1)
        acti1_1 = layers.Activation('elu')(norm1_1)
        pool1_1 = layers.MaxPooling2D(3, strides=2, padding='valid')(acti1_1)
        conv1_2 = layers.Conv2D(256, 5, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_1)
        norm1_2 = layers.BatchNormalization()(conv1_2)
        acti1_2 = layers.Activation('elu')(norm1_2)
        pool1_2 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_2)
        conv1_3 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1_2)
        acti1_3 = layers.Activation('elu')(conv1_3)
        conv1_4 = layers.Conv2D(384, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_3)
        acti1_4 = layers.Activation('elu')(conv1_4)
        conv1_5 = layers.Conv2D(256, 3, strides=1, padding='same', kernel_initializer='he_normal')(acti1_4)
        acti1_5 = layers.Activation('elu')(conv1_5)
        pool1_3 = layers.MaxPooling2D(3, 2, padding='valid')(acti1_5)
        conv1_6 = layers.Conv2D(512, 6, strides=1, padding='valid', kernel_initializer='he_normal')(pool1_3)
        acti1_6 = layers.Activation('elu')(conv1_6)
        drop1_1 = layers.Dropout(0.5)(acti1_6)
        conv1_7 = layers.Conv2D(2048, 1, strides=1, padding='same', kernel_initializer='he_normal')(drop1_1)
        acti1_7 = layers.Activation('elu')(conv1_7)
        flat1_1 = layers.Flatten()(acti1_7)
        drop1_2 = layers.Dropout(0.5)(flat1_1)

        from keras.applications import VGG16

        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_tensor=input1)
        flat = layers.Flatten()(conv_base.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        dense1 = layers.Dense(2048, activation='elu', kernel_initializer='he_normal')(drop1)
        drop2 = layers.Dropout(0.5)(dense1)
        conc1 = layers.Concatenate()([drop1_2, drop2])

        out = layers.Dense(250, activation='softmax')(conc1)

        self.model = models.Model(inputs=input1, outputs=out)

        set_trainable = False
        for layer in self.model.layers:
            if 'block' in layer.name:
                layer.trainable = False
            else:
                layer.trainable = True
            print("{}:{}".format(layer.name, layer.trainable))

    def load_model(self, model_path):
        self.model_name = 'my_load_model'

        self.model = models.load_model(model_path)