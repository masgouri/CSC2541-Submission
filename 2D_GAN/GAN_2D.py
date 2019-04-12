import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Add, BatchNormalization, Activation, Lambda, Concatenate
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D, Cropping2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#---------------- GPU Specs ----------------#
# modify the number to specify how memory much from the GPU to use
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# keep this to have channels first 
K.image_data_format() == "channels_first"

K.set_image_dim_ordering('th')

class WGAN():
    def __init__(self):
        # size of noise to feed into the generator
        self.random_dim = 100

        # optimizers for generator and discriminator
        self.adam_g = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        self.adam_d = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

        self.n_critic = 5
        self.clip_value = 0.01

        #---------------- Create Generator ----------------#
        inputs = Input(shape=(self.random_dim,))

        # number of resblocks to include
        resblocks = 0

        x = Dense(1024*4*4, input_dim=self.random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02))(inputs)
        x = LeakyReLU(0.2)(x)
        x = Reshape((1024, 4, 4))(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters=512, kernel_size=(5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #Apply num ResNet blocks, 
        for i in range(resblocks):
            x = res_block(x, 512, use_dropout=True)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)


        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters=1, kernel_size=(5, 5), padding='same')(x)
        x = Cropping2D(((53, 0),(53, 0)))(x)
        x = Activation('sigmoid')(x)

        # Add direct connection from input to output to improve training
        # outputs = Add()([x, inputs])
        # outputs = Lambda(lambda z: z/2)(outputs)

        self.generator = Model(inputs=inputs, outputs=x)
        self.generator.compile(loss=self.wasserstein_loss, optimizer=self.adam_g)

        #---------------- Create Discriminator ----------------#

        # change this for dif shapes
        inputs = Input(shape=(1, 203, 203))

        x = Conv2D(64, kernel_size=(5, 5), strides=2, padding="same", input_shape=(1, 203, 203))(inputs)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(128, kernel_size=(5, 5), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, kernel_size=(5, 5), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, kernel_size=(5, 5), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        self.discriminator = Model(inputs=inputs, outputs=x)
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=self.adam_d, metrics=['accuracy'])

        #---------------- Create Stacked Model ----------------#
        self.discriminator.trainable = False
        gan_in = Input(shape=(self.random_dim,))
        x = self.generator(gan_in)
        gan_out = self.discriminator(x)
        self.stacked = Model(inputs=gan_in, outputs=gan_out)
        self.stacked.compile(loss=self.wasserstein_loss, optimizer=self.adam_g)

    #---------------- Wasserstein Loss ----------------#
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    #---------------- Loss Plotting ----------------#
    def plot_loss(self, g_loss, d_loss):
        plt.figure(figsize=(10, 10))
        plt.plot(g_loss, label='Discriminator loss')
        plt.plot(d_loss, label='Generator loss')
        plt.title("Losses for the 2D self.Wasserstein GAN")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss/loss_wasserstein_%d.png' % epoch)

    #---------------- Image Generation ----------------#
    # adapted from https://hub.packtpub.com/generative-adversarial-networks-using-keras/
    def plot_images(self, epoch, examples=25, dim=(5, 5), figsize=(100, 100)):
        noise = np.random.normal(0, 1, size=[examples, randomDim])
        generatedImages = generator.predict(noise)
        # uncomment this line to print out real images
        # generatedImages=X_train[:25, :, :, :]
        plt.figure(figsize=figsize)
        for i in range(generatedImages.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('gen/generated_wasserstein_%d.png' % epoch)

    #---------------- Save Model ----------------#
    def save_model(self, gen, disc):
        gen.save('wgan_generator.h5')
        disc.save('wgan_discriminator.h5')
        return

    #---------------- ResBlocks ----------------#
    def res_block(self, input, filters, kernel_size=(5, 5), strides=1, use_dropout=False):
        # resblocks for improving generator (optional)
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding = 'same', strides=strides,)(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if use_dropout:
            x = Dropout(0.5)(x)

        x = Conv2D(filters=filters, kernel_size=kernel_size, padding = 'same', strides=strides,)(x)
        x = BatchNormalization()(x)

        merged = Add()([input, x])
        return merged

    #---------------- Train Model ----------------#
    def train(self, X_train, epochs, batch_size, save):

        # lists to track losses
        d_losses = []
        g_losses = []


        batch_count = X_train.shape[0] / batch_size
        print('Epochs:', epochs)
        print('Batch size:', batch_size)
        print('Batches per epoch:', batch_count)

        for e in range(1, epochs+1):
            # print('-'*15, 'Epoch %d' % e, '-'*15)
            for _ in tqdm(range(int(batch_count))):

                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

                # Generate fake images
                generated_images = self.generator.predict(noise)

                #print(image_batch.shape, generated_images.shape)
                X = np.concatenate([image_batch, generated_images])

                # Labels for generated and real data, label smoothing added
                y_dis = np.random.uniform(low=0, high=0.1, size=2*batch_size)
                y_dis[:batch_size] = np.random.uniform(low=0.9, high=1)
                
                ####### Train discriminator #######
                self.discriminator.trainable = True

                # number of extra times to train the discriminator (For WGAN)
                num_d_train = 2
                for _ in range(num_d_train):
                    self.discriminator.train_on_batch(X, y_dis)

                d_loss = self.discriminator.train_on_batch(X, y_dis)
                ####################################

                ####### Train generator ############
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                y_gen = np.ones(batch_size)
                self.discriminator.trainable = False

                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

                # number of extra times to train the generator (for DCGAN balance)
                num_g_train = 0
                for _ in range(num_g_train):
                    self.stacked.train_on_batch(noise, y_gen)

                g_loss = self.stacked.train_on_batch(noise, y_gen)
                ###################################

            print("generator loss: {} discriminator loss: {}".format(g_loss, d_loss))

            # Store loss of most recent batch from this epoch
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            if e % save == 0:
                self.plot_images(e)

        # uncomment to save model.. results in a huge file        
        # save_models(stacked, discriminator)

        # Plot losses from every epoch
        self.plot_loss(e, g_losses, d_losses)