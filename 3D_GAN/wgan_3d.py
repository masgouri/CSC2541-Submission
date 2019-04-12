from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.gridspec as gridspec
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf

# Share resources
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


class WGAN():
    def __init__(self):
        self.img_shape = (128, 128, 128, 1)
        self.latent_dim = 200
        self.channels = 1
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = Adam(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(8 * 8 * 8 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 8, 16)))
        model.add(Conv3D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling3D())
        model.add(Conv3D(16, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling3D())
        model.add(Conv3D(8, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling3D())
        model.add(Conv3D(4, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling3D())
        model.add(Conv3D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)


    def build_critic(self):

        model = Sequential()

        model.add(Conv3D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv3D(32, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv3D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv3D(128, kernel_size=4, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, X_train, epochs, batch_size=32, sample_interval=50):

        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        
        for epoch in range(epochs):
            for _ in range(self.n_critic):
  
                #  Train Discriminator

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                imgs = imgs.reshape((batch_size, 128, 128, 128, 1))
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            #  Train Generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            
            with open("res_wgan.txt", "a") as f:
                f.write("%d, %f, %f\n" % (epoch, 1-d_loss[0], 1-g_loss[0]))
          
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_train)


    def sample_images(self, epoch, x_train):
        rows = 3
        noise = np.random.normal(0, 1, (3, self.latent_dim))
        generated_images = self.generator.predict(noise)
        u = generated_images[0].shape[0]
        print("GENERATE IMAGE DIM==========>", generated_images.shape)
        
        gs = gridspec.GridSpec(rows, 3)
        plt.figure()
        for i in range(rows):
            d = generated_images[i].reshape(u, u, u)
            
            ax = plt.subplot(gs[i, 0])
            plt.imshow(np.rot90(d[64,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 1])
            plt.imshow(d[:,64,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 2])
            plt.imshow(np.rot90(d[:,:,64]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.colorbar()
            plt.clim(0,1)
        plt.savefig("gen/generated_%d.png" % epoch)
        plt.close()
        
        gs = gridspec.GridSpec(rows, 3)
        plt.figure()
        for i in range(rows):
            d = generated_images[i].reshape(u, u, u)
            ax = plt.subplot(gs[i, 0])
            plt.imshow(np.rot90(d[32,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 1])
            plt.imshow(d[:,32,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 2])
            plt.imshow(np.rot90(d[:,:,32]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.colorbar()
            plt.clim(0,1)
        plt.savefig("gen/generated_%d_32.png" % epoch)
        plt.close()

        gs = gridspec.GridSpec(rows, 3)
        plt.figure()
        for i in range(rows):
            d = generated_images[i].reshape(u, u, u)
            ax = plt.subplot(gs[i, 0])
            plt.imshow(np.rot90(d[96,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 1])
            plt.imshow(d[:,96,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 2])
            plt.imshow(np.rot90(d[:,:,96]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.colorbar()
            plt.clim(0,1)
        plt.savefig("gen/generated_%d_96.png" % epoch)
        plt.close()

        # Save real images as examples
        if epoch % 1000 == 0:
            img = x_train[np.random.randint(0, x_train.shape[0], size=1)]
            gs2 = gridspec.GridSpec(1, 3)
            plt.figure()
            d2 = img[0].reshape(128, 128, 128)
            ax2 = plt.subplot(gs2[0, 0])
            plt.imshow(np.rot90(d2[32,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 1])
            plt.imshow(d2[:,32,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 2])
            plt.imshow(np.rot90(d2[:,:,32]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            plt.savefig("gen/real_%d_32.png" % epoch)
            plt.close()
 
            gs2 = gridspec.GridSpec(1, 3)
            plt.figure()
            ax2 = plt.subplot(gs2[0, 0])
            plt.imshow(np.rot90(d2[64,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 1])
            plt.imshow(d2[:,64,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 2])
            plt.imshow(np.rot90(d2[:,:,64]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            plt.savefig("gen/real_%d.png" % epoch)
            plt.close()

            gs2 = gridspec.GridSpec(1, 3)
            plt.figure()
            ax2 = plt.subplot(gs2[0, 0])
            plt.imshow(np.rot90(d2[96,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 1])
            plt.imshow(d2[:,96,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 2])
            plt.imshow(np.rot90(d2[:,:,96]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            plt.savefig("gen/real_%d_96.png" % epoch)
            plt.close()
