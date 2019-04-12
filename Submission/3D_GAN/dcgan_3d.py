from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Flatten, Dropout, Add, Reshape
from keras.layers import BatchNormalization, Activation, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, Deconv3D, Conv1D, UpSampling1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import tensorflow as tf

# Share resources
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


def bin_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)


class GAN():
    def __init__(self, learning_rate_gen, learning_rate_disc, in_shape=(64, 64, 64, 1), out_shape=(64, 64, 64, 1), transfer=False, batch=64):
        # basic attributes
        self.learning_rate_gen = learning_rate_gen
        self.learning_rate_disc = learning_rate_disc
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.batch = batch
        self.generator_optimizer = Adam(self.learning_rate_gen, 0.5)
        self.discriminator_optimizer = Adam(self.learning_rate_disc, 0.5)

        # create the models, freeze the discriminator
        self.generator = self.make_generator()
        
        self.discriminator = self.make_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=self.discriminator_optimizer, metrics=['accuracy'])
        

        # Combined model
        self.discriminator.trainable = False
        z = Input(shape=(1,1,1,200))
        img = self.generator(z)
        validity = self.discriminator(img)
        self.stacked = Model(z, validity)
        self.stacked.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)
    

    # WIP
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def make_generator(self):
        out = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        model = Sequential()
        
        model.add(Deconv3D(filters=512, kernel_size=4, strides=1, kernel_initializer = 'glorot_normal', activation='relu',
                         input_shape=(1, 1, 1, 200), padding='valid'))
        model.add(BatchNormalization())
        model.add(Deconv3D(filters=256, kernel_size=4, strides=2, kernel_initializer = 'glorot_normal',
                  activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Deconv3D(filters=128, kernel_size=4, strides=2, kernel_initializer = 'glorot_normal',
                  activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Deconv3D(filters=64, kernel_size=4, strides=2, kernel_initializer = 'glorot_normal',
                  activation='relu', padding='same'))
        model.add(Deconv3D(filters=1, kernel_size=4, strides=2, kernel_initializer = 'glorot_normal',  
                  activation='relu', padding='same'))
        model.add(Activation(activation='sigmoid'))
        print(model.summary())
        noise = Input(shape=(1,1,1,200))
        img = model(noise)

        return Model(noise, img)


    def make_discriminator(self):
        input_shape = (self.out_shape[0], self.out_shape[1], self.out_shape[2], 1)
        model = Sequential()
        model.add(Conv3D(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer = 'glorot_normal',
                         activation='relu', input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=128, kernel_size=4, strides=2, padding='same', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=256, kernel_size=4, strides=2, padding='same', kernel_initializer = 'glorot_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=512, kernel_size=4, strides=2, padding='same', kernel_initializer = 'glorot_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=1, kernel_size=4, strides=1, padding='valid', kernel_initializer = 'glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())
        
        img = Input(input_shape)
        validity = model(img)

        return Model(img, validity)


    def generator_noise(self, batch, z_size=200, mu=0, sigma=0.2):
        # note: in paper is was 0 mean, 0.2 sigma
        noise = np.random.normal(mu, sigma, size=[batch, 1, 1, 1, z_size]).astype(np.float32)
        return noise
a

    def train(self, x_train, epochs, save, batch, output_file):
        
        batchCount = int(x_train.shape[0] / batch)
        # For plotting purposes
        gen_loss = []
        dis_loss_real = []
        dis_loss_fake = []
        dis_accuracy = 0

        for e in range(epochs):
            #--------- train discriminator ---------#
            print("Epoch is", e)
            
            # For debugging
            #for _ in tqdm(range(1)):
            for _ in tqdm(range(batchCount)):


                imgs = x_train[np.random.randint(0, x_train.shape[0], size=batch)]
                imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3], 1))
                
                # Generate random images + fake noise
                #imgs_noise = np.random.normal(0, 0.1, size=[batch, imgs.shape[1], imgs.shape[2], imgs.shape[3], 1])
                #imgs = imgs + imgs_noise  
                
                noise = self.generator_noise(batch)
                gen_imgs = self.generator.predict(noise)

                X = np.concatenate((imgs, gen_imgs))
                Y = np.zeros(2*batch)
                Y[:batch] = 1
                Y = Y.reshape((-1, 1, 1, 1, 1))

                # Train discriminator
                self.discriminator.trainable = True
                
                # Introduced some smoothing
                #d_loss_real = self.discriminator.train_on_batch(imgs, np.random.uniform(low=0.8, high=1, size=(batch,)).reshape(-1, 1, 1, 1, 1))
                #d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.random.uniform(low=0, high=0.2, size=(batch,)).reshape(-1,1,1,1,1))
                #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Update the discriminaor only if the accuracy is low
                if dis_accuracy <= 0.8:

                    d_loss_real = self.discriminator.train_on_batch(imgs, np.ones(batch).reshape(-1,1,1,1,1))
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros(batch).reshape(-1,1,1,1,1))
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    dis_accuracy = d_loss[1]
                else:
                    print("No need to train the discriminator: accuracy is ", str(dis_accuracy))
                    acc_1 = self.discriminator.evaluate(imgs, np.ones(batch).reshape(-1,1,1,1,1))
                    acc_2 = self.discriminator.evaluate(gen_imgs, np.zeros(batch).reshape(-1,1,1,1,1))
                    dis_accuracy = 0.5 * (acc_1[1] + acc_2[1])
                    d_loss[1] = dis_accuracy
                
                # Train generator
                noise2 = self.generator_noise(batch)
                y_2 = np.ones(batch)
                y_2 = y_2.reshape((-1, 1, 1, 1, 1))
                self.discriminator.trainable = False
                g_loss = self.stacked.train_on_batch(noise2, y_2)
            
            gen_loss.append(g_loss)
            dis_loss_real.append(d_loss_real)
            dis_loss_fake.append(d_loss_fake)

            with open("results.txt", "a") as txt_file:
                txt_file.write("%f, %f, %f, %f\n" % (g_loss, d_loss_real[0], d_loss_fake[0], d_loss[1]))            


            print("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (e, d_loss[0],100*d_loss[1], g_loss))

            if e % save == 0:
                self.save_imgs(self.generator, e, x_train)
            if e % 500 == 0:
                print('Saving model weights')
                self.save_model(self.generator, e, 'generator')
                self.save_model(self.discriminator, e, 'discriminator')
        
        np.save('gen_loss.npy', gen_loss)
        np.save('dis_loss_real.npy', dis_loss_real)
        np.save('dis_loss_fake.npy', dis_loss_fake)        


    def save_model(self, model, e, outfile):
        model.save_weights('{}_{}.h5'.format(outfile, e))
        return

    
    def save_imgs(self, model, epoch, x_train):
        #Save 3 generated images from 3 different angles for demonstration purposes.
        rows = 3
        noise = self.generator_noise(rows)
        generated_images = model.predict(noise)
        u = generated_images[0].shape[0]

        print("GENERATE IMAGE DIM==========>", generated_images.shape)
        
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
        plt.savefig("gen/generated_%d.png" % epoch)
        plt.close()
        
        gs = gridspec.GridSpec(rows, 3)
        plt.figure()
        for i in range(rows):
            d = generated_images[i].reshape(u, u, u)
            
            ax = plt.subplot(gs[i, 0])
            plt.imshow(np.rot90(d[16,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 1])
            plt.imshow(d[:,16,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 2])
            plt.imshow(np.rot90(d[:,:,16]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.colorbar()
            plt.clim(0,1)
        plt.savefig("gen/generated_%d_16.png" % epoch)
        plt.close()

        gs = gridspec.GridSpec(rows, 3)
        plt.figure()
        for i in range(rows):
            d = generated_images[i].reshape(u, u, u)
            ax = plt.subplot(gs[i, 0])
            plt.imshow(np.rot90(d[48,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 1])
            plt.imshow(d[:,48,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax = plt.subplot(gs[i, 2])
            plt.imshow(np.rot90(d[:,:,48]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.colorbar()
            plt.clim(0,1)
        plt.savefig("gen/generated_%d_48.png" % epoch)
        plt.close()

        # Save real images to compare
        if epoch % 100 == 0:
            img = x_train[np.random.randint(0, x_train.shape[0], size=1)]
            gs2 = gridspec.GridSpec(1, 3)
            plt.figure()
            d2 = img[0].reshape(64, 64, 64)
            ax2 = plt.subplot(gs2[0, 0])
            plt.imshow(np.rot90(d2[16,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 1])
            plt.imshow(d2[:,16,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 2])
            plt.imshow(np.rot90(d2[:,:,16]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            plt.savefig("gen/real_%d_16.png" % epoch)
            plt.close()
 

            gs2 = gridspec.GridSpec(1, 3)
            plt.figure()
            d2 = img[0].reshape(64, 64, 64)
            ax2 = plt.subplot(gs2[0, 0])
            plt.imshow(np.rot90(d2[32,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 1])
            plt.imshow(d2[:,32,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 2])
            plt.imshow(np.rot90(d2[:,:,32]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            plt.savefig("gen/real_%d.png" % epoch)
            plt.close()


            gs2 = gridspec.GridSpec(1, 3)
            plt.figure()
            d2 = img[0].reshape(64, 64, 64)
            ax2 = plt.subplot(gs2[0, 0])
            plt.imshow(np.rot90(d2[48,:,:], 2), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 1])
            plt.imshow(d2[:,48,:], cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            ax2 = plt.subplot(gs2[0, 2])
            plt.imshow(np.rot90(d2[:,:,48]), cmap=plt.cm.get_cmap('gray_r', 20))
            plt.clim(0,1)
            plt.savefig("gen/real_%d_48.png" % epoch)
            plt.close()
