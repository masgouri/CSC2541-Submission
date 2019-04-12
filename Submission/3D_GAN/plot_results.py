import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Plotting 3D GAN Losses')
    parser.add_argument('-i', '--input', type=str, help='Input file')
    parser.add_argument('-o', '--output', type=str, help='Output path')
    parser.add_argument('-w', '--wgan', type=int, default=1)
    return parser.parse_args()


def read_txt(input_path, wgan):
    dis_1 = []
    dis_2 = []
    gen = []
    with open(input_path, 'r') as f:
        for cnt, line in enumerate(f):
            nb = line.split(', ')
            dis_1.append(float(nb[1]))
            if wgan:
                gen.append(float(nb[2]))
            else:
                dis_2.append(float(nb[2]))
                gen.append(float(nb[3]))
    if wgan:
        return [dis_1], gen
    else:
        return [dis_1, dis_2], gen


def loss_plot(dis, gen, output_path, wgan):
    plt.figure()
    if wgan:
        plt.plot(dis[0], label="Critic_loss")
    else:
        plt.plot(dis[0], label="Discriminator_loss_real")
        plt.plot(dis[1], label="Discriminator_loss_fake")
    plt.plot(gen, label="Generator_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc= "upper right")

    plt.savefig(output_path)
    plt.close()


def main():
    args = get_args()
    dis, gen = read_txt(args.input)
    loss_plot(dis, gen, args.output, args.wgan)


if __name__ == '__main__':
    main()


