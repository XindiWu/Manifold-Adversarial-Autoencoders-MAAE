import os
import time
import torch
import argparse
import foolbox
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import auc
from torch.autograd import Variable

attacks = [
    ("fgsm", foolbox.attacks.FGSM),
    # ("pgd", foolbox.attacks.LinfinityBasicIterativeAttack),
    ("df", foolbox.attacks.DeepFoolAttack),
    ("cw", foolbox.attacks.CarliniWagnerL2Attack),
    # ("single", foolbox.attacks.SinglePixelAttack),
    # ("salt", foolbox.attacks.SaltAndPepperNoiseAttack)
]

from models import VAE
cuda = True if torch.cuda.is_available() else False

NUM_LABELS = 10
CHANNELS = 1
WIDTH = 28
HEIGHT = 28
BETA = 1.5

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    dataset = MNIST(
        root='data', train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
        download=True)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28*28) * 0.5 + 0.5, x.view(-1, 28*28) * 0.5 + 0.5, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + BETA * KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=NUM_LABELS if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y, training=True)
            else:
                recon_x, mean, log_var, z = vae(x, training=True)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                if args.conditional:
                    recon_x, mean, log_var, z, _ = vae(x, training=True)
                    # score_list = vae(x)
                    # print((score_list.argmax(1) == y).float().mean().item())

                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if iteration == len(data_loader)-1:
                    ptest(args, targeted_model=vae)

                if args.conditional:
                    c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                    x = vae.inference(n=c.size(0), c=c)
                else:
                    x = vae.inference(n=10)

                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p+1)
                    if args.conditional:
                        plt.text(
                            0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(28, 28).data.cpu().numpy())
                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')
                if not os.path.exists('checkpoints'):
                    os.mkdir('checkpoints')
                torch.save(vae.state_dict(), "checkpoints/vae_{:d}I{:d}.pth".format(epoch, iteration))


        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        g = sns.lmplot(
            x='x', y='y', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)

def plot(args, methods, distanceChoice=2, visualize=False):
    font = {'family': 'normal',
        'weight': 'bold',
        'size': 16}

    matplotlib.rc('font', **font)
    plt.style.use('bmh')

    #methods = ['Tloss','T+reg(0.8)']
    # methods = ['Base', 'D2(1, 4)', 'D2(1, 5)', 'D2(1, 6)', 'D2(1, 7)', 'D2(1, 7.5)'] # ImageNet

    methodNum = len(methods)
    attackNum = len(attacks)
    colors = ['k', 'g', 'b', 'c', 'y', 'r', 'm']

    '''
    :param folder: with /
    :param distance: 0: l1; 1: l2; 2: linf
    :return:
    '''
    rs = 1
    ls = [str(val) for val in methods]
    results = []
    distances = []
    for l in ls:
        # Show images
        if visualize:

            pretrained_model = "./checkpoints/vae_{}.pth".format(methods[-1])
            targeted_model = VAE(
                encoder_layer_sizes=args.encoder_layer_sizes,
                latent_size=args.latent_size,
                decoder_layer_sizes=args.decoder_layer_sizes,
                conditional=args.conditional,
                num_labels=NUM_LABELS if args.conditional else 0)
            targeted_model.load_state_dict(torch.load(pretrained_model))
            targeted_model.eval()

            attack_images_cw = np.load('advs_old/cw_{}.npy'.format(l))
            attack_images_df = np.load('advs_old/df_{}.npy'.format(l))
            attack_images_fgsm = np.load('advs_old/fgsm_{}.npy'.format(l))


            for attack_number, attack_image in enumerate(attack_images_cw):
                targeted_model(torch.Tensor(attack_images_fgsm[attack_number]), visualize=str(attack_number) + "fgsm")
                targeted_model(torch.Tensor(attack_images_df[attack_number]), visualize=str(attack_number) + "df")
                targeted_model(torch.Tensor(attack_images_cw[attack_number]), visualize=str(attack_number) + "cw")
                # plt.imshow(attack_images_fgsm[attack_number][0])
                # plt.show()
                # plt.imshow(attack_images_df[attack_number][0])
                # plt.show()
                # plt.imshow(attack_images_cw[attack_number][0])
                # plt.show()
        results.append(np.load('results_old/correct_predictions_{}.npy'.format(l)))
        distances.append(np.load('results_old/distance_{}.npy'.format(l))[:, :, distanceChoice]/2)

    c = -1
    fig = plt.figure(dpi=100, figsize=(18, 4))
    axs = [0 for i in range(len(attacks))]

    cleanAccuracy = []
    for i in range(methodNum):
        cleanAccuracy.append(np.mean(results[i][:, 0]))
    print(cleanAccuracy)

    for j in range(1, attackNum+1):
        c += 1
        axs[c] = fig.add_axes([0.05 + c * 0.19, 0.20, 0.17, 0.55])

        # minYaxis = np.inf
        # maxYaxis = 0

        for i in range(methodNum):
            correctPrediction = results[i][:, j]

            dis, accu = accuracyAndDistance(correctPrediction, distances[i][:, j])

            # axs[c].plot(x, results[i, 0, :], label=methods[i] + ' Clean', color=colors[i], ls=':')
            axs[c].plot(dis, accu, label=methods[i], color=colors[i], ls='-')
            axs[c].title.set_text(attacks[j-1][0])
            axs[c].set_xlabel('Linf Bound')

            print(auc(dis, accu))

        print()

        # axs[c].set_xlim(0.1, 1)
        axs[c].set_ylim(0, 1)
        if c == 0:
            axs[c].set_ylabel('Accuracy')


    plt.legend(loc="upper center", bbox_to_anchor=(0, 1.5), ncol=methodNum, fancybox=True)
    plt.savefig("differentIte_old.pdf", dpi=350)



def loadData():
    if not os.path.exists("../../data/MNIST/test100/"):

        dataloader = torch.utils.data.DataLoader(
            MNIST(
                root="data",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=1,
            num_workers=4,
            shuffle=True,
        )
        count = np.zeros(10)
        teda = []
        tela = []
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.data.numpy()
            labels = labels.numpy()
            labels = int(labels)
            if count[labels] < 10:
                teda.append(imgs)
                tela.append(labels)
                count[labels] += 1
        teda = np.array(teda).reshape((-1, CHANNELS, WIDTH, HEIGHT))
        tela = np.array(tela)
        print(teda.shape)
        print(tela.shape)
        os.makedirs("../../data/MNIST/test100/")
        np.save("../../data/MNIST/test100/testdata.npy", teda)
        np.save("../../data/MNIST/test100/testlabel.npy", tela)
    else:
        teda = np.load("../../data/MNIST/test100/testdata.npy")
        tela = np.load("../../data/MNIST/test100/testlabel.npy")

    SMALL_TEST = True
    if SMALL_TEST:
        from random import sample
        examples = sample(range(len(tela)), 100)
        return teda[examples], tela[examples]
    return teda.reshape((-1, CHANNELS, WIDTH, HEIGHT)), tela

def accuracyAndDistance(correctPrediction, distance):
    # maxiD = np.max(distance)
    distance = distance
    total = float(distance.shape[0])
    print(total)
    accus = []
    dis = []
    for i in range(101):
        step = i/100.0
        index = np.where(distance <= step)[0]
        remain = total - len(index)
        predict = np.sum(correctPrediction[index])
        acc = (predict+remain)/total
        dis.append(step)
        accus.append(acc)
    return dis, accus

def distance3(a, b):
    a = a.reshape(-1,1)+1
    b = b.reshape(-1,1)+1
    return np.linalg.norm(a - b, ord=1), np.linalg.norm(a - b, ord=2), np.linalg.norm(a - b, ord=np.inf)

def ptest(args, checkpoints=None, targeted_model=None, training_set=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    trainacc = []
    testacc = []
    if checkpoints is None:
        checkpoints = [targeted_model]
    for checkpoint in checkpoints:
        trainloader = torch.utils.data.DataLoader(
            MNIST(
                "data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )

        testloader = torch.utils.data.DataLoader(
            MNIST(
                "data",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=True,
        )

        if targeted_model is None:
            pretrained_model = "./checkpoints/vae_{}.pth".format(checkpoint)
            targeted_model = VAE(
                encoder_layer_sizes=args.encoder_layer_sizes,
                latent_size=args.latent_size,
                decoder_layer_sizes=args.decoder_layer_sizes,
                conditional=args.conditional,
                num_labels=NUM_LABELS if args.conditional else 0).to(device)
            targeted_model.load_state_dict(torch.load(pretrained_model))
            targeted_model.eval()
            if cuda:
                targeted_model.cuda()

        if training_set:
            correct_num = 0
            total_num = 0
            for i, (imgs, labels) in enumerate(trainloader):
                if cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                imgs = Variable(imgs.type(Tensor), requires_grad=False)
                # print(targeted_model(imgs).argmax(1), labels)
                correct_num += np.sum((targeted_model(imgs).argmax(1) == labels).detach().cpu().numpy())
                total_num += len(imgs)
                #print(np.sum((targeted_model(imgs).argmax(1) == labels).detach().cpu().numpy()))
            print("Accuracy in training set: ", correct_num / total_num)
            trainacc.append(correct_num / total_num)

        correct_num = 0
        total_num = 0
        for i, (imgs, labels) in enumerate(testloader):
            if cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            # print(targeted_model(imgs).argmax(1), labels)
            imgs = Variable(imgs.type(Tensor), requires_grad=False)
            correct_num += np.sum((targeted_model(imgs).argmax(1) == labels).detach().cpu().numpy())
            total_num += len(imgs)
        print("Accuracy in testing set: ", correct_num / total_num)
        testacc.append(correct_num / total_num)
    return {'train': trainacc, 'test': testacc}


def test(args, checkpoints):
    '''
    Before running this function, you must run 'train_target_model.py' to train a cb3d model first
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for checkpoint in checkpoints:
        print("CUDA Available: ", torch.cuda.is_available())
        XTest, yTest = loadData()
        print(np.max(XTest), np.min(XTest))

        pretrained_model = "./checkpoints/vae_{}.pth".format(checkpoint)
        targeted_model = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            decoder_layer_sizes=args.decoder_layer_sizes,
            conditional=args.conditional,
            num_labels=NUM_LABELS if args.conditional else 0).to(device)
        targeted_model.load_state_dict(torch.load(pretrained_model))
        targeted_model.denormalize = True
        targeted_model.eval()
        if cuda:
            targeted_model.cuda()

        fmodel = foolbox.models.PyTorchModel(targeted_model, bounds=(np.min(XTest), np.max(XTest)), num_classes=NUM_LABELS, channel_axis=1)

        Xadv = [np.zeros_like(XTest) for attack in attacks]

        start = time.time()
        for i in range(XTest.shape[0]):
            img = XTest[i].reshape([CHANNELS, WIDTH, HEIGHT])
            for j, attack in enumerate(attacks):
                result = attack[1](fmodel)(input_or_adv=img, label=int(yTest[i]))
                Xadv[j][i] = img if result is None else result
                print(j, end=",")
            print('\r processed ', i+1, '/', XTest.shape[0], (time.time()-start)/60.0, 'minutes passed')

        print()
        print('Total cost', (time.time()-start)/3600.0, 'hours')

        if not os.path.exists('advs_old'):
            os.mkdir('advs_old')
        for i, attack in enumerate(attacks):
            np.save('advs_old/{}_{}'.format(attack[0], checkpoint), Xadv[i])

        distances = np.zeros([XTest.shape[0], len(attacks) + 1, 3])
        correct_predictions = np.zeros([XTest.shape[0], len(attacks) + 1])

        for i in range(XTest.shape[0]):
            for j, attack in enumerate(attacks):
                distances[i, 1 + j, :] = distance3(Xadv[j][i], XTest[i])

        XTest = Variable(torch.from_numpy(XTest), requires_grad=False)
        yTest = Variable(torch.from_numpy(yTest), requires_grad=False)

        Xadv = [Variable(torch.from_numpy(np.array(Xadv_example)), requires_grad=False) for Xadv_example in Xadv]
        if cuda:
            XTest = XTest.cuda()
            yTest = yTest.cuda()
            Xadv = [Xadv_example.cuda() for Xadv_example in Xadv]

        with torch.no_grad():
            correct_predictions[:,0] = (targeted_model(XTest).argmax(1).view(-1) == yTest.view(-1)).detach().cpu().numpy()
            print("Raw image: ", np.mean(correct_predictions[:, 0]))
            for i, _ in enumerate(attacks):
                correct_predictions[:, i + 1] = (targeted_model(Xadv[i]).argmax(1).view(-1) == yTest.view(-1)).detach().cpu().numpy()
                print("fgsm image: ", np.mean(correct_predictions[:, i + 1]))

        if not os.path.exists('results_old'):
            os.mkdir('results_old')
        np.save('results_old/correct_predictions_{}'.format(checkpoint), correct_predictions)
        np.save('results_old/distance_{}'.format(checkpoint), distances)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=int, default=256)
    parser.add_argument("--decoder_layer_sizes", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument("--test", default=[], nargs='*', help="Run adv_test procedure")
    parser.add_argument("--plot", default=[], nargs='*', help="Run plot procedure")
    parser.add_argument("--ptest_full", default=[], nargs='*', help="Run full performance test procedure")
    parser.add_argument("--ptest", default=[], nargs='*', help="Run performance test procedure")

    args = parser.parse_args()

    if len(args.test) > 0:
        test(args, args.test)
    elif len(args.ptest) > 0:
        ptest(args, args.ptest)
    elif len(args.ptest_full) > 0:
        ptest(args, args.ptest, training_set=True)
    elif len(args.plot) > 0:
        plot(args, args.plot)
    else:
        main(args)
