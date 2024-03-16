import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time as tm
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
from model import RetNet, Transformer, patch_transform
from data import get_loader
import os

def train(model, epochs, size_batch, size_patch, dataset, path_model, args):
    loader_train, _ = get_loader(dataset, size_batch)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, verbose=True)

    accuracy_best = 0

    for epoch in range(epochs):
        start_time = tm.perf_counter()
        loss_running = []

        model.train()
        for step, (inputs, labels) in enumerate(loader_train):
            optimizer.zero_grad()

            inputs = patch_transform(inputs, size_patch)
            outputs = model(inputs.to('cuda'))

            loss = criterion(outputs, labels.to('cuda'))
            loss.backward(retain_graph=True)

            optimizer.step()
            loss_running.append(loss.item())

            if step % 100 == 0:
                print(f'[{epoch+1:<3} / {step+1:>6}]\tloss: {np.mean(loss_running):.6f}')

        accuracy = test(model, size_batch, size_patch, dataset)

        if accuracy_best < accuracy:
            accuracy_best = accuracy
            th.save(model.state_dict(), path_model + f'/{dataset}_{model.info()}.pth')

        end_time = tm.perf_counter()
        print(f'epoch: {epoch+1:<3}\taccuracy: {accuracy:.4f}\tbest accuarcy: {accuracy_best:.4f}\ttime: {end_time-start_time:.2f}')

        cos_decay.step()

    th.save(model.state_dict(), path_model + f'/{dataset}_{model.info()}_{accuracy_best:.4f}.pth')


def test(model, size_batch, size_patch, dataset):
    _, loader_test = get_loader(dataset, size_batch)

    with th.no_grad():
        model.eval()

        list_true = []
        list_pred = []
        for inputs_test, labels_test in loader_test:
            inputs_test = patch_transform(inputs_test, size_patch)
            outputs_test = model(inputs_test.to('cuda'))

            _, labels_pred = th.max(outputs_test.data, 1)
            list_true += labels_test.tolist()
            list_pred += labels_pred.tolist()

        accuracy = accuracy_score(y_true=list_true, y_pred=list_pred)
        # cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument("--embed_dim", type=int, default=16, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="Img size")
    parser.add_argument("--size_patch", type=list, default=[4, 4], help="Patch Size")
    parser.add_argument("--num_patches", type=int, default=16, help="Patch Size")
    parser.add_argument("--n_channels", type=int, default=256, help="Number of channels")
    parser.add_argument("--size_batch", type=int, default=256, help="Patch Batch")
    parser.add_argument('--dataset', type=str, default='mnist', help=['mnist', 'fmnist', 'cifar'])
    parser.add_argument('--epochs', type=int, default=49)
    # 是否使用位置编码
    parser.add_argument('--pos', type=bool, default=False, help='Use positional encoding')
    parser.add_argument('--model', type=str, default='retnet', help=['retnet', 'transformer'])
    parser.add_argument('--path_model', type=str, default='./model')
    
    
    # model = TypicalTransformer(8, 49, 16)
    # model = RetentiveTransformer(8, 49, 16, 0.9)

    # epochs = 50
    # size_batch = 256
    # size_patch = [4, 4]
    # dataset = 'mnist'
    # path_model = 'model'

    args = parser.parse_args()
    path_model = os.path.join('model', args.dataset)

    if args.model == 'retnet':
        if args.dataset == 'cifar':
        # RetNet(num_layer, num_head, num_sequence, num_feature, args)
            model = RetNet(1, 1, 64, 48, args).cuda()
        else:
            model = Transformer(1,1, 49, 16, args).cuda()
    else:
        if args.dataset == 'cifar':
            # Transformer(num_layer, num_head, num_sequence, num_feature, args)
            model = Transformer(1,1, 64, 48, args).cuda()
        else:
            model = Transformer(1,1, 49, 16, args).cuda()

    # print(model)
    print(args)
    # epochs = 50
    # size_batch = 256
    # size_patch = [4, 4]
    # dataset = 'mnist'
    # path_model = 'model'

    # train(model, epochs, size_batch, size_patch, dataset, path_model)
    train(model, args.epochs, args.size_batch, args.size_patch, args.dataset, args.path_model, args)

    # model.load_state_dict(th.load(path_model))
    # acc = test(model, size_batch, size_patch, dataset)
    # print(acc)