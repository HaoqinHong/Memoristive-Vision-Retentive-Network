import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time as tm

from sklearn.metrics import confusion_matrix, accuracy_score
from model import RetNet, Transformer, RetNetGated, patch_transform
from data import get_loader


def train(model, epochs, size_batch, size_patch, dataset, path_model):
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

    return accuracy_best


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
    results = []
    for i in range(50):
        model = Transformer(num_layer=1, num_head=1, num_sequence=49, num_feature=16, pos=False)
        # model = RetNet(num_layer=1, num_head=1, num_sequence=49, num_feature=16, pos=False)
        # model = RetNetGated(num_layer=1, num_head=1, num_sequence=49, num_feature=16, pos=False)
        model.to('cuda')

        epochs = 50
        size_batch = 196
        size_patch = [4, 4]
        dataset = 'mnist'
        path_model = 'model'

        acc = train(model, epochs, size_batch, size_patch, dataset, path_model)

        results.append(acc)

    with open(f"{dataset}_{model.info()}_{tm.strftime('%d%H%M')}.txt", 'w') as fp:
        print(results, file=fp)
        print(np.mean(results), file=fp)
    # model.load_state_dict(th.load(path_model))
    # acc = test(model, size_batch, size_patch, dataset)
    # print(acc)
