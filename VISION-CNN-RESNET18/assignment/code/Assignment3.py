# NECESSARY PYTHON LIBRARIES
import torch
import torchvision
import torchvision.transforms as tf
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Sequential, Flatten, Dropout, Linear, ReLU, BatchNorm2d, LogSoftmax, NLLLoss
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from collections import OrderedDict

# BATCH SIZE, LEARNING RATE, CATEGORIES IN DATASET, AND NUMBER OF EPOCH
BATCH_SIZE = 150
LEARNING_RATE = 0.001
CATEGORIES = ['airport_inside', 'artstudio', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore',
              'bowling', 'buffet', 'casino', 'church_inside', 'classroom', 'closet', 'clothingstore',
              'computerroom']
EPOCHS = 7

# TRAINING FOR CNN MODEL
def train(model, device, train_loader, optimizer, epoch, loss_criteria):

    model.train()
    train_loss = 0
    correct = 0

    print('Epoch:', epoch)

    for batch_index, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criteria(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        correct += torch.sum(target == predicted).item()

        print('\tTraining batch {} loss: {:.4f}'.format(batch_index + 1, loss.item()))

    avg_train_loss = train_loss / (batch_index + 1)
    train_acc = 100. * correct / len(train_loader.dataset)
    print('Training set average loss: {:.4f}, Training accuracy {:.1f}'.format(avg_train_loss, train_acc))

    return avg_train_loss, train_acc

# TESTING AND EVALUATING CNN MODEL
def test(model, device, test_loader, loss_criteria):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_criteria(output, target)
            test_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    avg_val_loss = test_loss / batch_count
    val_acc = 100. * correct / len(test_loader.dataset)
    print('Validation set average loss: {:.4f}, Validation accuracy: {:.1f}'.format(avg_val_loss,
                                                                                    val_acc))

    return avg_val_loss, val_acc

# MAIN FUNCTION
def main():

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    transform = tf.Compose([tf.ToTensor(),
                            tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_set = torchvision.datasets.ImageFolder(
        root="C:/Users/can_a/PycharmProjects/Assignment3/train",
        transform=transform
    )

    test_set = torchvision.datasets.ImageFolder(
        root="C:/Users/can_a/PycharmProjects/Assignment3/test",
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False
    )

    class CNN(nn.Module):
        def __init__(self, num_of_class=15):
            super().__init__()

            self.model = Sequential(
                Conv2d(3, 8, kernel_size=11, stride=4, padding=0), ReLU(inplace=True),
                Conv2d(8, 16, kernel_size=7, stride=1, padding=2), ReLU(inplace=True),
                MaxPool2d(2, 2),

                Conv2d(16, 32, kernel_size=4, stride=1, padding=1), ReLU(inplace=True),
                Conv2d(32, 64, kernel_size=4, stride=1, padding=1), ReLU(inplace=True),
                Conv2d(64, 128, kernel_size=4, stride=1, padding=1), ReLU(inplace=True),
                MaxPool2d(2, 2),
            ).to(device)

            self.classifier = Sequential(
                Flatten(),
                Linear(3200, 1600),
                ReLU(inplace=True),
                Dropout(0.50),

                Linear(1600, num_of_class),
                ReLU(inplace=True)
            ).to(device)

        def forward(self, x):
            x = self.model(x)
            x = self.classifier(x)
            return x

    model = CNN(num_of_class=15).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_criteria = nn.CrossEntropyLoss()

    epoch_list = []
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, loss_criteria)
        test_loss = test(model, device, test_loader, loss_criteria)
        epoch_list.append(epoch)
        training_loss.append(train_loss[0])
        validation_loss.append(test_loss[0])
        training_acc.append(train_loss[1])
        validation_acc.append(test_loss[1])

    total_acc = 0
    for i in validation_acc:
        total_acc += i/EPOCHS
    print('Accuracy of the my architecture: {:.1f}'.format(total_acc))

    plt.figure(figsize=(15, 15))
    plt.plot(epoch_list, training_loss)
    plt.plot(epoch_list, validation_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Change -> {} Batch Size, {} Learning Rate'.format(BATCH_SIZE, LEARNING_RATE), fontsize=15)
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.plot(epoch_list, training_acc)
    plt.plot(epoch_list, validation_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Change -> {} Batch Size, {} Learning Rate'.format(BATCH_SIZE, LEARNING_RATE), fontsize=15)
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

    labels = []
    predictions = []
    model.eval()

    for data, target in test_loader:
        for label in target.data.numpy():
            labels.append(label)
        for prediction in model(data).data.numpy().argmax(1):
            predictions.append(prediction)

    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)
    plt.figure(figsize=(16, 16))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plt.show()


    resnet18 = torchvision.models.resnet18(pretrained=True).to(device)

    for param in resnet18.parameters():
        param.requires_grad = False

    last_layer = Sequential(OrderedDict([
        ('conv1', Conv2d(256, 50, kernel_size=3, stride=4, padding=0)),
        ('conv2', Conv2d(50, 25, kernel_size=3, stride=1, padding=2)),
        ('relu', ReLU()),
        ('pool1', MaxPool2d(2, 2)),
        ('conv3', Conv2d(25, 50, kernel_size=1, stride=1, padding=1)),
        ('conv4', Conv2d(50, 256, kernel_size=1, stride=1, padding=1)),
        ('conv5', Conv2d(256, 512, kernel_size=1, stride=1, padding=1)),
        ('relu', ReLU())
    ]))
    fc = Sequential(OrderedDict([
        ('fc1', Linear(512, 256)),
        ('relu', ReLU()),
        ('fc2', Linear(256, 15)),
        ('output', LogSoftmax(dim=1))
    ]))

    fc = Sequential(OrderedDict([
        ('fc1', Linear(512, 256)),
        ('relu', ReLU()),
        ('fc2', Linear(256, 15)),
        ('output', LogSoftmax(dim=1))
    ]))

    resnet18.layer4 = last_layer
    resnet18.fc = fc
    resnet18.to(device)

    optimizer_fc = Adam(resnet18.fc.parameters(), lr=0.001)
    criterion = NLLLoss()

    epoch_list_rn = []
    training_loss_rn = []
    validation_loss_rn = []
    training_acc_rn = []
    validation_acc_rn = []

    for e in range(1, EPOCHS + 1):
        train_loss_rn = train(resnet18, device, train_loader, optimizer_fc, e, criterion)
        test_loss_rn = test(resnet18, device, test_loader, criterion)
        epoch_list_rn.append(e)
        training_loss_rn.append(train_loss_rn[0])
        validation_loss_rn.append(test_loss_rn[0])
        training_acc_rn.append(train_loss_rn[1])
        validation_acc_rn.append(test_loss_rn[1])

    total_acc_rn = 0
    for i in validation_acc_rn:
        total_acc_rn += i / EPOCHS
    print('Accuracy of the ResNet18 architecture: {:.1f}'.format(total_acc_rn))

    labels_rn = []
    predictions_rn = []
    resnet18.eval()

    for data, target in test_loader:
        for label in target.data.numpy():
            labels_rn.append(label)
        for prediction in resnet18(data).data.numpy().argmax(1):
            predictions_rn.append(prediction)

    cm = confusion_matrix(labels_rn, predictions_rn)
    df_cm = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)
    plt.figure(figsize=(16, 16))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plt.show()


if __name__ == '__main__':
    main()
