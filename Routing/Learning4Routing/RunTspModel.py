import pickle
import os
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from Network import *
from TspUntils import *

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def masked_accuracy(output, target, mask):
    with torch.no_grad():
        masked_output = torch.masked_select(output, mask)
        masked_target = torch.masked_select(target, mask)
        accuracy = masked_output.eq(masked_target).float().mean()
        return accuracy

def train():
    for epoch in range(epochs):
        # Train
        model.train()
        for batch_idx, (seq, length, target) in enumerate(train_loader):
            seq, length, target = seq.to(device), length.to(device), target.to(device)
            optimizer.zero_grad()
            log_pointer_score, argmax_pointer, mask = model(seq, length)
            # (batch * max_seq_len, max_seq_len)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            # (batch_size, max_seq_len)
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            train_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())

            if batch_idx % 20 == 0:
                print('Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'
                    .format(epoch, (batch_idx+1) * len(seq), len(train_loader.dataset),
                        100. * (batch_idx+1) / len(train_loader), train_loss.avg, train_accuracy.avg))

        # Test
        model.eval()
        for seq, length, target in test_loader:
            seq, length, target = seq.to(device), length.to(device), target.to(device)
            log_pointer_score, argmax_pointer, mask = model(seq, length)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            test_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            test_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())
        print('Epoch {}: Test\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, test_loss.avg, test_accuracy.avg))

if __name__ == '__main__':
    min_length = 5
    max_length = 10
    batch_size1 = 256
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True if use_cuda else False
    emb_dim = 100
    epochs = 100

    # 导入模型
    model = PointerNetwork(input_dim=2, embedding_dim=emb_dim, hidden_size=emb_dim).to(device)
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()
    lr = 0.01
    wd = 1e-5
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    print("正在加载训练数据集")
    filename = "../data/tsp_train_data.pkl"
    with open(filename, 'rb') as f:
        train_set = pickle.load(f) 

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size1, shuffle=True, collate_fn=sparse_seq_collate_fn)
    print("加载训练数据集完成，开始加载测试数据集")
    filename = "../data/tsp_test_data.pkl"
    with open(filename, 'rb') as f:
        test_set = pickle.load(f)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size1, shuffle=False, collate_fn=sparse_seq_collate_fn)
    print("加载测试数据集完成")

    # 训练模型
    is_train = True
    if is_train:
        train()
        path = '../model/'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = "my_model.pkl"
        f = open(path+filename, 'wb')
        pickle.dump(model, f)
        f.close()
    else:
        pass


