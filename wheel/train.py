import warnings
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

from model import *
from utils import *

warnings.filterwarnings("ignore")

same_seeds(0)

if __name__ == "__main__":
    ## 基本设置
    print(1)
    opt = func_config()
    func_folder(opt)
    writer = SummaryWriter(log_dir=opt.logdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    val_dataset = torchvision.datasets.MNIST(root='./data/',
                                             train=False,
                                             transform=transforms.ToTensor())

    ## Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True)  # 训练时，数据打乱

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False)  # 测试时，不打乱

    ## 模型初始化
    model = ConvNet(opt.num_classes).to(device)

    ## 优化器和学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=opt.lr)

    ## 查看模型基本状况
    print("模型参数个数：", sum([param.numel() for param in model.parameters() if param.requires_grad]))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    ## 训练
    for epoch in range(opt.num_epochs):
        print("\n----", epoch, "----\n")

        # 训练
        model.train()

        record = {"loss_0": AverageMeter()}

        for batch, (x, y) in enumerate(tqdm(train_loader, total=len(train_loader))):

            batches_done = len(train_loader) * epoch + batch

            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = func_loss(x, y, logits)

            (loss['loss_0'] / opt.ga_batches).backward()

            if batches_done % opt.ga_batches == 0:
                optimizer.step()
                optimizer.zero_grad()

            record["loss_0"].update(loss["loss_0"].cpu().detach())

            writer.add_scalars("batch代价函数", {"训练集": loss["loss_0"].cpu().detach()}, batches_done)

        scheduler.step()
        writer.add_scalars("总代价函数", {"训练集": record["loss_0"].avg}, epoch)

        # 验证
        if epoch % opt.val_epochs == 0:
            model.eval()

            record = {"loss_0": AverageMeter()}

            for batch, (x, y) in enumerate(tqdm(val_loader, total=len(val_loader))):
                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    logits = model(x)
                    loss = func_loss(x, y, logits)

                record["loss_0"].update(loss["loss_0"].cpu().detach())

            writer.add_scalars("总代价函数", {"验证集": record["loss_0"].avg}, epoch)

        if epoch % opt.save_epochs == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_%d.pth" % epoch)
