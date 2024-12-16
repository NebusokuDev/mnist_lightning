import multiprocessing
from pathlib import Path

import torch
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Module, CrossEntropyLoss, MaxPool2d, ReLU, Conv2d, Sequential, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.datasets import MNIST


class NanoFCNet(Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        return torch.log_softmax(x, dim=1)


class NanoConvNet(Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            Conv2d(1, 32, 3, 1, 1),
            MaxPool2d(2, 2),
            ReLU()
        )

        self.classifier = Sequential(
            Linear(32 * 14 * 14, 10)  # 10クラスの分類
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LitMNIST(LightningModule):
    def __init__(self, model: Module, criterion: Module = None):
        super(LitMNIST, self).__init__()
        self.model = model
        self.criterion = criterion or CrossEntropyLoss()
        self.train_accuracy = MulticlassAccuracy(num_classes=10)
        self.val_accuracy = MulticlassAccuracy(num_classes=10)
        self.test_accuracy = MulticlassAccuracy(num_classes=10)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)
        acc = self.train_accuracy(predict, label)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        loss = self.criterion(predict, label)
        acc = self.val_accuracy(predict, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        predict = self.forward(image)
        acc = self.test_accuracy(predict, label)

        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_acc": acc}


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64, num_workers: int = multiprocessing.cpu_count()):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # データ変換
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # データセットのダウンロード
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # データセットの分割
        if stage == "fit" or stage is None:
            self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_val = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    # モデルとデータモジュールの初期化
    model = NanoConvNet()
    lit_model = LitMNIST(model)
    datamodule = MNISTDataModule()

    # 保存ディレクトリとチェックポイントコールバック
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="mnist-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="min"
    )

    # トレーナーの作成と実行
    trainer = Trainer(
        max_epochs=20,
        accelerator="auto",
        callbacks=[checkpoint_callback]
    )
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)

    # ベストモデルのパス
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    # モデルのロードとテスト
    best_model = LitMNIST.load_from_checkpoint(best_model_path, model=NanoFCNet())
    trainer.test(best_model, datamodule=datamodule)
