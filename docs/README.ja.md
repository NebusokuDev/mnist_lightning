# MNIST分類モデル - PyTorch Lightningによる実装

- [English](README.md)
- [日本語](README.ja.md)

このリポジトリは、PyTorch Lightningを使用してMNISTデータセットに対する画像分類モデルを訓練するサンプル実装です。2つの異なるモデル、`NanoFCNet`（全結合ネットワーク）および`NanoConvNet`（畳み込みニューラルネットワーク）を使用して、MNISTデータセットに基づいた分類タスクを行います。

## 目次

- [概要](#概要)
- [依存関係](#依存関係)
- [使用方法](#使用方法)
  - [セットアップ](#セットアップ)
  - [モデルのトレーニング](#モデルのトレーニング)
  - [ベストモデルのテスト](#ベストモデルのテスト)
- [ファイル構成](#ファイル構成)
- [ライセンス](#ライセンス)

## 概要

このプロジェクトでは、次の2つのモデルを使用してMNISTの数字画像を分類します。

- `NanoFCNet`: 全結合層のみを使用したシンプルなニューラルネットワーク。
- `NanoConvNet`: 畳み込み層を含むより複雑な構造のニューラルネットワーク。

PyTorch Lightningを使ってモデルのトレーニングと検証を簡単に管理します。最適なモデルはチェックポイント機能を使って保存されます。

## 依存関係

このプロジェクトを実行するための依存関係は以下の通りです:

- Python 3.7以上
- PyTorch
- PyTorch Lightning
- TorchMetrics
- torchvision
- matplotlib（必要に応じてグラフの描画に使用）

依存関係をインストールするには、以下のコマンドを実行してください:

```bash
pip install -r requirements.txt
```

`requirements.txt`の内容は以下の通りです:

```txt
torch>=2.0.0
pytorch-lightning>=2.0.0
torchmetrics
torchvision
matplotlib
```

## 使用方法

### セットアップ

まず、必要なデータセットをダウンロードします。MNISTデータセットは、`MNISTDataModule`クラスを使用して自動的にダウンロードされます。

```bash
python train.py
```

### モデルのトレーニング

`train.py`を実行することで、`NanoConvNet`（畳み込みネットワーク）モデルが訓練されます。訓練には、PyTorch Lightningの`Trainer`を使用しています。訓練中には、検証データセットでの精度をモニタリングし、最良のモデルを保存します。

トレーニングプロセス中に、最良のモデルが`./checkpoints`ディレクトリに保存されます。

### ベストモデルのテスト

トレーニングが完了した後、保存された最良のモデルをテストするために、以下のコードを実行できます。

```python
# ベストモデルのロード
best_model = LitMNIST.load_from_checkpoint(best_model_path, model=NanoFCNet())

# テストの実行
trainer.test(best_model, datamodule=datamodule)
```

## ファイル構成

```
(...)
  ├── data/                   # MNISTデータセットの格納先
  ├── train.py                # モデルのトレーニングを行うスクリプト
  └── README.md               # このドキュメント
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細については、`LICENSE`ファイルを参照してください。