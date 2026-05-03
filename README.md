# MNIST Diffusion Model / Flow Matching

MNISTを対象に、画像生成モデルの学習と推論を行うための学習用プロジェクトです。

このプロジェクトでは、次の2つを実装しています。

- Diffusion Model: DDPM形式のノイズ予測モデル
- Flow Matching: ノイズ分布からデータ分布へ移す速度場モデル

両者は、同じ時間条件付きU-Netを使います。違いは、学習時に何を予測させるかと、推論時にどのように画像を生成するかです。

さらに、生成したい数字 (0-9) を指定して画像を生成できる **数字条件付きサンプリング** に対応しています。実装は **Classifier-Free Guidance (CFG)** で、学習時にラベルを確率的にドロップして条件あり/無条件の両方を1つのモデルで学習し、推論時に `guidance-scale` で条件強度を調整します。

## 想定環境

想定している実行環境は次の通りです。

- CPU: 24コア
- GPU: NVIDIA GeForce RTX 4080
- メインメモリ: 32GB
- Python: 3.10以降を推奨
- PyTorch + torchvision

GPUが使える場合は自動的にCUDAを使います。CPUだけでも動きますが、学習時間は長くなります。

## セットアップ

ローカル環境（venv）で動かす場合は次の手順です。

```bash
cd project

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Windows PowerShellの場合は、仮想環境の有効化だけ次のように置き換えてください。

```powershell
.venv\Scripts\Activate.ps1
```

CUDA版PyTorchを明示的に入れたい場合は、PyTorch公式サイトの案内に従って環境に合うコマンドを使ってください。

Dockerで動かす場合は、同梱の `Dockerfile` と `scripts/run_container_wsl.sh` を利用できます。

```bash
bash scripts/run_container_wsl.sh
```

学習・生成は `scripts/` 配下のシェルスクリプトからも実行できます。

```bash
bash scripts/train_diffusion.sh
bash scripts/sample_diffusion.sh
bash scripts/train_flow.sh
bash scripts/sample_flow.sh
```

## ディレクトリ構成

```text
project/
├── README.md
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── run_container_wsl.sh
│   ├── train_diffusion.sh
│   ├── sample_diffusion.sh
│   ├── train_flow.sh
│   └── sample_flow.sh
└── src/
    ├── train_diffusion.py
    ├── sample_diffusion.py
    ├── train_flow.py
    ├── sample_flow.py
    └── mnist_gen/
        ├── __init__.py
        ├── data.py
        ├── diffusion.py
        ├── flow_matching.py
        ├── models.py
        └── utils.py
```

学習出力（チェックポイントや生成サンプル）は、リポジトリ直下の `outputs/` 以下に保存されます。

## Diffusion Modelの学習

```bash
python src/train_diffusion.py --epochs 20 --batch-size 256 --num-workers 8
```

学習済みモデルは次に保存されます。

```text
outputs/diffusion/checkpoints/last.pt
outputs/diffusion/checkpoints/best.pt
```

RTX 4080であれば、MNISTではこの設定で十分軽く動きます。可読性を優先して、EMAや混合精度学習は入れていません。

## Diffusion Modelによる生成

```bash
python src/sample_diffusion.py \
  --checkpoint outputs/diffusion/checkpoints/best.pt \
  --num-samples 64
```

生成結果は次に保存されます。

```text
outputs/diffusion/samples/diffusion_samples.png
```

数字を指定して生成したい場合は、`--label` と `--guidance-scale` を渡します。

```bash
python src/sample_diffusion.py \
  --checkpoint outputs/diffusion/checkpoints/best.pt \
  --num-samples 16 \
  --label 7 \
  --guidance-scale 3.0
```

`--label` を省略すると、`0..9` を循環するラベルでまとめて生成します。`--label 7` のように指定した場合、出力ファイル名には `_label7` が付与されます。

Diffusionの推論では、ランダムノイズから始めて、学習済みモデルで少しずつノイズを取り除きます。

## Flow Matchingの学習

```bash
python src/train_flow.py --epochs 20 --batch-size 256 --num-workers 8
```

学習済みモデルは次に保存されます。

```text
outputs/flow/checkpoints/last.pt
outputs/flow/checkpoints/best.pt
```

Flow Matchingでは、ノイズ画像 `x0` と本物画像 `x1` を結ぶ直線経路を考えます。時刻 `t` の中間画像は次の式で作ります。

```text
x_t = (1 - t) x0 + t x1
```

このとき、目標となる速度は次のようになります。

```text
v = x1 - x0
```

モデルは、中間画像 `x_t` と時刻 `t` から、この速度 `v` を予測するように学習します。

## Flow Matchingによる生成

```bash
python src/sample_flow.py \
  --checkpoint outputs/flow/checkpoints/best.pt \
  --num-samples 64 \
  --steps 100
```

生成結果は次に保存されます。

```text
outputs/flow/samples/flow_samples.png
```

数字を指定して生成したい場合は、`--label` と `--guidance-scale` を渡します。

```bash
python src/sample_flow.py \
  --checkpoint outputs/flow/checkpoints/best.pt \
  --num-samples 16 \
  --steps 100 \
  --label 3 \
  --guidance-scale 3.0
```

Flow Matchingの推論では、ノイズ画像から始めて、学習済み速度場に沿って常微分方程式をEuler法で積分します。CFG有効時は条件あり/無条件の速度を `(1+w)·v_cond − w·v_uncond` で合成します。

## 主な引数

学習スクリプトでよく使う引数は次の通りです。

```bash
--epochs        学習エポック数
--batch-size    バッチサイズ
--lr            学習率
--num-workers   DataLoaderのワーカー数
--base-channels U-Netの基本チャンネル数
--out-dir       出力先
```

Diffusionでは追加で次を指定できます。

```bash
--timesteps     拡散ステップ数。標準は1000
```

Flow Matchingの生成では追加で次を指定できます。

```bash
--steps         Euler法の積分ステップ数。標準は100
```

数字条件付き (CFG) に関する引数は次の通りです。

学習時:

```bash
--num-classes   クラス数。MNISTは10。0を指定すると無条件モデルとして学習
--p-uncond      学習時にラベルを無条件トークンに置換する確率。標準は0.1
```

サンプリング時:

```bash
--label           生成する数字 (0-9)。省略時は 0..9 を循環
--guidance-scale  CFGの強度 w。0で無条件、値を上げるほどラベルへの追従が強くなる。標準は3.0
--num-classes     省略時はチェックポイントから復元
```

## 勉強するときの見方

`src/mnist_gen/models.py` に共通の時間条件付きU-Netがあります。DiffusionとFlow Matchingの違いを比較したい場合は、まず次の2つを見ると分かりやすいです。

```text
src/mnist_gen/diffusion.py
src/mnist_gen/flow_matching.py
```

Diffusionでは、モデルの出力を「加えられたノイズ」と見なします。Flow Matchingでは、モデルの出力を「画像を動かす速度」と見なします。どちらも、入力は「時刻付きの画像」で、出力は画像と同じ形のテンソルです。

## 注意

MNISTは画像サイズが小さいため、この実装ではシンプルなU-Netで十分です。CIFAR-10や高解像度画像に拡張する場合は、モデル規模、正規化、サンプリング手法、EMA、混合精度学習などを追加する必要があります。
