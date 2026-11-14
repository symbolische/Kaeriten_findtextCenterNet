# 公開手順（FORKING.md）

パイロット版を GitHub で公開するための手順です。作業済みのツリーは
`fork_pilot/tree/`（本家 lithium0003/findtextCenterNet の clone）にあり、
追加分はブランチ **`pilot-kaeriten`** にコミットしてあります。

## 前提となる調査結果：本家の重みは git 管理外

本家の `model.pt`（約 1.0 GB）と `model3.pt`（約 664 MB）は、**git では
管理されていません**（LFS でも通常の blob でもありません。本家の
`git ls-files` に存在せず、`.gitattributes` もありません）。
本家 README の通り、利用者が Hugging Face
（`https://huggingface.co/lithium0003/findtextCenterNet`）からダウンロードして
リポジトリ直下に置く運用です。

したがって fork 側も同じ運用でよく、**Git LFS は不要**です。
私たちの `kaeriten_model.pt` は約 385 KB なので、通常の blob として
そのままコミット済みです。

> 注意: 手元の作業コピーで `model.pt` / `model3.pt` をリポジトリ直下に
> 置いた場合、それらを **コミットしないでください**（GitHub の 1 ファイル
> 100 MB 制限に抵触し、push が拒否されます）。

## 手順

### 1. GitHub で本家を fork する

ブラウザで https://github.com/lithium0003/findtextCenterNet を開き、
**Fork** ボタンから自分のアカウントに fork を作ります。
（clone から新規リポジトリを作るのではなく、必ず Fork を使ってください。
upstream との fork 関係が保たれ、将来の本家更新の取り込みや PR が可能になります。）

### 2. 手元のツリーに自分の fork を remote として追加する

```sh
cd fork_pilot/tree
git remote add myfork https://github.com/<あなたのユーザー名>/findtextCenterNet.git
git fetch myfork
```

（`origin` は本家 lithium0003 のままにしておきます。）

### 3. ブランチを push する

```sh
git push myfork pilot-kaeriten
```

これで fork 側に `pilot-kaeriten` ブランチが公開されます。
コミットは追加ファイル 5 点のみです:

- `kaeriten.py`
- `kaeriten_model.pt`
- `run_ocr_kaeriten.py`
- `README_pilot.md`
- `FORKING.md`（この文書）

なお、このコミットの author は作業用の仮名義
（`kanbun-pilot <kanbun-pilot@localhost>`）で作ってあります。
気になる場合は push 前に `git commit --amend --reset-author` 等で
自分の名義に直してください。

### 4. GitHub 上での仕上げ（任意）

- リポジトリの説明（About）に「findtextCenterNet に返り点認識を追加した
  パイロット版」等を書くと見つけてもらいやすくなります。
- 説明の起点として `README_pilot.md` へのリンクを README 冒頭に 1 行足す
  改変は任意です（現行ツリーでは本家 README は無改変です）。
- デフォルトブランチを `pilot-kaeriten` に変えるかどうかはお好みで
  （変えると訪問者が最初にパイロット版を見ます）。

## 利用者側の導入（参考）

fork を clone した利用者は、本家 README の手順で `model.pt` / `model3.pt` を
Hugging Face から取得してリポジトリ直下に置けば、`run_ocr_kaeriten.py` が
そのまま動きます（詳細は `README_pilot.md` の「使い方」）。
