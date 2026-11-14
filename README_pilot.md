# findtextCenterNet + 返り点（パイロット版）

## English summary

This is a pilot fork of [lithium0003/findtextCenterNet](https://github.com/lithium0003/findtextCenterNet)
(MIT, © 2022 lithium0003) that adds **kaeriten** (返り点, the reading-order marks of
kanbun kundoku) recognition for modern typeset kanbun (近代活字漢文, vertical writing).
The stock OCR and its ruby extraction are used unchanged — ruby and okurigana are
treated uniformly as okurigana — and a small 24-class CNN classifies the kaeriten
slot at the lower-left of each recognised kanji box. Outputs are an extended JSON
(`*.kaeriten.json`) and an Aozora Bunko format text (`*.aozora.txt`) using the
official Aozora kunten notation (kaeriten as ［＃…］, okurigana as ［＃（…）］). The kaeriten weights
were trained only on 『古事類苑』 (Kojiruien) and 『大日本佛教全書』
(Dai-Nihon Bukkyō Zensho) — no other corpus. On held-out material the
classifier scores mark exact 0.966 (bitonal) / 0.950 (color) on Kojiruien
pages, 0.949 on four never-trained Dai-Nihon Bukkyō Zensho books, and
0.972 on eight further unseen Honkoku books. Some rare kaeriten classes
have no training examples at all and cannot be predicted; see the
limitations section. Warichu （割注） is out of scope.

---

## この fork が追加するもの

本家 findtextCenterNet は漢字・ルビ（送り仮名）の OCR を行いますが、訓読に必須の
**返り点**（レ点・一二三・上下など）は出力しません。このパイロット版は、本家の
推論結果に対して後処理モジュールを追加し、返り点を付与します。

追加ファイルは次の 4 つだけです（本家のファイルは一切変更していません）。

| ファイル | 内容 |
|---|---|
| `kaeriten.py` | 返り点後処理モジュール本体 |
| `kaeriten_model.pt` | 返り点スロット分類器の重み（約 385 KB） |
| `run_ocr_kaeriten.py` | 本家 OCR → 返り点付与を 1 コマンドで実行するラッパー |
| `README_pilot.md` | この文書 |

## 仕組み

1. 本家の OCR（`run_ocr.py`）で本文・ルビのボックスと文字列を得る（**無改変**）。
   ルビと送り仮名は区別せず、すべて「送り仮名」として本家のルビ抽出をそのまま使う。
2. 本文の各文字ボックスについて、その左下にある「返り点スロット」領域を
   元画像から切り出し（文字幅・行ピッチに対する相対ウィンドウ、48×48）、
   24 クラスの小規模 CNN（約 9.6 万パラメータ）で分類する。
   クラスは `none`（返り点なし）と、レ・一・二・三・四・五・上・中・下・天・地・人・
   甲・乙・丙・丁・戊 の単独点、および合符の一レ・二レ・三レ・上レ・中レ・天レ。
3. 結果を拡張 JSON と青空文庫形式テキストに書き出す。

## 出力と青空文庫形式の規約

入力 `page.png` に対して次を出力します。

- `page.png.kaeriten.json` — 本家の OCR JSON の各 box に `kaeriten`（予測した
  返り点、なければ空文字列）と `kaeriten_score`（softmax 確率）を追加したもの。
  本家の `text` / `aozora` / `noruby` フィールドは変更しません。ページ全体の
  返り点付き文字列も `aozora_kaeriten` フィールドに入ります。
- `page.png.aozora.txt` — 青空文庫形式テキスト。

青空文庫形式での訓点の置き方は、青空文庫の公式記法
（[注記一覧「訓点」](https://www.aozora.gr.jp/annotation/kunten.html)）に従います。

- **返り点**は、その字の直後に `［＃…］` として置きます。レ点はカタカナの
  「レ」です。合符（一レ・上レなど）は `［＃一レ］` のように 1 つの注記に
  おさめます。例: `爲［＃レ］義`
- **送り仮名**は、訓点送り仮名として、その字の直後に `［＃（…）］` として
  置きます。ルビの `《…》` 表記は使いません。本家がルビとして抽出した
  仮名がこれに当たります。例: `合［＃（アツマル）］`
- 返り点と送り仮名が同じ字に付くときは、公式記法どおり**送り仮名を先、
  返り点を後**に書きます。例: `問［＃（い）］［＃レ］`
- 本家がルビグループ（親文字の連続＋仮名の連続）として抽出した仮名は、
  縦位置が最も近い親文字の送り仮名として帰属させます。
- 行は改行、ブロック間は空行で区切ります（本家の組立と同じ）。

出力例（『続浄土宗全書』 Z02_0042、実際の出力から抜粋）:

```
○言經者經也。如［＃レ］科レ文本。以［＃二］同字［＃一］釋［＃二］※字義［＃一］是人師常規。故
鸞師論注上（十）云。設者説也。天台觀經疏上（六）云。觀ハ者觀也。又非［＃二］
```

注意: 再構成した文字列では、本家がボックス間の間隔から入れていた空白が
落ちます（拡張 JSON 内の本家 `aozora` は空白つきのまま保持されます）。
また、本家 OCR がルビとして抽出しなかった送り仮名（本文ボックスとして
並ぶ小書き仮名など。上の例の「觀ハ」の「ハ」）や、返り点そのものを文字
として読んでしまったもの（上の例の「科レ」の「レ」）は、本文にそのまま
残ります。

## 対象範囲と制限

- **近代活字・縦書きの漢文**を対象とします（明治以降の翻刻・活字本を想定）。
- **割注は対象外**です。割注内の文字や返り点は扱いません。
- 左ルビ・再読文字は対象外です。公式記法では再読文字の左ルビを
  `［＃「未」の左に「ザル」のルビ］` のような記述式の注記で表しますが、
  本パイロット版では検出も変換もしません（本家の抽出のまま本文に残ります）。
- 返り点の分類は本家 OCR の文字ボックス位置に依存します。ボックスがずれた
  文字では返り点も誤ります。
- 合符（一レ・上レ など）は 1 クラスとして予測します。甲レ・下レ は学習データに
  ほぼ現れないためクラスに含みません。
- **予測できない返り点があります（学習データ側の制約）**。学習に使った
  『古事類苑』『大日本佛教全書』の範囲では、次の返り点は出現が皆無のため
  本モデルは原理的に出力できません：**天・地・人（単独）、天レ・中レ・三レ、
  戊**。また **二レ・丁・五・丙** は学習例が 1–3 件しかなく実用になりません。
  **四・甲・乙** も学習例が 11–18 件と少なく、精度は限定的です。
  これらは資料側の出現数そのものが僅少なためで、将来の追加学習でも
  解消は見込みにくい制約です。主要な返り点（レ・一・二・三・上・中・下・
  一レ・上レ）は十分な学習例があります。

## 精度（実測）

いずれも分類器の学習に使っていない保留資料での実測です
（mark exact = 返り点のあるユニットでの一致率）。

- **古事類苑 保留 10 頁**（2,991 ユニット・返り点 685 件）:
  mark exact **0.966**（二値化頁）/ **0.950**（カラー頁）、
  全体一致率 0.984 / 0.984
- **『大日本佛教全書』保留 4 資料**（『明要鈔』『本朝諸宗要集』
  『伝律図源解集』『仁王経問答』、30,612 ユニット・返り点 5,230 件）:
  mark exact **0.949**、資料別 0.915 – 0.981
- **本刻未見書 8 書（EV01–08）**（返り点 1,059 件）:
  mark exact **0.972**
- 参考（学習に使っていないが本モデルの対象外の書）:
  『続浄土宗全書』保留 8 頁では mark exact 0.870 でした。
  書物・活字が変わると精度は下がります。

評価は `clf_common` 系の GT 位置合わせプロトコル（転記 GT との照合）によります。

## 使い方

前提: 本家 README の通り、Python 環境と本家の学習済み重み
`model.pt` / `model3.pt`（Hugging Face からダウンロードしてリポジトリ直下に配置）
が必要です。

```sh
# 本家 OCR + 返り点付与をまとめて実行
python run_ocr_kaeriten.py page.png

# 本家の OCR を済ませてある場合は後処理だけでも実行できます
python kaeriten.py page.png            # page.png.json を読み、2 つの出力を書く
```

## 重みの出典

返り点分類器（`kaeriten_model.pt`）の学習データは、次の 2 資料のみです
（合計 304,817 スロット、『続浄土宗全書』など他の資料のデータは含みません）。

- **『古事類苑』**（40,177 件）：国立国会図書館デジタルコレクションの
  頁画像と、データベース転記テキストとの照合で生成したラベル
  （引用箇所を選んだセレクティブ収穫）。
- **『大日本佛教全書』**（264,640 件）：東京大学 dheq.org が公開する
  SAT 研究会「みんなで翻刻」プロジェクトの翻刻テキスト
  （CC BY-SA、頁マーカー付き）を GT とし、国立国会図書館デジタル
  コレクションの同一頁画像と位置合わせして生成したラベル。
  翻刻テキストが存在しない資料・箇所は学習に含みません。

ラベルはいずれも人手の転記テキスト由来の実ラベル（銀ラベルではありません）です。

## ライセンス

- 本家 findtextCenterNet: MIT License（© 2022 lithium0003、`LICENSE` 参照）。
- 本パイロット版の追加分（`kaeriten.py`、`run_ocr_kaeriten.py`、本文書、
  `kaeriten_model.pt`）も MIT License とします。
