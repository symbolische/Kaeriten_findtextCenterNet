#!/usr/bin/env python3
"""kaeriten.py — findtextCenterNet への返り点（訓点）後処理モジュール。

本家 findtextCenterNet の OCR 出力 JSON（<画像>.json）と元画像を受け取り、
本文の各文字ボックスについて「返り点スロット」（その文字の左下に置かれる
返り点のための領域）を元画像から切り出して 24 クラスの小規模 CNN で分類し、
予測した返り点を書き戻す。

出力:
  1. 拡張 JSON (<画像>.kaeriten.json)
     - 元 JSON の各 box に 'kaeriten'（予測した返り点。無ければ ''）と
       'kaeriten_score'（予測クラスの softmax 確率）を追加したもの。
     - ページ全体の文字列 'aozora_kaeriten'（下記 2 の内容）も追加する。
     - 元の 'text' / 'aozora' / 'noruby' は変更しない。
  2. 青空文庫形式テキスト (<画像>.aozora.txt)
     - 青空文庫「注記一覧 訓点」（https://www.aozora.gr.jp/annotation/kunten.html）
       の記法に従う。返り点は漢字の直後に ［＃…］、訓点送り仮名は
       ［＃（…）］ とし、両者が付く字では送り仮名を先、返り点を後に書く。
       合符（一レ・上レなど）は ［＃一レ］ のように 1 つの注記にする。
       例: 爲［＃レ］ / 問［＃（い）］［＃レ］
     - 本家がルビとして抽出した仮名は、訓点送り仮名として扱う（ルビの
       《…》表記は使わない）。グループ化された仮名は、縦位置が最も近い
       親文字に帰属させる。
     - ルビとして抽出されなかった仮名（本文ボックスとして並ぶ小書き仮名
       など）は本文にそのまま残る（本家の抽出範囲に依存する制限）。
     - 行は改行、ブロック間は空行で区切る（本家のページ文字列の組立と同じ）。
     - 注意: 空白文字はボックス情報に含まれないため、再構成した文字列では
       空白が落ちる（拡張 JSON 内の本家 'aozora' はそのまま保持される）。

分類器・幾何について:
  - スロットの切り出し幾何（文字中心 cx, cy、文字幅 w、行ピッチ pitch からの
    相対ウィンドウ、48x48 へのリサイズ）は学習時と同一のものを用いる。
  - インク抽出は、二値画像では学習時と同じ固定閾値、グレースケール画像では
    ページ単位の Otsu 二値化（学習時の条件に合わせたもの）。
  - 分類対象は本文ボックス（ruby=0 かつ w>=28）のみ。ルビ文字・小文字には
    返り点を付けない。

使い方:
  python kaeriten.py page.png                # page.png.json を読み、2 出力を書く
  python kaeriten.py page.png page.png.json  # OCR JSON を明示する場合
  python kaeriten.py page.png --model kaeriten_model.pt

依存: numpy, Pillow, PyTorch（本家の推論環境と同じ）。
"""

import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

# 24 クラス（学習時と同一の順序）
CLASSES = ['none', 'レ', '一', '二', '三', '上', '中', '下', '天', '地', '人',
           '一レ', '上レ', '天レ',
           '四', '五', '甲', '乙', '丙', '丁', '戊', '二レ', '中レ', '三レ']

MAIN_MIN_W = 28.0  # 本文ボックスとみなす最小幅（学習時と同一）


class Net(nn.Module):
    """返り点スロット分類器（学習時と同一の構造）。"""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        return self.fc(self.body(x).flatten(1))


def otsu_threshold(gray):
    """ページ単位 Otsu 閾値（同点の場合は最小の t を返す）。"""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / gray.size
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma2 = (mu_t * omega - mu) ** 2 / denom
    return int(np.nanargmax(sigma2))


def page_ink(gray):
    """学習時と同条件のインク（二値: True）配列を返す。

    - 二値画像（階調が 2 値以下）: 学習時と同じ固定閾値 gray < 140。
    - グレースケール画像: ページ単位 Otsu で二値化してからインクとする
      （グレースケール頁の学習データ作成時と同じ条件）。
    """
    if len(np.unique(gray)) <= 2:
        return gray < 140
    return gray <= otsu_threshold(gray)


def page_pitch(boxes):
    """行ピッチ（同一行内の本文ボックス cy 間隔の中央値）。学習時と同一の算出。"""
    lines = OrderedDict()
    for b in boxes:
        lines.setdefault((b['blockidx'], b['lineidx']), []).append(b)
    sp = []
    for bs in lines.values():
        bs = sorted(bs, key=lambda b: b['subidx'])
        yy = [b['cy'] for b in bs if b['w'] >= MAIN_MIN_W]
        sp += [yy[i + 1] - yy[i] for i in range(len(yy) - 1)]
    if sp:
        return float(np.median(sp))
    mains = [b['h'] for b in boxes if b['w'] >= MAIN_MIN_W]
    return float(np.median(mains)) if mains else 40.0


def crop_context(ink, cx, cy, w, pitch, S=48):
    """返り点スロットの context crop（学習時の切り出しと同一幾何）。

    文字の左下（縦書きでは行の継ぎ目側）に中心を置いた相対ウィンドウを
    切り出し、SxS の float 配列（ink=1）にする。
    """
    x0 = int(cx - 0.78 * w)
    x1 = int(cx + 0.38 * w)
    y0 = int(cy - 0.18 * pitch)
    y1 = int(cy + 1.18 * pitch)
    H, W = ink.shape
    canvas = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.float32)
    sx0, sx1 = max(0, x0), min(W, x1)
    sy0, sy1 = max(0, y0), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = ink[sy0:sy1, sx0:sx1].astype(np.float32)
    im = Image.fromarray((canvas * 255).astype(np.uint8)).resize((S, S), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def load_model(model_path):
    net = Net()
    state = torch.load(model_path, map_location='cpu')
    net.load_state_dict(state)
    net.eval()
    return net


def predict_kaeriten(image_path, ocr, net):
    """OCR JSON の各ボックスに対する返り点予測を返す。

    戻り値: {box のインデックス: (返り点文字列 or '', 確率)}
    分類対象外のボックスは ('', 0.0)。
    """
    gray = np.asarray(Image.open(image_path).convert('L'))
    ink = page_ink(gray)
    pitch = page_pitch(ocr['box'])
    targets = [i for i, b in enumerate(ocr['box'])
               if b.get('ruby', 0) == 0 and b['w'] >= MAIN_MIN_W]
    result = {i: ('', 0.0) for i in range(len(ocr['box']))}
    if not targets:
        return result
    crops = np.stack([crop_context(ink, ocr['box'][i]['cx'], ocr['box'][i]['cy'],
                                   ocr['box'][i]['w'], pitch) for i in targets])
    X = torch.from_numpy(crops)[:, None]
    with torch.no_grad():
        for s in range(0, len(X), 256):
            logits = net(X[s:s + 256])
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for j, i in enumerate(targets[s:s + 256]):
                cls = CLASSES[int(pred[j])]
                result[i] = ('' if cls == 'none' else cls, float(conf[j]))
    return result


# ---------------------------------------------------------------------------
# 青空文庫形式へのシリアライズ
# ---------------------------------------------------------------------------

def decode_ruby(text, outtype='aozora'):
    """本家 util_func.decode_ruby と同一の変換（互換用に残置）。

    本モジュールの青空文庫形式テキストは build_aozora_text が直接生成し、
    この関数は使わない。
    """
    if outtype == 'aozora':
        text = re.sub('\uFFF9(.*?)\uFFFA(.*?)\uFFFB', r'｜\1《\2》', text)
    elif outtype == 'noruby':
        text = re.sub('\uFFF9(.*?)\uFFFA(.*?)\uFFFB', r'\1', text)
    return text


def build_aozora_text(boxes, kaeriten):
    """ボックス列から、青空文庫形式（訓点注記）の文字列を作る。

    記法は青空文庫「注記一覧 訓点」に従う
    （https://www.aozora.gr.jp/annotation/kunten.html）:
      - 返り点は、漢字の直後に ［＃…］ と書く。合符（一レ・上レなど）は
        ［＃一レ］ のように 1 つの注記におさめる。レ点はカタカナの「レ」。
      - 訓点送り仮名は、漢字の直後に ［＃（…）］ と書く。
      - 返り点と訓点送り仮名が同じ字に付くときは、送り仮名を先、
        返り点を後に書く。例: 問［＃（い）］［＃レ］

    本家がルビとして抽出した仮名は、すべて訓点送り仮名として扱う。
    ルビグループ（親文字の連続＋仮名の連続）の仮名は、縦位置（cy）が
    最も近い親文字に帰属させる。親文字を伴わずに現れたルビ文字は、
    同じ行の直前の文字の送り仮名とする（行頭で直前が無い場合は
    そのまま本文に出す）。
    行は '\n'、ブロック間は空行（本家のページ文字列の組立と同じ）。
    """
    blocks = OrderedDict()
    for i, b in enumerate(boxes):
        blocks.setdefault(b['blockidx'], OrderedDict()).setdefault(b['lineidx'], []).append(i)

    def render(unit):
        s = unit['text']
        if unit['oku']:
            s += '［＃（' + ''.join(unit['oku']) + '）］'
        if unit['kaeri']:
            s += '［＃' + unit['kaeri'] + '］'
        return s

    block_texts = []
    for lines in blocks.values():
        line_strs = []
        for idxs in lines.values():
            idxs = sorted(idxs, key=lambda i: boxes[i]['subidx'])
            parts = []  # ('unit', unit dict) | ('raw', str)
            units = []  # unit dicts（出現順）

            def new_unit(i):
                u = {'text': boxes[i]['text'], 'cy': boxes[i]['cy'],
                     'oku': [], 'kaeri': kaeriten[i][0]}
                units.append(u)
                parts.append(('unit', u))
                return u

            k = 0
            while k < len(idxs):
                i = idxs[k]
                b = boxes[i]
                if b.get('rubybase', 0) == 1:
                    # 親文字の連続 + それに続くルビ文字の連続 を 1 グループとする
                    bases = []
                    while k < len(idxs) and boxes[idxs[k]].get('rubybase', 0) == 1:
                        bases.append(idxs[k])
                        k += 1
                    rubys = []
                    while k < len(idxs) and boxes[idxs[k]].get('ruby', 0) == 1:
                        rubys.append(idxs[k])
                        k += 1
                    base_units = [new_unit(j) for j in bases]
                    for r in rubys:
                        tgt = min(base_units,
                                  key=lambda u: abs(boxes[r]['cy'] - u['cy']))
                        tgt['oku'].append(boxes[r]['text'])
                    continue
                if b.get('ruby', 0) == 1:
                    # 親文字を伴わないルビ文字: 直前の文字の送り仮名とする
                    run = []
                    while k < len(idxs) and boxes[idxs[k]].get('ruby', 0) == 1:
                        run.append(idxs[k])
                        k += 1
                    if units:
                        for r in run:
                            units[-1]['oku'].append(boxes[r]['text'])
                    else:
                        parts.append(('raw', ''.join(boxes[r]['text'] for r in run)))
                    continue
                new_unit(i)
                k += 1
            line_strs.append(''.join(render(v) if kind == 'unit' else v
                                     for kind, v in parts))
        block_texts.append(''.join(s + '\n' for s in line_strs))
    return '\n'.join(block_texts).rstrip('\n')


def process(image_path, json_path=None, model_path=None, out_json=None, out_txt=None):
    if json_path is None:
        json_path = image_path + '.json'
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'kaeriten_model.pt')
    if out_json is None:
        out_json = image_path + '.kaeriten.json'
    if out_txt is None:
        out_txt = image_path + '.aozora.txt'

    with open(json_path, encoding='utf-8') as f:
        ocr = json.load(f)
    net = load_model(model_path)
    kaeriten = predict_kaeriten(image_path, ocr, net)

    for i, b in enumerate(ocr['box']):
        b['kaeriten'], b['kaeriten_score'] = kaeriten[i]
    aozora = build_aozora_text(ocr['box'], kaeriten)
    ocr['aozora_kaeriten'] = aozora

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(ocr, f, indent=2, ensure_ascii=False)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(aozora + '\n')

    n_marks = sum(1 for k, _ in kaeriten.values() if k)
    return {'boxes': len(ocr['box']), 'kaeriten': n_marks,
            'out_json': out_json, 'out_txt': out_txt}


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    model_path = None
    for a in argv[1:]:
        if a.startswith('--model='):
            model_path = a.split('=', 1)[1]
    if not args:
        print('usage: python kaeriten.py <page_image> [<ocr_json>] [--model=<path>]')
        return 1
    image_path = args[0]
    json_path = args[1] if len(args) > 1 else None
    info = process(image_path, json_path, model_path)
    print(f"{image_path}: boxes={info['boxes']} kaeriten={info['kaeriten']} "
          f"-> {info['out_json']}, {info['out_txt']}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
