#!/usr/bin/env python3
"""run_ocr_kaeriten.py — 本家の OCR と返り点後処理を 1 コマンドで実行するラッパー。

使い方:
  python run_ocr_kaeriten.py page.png [--resize=1.0]

処理:
  1. 本家の run_ocr.py で OCR を実行し、page.png.json を生成する
     （model.pt / model3.pt がリポジトリ直下に必要。本家 README 参照）。
  2. kaeriten.py で返り点を付与し、page.png.kaeriten.json と
     page.png.aozora.txt（青空文庫形式）を生成する。

本家のファイルは一切変更しない。
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    opts = [a for a in argv[1:] if a.startswith('--')]
    if not args:
        print('usage: python run_ocr_kaeriten.py <page_image> [--resize=1.0]')
        return 1
    image = os.path.abspath(args[0])

    # 1) 本家 OCR（cwd はリポジトリ直下: モデル配置とモジュール解決のため）
    cmd = [sys.executable, os.path.join(HERE, 'run_ocr.py'), image] + opts
    print('+', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print('run_ocr.py failed with exit code', r.returncode)
        return r.returncode
    json_path = image + '.json'
    if not os.path.exists(json_path):
        print('OCR output not found:', json_path)
        return 1

    # 2) 返り点後処理
    sys.path.insert(0, HERE)
    import kaeriten
    info = kaeriten.process(image, json_path,
                            model_path=os.path.join(HERE, 'kaeriten_model.pt'))
    print(f"kaeriten: boxes={info['boxes']} kaeriten={info['kaeriten']} "
          f"-> {info['out_json']}, {info['out_txt']}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
