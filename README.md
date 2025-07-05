# gazou_kiritori
画像切り取りツール、lora素材作成に使いやすいかも。
ドラッグや固定サイズで画像を切り抜き、簡単に保存できます。
## 特徴
- 画像のドラッグ＆ドロップで読み込み
- 固定サイズや自由な矩形でのトリミング
- 保存先フォルダの指定が可能
- シンプルで使いやすいインターフェース

## 使い方

1. Python（バージョン3.9以上推奨）をインストール
2. このリポジトリをダウンロードまたはクローン
3. 仮想環境を作成し、必要なライブラリをインストール
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

markdown
コピーする
編集する
4. アプリを起動
python gazou_kiritori.py

yaml
コピーする
編集する
または  
`run_gazou_kiritori.bat` をダブルクリック

## 必要なライブラリ

- PyQt6
- Pillow
- その他（requirements.txt参照）

## ライセンス

MIT License  
自由に改造・配布OKです！
