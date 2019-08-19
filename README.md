このリポジトリは目次ページ(table of contents,TOC)認識モデルのリポジトリです。

# Inference
inference_xception.pyを実行時に、以下のように入力ディレクトリと出力ディレクトリを指定してください。
python inference_xception,py --input_dir (入力ディレクトリ) --output_dir (出力ディレクトリ)

出力ディレクトリにTOC,notTOCディレクトリが作成され、目次かどうかに応じて入力ディレクトリの画像が振り分けられます。

# Training
datasetディレクトリ内の
traindata,valdataそれぞれに、
TOC:目次ページ
notTOC:非目次ページ
の画像を入れてください。valdataはバリデーション用のデータセットを入れてください。
xception_mokujiclassify_g.pyを実行すると学習が始まります。
学習済みの重みファイルはepochごとにcheckpoints_xcpディレクトリに出力されます。

