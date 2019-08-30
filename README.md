## 環境のセットアップ

```shell script
pip install keras
pip install tensorflow-gpu==1.13.1
pip install matplotlib
```

## LSTMについて

### 一般的な使われ方
LSTMのInputとOutputについて、以下の二つのパターンに分けられる。
- n入力n出力：系列データn個に対し、n個のデータを出力する
    + ５個の数値ベクトル（例[1,3,6,1,2]）を入力に、積算（例[1,4,10,11,13])）を求める問題
- n入力1出力：系列データn個に対し、1個のデータを出力する
    + 音楽のイントロ５秒を聞いて、何の曲かを当てるような問題

※LTSMに内包するノード数を最終的に一つの数字にするため、最後に出力１の全結合層をかませる。

n入力1出力も、基本はn入力n出力し、最後の一つの値を取り出している。  
そしてその最後の一つを損失関数で最適化するといった仕組みである。


n入力n出力は、出力全てが合うように最適化されるが、n入力1出力は最後の答えが合えばその過程はなんでも良い、
というような最適化のされ方をする。


### 応用的な使われ方　～LSTMをAutoEncoderのEncoderのように使う。～
後述するseq2seqでも使われている使われ方。
LSTMは、過去の状態を内部に記録する構造になっている。
n入力1出力をした最後のLTSMの状態を入力ベクトルの状態が圧縮されたベクトルとみることができる。  
この状態を用いることによって、有効活用することで、別の用途に使える。

SlideShareに公開されている[Kenji Uraiさんの資料](https://www.slideshare.net/KenjiUrai/kenji-urailstm)の次の図が、
イメージとしてわかりやすい。※この図自身はLSTMではないが内部状態を保存しているイメージが伝わると思う。
![図](https://image.slidesharecdn.com/urailstm-161223090443/95/lstm-long-shortterm-memory-12-638.jpg?cb=1483077109)


## LSTMの使われ方
LSTMが通常のニューラルネットの判別問題と違う使われ方がなされる場合もある。  
例えば、株価の予測である。  
LSTMで予測した次の値を、LSTMの入力に使う。例えば、次のような感じ。
```python
# 0-9までの1刻みのベクトルを生成
input_data = np.array(0,10)

# 0-10まで
predict_result = model.predict(input_data)
for i in np.arage(1,10):
    input_data = np.append(input_data, predict_result)
    input_data = np.delete(input_data,0)
    # modelはkerasの学習済みモデル
    predict_result = model.predict(input_data)
```

## 個人的に面白いと思った使われ方
seq2seqというやり方。  
英語から日本語などの翻訳に使われる。  
上記のAutoEncorderのEncoderのように使う方法の一つである。

EncorderとDecorderともに、LSTMである。  
英語の文をLSTMのn入力1出力を用いて、一旦内部状態である意味ベクトルにし、同一の意味ベクトルを初期状態にしたLSTMを使って日本語の文章を生成していく。
下記のwebサイトが参考になった。  
- https://keras.io/examples/lstm_seq2seq/
- http://higepon.hatenablog.com/entry/20171210/1512887715

似たようなものとしてAttentionがある。  
これについては、また後に記載する。

## コードで実践
実際にいくつか練習してみる。
- [simple_sin_prediction.py](./simple_sin_prediction.py)：シンプルにsin関数を学習し、次の点を推論する。
- [functional_api_sin_prediction.py](functional_api_sin_prediction.py)：functional_apiでsimple_sin_predictionと同じことをする。
- [functional_api_linear_prediction.py](./functional_api_linear_prediction.py)：直線を学習し、続きをかけるか検証する。