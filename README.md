# midibert

Music Transformer をいじってBERTにする

## トークン化

[PAD] 0  
[CLS] 1  
[MASK] 2  
[bar] 3 ← BERTの[SEP]  
[start] 4 ~ 51　(48)  
[duration] 52 ~ 67 (16)  
[pitch] 68 ~ 151 (84)  
[velocity] 152 ~ 167 (16)  

## 学習曲線

<img src="loss.png" width="600px"/>

## 変更点

tensorflowのversionを上げる  
FFNの、ReLU → GELU  
不必要な処理を消す  
