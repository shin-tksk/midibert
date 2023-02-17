# midibert

Music Transformer をいじってBERTにする

## トークン化

[PAD] 0  
[CLS] 1  
[MASK] 2  
[bar] 3 ← BERTの[SEP]  
[start] 4 ~ 99  
[duration] 100 ~ 115  
[pitch] 116 ~ 243  
[velocity] 244 ~ 259  
[tempo] 260 ~ 295  
[chord] 296 ~ 319  
[instrument]  

## もうちょい小さく

[PAD] 0  
[CLS] 1  
[MASK] 2  
[bar] 3 ← BERTの[SEP]  
[start] 4 ~ 51　(48)  
[duration] 52 ~ 67 (16)  
[pitch] 68 ~ 163 (96)  
[velocity] 164 ~ 179 (16)  

##  追加する処置

tensorflowのversionを上げる  
FFNの、ReLU → GELU  
Adam → RAdam  
不必要な処理を消す（処理速度をあげる）  
