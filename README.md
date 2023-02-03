# midibert

Music Transformer をいじってBERTにする

## 現在の工夫点

[bar] 0 ← BERTの[SEP]  
[start] 1 ~ 96  
[duration] 97 ~ 112  
[pitch] 113 ~ 240  
[velocity] 241 ~ 256  
[tempo] 257 ~ 292  
[chord] 293 ~ 316  
[PAD] 317  

## 追加する処理

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
