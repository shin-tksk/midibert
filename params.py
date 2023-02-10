import midi_processor.processor as sequence
# max_seq = 2048
max_seq=2048
l_r = 0.001
embedding_dim = 256
num_attention_layer = 6
batch_size = 8
loss_type = 'categorical_crossentropy'
event_dim = sequence.RANGE_START + sequence.RANGE_DURATION + sequence.RANGE_PITCH + sequence.RANGE_BAR + sequence.RANGE_VELOCITY + sequence.RANGE_BPM + sequence.RANGE_CHORD
#event_dim = 500
pad_token = 0
cls_token = 1
mask_token = 2
vocab_size = event_dim + 3
