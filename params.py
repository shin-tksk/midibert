import midi_processor.processor as sequence
max_seq=2048
l_r = 0.001
embedding_dim = 128
num_attention_layer = 8
batch_size = 32
loss_type = 'categorical_crossentropy'
#event_dim = sequence.RANGE_START + sequence.RANGE_DURATION + sequence.RANGE_PITCH + sequence.RANGE_BAR + sequence.RANGE_VELOCITY + sequence.RANGE_BPM + sequence.RANGE_CHORD
event_dim = sequence.RANGE_START + sequence.RANGE_DURATION + sequence.RANGE_PITCH + sequence.RANGE_BAR + sequence.RANGE_VELOCITY
pad_token = 0
cls_token = 1
mask_token = 2
vocab_size = event_dim + 3
