import numpy as np
import os
import miditoolkit
import pprint
import numpy as np
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
from operator import itemgetter
from midi_processor.chord import detect_chord


RANGE_START = 48
RANGE_DURATION = 16
RANGE_PITCH = 84
RANGE_BAR = 1
RANGE_VELOCITY = 16 # 40 ~ 115 5刻み
RANGE_BPM = 0 # 25 ~ 200 5刻み
RANGE_CHORD = 0

#RANGE_INSTRUMENT = 11
RANGE_INSTRUMENT = 0

class Note:

    def __init__(self, name, start, duration, pitch, velocity, instrument):
        self.name = name
        self.start = start
        self.duration = duration
        self.pitch = pitch
        self.velocity = velocity
        self.instrument = instrument

    def __repr__(self):
        return "<Note : name = {}, start = {}, duration = {}, pitch = {}, velocity = {}, instrument = {}>".format(self.name, self.start, self.duration, self.pitch, self.velocity, self.instrument)

class Event:

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return "<Event : name = {}, value = {}>".format(self.name, self.value)

## encode ##
def midi2note(midi):

    bpm_list = midi.tempo_changes
    tick = midi.ticks_per_beat

    note_list = []

    cur_time = -1
    cur_bpm = 0
    count = -1
    
    '''
    for b in bpm_list:

        bpm = int(b.tempo // 5) * 5
        start = int(b.time / tick * 24)

        if bpm < 25:
            bpm = 25
        elif bpm > 200:
            bpm = 200

        if bpm != cur_bpm:

            if start == cur_time:
                note_list.pop(count)
                count -= 1

            note_list.append(Note('bpm', start-0.01, None, bpm, None, None))
            cur_bpm = bpm
            cur_time = start  
            count += 1 
    '''
    max_end = 0
    for i in midi.instruments:
        
        # instrument off
        
        if i.is_drum is False:
            
            for n in i.notes:
                start = n.start / tick * 12
                end = n.end / tick * 12
                velocity = n.velocity - n.velocity % 5
                #print(start,end)
                if 24 <= n.pitch < RANGE_PITCH + 24:
                    note = Note('note', start, end-start, n.pitch-24, velocity, 0)
                    note_list.append(note)
                    #print(n)
                    #print(end)
                    if  end > max_end:
                        max_end = end
                else:
                    #bar = int(np.round(start) // 48)
                    #start = int(np.round(start) % 48)
                    #print(bar, start, n.pitch)
                    continue
        
        '''
        # instrument on
        if i.is_drum is False and i.program < 81:
            
            for n in i.notes:
        
                n.start = int(n.start / tick * 12)
                n.end = int(n.end / tick * 12)
                n.velocity = n.velocity - n.velocity % 5
                note = Note('note', n.start, n.end-n.start, n.pitch, n.velocity, i.program)
                note_list.append(note)
        
        elif i.is_drum is True and i.program == 0:
            
            for n in i.notes:
        
                n.start = int(n.start / tick * 12)
                n.end = int(n.end / tick * 12)
                n.velocity = n.velocity - n.velocity % 5
                note = Note('note', n.start, n.end-n.start, n.pitch, n.velocity, 81)
                note_list.append(note)
        '''
    #pprint.pprint(note_list)
    note_list = sorted(note_list, key = lambda k: k.start)
    #pprint.pprint(note_list)
    #print(max_end)
    return note_list

def chord2note(notes):

    vector = np.zeros(12)
    counter = np.zeros(12)
    num = 1
    chord_list = []
    note_list = []

    for n in notes:

        if n.name != 'note':
            continue

        else:
            if n.start >= 48 * num:

                counter = np.where(counter == 0, 1, counter)
                chord = vector / counter
                chord_num = detect_chord(chord)
                #print(chord_num)
                chord_list.append(Note('chord', 48 * (num - 1) - 0.001, None, chord_num, None, None))
                vector = np.zeros(12)
                counter = np.zeros(12)
                num += 1
                
            idx = n.pitch % 12
            vector[idx] += n.velocity
            counter[idx] += 1

    note_list = notes + chord_list
    note_list = sorted(note_list, key = lambda k: k.start)
    #pprint.pprint(note_list)
    return note_list

def note2event(notes):

    event_list = []
    cur_bar = -1
    cur_inst = -1
    dur_list = [1,2,3,4,6,8,12,16,24,32,48,96,120,144,168,192]

    for n in notes:

        bar = int(np.round(n.start) // 48)
        start = int(np.round(n.start) % 48)
        #print(bar,start)
        if bar > cur_bar:
            for i in range(bar - cur_bar):
                event = Event('bar', cur_bar + i + 1)
                event_list.append(event)
            cur_bar = bar 
        #if n.name == 'bpm':
        #    event_list.append(Event('bpm', (n.pitch - 25) // 5))

        #elif n.name == 'chord':
        #    if n.pitch is not None:
        #        event_list.append(Event('chord', n.pitch))
        #    else:
        #        continue

        if n.name == 'note':

            '''
            inst = n.instrument // 8
            if inst != cur_inst:
                event_list.append(Event('instrument', inst))
                cur_inst = inst
            '''

            event_list.append(Event('start', start))

            duration = np.abs(np.asarray(dur_list) - n.duration).argmin()
            event_list.append(Event('duration', duration))
            '''
            if duration > RANGE_DURATION:
                event_list.append(Event('duration', RANGE_DURATION))
            elif duration == 0:
                event_list.append(Event('duration', 1))
            else:
                event_list.append(Event('duration', duration))
            '''

            pitch = n.pitch
            event_list.append(Event('pitch', pitch))

            '''
            instrument = n.instrument
            event_list.append(Event('instrument', instrument // 8))
            '''

            velocity = n.velocity
            if velocity > 110:
                event_list.append(Event('velocity', 15))
            elif velocity <= 40:
                event_list.append(Event('velocity', 0))
            else:
                event_list.append(Event('velocity', (velocity - 40) // 5))
        
        #else:
        #    print('error')
        #break 
    #pprint.pprint(event_list[:50])
    return event_list

def event2word(events):

    word_list = []

    for e in events:
        #print(e)
        if e.name == 'bar':
            word_list.append(3)
        elif e.name == 'start':
            word_list.append(3 + e.value + RANGE_BAR)
        elif e.name == 'duration':
            word_list.append(3 + e.value + RANGE_BAR + RANGE_START)
        elif e.name == 'pitch':
            word_list.append(3 + e.value + RANGE_BAR + RANGE_START + RANGE_DURATION )
        elif e.name == 'velocity':
            word_list.append(3 + e.value + RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH)
        #elif e.name == 'bpm':
        #    word_list.append(3 + e.value + RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH + RANGE_VELOCITY)
        #elif e.name == 'chord':
        #    word_list.append(3 + e.value + RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH + RANGE_VELOCITY + RANGE_BPM)
        #elif e.name == 'instrument':
        #    word_list.append(e.value + RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH + RANGE_VELOCITY + RANGE_BPM)
        else:
            print('error')
    
    return word_list

def encode_midi(path):
    try:
        midi = miditoolkit.midi.parser.MidiFile(path)
        notes = midi2note(midi)
        #notes = chord2note(notes)
        #pprint.pprint(notes[:50])
        events  = note2event(notes)
        #pprint.pprint(events[:50])
        words = event2word(events)
        #print()
        #print(words[:20])
        return words
    
    except:
        print()
        print(path)


## decode ##
'''
RANGE_START = 96
RANGE_DURATION = 16
RANGE_PITCH = 128
RANGE_BAR = 1
RANGE_VELOCITY = 16 # 40 ~ 115 5刻み
RANGE_BPM = 36 # 25 ~ 200 5刻み
'''

def word2event(words):
    
    event_list = []

    for w in words:

        if w < 3:
            continue

        elif w == 3:
            event_list.append(Event('bar', 0))

        elif w < 3 + RANGE_BAR + RANGE_START:
            event_list.append(Event('start', w - 3 - RANGE_BAR))

        elif w < 3 + RANGE_BAR + RANGE_START + RANGE_DURATION:
            event_list.append(Event('duration', w - 3 - RANGE_BAR - RANGE_START))

        elif w < 3 + RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH:
            event_list.append(Event('pitch', w - 3 - RANGE_BAR - RANGE_START - RANGE_DURATION))

        elif w < 3 + RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH + RANGE_VELOCITY:
            event_list.append(Event('velocity', ((w - 3 - RANGE_BAR - RANGE_START - RANGE_DURATION - RANGE_PITCH) * 5 ) + 40))
        
        #elif w < 3 + RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH + RANGE_VELOCITY + RANGE_BPM:
        #    event_list.append(Event('bpm', w - 3 - RANGE_BAR - RANGE_START - RANGE_DURATION - RANGE_PITCH - RANGE_VELOCITY))

        #elif w < 3 + RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH + RANGE_VELOCITY + RANGE_BPM + RANGE_CHORD:
        #    event_list.append(Event('chord', w - 3 - RANGE_BAR - RANGE_START - RANGE_DURATION - RANGE_PITCH - RANGE_VELOCITY - RANGE_BPM))

        #elif w < RANGE_BAR + RANGE_START + RANGE_DURATION + RANGE_PITCH + RANGE_VELOCITY + RANGE_BPM + RANGE_INSTRUMENT:
        #    event_list.append(Event('instrument', w - RANGE_BAR - RANGE_START - RANGE_DURATION - RANGE_PITCH - RANGE_VELOCITY - RANGE_BPM))
        
        else:
            continue
    #pprint.pprint(event_list[:20])
    return event_list

def event2note(events):

    cur_bar = 0
    note_list = [] 
    dur_list = [1,2,3,4,6,8,12,16,24,32,48,96,120,144,168,192]

    for i in range(len(events)-3): # inst off
    #for i in range(len(events)-4): # inst on
        
        if events[i].name == 'bar':
            cur_bar += 1

        elif events[i].name == 'bpm':

            if events[i+1].name == 'start':
                start = cur_bar * 96 + events[i+1].value
                pitch = events[i].value * 5 + 25
            elif events[i+1].name == 'bar':
                start = (cur_bar + 1) * 96
                pitch = events[i].value * 5 + 25
            else:
                start = cur_bar * 96
                pitch = events[i].value * 5 + 25

            note_list.append(Note('bpm', start, 0, pitch, 0, 0))

        elif events[i].name == 'start':
            if events[i+1].name == 'duration' and events[i+2].name == 'pitch': # inst off
            #if events[i+1].name == 'duration' and events[i+2].name == 'pitch' and events[i+3].name == 'instrument': # inst on

                start = events[i].value + 48 * cur_bar
                duration = dur_list[events[i+1].value]
                pitch = events[i+2].value + 24

                # inst on
                '''
                #instrument = events[i+3].value
                #if events[i+4].name == 'velocity':
                #    velocity = events[i+4].value
                '''
                # inst off
                instrument = 0
                if events[i+3].name == 'velocity':
                    velocity = events[i+3].value

                else:
                    velocity = 80
                
                note_list.append(Note('note', start, duration, pitch, velocity, instrument))
    #pprint.pprint(note_list[:10])
    return note_list

def save_midi(notes,path):

    midi = mid_parser.MidiFile()

    piano = ct.Instrument(program=0, is_drum=False, name='piano')
    midi.instruments = [piano] 

    
    '''
    chromatic = ct.Instrument(program=12, is_drum=False, name='chromatic')
    organ = ct.Instrument(program=16, is_drum=False, name='organ')
    guitar = ct.Instrument(program=26, is_drum=False, name='guitar')
    bass = ct.Instrument(program=33, is_drum=False, name='bass')
    string = ct.Instrument(program=40, is_drum=False, name='string')
    ensemble = ct.Instrument(program=48, is_drum=False, name='ensemble')
    brass = ct.Instrument(program=56, is_drum=False, name='brass')
    reed = ct.Instrument(program=64, is_drum=False, name='reed')
    pipe = ct.Instrument(program=72, is_drum=False, name='pipe')
    drum = ct.Instrument(program=0, is_drum=True, name='drum')
    midi.instruments = [piano, chromatic, organ, guitar, bass, string, ensemble, brass, reed, pipe, drum]
    '''

    for n in notes:
        
        if n.name == 'note':

            end = (n.start + n.duration) * 480 // 12
            start = n.start * 480 // 12
            velocity = n.velocity

            note = ct.Note(start=start, end=end, pitch=n.pitch, velocity=velocity)
            num = n.instrument
            
            midi.instruments[num].notes.append(note)
        
        elif n.name == 'bpm':

            time = n.start * 480 // 24
            tempo = n.pitch
            midi.tempo_changes.append(ct.TempoChange(tempo,time))

    #if midi.tempo_changes[0].time != 0:
    #    midi.tempo_changes[0].time = 0
    midi.tempo_changes.append(ct.TempoChange(120,0))
        
    midi.dump(path)
    
    return midi

def decode_midi(words,path):

    events = word2event(words)
    notes  = event2note(events)
    midi = save_midi(notes,path)

    return midi

if __name__ == '__main__':
    #0679868b152984be7b3b73a1ca5b204e.mid
    words = encode_midi('midi/yoru/just.mid')
    decode_midi(words,'result/test/gen.mid')
