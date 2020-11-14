#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Arctic Space Technologies.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#


import pyldpc
import numpy as np
import socket
from threading import Thread #, Lock
from collections import deque
from gnuradio import gr
from . import plop_utils
import sys
import time
import math

class ENCODING_SCHEMES:
    BCH = 'bch'
    LDPC1 = 'ldpc1'
    LDPC2 = 'ldpc2'

class CARRIER_MODULATION_MODES:
    CMM_1 = 1 # Unmodulated CARRIER only
    CMM_2 = 2 # CARRIER modulated with ACQUISITION SEQUENCE
    CMM_3 = 3 # CARRIER modulated with CLTU
    CMM_4 = 4 # CARRIER modulated with the IDLE SEQUENCE

    CARRIER_ONLY  = CMM_1
    ACQ_SEQUENCE  = CMM_2
    IDLE_SEQUENCE = CMM_4 # <-- Attn
    CLTU          = CMM_3

    @staticmethod
    def cmm_handler_mappings(obj):
        return {
                   CARRIER_MODULATION_MODES.CARRIER_ONLY:  obj.carrier_only,
                   CARRIER_MODULATION_MODES.ACQ_SEQUENCE:  obj.acquisition_sequence,
                   CARRIER_MODULATION_MODES.IDLE_SEQUENCE: obj.cltu_or_idle_transmit,
                   CARRIER_MODULATION_MODES.CLTU:          obj.cltu_or_idle_transmit
               }

class Plop2Listener(Thread):
    def __init__(self, *args, **kwargs):
        Thread.__init__(self)
        self.killed = False
        
        # TCP configs
        self.bind_addr = kwargs['bind_addr']
        self.bind_port = kwargs['bind_port']
        self.buffer_size = kwargs['buffer_size']
        self.encoding_scheme = kwargs['encoding_scheme']

        # Reference to parent block
        self.gr_block_object = kwargs['gr_block_object']
        
        # Start sequence bitstream
        self.p2 = Plop2Encoder( encoding_scheme = self.encoding_scheme,
                                randomization_requested = kwargs['randomization_requested'] )
        self.CLTUs = deque()
        

    def fetch_cltu_bit_if_available(self):
        if(len(self.CLTUs)>=1):
            result = self.CLTUs.popleft()
            return result
        return None

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.bind_addr, self.bind_port))
        self.s.listen(1)
        while(1):
            conn, addr = self.s.accept()
            self.conn = conn
            self.addr = addr
            print('Client connected: ', addr)
            self.gr_block_object.begin_acquisition()
            while 1:
                data = conn.recv(self.buffer_size) 
                if data is None:
                    break
                data = plop_utils.bittify8(data)
                self.CLTUs.extend(self.p2.make_cltu(data))
            conn.close()
            while (leftover_bits := len(self.CLTUs)) >= 1:
                sleep(leftover_bits / self.gr_block_object.sample_rate)
            while self.gr_block_object.current_state!=CARRIER_MODULATION_MODES.IDLE_SEQUENCE:
                sleep(5/self.gr_block_object.sample_rate)
            self.gr_block_object.current_state = CARRIER_MODULATION_MODES.CARRIER_ONLY

    def start(self): 
        self.__run_backup = self.run 
        self.run = self.__run       
        Thread.start(self) 

    def __run(self): 
        sys.settrace(self.globaltrace) 
        self.__run_backup() 
        self.run = self.__run_backup 

    def globaltrace(self, frame, event, arg): 
        if event == 'call': 
          return self.localtrace 
        else: 
          return None

    def localtrace(self, frame, event, arg): 
        if self.killed: 
          if event == 'line': 
            raise SystemExit() 
        return self.localtrace 

    def kill(self): 
        self.killed = True


class plop2(gr.sync_block):
    """
    docstring for block plop2 
    """
    def __init__(self,  sample_rate=1200,
                        latency_protection=50,
                        syncword='1010101010101010',
                        encoding='ldpc1',
                        randomizer_on=True,
                        bind_addr='127.0.0.1', bind_port=4005, tcp_recv_buf_size=4096):

        gr.sync_block.__init__(self,
            name="plop2",
            in_sig=None, #np.byte
            out_sig=[np.byte])
        # self.bind_addr = bind_addr
        # self.bind_port = bind_port
        self.current_state = CARRIER_MODULATION_MODES.CMM_1
        self.state_handler = CARRIER_MODULATION_MODES.cmm_handler_mappings(self)
        params_dict = { 'bind_addr'  : bind_addr,
                        'bind_port'  : bind_port,
                        'buffer_size': tcp_recv_buf_size,
                        'encoding_scheme': encoding,
                        'randomization_requested': randomizer_on,
                        'gr_block_object': self
                        }
        self.listener = Plop2Listener(**params_dict)
        self.listener.start()
        self.sample_rate = int(sample_rate)
        self.latency_protection = latency_protection
        assert(syncword.count('0') + syncword.count('1')) == len(syncword), "Acq Seq string non-binary"
        self.syncword = np.array(list(syncword), dtype=np.uint64)
        self.bytes_sent = 0
        self.init_time_ns = time.time_ns()

    #################################
    ## CMM state change methods begins
    #################################

    def carrier_only(self):
        return np.uint8(0)

    def begin_acquisition(self):
        self.acquisition_sequence_bit = np.uint8(0)
        self.current_state = CARRIER_MODULATION_MODES.ACQ_SEQUENCE
        self.acq_bits_sent = 0
        print("CMM set to ACQ_SEQUENCE. Time = ", time.time())

    def acquisition_sequence(self):
        if(self.acq_bits_sent >= 16):
            return self.cltu_or_idle_transmit()
        self.acquisition_sequence_bit = self.acquisition_sequence_bit^1
        self.acq_bits_sent = self.acq_bits_sent + 1
        return np.uint8(self.acquisition_sequence_bit)

    def begin_idling(self):
        self.current_state = CARRIER_MODULATION_MODES.IDLE_SEQUENCE
        self.idle_sequence_bit = 1 # 231xb0.pdf sec 7.2.4. Gets inverted before sending, begins with 0
        # print("CMM set to IDLE_SEQUENCE. Time = ", time.time())

    def cltu_or_idle_transmit(self):
        outbit = self.listener.fetch_cltu_bit_if_available()
        if(outbit==None):
            if(self.current_state != CARRIER_MODULATION_MODES.IDLE_SEQUENCE):
                self.begin_idling()
            self.idle_sequence_bit = self.idle_sequence_bit^1
            outbit = self.idle_sequence_bit
        else:
            # if(self.current_state != CARRIER_MODULATION_MODES.CLTU):
            #     print("CMM set to CLTU. Time = ", time.time())
            self.current_state = CARRIER_MODULATION_MODES.CLTU
        return outbit
    #################################
    ## CMM state change methods end
    #################################


    #def forecast(self, noutput_items, ninput_items_required):
    #    #setup size of input_items[i] for work call
    #    for i in range(len(ninput_items_required)):
    #        ninput_items_required[i] = 0

    def get_n_bytes_to_send(self):
        time_now = time.time_ns()
        time_dif = time_now - self.init_time_ns
        target_sample_count = math.floor(self.sample_rate*time_dif/1000000000)
        return (target_sample_count +self.latency_protection - self.bytes_sent)

    def work(self, input_items, output_items):
        n_bytes_to_send = self.get_n_bytes_to_send()
        for i in range(n_bytes_to_send):
            output_items[0][i] = self.state_handler[self.current_state]()
        self.bytes_sent += n_bytes_to_send
        #if(n_bytes_to_send!=len(output_items[0])):
        #    print("Wierd problem")
        #    print('n_bytes_to_send={0}, len(...)={1}'.format(n_bytes_to_send, len(output_items[0])))
        return n_bytes_to_send #len(output_items[0])


CLTU_START_SEQUENCE_BCH  = [0xEB90 ]
CLTU_START_SEQUENCE_LDPC = [0x0347, 0x76C7, 0x2728, 0x95B0]

TRANSFER_FRAME_LENGTHS = {'bch'  : 56, 'ldpc1':  64, 'ldpc2': 256} #k
CODEWORD_LENGTHS       = {'bch'  : 64, 'ldpc1': 128, 'ldpc2': 512} #n
GENERATOR_MAT_MAKER    = {'bch'  : None,
                          'ldpc1': plop_utils.get_ldpc_code1_G_uint64, #Make Generator matrix
                          'ldpc2': plop_utils.get_ldpc_code2_G_uint64
                          }
class Plop2Encoder:
    def __init__(self, **kwargs):
        self.encoding_scheme = kwargs['encoding_scheme']
        # Helper flags
        self.usingBCH   = (self.encoding_scheme=='bch'  )
        self.usingLDPC1 = (self.encoding_scheme=='ldpc1')
        self.usingLDPC2 = (self.encoding_scheme=='ldpc2')
        if(self.usingBCH and kwargs['randomization_requested']==False):
            # Randomization managed for only for BCH, mandatory for LDPC
            self.randomization_enabled = False
            self.codeword_maker = self.generate_codeword
        else:
            self.randomization_enabled = True
            self.codeword_maker = self.randomize_transfer_frame_and_generate_codeword
            plop_utils.btg_counter = 0
        
        self.k = TRANSFER_FRAME_LENGTHS[self.encoding_scheme]
        self.n = CODEWORD_LENGTHS[self.encoding_scheme]
        self.G = GENERATOR_MAT_MAKER[self.encoding_scheme]()

        self.start_seq = CLTU_START_SEQUENCE_BCH if self.usingBCH else CLTU_START_SEQUENCE_LDPC
        self.start_seq = plop_utils.bittify16([self.start_seq])[0]
        self.tail_seq  = None

        if self.usingBCH:
            print("Exitting due to invalid encoding_scheme")
            sys.exit(1001)
        # self.G = np.transpose(np.concatenate((np.identity(64, dtype=np.uint8), plop_utils.bittify16(W)), axis=1))
        # print("Plop2Encoder::init. W shape is {}".format(np.shape(W)))
        # print("Plop2Encoder::init. G shape is {}".format(np.shape(self.G)))

    def generate_codeword(self, ibits):
        return pyldpc.utils.binaryproduct(ibits, self.G)

    def randomize_transfer_frame_and_generate_codeword(self, ibits):
        pseudo_random_seq = plop_utils.bit_transition_generator(self.k)
        return self.generate_codeword(ibits^pseudo_random_seq)

    def make_cltu(self, data):
        if data is None or len(data)==0:
            return None
        # zero padding to make length multiple of self.k
        k = self.k
        n_excess_bits = len(data)%k
        if(n_excess_bits != 0):
            data = np.pad( data, (0, k-n_excess_bits ) )
        n_cltus = int(len(data)/k)
        CLTUs = []
        transfer_frames = data.reshape((n_cltus, k))
        for i in range(n_cltus):
            codeword = self.codeword_maker(transfer_frames[i])
            CLTUs.extend(self.start_seq)
            CLTUs.extend(codeword)
            if(self.tail_seq is not None):
                CLTUs.extend(self.tail_seq)
            # CLTUs.extend(np.concatenate([self.start_seq, codeword]))
        return CLTUs
        # Avoid complexity when bits fit in one CLTU
        # if(len(data)==64):
        #     transfer_frame = data if self.p2.usingBCH() else self np.bitwise_xor(data, self.p2.get_n_bytes_to_send())
        #     codeword = self.p2.make_codeword(transfer_frame)
        #     cltu = np.concatenate([self.start_seq, codeword])
        #     self.CLTUs.extend(cltu)
        #     return
        
        # Otherwise, break into smaller chunks called transfer frames.
