from re import I
import sys
import socket
import numpy as np
import torch as torch
from torch.multiprocessing import Process, Pipe, Queue
from spiketag.utils import Timer
from time import time
from spiketag.utils import EventEmitter
from spiketag.analysis.core import get_hd
from spiketag.utils import FIFO
# from spiketag.fpga.memory_api import read_mem_16
from ..rotenc import Rotenc
import threading
import nidaqmx
from nidaqmx.constants import AcquisitionType

ENABLE_PROFILER = False

# Lab
host_ip = '192.168.2.2'
pynq_ip = '192.168.2.99'

# Test
#host_ip = '10.102.20.34'
# pynq_ip = '127.0.0.1'
# verbose = True

is_close = lambda pos, cue_pos, radius: (pos[:2]-cue_pos[:2]).norm()/100 < radius



class Jovian_Stream(str):
    def parse(self):
        line = self.__str__()
        _line = line.split(',')
        try:
            _t,_x,_y,_ball_vel = int(_line[0]), int(_line[1]), int(_line[2]), int(_line[7])
            _coord = [_x, _y, 0]
            return _t, _coord, _ball_vel
        except:
            _t,_info = int(_line[0]), _line[1]
            return _t, _info

        
def rotate(pos, theta=0):
    x = theta/360*2*np.pi
    R = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    return R.dot(pos)

def boundray_check(pos, bx=[-50.0, 50.0], by=[-50.0, 50.0]):
    new_pos = pos
    animal_size = 5
    if pos[0] < bx[0] + animal_size:
        new_pos[0] = bx[0] + animal_size
    if pos[0] > bx[1] - animal_size:
        new_pos[0] = bx[1] - animal_size
    if pos[1] < by[0] + animal_size:
        new_pos[1] = by[0] + animal_size
    if pos[1] > by[1] - animal_size:
        new_pos[1] = by[1] - animal_size
    return new_pos

def _core_jovian_process(touch_radius, shared_cue_dict, direction, cnt, current_pos, current_hd, ball_vel, event_queue):#,ready_event, start_gate):
    # soc_input, buf, buffer, buffering, emit, log, touch_radius, shared_cue_dict, rot,cnt, current_pos, current_hd, ball_vel
        #print("prosess start")
        '''jovian reading process that use 
           a multiprocessing pipe + a jovian instance 
           as input parameters
        '''
               # the buf generator
        
        syncer='Jovian' #'NI','EXT'
        buf = None        # the buf generator
        buffer = ''       # the content
        buffering = False # the buffering state
        loop_number=0

        if syncer != 'EXT':
            write_task = nidaqmx.Task()
        if syncer != 'Jovian':
            read_task = nidaqmx.Task()
        if syncer == 'NI':
            SAMPLE_RATE = 1000.0  # 1ms resolution
            SEQUENCE_S = 2049.0 
            PULSE_S = 0.5
            TRIGGER_US = [
                0, 1000000, 3000000, 64000000, 105000000, 181000000, 266000000, 284000000, 
                382000000, 469000000, 531000000, 545000000, 551000000, 614000000, 712000000, 
                726000000, 810000000, 830000000, 846000000, 893000000, 983000000, 1024000000, 
                1113000000, 1196000000, 1214000000, 1242000000, 1257000000, 1285000000, 
                1379000000, 1477000000, 1537000000, 1567000000, 1634000000, 1697000000, 
                1718000000, 1744000000, 1749000000, 1811000000, 1862000000, 1917000000, 
                1995000000, 2047000000
            ]
            TRIGGER_S = [t / 1_000_000.0 for t in TRIGGER_US]
            NUM_SAMPLES = int(SAMPLE_RATE * SEQUENCE_S)
            SAMPLES = int(SAMPLE_RATE * SEQUENCE_S)
            output_waveform = np.zeros(NUM_SAMPLES, dtype=bool)
            for t in TRIGGER_S:
                start_idx = int(t * SAMPLE_RATE)
                end_idx = int((t + PULSE_S) * SAMPLE_RATE)
                output_waveform[start_idx:end_idx] = True
        if syncer == 'Jovian':
            TRIGGER_US = [
                0, 1000000, 3000000, 64000000, 105000000, 181000000, 266000000, 284000000, 
                382000000, 469000000, 531000000, 545000000, 551000000, 614000000, 712000000, 
                726000000, 810000000, 830000000, 846000000, 893000000, 983000000, 1024000000, 
                1113000000, 1196000000, 1214000000, 1242000000, 1257000000, 1285000000, 
                1379000000, 1477000000, 1537000000, 1567000000, 1634000000, 1697000000, 
                1718000000, 1744000000, 1749000000, 1811000000, 1862000000, 1917000000, 
                1995000000, 2047000000
            ]
            TRIGGER_S = [t / 1_000_000.0 for t in TRIGGER_US]
            SEQUENCE_S = 2049.0 
            PULSE_S=0.5
            end_pulse=0
            loop_state=True
        
       
        flag=True
        
        sync_count=0
            
        def readbuffer():
            nonlocal buf,buffer,buffering
            buffer = soc_input.recv(256).decode("utf-8")
            buffering = True
            while buffering:
                if '\n' in buffer:
                    (line, buffer) = buffer.split("\n", 1)
                    yield Jovian_Stream(line + "\n")
                else:
                    more = soc_input.recv(256).decode("utf-8")
                    if not more:
                        buffering = False
                    else:
                        buffer += more
            if buffer:
                yield Jovian_Stream(buffer)

        def readline():
            nonlocal buf,buffer,buffering
            if buf is None:
                buf = readbuffer()
                return buf.__next__()
            else:
                return buf.__next__()

        def read_routine():
            '''
            read current_pos, current_hd, ball_vel, current_cue_pos into shared memory
            and put them into log
            '''
            _t, _coord, _ball_vel = readline().parse()
            current_pos[:]  = torch.tensor(_coord)
            current_hd[:]   = direction
            ball_vel[:]     = _ball_vel
            _cue_name_0, _cue_name_1 = list(shared_cue_dict.keys())
            event_queue.put(('read','ani_pos: {}, [{:.2f}, {:.2f}, {:.2f}], {:.2f}, {:.1f}'.format(_t, 
                                                                                         current_pos[0],
                                                                                         current_pos[1],
                                                                                         current_pos[2], 
                                                                                         current_hd[0], 
                                                                                         ball_vel[0])))
    
            event_queue.put(('read','cue_pos: [{0:.2f},{1:.2f},{2:.2f}],[{3:.2f}, {4:.2f}, {5:.2f}]'.format(shared_cue_dict[_cue_name_0][0],
                                                                                                  shared_cue_dict[_cue_name_0][1],
                                                                                                  shared_cue_dict[_cue_name_0][2],
                                                                                                  shared_cue_dict[_cue_name_1][0],
                                                                                                  shared_cue_dict[_cue_name_1][1],
                                                                                                  shared_cue_dict[_cue_name_1][2], 
                                                                                             )))

        def task_routine():
            '''
            jov emit necessary event to task by going through `task_routine` at each frame (check _jovian_process)
            One can flexibly define his/her own task_routine. 
            It provides the necessary event for the task fsm at frame rate. 
            '''
            cnt.add_(1)
            if cnt == 1:
                 event_queue.put(('start',None))
            # if self.cnt%2 == 0:
            event_queue.put(('frame',None))
            check_touch_agent_to_cue()  # JUMPER, one_cue, two_cue, moving_cue etc..
            check_touch_cue_to_cue()    # JEDI

        def check_touch_agent_to_cue():
            for _cue_name in shared_cue_dict.keys():
                # event_queue.put(('log_info','{}'.format(is_close(current_pos, torch.tensor(shared_cue_dict[_cue_name]), touch_radius))))
                if is_close(current_pos, torch.tensor(shared_cue_dict[_cue_name]), touch_radius):
                    # event_queue.put(('touch','Im touched!'))
                    event_queue.put(('touch', ( _cue_name, shared_cue_dict[_cue_name] )))

        def check_touch_cue_to_cue():
            # here let's assume that there are only two cues to check
            _cue_name_0, _cue_name_1 = list(shared_cue_dict.keys())
            if is_close(torch.tensor(shared_cue_dict[_cue_name_0]), 
                              torch.tensor(shared_cue_dict[_cue_name_1]), touch_radius):
                event_queue.put(('touch', ( _cue_name_0 + '->' + _cue_name_1, shared_cue_dict[_cue_name_0] )))



        def sync_routine():
            nonlocal flag, sync_count,syncer,start_loop,end_pulse,loop_state
            data=[0]
            if syncer != 'Jovian':
                data = read_task.read(number_of_samples_per_channel=-1)
            if syncer == 'Jovian':
                if time()-start_loop>SEQUENCE_S:
                    start_loop=time()
                    loop_state=True
                if loop_state:    
                    if time()-start_loop>=TRIGGER_S[sync_count%len(TRIGGER_S)]:
                        if not end_pulse:
                            data=[1]
                            End_pulse=TRIGGER_S[sync_count%len(TRIGGER_S)]+PULSE_S
                        else:
                            print("pulse length is too long")
    
                        if sync_count%len(TRIGGER_S)==len(TRIGGER_S)-1:
                            loop_state=False
                        
                
                if end_pulse:
                    if time()-start_loop>end_pulse:
                        data=[0]
                        write_task.write(False)
                        end_pulse=0
                    
            if any(data):
                if flag:
                    sync_count += 1
                    #sync_queue.put(('log_info',f'sync_status: high, sync_count: {sync_count}'))
                    event_queue.put(('sync',f'sync_status: high, sync_count: {sync_count}'))

                    flag=False
                    if syncer == 'Jovian':
                        write_task.write(True)
            else:
                flag=True

        soc_input = socket.create_connection((host_ip, '22224'), timeout=1)
        soc_input.setblocking(False)
        soc_input.settimeout(0.8)
        soc_input.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            

        try:
            if syncer != 'EXT':
                write_task.do_channels.add_do_chan("PXI1Slot4/port0/line0")
                if syncer == 'NI':
                    write_task.timing.cfg_samp_clk_timing(
                    rate=SAMPLE_RATE, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=SAMPLES)
    
            # Setup Input (Line 1)
            if syncer != 'Jovian':
                read_task.di_channels.add_di_chan("PXI1Slot4/port0/line5")
                read_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=SAMPLES)
    
            # ready_event.set()
            # # Wait for the main process to signal "Start"
            
            # start_gate.wait()
            # soc_input.shutdown(2)
            # init_soc()
            if syncer == 'NI':
                write_task.write(output_waveform, auto_start=False)
                print("Starting Loop. Press Ctrl+C to stop.")
            if syncer != 'Jovian':
                read_task.start()
            if syncer != 'EXT':
                write_task.start()
            if syncer == 'Jovian':
                start_loop = time()
    
        
            while True:
                    tic = time()
                    sync_routine()
                    read_routine()
                    task_routine()
                    toc = time()
                    secs = (toc - tic)
                    # self.jovian_time += secs
                    tictoc = secs * 1000
                    # jovian_queue.put(('log_info',f'loop {loop_number} takes: {tictoc:.2f} ms')) # current jovian time: {self.jovian_time:.4f} s
                    event_queue.put(('jovian',f'loop {loop_number} takes: {tictoc:.2f} ms')) # current jovian time: {self.jovian_time:.4f} s

                    loop_number += 1
            # while True:
            #     with Timer('', verbose=ENABLE_PROFILER):
            #         try:
            #             _t, _coord, _ball_vel = readline().parse()
            #             _cue_name_0, _cue_name_1 = list(shared_cue_dict.keys())
            #             if type(_coord) is list:
            #                 current_pos[:]  = torch.tensor(_coord)
            #                 current_hd[:]   = direction
            #                 ball_vel[:]     = _ball_vel
            #                 event_queue.put(('log_info','{}, {}, {}, {}'.format(_t, current_pos.numpy(), 
            #                                                         current_hd.numpy(), 
            #                                                         _ball_vel)))
            #                 event_queue.put(('log_info','cue_pos:, {},{}'.format(shared_cue_dict[_cue_name_0],shared_cue_dict[_cue_name_1])))
            #                 task_routine()
            #             else:
            #                 event_queue.put(('log_waarn','{}, {}'.format(_t, _coord)))
    
            #         # except Exception as e:
            #             # self.log.warn(f'jovian recv process error:{e}')
            #         except:
            #             event_queue.put(('log_info','jovian recv process error'))

        except Exception as e:
            event_queue.put(('log_warn',f"Subprocess error: {e}"))
        finally:
            # THIS IS THE CRITICAL PART
            if soc_input:
                print("Closing Jovian socket and MSA...")
                soc_input.shutdown(socket.SHUT_RDWR)
                soc_input.close()
                if syncer != 'EXT':
                    write_task.stop()
                    write_task.close()
                if syncer != 'Jovian':
                    read_task.stop()
                    read_task.close()
            






class Jovian(EventEmitter):
    '''
    Jovian is the abstraction of Remote virtual reality engine, it does following job:
    0.  jov = Jovian()                                      # instance
    1.  jov.readline().parse()                              # read from mouseover
    2.  jov.start(); jov.stop()                             # start reading process in an other CPU
    3.  jov.set_trigger(); jov.examine_trigger();           # set and examine the trigger condition (so far only touch) based on both current input and current state
    4.  jov.teleport(prefix, target_pos, target_item)       # execute output (so far only teleport)

    Jovian is a natrual event emit, it generate two `events`:
    1. touch      (according to input and trigger condition, it touches something)
    2. teleport   (based on the task fsm, something teleports)

    Jovian object communicate with other sub-process via shared memory, which is defined in `shared_mem_init` function, 
    and used by maze_view and task:
    1. maze_view.connect(self.jov) 
    2. task = globals()[self.task_name](self.jov)
    '''
    def __init__(self):
        super(Jovian, self).__init__()
        self.event_queue = Queue()
        self.socket_init()
        # self.buf_init()
        self.shared_mem_init()
        self.rotenc_init()
        # self.sync_init()

        self._stop_event = threading.Event()
        self.update_thread = None

        # self.ready_event = Event()  # Subprocess says: "I'm connected"
        # self.start_gate = Event()   # Main process says: "Go!"

        # self.jovian_process = Process(target=_jovian_process,args=(self.log, self.touch_radius, self.shared_cue_dict, self.rot.direction, self.cnt, self.current_pos, self.current_hd, self.ball_vel,self.event_queue,),name='jovian')
        # self.jovian_process.daemon = True
        # self.jovian_process.start() 
        # self.log.info("Subprocess spawned and initializing jovian_process in background...")

    def sync_init(self):
        self.sync_status = 0
        self.sync_count = 0
        self.sync_start = False

    def socket_init(self):
        ### mouseover server connection
        # self.input = socket.create_connection((host_ip, '22224'), timeout=1)
        # self.input.setblocking(False)
        # self.input.settimeout(0.8)
        # self.input.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.output = socket.create_connection((host_ip, '22223'), timeout=1)
        self.output.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.output_control = socket.create_connection((host_ip, '22225'), timeout=1)
        self.enable_output()

        ### pynq server connection
        try:
            self.pynq = socket.create_connection((pynq_ip, '2222'), timeout=1)
            self.pynq.setblocking(1)
            self.pynq.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.pynq_connected = True
            self.socks = [self.output, self.output_control, self.pynq]
        except:
            self.pynq_connected = False
            self.socks = [self.output, self.output_control]


    def rotenc_init(self):
        '''
        init the rotenc
        '''
        self.rot = Rotenc()


    # def buf_init(self):
    #     self.buf = None        # the buf generator
    #     self.buffer = ''       # the content
    #     self.buffering = False # the buffering state
    #     self.loop_number = 0
    #     # self.jovian_time = 0


    def shared_mem_init(self):
        '''
        once a shared variable is initialized, it must be filled here first and then can be accessed by other processs
        '''
        # trigger task using frame counter
        self.cnt = torch.empty(1,)
        self.cnt.share_memory_()
        self.cnt.fill_(0)

        # current position of animal
        self.current_pos = torch.empty(3,)
        self.current_pos.share_memory_()
        self.current_pos.fill_(0)

        # the influence radius of the animal
        self.touch_radius = torch.empty(1,)
        self.touch_radius.share_memory_()
        self.touch_radius.fill_(15)

        # bmi position (decoded position of the animal)
        self.bmi_pos = torch.empty(2,)
        self.bmi_pos.share_memory_()
        self.bmi_pos.fill_(0) 

        # bmi head-direction (inferred head direction at bmi_pos)
        self.hd_window = torch.empty(1,)  # time window(seconds) used to calculate head direction
        self.hd_window.share_memory_()
        self.hd_window.fill_(0)

        self.ball_vel = torch.empty(1,)
        self.ball_vel.share_memory_()
        self.ball_vel.fill_(0)
        
        self.bmi_hd = torch.empty(1,)       # calculated hd sent to Jovian for VR rendering
        self.bmi_hd.share_memory_()
        self.bmi_hd.fill_(0)         
        
        self.current_hd = torch.empty(1,)   # calculated hd (same as bmi_hd) sent to Mazeview for local playground rendering
        self.current_hd.share_memory_()
        self.current_hd.fill_(0)

        # bmi radius (largest teleportation range)
        self.bmi_teleport_radius = torch.empty(1,)
        self.bmi_teleport_radius.share_memory_()
        self.bmi_teleport_radius.fill_(0)    

        # rw counter added by shinsuke
        self.rw_cnt=torch.empty(1,)
        self.rw_cnt.share_memory_()
        self.rw_cnt.fill_(0)

    def reset(self):
        [conn.shutdown(2) for conn in self.socks]
        # self.buf_init()
        self.socket_init()


    def enable_output(self, enable=True):
        if enable:
            self.output_control.send(b'1')
        else:
            self.output_control.send(b'1')

    def _update_loop(self):
        """Internal loop that runs in a Thread to trigger EventEmitter callbacks."""
        # self.emit('start')
        
        while not self._stop_event.is_set():
            try:
                # Check for events sent from the subprocess
                while not self.event_queue.empty():
                    event_name, args = self.event_queue.get_nowait()
                    if event_name == 'read':
                        self.read_routine(args)
                    elif event_name == 'sync':
                        self.sync_routine(args)
                    elif event_name == 'jovian':
                        self._jovian_process(args)
                    elif event_name == 'log_warn':
                        self.log.warn(args)
                        
                    elif args is None:
                        self.emit(event_name)
                        # self.log.info(event_name)
                    else:
                        self.emit(event_name, args= args)
                        
            except Exception:
                pass
            self._stop_event.wait(0.001) # Small sleep to prevent 100% CPU usage

    def read_routine(self,args):
        self.log.info(args)

    def sync_routine(self,args):
        self.log.info(args)

    def _jovian_process(self,args):
        self.log.info(args)

    # def sync_routine(self):
    #     while not self._stop_event_s.is_set():
    #         try:
    #             # Check for events sent from the subprocess
    #             while not self.sync_queue.empty():
    #                 event_name, args = self.sync_queue.get_nowait()
    #                 if event_name == 'log_info':
    #                     self.log.info(args)
    #                 elif event_name == 'log_warn':
    #                     self.log.warn(args)
                        
    #                 elif args is None:
    #                     self.emit(event_name)
    #                     # self.log.info(event_name)
    #                 else:
    #                     self.emit(event_name, args= args)
                        
    #         except Exception:
    #             pass
    #         self._stop_event_s.wait(0.001) # Small sleep to prevent 100% CPU usage

    # def _jovian_process(self):
    #     while not self._stop_event_j.is_set():
    #         try:
    #             # Check for events sent from the subprocess
    #             while not self.jovian_queue.empty():
    #                 event_name, args = self.jovian_queue.get_nowait()
    #                 if event_name == 'log_info':
    #                     self.log.info(args)
    #                 elif event_name == 'log_warn':
    #                     self.log.warn(args)
                        
    #                 elif args is None:
    #                     self.emit(event_name)
    #                     # self.log.info(event_name)
    #                 else:
    #                     self.emit(event_name, args= args)
                        
    #         except Exception:
    #             pass
    #         self._stop_event_j.wait(0.001) # Small sleep to prevent 100% CPU usage
        
    # def readbuffer(self):
    #     self.buffer = self.input.recv(256).decode("utf-8")
    #     self.buffering = True
    #     while self.buffering:
    #         if '\n' in self.buffer:
    #             (line, self.buffer) = self.buffer.split("\n", 1)
    #             yield Jovian_Stream(line + "\n")
    #         else:
    #             more = self.input.recv(256).decode("utf-8")
    #             if not more:
    #                 self.buffering = False
    #             else:
    #                 self.buffer += more
    #     if self.buffer:
    #         yield Jovian_Stream(self.buffer)


    # def readline(self):
    #     if self.buf is None:
    #         self.buf = self.readbuffer()
    #         return self.buf.__next__()
    #     else:
    #         return self.buf.__next__()

    # def _jovian_process(self):
    #     '''jovian reading process that use 
    #        a multiprocessing pipe + a jovian instance 
    #        as input parameters
    #     '''
    #     while True:
    #         tic = time()
    #         self.sync_routine()
    #         self.read_routine()
    #         self.task_routine()
    #         toc = time()
    #         secs = (toc - tic)
    #         # self.jovian_time += secs
    #         tictoc = secs * 1000
    #         self.log.info(f'loop {self.loop_number} takes: {tictoc:.2f} ms') # current jovian time: {self.jovian_time:.4f} s
    #         self.loop_number += 1

    @property
    def current_cue_pos(self):
        _cue_pos_0 = self.shared_cue_dict[self._cue_name_0]
        _cue_pos_0[:2] = (_cue_pos_0[:2]-self.maze_origin[:2])/self.maze_scale
        _cue_pos_1 = self.shared_cue_dict[self._cue_name_1]
        _cue_pos_1[:2] = (_cue_pos_1[:2]-self.maze_origin[:2])/self.maze_scale
        cue_pos = np.concatenate((_cue_pos_0, _cue_pos_1))
        return cue_pos

                                                     

    # def sync_routine(self):
    #     '''
    #     sync routine: read FPGA-NSP state from `bmi.fpga.mem_16[0]`
    #     - 0: SPI is not running and sync pulse is low
    #     - 1: SPI is running and sync pulse is low
    #     - 2: SPI is not running and sync pulse is high (impossible state)
    #     - 3: SPI is running and sync pulse is high

    #     # ! sync_pulse width is 100ms so this sync_routine should be called at least 10Hz

    #     self.sync_state: keep track of the FPGA-NSP state
    #     self.sync_count: keep track of the sync pulse count

    #     log.info('SY') and self.sync_count increasement 
    #     should only be triggered when the FPGA-NSP state changes from 1 to 3
    #     '''
    #     new_sync_status = read_mem_16(0)

    #     # x = np.array([new_sync_status], dtype=np.int16)
    #     # f_sync = open('./sync.bin', 'ab+')
    #     # f_sync.write(x.tobytes())
    #     # f_sync.close()

    #     # the first pulse state transition (0->3)
    #     # the second and later pulse state transition (1->3)
    #     if new_sync_status == 3 and self.sync_status == 0: # capture the rising edge
    #         self.sync_start = True
    #         self.sync_count = 1
    #         self.log.info(f'sync_status: start, sync_count: {self.sync_count}')
    #     if new_sync_status == 3 and self.sync_status == 1 and self.sync_start:
    #         self.sync_count += 1
    #         self.log.info(f'sync_status: high, sync_count: {self.sync_count}')
    #     self.sync_status = new_sync_status

    def set_bmi(self, bmi, pos_buffer_len=30):
        '''
        This set BMI, Its binner and decoder event for JOV to act on. The event flow:
        bmi.binner.emit('decode', X) ==> jov
        customize the post decoding calculation inside the function
        `on_decode(X)` where the X is sent from the bmi.binner, but the `self` here is the jov

        set_bmi connect the event flow from
                 decode(X)                shared variable
             y=dec.predict_rt(X)         (bmi_pos, bmi_hd)
        bmi =====================> jov ====================> task
        '''
        ## Set the BMI buffer for smoothing both pos and hd
        self.bmi = bmi
        self.bmi_pos_buf = np.zeros((pos_buffer_len, 2))
        hd_buffer_len = int(self.hd_window.item()/self.bmi.binner.bin_size)
        self.bmi_hd_buf  = np.zeros((hd_buffer_len, 2))
        self.bmi_hd_buf_ring = np.zeros((hd_buffer_len, ))
        self.log.info('Initiate the BMI decoder and playground jov connection')
        self.log.info('position buffer length:{}'.format(pos_buffer_len))

        ## Set the real-time posterior placehodler
        dumb_X = np.zeros((self.bmi.binner.B, self.bmi.binner.N))
        self.perm_idx = np.random.permutation(dumb_X.shape[1])
        if hasattr(self.bmi.dec, 'model'):
            print(self.bmi.dec.model)
            self.bmi.dec.model.share_memory();
            post_2d = np.random.randn(*self.bmi.dec.pc.O.shape)
        else:
            _, post_2d = self.bmi.dec.model.predict_rt(dumb_X, neuron_idx=self.bmi.dec.neuron_idx)
        self.current_post_2d = torch.empty(post_2d.shape)
        self.current_post_2d.share_memory_()
        self.log.info('The decoder binsize:{}, the B_bins:{}'.format(self.bmi.binner.bin_size, self.bmi.binner.B))
        self.log.info('The decoder input (spike count bin) shape:{}'.format(dumb_X.shape))
        self.log.info('The decoder output (posterior) shape: {}'.format(self.current_post_2d.shape))
        self.speed_fifo = FIFO(depth=39)
        
        # self.bmi.dec.drop_neuron(np.array([7,9]))

        # self.scv_save= []
        self.dec_pos_save=[]
        self.cue_pos_save=[]
        self.animal_pos_save=[]
        self.animal_hdv_save=[]
        self.bmi_pos_save=[]

        

        @self.bmi.binner.connect
        def on_decode(X):
            '''
            This event is triggered every time a new bin is filled (based on BMI output timestamp)
            '''
            # print(self.binner.nbins, self.binner.count_vec.shape, X.shape, np.sum(X))
            with Timer('decoding', verbose=False): # use verbose to see the decoding time, rouphly 16 ms for 500 units
                # ----------------------------------
                # 1. Ring decoder for the head direction
                # ----------------------------------
                # hd = self.bmi.dec.predict_rt(X) # hd should be a angle from [0, 360]
                # self.bmi_hd_buf_ring = np.hstack((self.bmi_hd_buf_ring[1:], hd))
                # # print(self.bmi_hd_buf_ring)
                # self.bmi_hd[:] = torch.tensor(self.bmi_hd_buf_ring.mean()) 

                # ----------------------------------
                # 2. Bayesian decoder for the position
                # ----------------------------------
                # if X.sum(axis=0)>2:
                # _X = X[:, self.perm_idx]

                ### save scv to file ###
                # f_scv.write(X.tobytes())
                # if not len(self.scv_save):
                #     self.scv_save = X.reshape(X.shape[1],X.shape[2])
                # else:
                #     self.scv_save = np.vstack((self.scv_save,X.reshape(X.shape[1],X.shape[2])[-1]))

                if hasattr(self.bmi.dec, 'model'):
                    self.log.info(f'use the bmi model:{X.shape},{X.dtype}')
                    # y = self.bmi.dec.predict_rt(X, cuda=False, mode='eval', bn_momentum=0.9);
                    # X=X.copy().reshape(1, X.shape[0], X.shape[1])
                    y = self.bmi.dec.predict_rt(X, cuda=True, mode='eval', bn_momentum=0.1);
                    y=y#+self.bmi.dec.pos_mean_original 
                    self.bmi.model_output[:] = torch.from_numpy(y)
                    self.log.info(f'bmi model output:{y}')
                    post_2d = self.bmi.dec.pc.real_pos_2_soft_pos(self.bmi.model_output.numpy(), kernel_size=7)
                else:
                    self.log.info(f'cannot find bmi model:{X.shape}')
                    y, post_2d = self.bmi.dec.predict_rt(X)
                    self.log.info(f'post_2d:{post_2d.shape}')

                post_2d /= post_2d.sum()
                max_posterior = post_2d.max()
                
                ### save decoded position to file ###
                # f_dec_pos.write(y.tobytes())
                if not len(self.dec_pos_save):
                    self.dec_pos_save = y

                else:
                    self.dec_pos_save = np.vstack((self.dec_pos_save,y))


                ### save posterior to file ###
                # f_post = open('./post_2d.bin', 'ab+')
                # f_post.write(post_2d.tobytes())
                # f_post.close()

                ### save cue_pos to file ###
                # f_cue_pos.write(self.current_cue_pos.tobytes())
                if not len(self.cue_pos_save):
                    self.cue_pos_save = self.current_cue_pos
                else:
                    self.cue_pos_save = np.vstack((self.cue_pos_save, self.current_cue_pos))
                

                ### save animal_pos to file ###
                animal_pos = (self.current_pos.numpy()[:2]-self.maze_origin[:2])/self.maze_scale
                # f_animal_pos.write(animal_pos.tobytes())
                if not len(self.animal_pos_save):
                    self.animal_pos_save = animal_pos
                else:
                    self.animal_pos_save = np.vstack((self.animal_pos_save,animal_pos))

                ### save animal_hdv to file ###
                animal_hdv = np.concatenate((self.current_hd.numpy(), 
                                             self.ball_vel.numpy()/14e-3/100))
                # f_animal_hdv.write(animal_hdv.tobytes())
                if not len(self.animal_hdv_save):
                    self.animal_hdv_save = animal_hdv
                else:
                    self.animal_hdv_save = np.vstack((self.animal_hdv_save,animal_hdv))



                ### Key: filter out criterion ###
                if X.sum()>2:
                    self.current_post_2d[:] = torch.tensor(post_2d) * 1.0
                    
                # #################### just for dusty test #########################
                # y += np.array([263.755, 263.755])
                # y -= np.array([253.755, 253.755])
                # y -= np.array([318.529, 195.760])
                # y /= 4.5
                # ##################################################################
                ball_vel_thres = self.bmi_teleport_radius.item()
                self.speed_fifo.input(self.ball_vel.numpy()[0])
                # self.log.info('FIFO:{}'.format(self.speed_fifo.numpy()))
                speed = self.speed_fifo.mean()/14e-3/100
                self.log.info('speed:{}, threshold:{}'.format(speed, ball_vel_thres))
                self.log.info('max_post:{}, post_thres:{}'.format(max_posterior, self.bmi.posterior_threshold))
                # current_speed = self.speed_fifo.mean()
                try:
                    if self.bmi.bmi_update_rule == 'moving_average':
                        # # rule1: decide the VR output by FIFO smoothing
                        if speed < ball_vel_thres and X.sum()>2 and max_posterior>self.bmi.posterior_threshold:
                            self.bmi_pos_buf = np.vstack((self.bmi_pos_buf[1:, :], y))
                            _teleport_pos = np.mean(self.bmi_pos_buf, axis=0)
                            self.log.info('_teleport_pos:{}'.format(_teleport_pos))
                        else:
                            _teleport_pos = self.bmi_pos.numpy()
                            
                    elif self.bmi.bmi_update_rule == 'fixed_length':
                        # # rule2: decide the VR output by fixed length update
                        u = (y-self.bmi_pos.numpy())/np.linalg.norm(y-self.bmi_pos.numpy())
                        tao = 5
                        if speed < ball_vel_thres and X.sum()>2 and max_posterior>self.bmi.posterior_threshold:
                            tao = 5 # cm
                            _teleport_pos = self.bmi_pos.numpy() + tao*u 
                        else:
                            _teleport_pos = self.bmi_pos.numpy()

                    elif self.bmi.bmi_update_rule == 'randomized_control':
                        # # rule1: decide the VR output by FIFO smoothing
                        if speed < ball_vel_thres and X.sum()>2 and max_posterior>self.bmi.posterior_threshold:
                            last_mean_pos = np.mean(self.bmi_pos_buf, axis=0)
                            self.bmi_pos_buf = np.vstack((self.bmi_pos_buf[1:, :], y))
                            mean_pos = np.mean(self.bmi_pos_buf, axis=0)
                            diff_pos = mean_pos - last_mean_pos
                            distance = np.linalg.norm(diff_pos)
                            theta = np.random.uniform(low=0.0, high=2*np.pi)
                            new_pos  = self.bmi_pos.numpy() + np.array([distance*np.cos(theta), distance*np.sin(theta)])  # current position + randomly rotated distance
                            _teleport_pos  = boundray_check(new_pos)     # make sure it is inside the maze
                            self.log.info('_teleport_pos:{}'.format(_teleport_pos))
                        else:
                            _teleport_pos = self.bmi_pos.numpy()    
                            
                    # # set shared variable
                    # _teleport_pos = rotate(_teleport_pos, theta=0)
                                    ### save cue_pos to file ###
                    current_bmi_pos = _teleport_pos.astype(np.float32)
                    # f_bmi_pos.write(current_bmi_pos.tobytes())
                    if not len(self.bmi_pos_save):
                        self.bmi_pos_save = current_bmi_pos
                    else:
                        self.bmi_pos_save = np.vstack((self.bmi_pos_save, current_bmi_pos))



                    
                    self.bmi_pos[:] = torch.tensor(_teleport_pos)

                    # self.bmi_hd_buf = np.vstack((self.bmi_hd_buf[1:, :], _teleport_pos))
                    # window_size = int(self.hd_window[0]/self.bmi.binner.bin_size)
                    # hd, speed = get_hd(trajectory=self.bmi_hd_buf[-window_size:], speed_threshold=0.6, offset_hd=0)
                        # hd = 90
                        # if speed > .6:
                            # self.bmi_hd[:] = torch.tensor(hd)      # sent to Jovian
                            # self.current_hd[:] = torch.tensor(hd)  # sent to Mazeview
                        # self.emit('bmi_update', pos=self.teleport_pos)
                        # self.log.info('\n')
                    self.log.info('BMI output(x,y,speed,ball_thres): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(_teleport_pos[0],
                                                                                                                _teleport_pos[1], 
                                                                                                                speed, 
                                                                                                                ball_vel_thres))
                except Exception as e:
                    self.log.warn('BMI error: {}'.format(e))
                    pass
                    

    def set_trigger(self, shared_cue_dict):
        '''shared_cue_dict is a a shared memory dict between processes contains cue name and position:
           shared_cue_dict := {cue_name: cue_pos, 
                               ...}
        '''
        self.shared_cue_dict = shared_cue_dict
        self._cue_name_0, self._cue_name_1 = list(self.shared_cue_dict.keys())
        self.log.info('-----------------------------------------------------------------------------------------')
        self.log.info('jovian and maze_view is connected, they starts to share cues position and transformations')
        self.log.info(f'cue names: {list(self.shared_cue_dict.keys())}')
        self.log.info('-----------------------------------------------------------------------------------------')


    # def task_routine(self):
    #     '''
    #     jov emit necessary event to task by going through `task_routine` at each frame (check _jovian_process)
    #     One can flexibly define his/her own task_routine. 
    #     It provides the necessary event for the task fsm at frame rate. 
    #     '''
    #     self.cnt.add_(1)
    #     if self.cnt == 1:
    #         self.emit('start')
    #     # if self.cnt%2 == 0:
    #     self.emit('frame')
    #     self.check_touch_agent_to_cue()  # JUMPER, one_cue, two_cue, moving_cue etc..
    #     self.check_touch_cue_to_cue()    # JEDI

    # def check_touch_agent_to_cue(self):
    #     for _cue_name in self.shared_cue_dict.keys():
    #         # self.log.info(f'touch with touch_radius: {self.touch_radius}')
    #         if is_close(self.current_pos, torch.tensor(self.shared_cue_dict[_cue_name]), self.touch_radius):
    #             self.emit('touch', args=( _cue_name, self.shared_cue_dict[_cue_name] ))

    # def check_touch_cue_to_cue(self):
    #     # here let's assume that there are only two cues to check
    #     _cue_name_0, _cue_name_1 = list(self.shared_cue_dict.keys())
    #     if is_close(torch.tensor(self.shared_cue_dict[_cue_name_0]), 
    #                       torch.tensor(self.shared_cue_dict[_cue_name_1]), self.touch_radius):
    #         # self.log.info(f'touch with touch_radius: {self.touch_radius}')
    #         self.emit('touch', args=( _cue_name_0 + '->' + _cue_name_1, self.shared_cue_dict[_cue_name_0] ))

    def bmi_close(self):
        f_scv = open('./scv.bin', 'ab+')
        f_dec_pos = open('./dec_pos.bin', 'ab+')
        f_cue_pos = open('./cue_pos.bin', 'ab+')
        f_animal_pos = open('./animal_pos.bin', 'ab+')
        f_animal_hdv = open('./animal_hdv.bin', 'ab+')
        f_bmi_pos = open('./bmi_pos.bin', 'ab+')
        # f_t_scv = open('./t_scv.bin', 'ab+')
        f_miss_spk = open('./miss_spk.bin', 'ab+')
        # f_spk_report = open('./spk_report.bin', 'ab+')


        if len(self.bmi.binner.scv_save):
            self.bmi.binner.scv_save=np.array(self.bmi.binner.scv_save)
            f_scv.write(self.bmi.binner.scv_save.tobytes())
            self.bmi.binner.scv_save = []
        if len(self.dec_pos_save):
            f_dec_pos.write(self.dec_pos_save.tobytes())
            self.dec_pos_save = []
        if len(self.cue_pos_save):
            f_cue_pos.write(self.cue_pos_save.tobytes())
            self.cue_pos_save = []
        if len(self.animal_pos_save):
            f_animal_pos.write(self.animal_pos_save.tobytes())
            self.animal_pos_save=[]
        if len(self.animal_hdv_save):
            f_animal_hdv.write(self.animal_hdv_save.tobytes())
            self.animal_hdv_save = []
        if len(self.bmi_pos_save):
            f_bmi_pos.write(self.bmi_pos_save.tobytes())
            self.bmi_pos_save = []
        if len(self.bmi.binner.update_t_save):
            self.bmi.binner.update_t_save = np.array(self.bmi.binner.update_t_save)
            # f_t_scv.write(self.bmi.binner.update_t_save.tobytes())
            np.save('t_scv.npy', self.bmi.binner.update_t_save)
            self.bmi.binner.update_t_save = []
        if len(self.bmi.binner.missed_spikes_save):
            self.bmi.binner.missed_spikes_save = np.array(self.bmi.binner.missed_spikes_save)
            f_miss_spk.write(self.bmi.binner.missed_spikes_save.tobytes())
            self.bmi.binner.missed_spikes_save = []
        if len(self.bmi.binner.report_bin):
            self.bmi.binner.report_bin = np.array(self.bmi.binner.report_bin)
            # f_spk_report.write(self.bmi.binner.report_bin.tobytes())
            np.save('spk_report.npy', self.bmi.binner.report_bin)
            self.bmi.binner.report_bin = []

        
        
        f_scv.close()
        f_dec_pos.close()
        f_cue_pos.close()
        f_animal_pos.close()
        f_animal_hdv.close()
        f_bmi_pos.close()
        # f_t_scv.close()
        f_miss_spk.close()
        # f_spk_report.close()



    
    def start(self):
        self.rot.start()
        self.pipe_jovian_side, self.pipe_gui_side = Pipe()
        # self.jovian_process = Process(target=self._jovian_process, name='jovian') #, args=(self.pipe_jovian_side,)
        self.jovian_process = Process(target=_core_jovian_process,args=(self.touch_radius, self.shared_cue_dict, self.rot.direction, self.cnt, self.current_pos, self.current_hd, self.ball_vel,self.event_queue, ),name='jovian')
        self.jovian_process.daemon = True
        self.reset() # !!! reset immediately before start solve the first time jam issue

        
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.jovian_process.start()  

    def stop(self):
        self._stop_event.set()
        
        if self.update_thread:
            self.update_thread.join()
       
        
        self.jovian_process.terminate()
        self.jovian_process.join()
        self.cnt.fill_(0)
        self.rot.stop()

    def get(self):
        return self.pipe_gui_side.recv().decode("utf-8")


    def toggle_motion(self):
        cmd = "console.toggle_motion()\n"
        self.output.send(cmd.encode())
    
    #shinsuke added.
    def toggle_blanking(self):
        cmd = "console.toggle_blanking()\n"
        self.output.send(cmd.encode())

    def toggle_motion_and_blanking(self):
        cmd="console.toggle_motion_and_blanking()\n"
        self.output.send(cmd.encode())

    def set_alpha(self,target_item, alpha):
        if alpha > 0.7:
            alpha = 0.7
        cmd="model.set_alpha('{}',{})\n".format(target_item,alpha)
        self.output.send(cmd.encode())

    def teleport(self, prefix, target_pos, head_direction=None, target_item=None):
        '''
           Jovian abstract (output): https://github.com/chongxi/playground/issues/6
           Core function: This is the only function that send `events` back to Jovian from interaction 
        '''
        try:
            x, y, z = target_pos # the coordination
        except:
            x, y = target_pos
            z = 0

        if head_direction is None:
            v = 0
        else:
            v = head_direction

        if prefix == 'console':  # teleport animal, target_item is None
            cmd = "{}.teleport({},{},{},{})\n".format('console', x, y, 5, v)
            self.output.send(cmd.encode())

        elif prefix == 'model':  # move cue
            with Timer('', verbose = ENABLE_PROFILER):
                z += self.shared_cue_height[target_item]
                cmd = "{}.move('{}',{},{},{})\n".format('model', target_item, x, y, z)
                self.output.send(cmd.encode())
                bottom = z - self.shared_cue_height[target_item]
                self.shared_cue_dict[target_item] = self._to_jovian_coord(np.array([x,y,bottom], dtype=np.float32))
                # shared_cue_dict is used in `maze_view.cue_update`

    def move_to(self, x, y, z=5, hd=0, hd_offset=0): 
        '''
        x,y = 0,0 # goes to the center (Jovian protocol)
        hd_offset = jov.rot.direction # the body direction
        '''
        cmd="{}.teleport({},{},{},{})\n".format('console', x, y, z, hd+hd_offset) 
        self.output.send(cmd.encode()) 


    def reward(self, time):
        self.log.info('reward {}'.format(time))
        t_rw_cnt=self.rw_cnt.numpy()
        self.rw_cnt.fill_(t_rw_cnt[0]+1)
        try:
            cmd = 'reward, {}'.format(time)
            self.pynq.send(cmd.encode())
        except:
            self.log.info('fail to send reward command - pynq connected: {}'.format(self.pynq_connected))


    #Shinsuke added
    def sw_switch(self, on_off):
        self.log.info('sweet_{}'.format(on_off))
        try:
            cmd = 'rw_switch, {}'.format(on_off)
            self.pynq.send(cmd.encode())
        except:
            self.log.info('fail to send reward command - pynq connected: {}'.format(self.pynq_connected))

    def air_puff(self, on_off):
        self.log.info('air_puff_{}'.format(on_off))
        try:
            cmd = 'air_puff, {}'.format(on_off)
            self.pynq.send(cmd.encode())
        except:
            self.log.info('fail to send reward command - pynq connected: {}'.format(self.pynq_connected))

    def RD_switch(self, S_L):
        self.log.info('RD_{}'.format(S_L))
        try:
            cmd = 'RD_switch, {}'.format(S_L)
            self.pynq.send(cmd.encode())
        except:
            self.log.info('fail to send reward command - pynq connected: {}'.format(self.pynq_connected))

    def JEDI_reward(self, time, onset, refractory):
        self.log.info('JEDI_reward {} {} {}'.format(time, onset, refractory))
        t_rw_cnt=self.rw_cnt.numpy()
        self.rw_cnt.fill_(t_rw_cnt[0]+1)
        try:
            cmd = 'JEDI_reward,{} {} {}'.format(time, onset, refractory)
            self.pynq.send(cmd.encode())
        except:
            self.log.info('fail to send reward command - pynq connected: {}'.format(self.pynq_connected))


