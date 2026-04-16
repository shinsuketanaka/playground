from ..utils.utils import EventEmitter, Timer
import numpy as np
import time





class Binner(EventEmitter):
    """
    The binner takes real-time spike input from the BMI and output when B time bins are ready for the decoder
    To decode the output in a given time bin, we used the spike count of N neurons in B time bins output from the binner

    binner = Binner(bin_size=33.33, n_id=N, n_bin=B)    # binner initialization (space and time)
    binner.input(bmi_output, type='individual_spike')   # internal state update triggered

    Parameters:

    N N neurons give rise to N spike count in each bin
    B B bins
    bin_size The time span (ms) to compute the spike count of each bin
    Internal States:

    count_vec: shape = (B,N) B time bins and N neurons
    nbins (+1 when the input timestamps goes to the next bin, its number is the current bin)
    output (emitted variable to the decoder, N neuron's spike count in previous B bins)

    https://github.com/chongxi/spiketag/issues/47
    """
    def __init__(self, bin_size, n_id, n_bin, sampling_rate=30000, exclude_first_unit=False):
        super(Binner, self).__init__()
        self.bin_size = bin_size
        self.N = n_id
        self.B = n_bin
        self.count_vec = np.zeros((self.B, self.N))
        
        # self.nbins = 1 # self.nbins-1 is the index of the last bin
        self.fs = sampling_rate
        # self.dt = 1/self.fs   # each frame is 1/25000:40us, which is the resolution of timestamp
        # self.last_bin = 0
        # self.exclude_first_unit = exclude_first_unit
        self.exclude_first_unit =False

        self.t_grace = 0.005 # grace period in seconds to wait for the late spike to 
        # self.last_bin_time = 0  # Last bin time in seconds
        # self.current_bin_time = 0  # Current bin time in seconds
        self.current_time = 0
        self.first_call = True
        self.first_frame = True
        
        # Timer for bin updating
        self.last_update_time = None
        #self.pred_history = deque(maxlen=5)
        self.last_decoded_time = None
        self.current_bin_ephys_time = None
        self.next_bin_ephys_start_time = None
        self.early_spike_buffer = {
            'time_stamps': np.array([], dtype=np.int64),
            'unique_ids': np.array([], dtype=np.int32)
        }
        self.scv_save=[]
        self.update_t_save=[]
        self.missed_spikes_save = []
        self.report_bin=[]
    # def input(self, bmi_output, type='individual_spike'):
    #     '''
    #     each bmi_output is a spike with its timestamp and spike id
    #     each time a bmi_output arrive, this function is triggered
    #     when nbins grows, the binner emits the `decode` event with its `_output`
    #     '''
    #     self.current_time = bmi_output.timestamp*self.dt
    #     self.current_bin = int(self.current_time//self.bin_size) # devided by [bin_size], current_bin is abosolute bin
    #     spk_id = int(bmi_output.spk_id)

    #     if self.current_bin < self.B:                                                 # within B, no new bin
    #         self.count_vec[self.current_bin, spk_id] += 1                  # update according to current_bin
    #     elif self.current_bin >= self.B and self.current_bin==self.last_bin:          # current_bin 
    #         self.count_vec[-1, spk_id] += 1
    #     elif self.current_bin >= self.B and self.current_bin>self.last_bin:           # key: current_bin>last_bin means a input to decoder is completed 
    #         self.emit('decode', X=self.output)                                        # output count_vec for decoding
    #         self.count_vec = np.vstack((self.count_vec[1:], np.zeros((1, self.N))))   # roll and append new bin (last row)
    #         self.count_vec[-1, spk_id] += 1                                # update the newly appended bin (last row)

    #     # print(self.count_vec.shape, self.current_bin, self.last_bin)
    #     self.last_bin = self.current_bin

    
    def input(self, bmi_output, type='individual_spike'):

        """
        Process all spikes received since last update
        """
        current_time = time.time()
        time_stamps = bmi_output['time_stamps']
        unique_ids = bmi_output['unique_ids']
        if self.first_call:
            self.last_update_time = current_time
            self.first_call = False
            self.current_bin_ephys_time = time_stamps[0] # store the first spike time for the first bin start time (unit is not seconds, it is timepoint count), it is from the spikeglx, so it is the real time (already calibrated based on the first sync time)
            self.next_bin_ephys_start_time = self.current_bin_ephys_time + self.bin_size * self.fs

        time_elapsed = current_time - self.last_update_time
        # self.log.info('current time: {}, time_elapsed: {}'.format(current_time,time_elapsed))

        
        # Check if we need to roll to a new bin (every t_step seconds)
        if time_elapsed >= self.bin_size + self.t_grace:
            # decode here:
            # X = self.scv.copy().reshape(1, self.scv.shape[0], self.scv.shape[1])
            self.update_t_save.append(self.current_bin_ephys_time)
            self.scv_save.append(self.count_vec[-1].copy())
            
            self.emit('decode', X=self.output)
            # Initialize the new bin with zeros
            self.last_update_time = current_time
            self.current_bin_ephys_time = time_stamps[0]
            self.next_bin_ephys_start_time = self.current_bin_ephys_time + self.bin_size * self.fs
            new_bin = np.zeros((1, self.N))

            # Check the buffer to see if any "early" spikes now belong in this new bin
            if self.early_spike_buffer['time_stamps'].size > 0:
                buffered_ts = self.early_spike_buffer['time_stamps']
                buffered_ids = self.early_spike_buffer['unique_ids']

                # Find which buffered spikes belong in the NEW bin
                in_new_bin_mask = (self.current_bin_ephys_time <= buffered_ts) & (buffered_ts < self.next_bin_ephys_start_time)
                
                # Find which spikes are STILL too early and must remain in the buffer
                still_early_mask = buffered_ts >= self.next_bin_ephys_start_time

                # Add spikes that fall in the new bin to its count
                if np.any(in_new_bin_mask):
                    np.add.at(new_bin[0, :], buffered_ids[in_new_bin_mask], 1)
                    # print(f"Processed {np.sum(in_new_bin_mask)} buffered spikes into new bin.")
                    # self.log.info(f"Processed {np.sum(in_new_bin_mask)} buffered spikes into new bin.")

                # Spikes that are not in the new bin and are not "still early" have been missed. 
                # This can happen if the processing loop is slow and skips a bin entirely.
                missed_spikes = len(buffered_ts) - np.sum(in_new_bin_mask) - np.sum(still_early_mask)
                if missed_spikes > 0:
                    # print(f"WARNING: Discarded {missed_spikes} buffered spikes that are now too old.")
                    # self.log.info(f"WARNING: Discarded {missed_spikes} buffered spikes that are now too old.")
                    self.missed_spikes_save.append(missed_spikes.copy())
                # Update the buffer to only contain the spikes that are still early
                self.early_spike_buffer['time_stamps'] = buffered_ts[still_early_mask]
                self.early_spike_buffer['unique_ids'] = buffered_ids[still_early_mask]
            
            # Append the new bin, which now contains the counts from the buffered spikes
            self.count_vec = np.vstack((self.count_vec[1:], new_bin))
            
            # self.current_bin_time += self.bin_size
            # self.current_time += time_elapsed
            #np.save('scv.npy', self.scv_save)

            




        

        

        # Process all spikes into the current (last) bin
        if len(time_stamps) > 0:
            # Vectorized NumPy operations to count spikes in the current bin
            in_bin_mask = (self.current_bin_ephys_time <= time_stamps) & (time_stamps < self.next_bin_ephys_start_time)
            # Count spikes that are out of the current bin
            late_mask = time_stamps < self.current_bin_ephys_time # late spikes means the spikes that should be in the last bin, but they came too late to be counted, this indicates the grace period is too short
            early_mask = time_stamps >= self.next_bin_ephys_start_time # early spikes means the spikes that should be in the next bin, but they came too early to be counted, this indicates the grace period is too long
            early_count = np.sum(early_mask)
            late_count = np.sum(late_mask)
            in_bin_count = np.sum(in_bin_mask)
            # if not len(self.report_bin):
            #     self.report_bin = np.array([self.current_bin_ephys_time, early_count, late_count, in_bin_count])
            # else:
            #     self.report_bin = np.vstack((self.report_bin, np.array([self.current_bin_ephys_time, early_count, late_count, in_bin_count])))
            # Process in-bin spikes
            if in_bin_count > 0:
                np.add.at(self.count_vec[-1, :], unique_ids[in_bin_mask], 1)

            # Buffer early spikes
            if early_count > 0:
                self.early_spike_buffer['time_stamps'] = np.concatenate([self.early_spike_buffer['time_stamps'], time_stamps[early_mask]])
                self.early_spike_buffer['unique_ids'] = np.concatenate([self.early_spike_buffer['unique_ids'], unique_ids[early_mask]])

            # Report statistics
            # print(f"Spikes: {in_bin_count} in-bin, {early_count} early (buffered), {late_count} late (discarded)")
            # self.log.info(f"Spikes: {in_bin_count} in-bin, {early_count} early (buffered), {late_count} late (discarded)")
            if late_count>0:
                self.report_bin.append([
                    self.current_bin_ephys_time, 
                    early_count, 
                    late_count, 
                    in_bin_count
                ])

    
    
    @property
    def output(self):
        # first column (unit) is the noise
        # Warning: because the binner never send the unit#0 (noise) to the
        # decoder, we should also exclude unit#0 when building the decoder.
        self._output=self.count_vec.copy().reshape(1, self.count_vec.shape[0], self.count_vec.shape[1])
        if self.exclude_first_unit:
            self._output = self._output[:, 1:] 
        else:
            self._output = self._output

        return self._output
