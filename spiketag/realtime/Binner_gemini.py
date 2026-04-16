import numpy as np
import time
from ..utils.utils import EventEmitter

class Binner(EventEmitter):
    def __init__(self, bin_size, n_id, n_bin, sampling_rate=30000, exclude_first_unit=False):
        super(Binner, self).__init__()
        self.bin_size = bin_size
        self.N = n_id
        self.B = n_bin
        self.fs = sampling_rate
        self.bin_size_samples = bin_size * self.fs
        
        self.count_vec = np.zeros((self.B, self.N))
        self.exclude_first_unit = exclude_first_unit
        self.t_grace = 0.005 

        self.last_update_time = None
        self.current_bin_ephys_time = None
        self.next_bin_ephys_start_time = None
        
        # Use lists for "save" variables; appending to lists is O(1), np.append is O(n)
        self.scv_save = []
        self.update_t_save = []
        self.missed_spikes_save = []
        self.report_bin = []

        # Buffer for early spikes
        self.early_ts_buffer = []
        self.early_ids_buffer = []
        
        self.first_call = True

    def input(self, bmi_output, type='individual_spike'):
        current_real_time = time.time()
        time_stamps = bmi_output['time_stamps']
        unique_ids = bmi_output['unique_ids']

        if self.first_call:
            self.last_update_time = current_real_time
            self.current_bin_ephys_time = time_stamps[0]
            self.next_bin_ephys_start_time = self.current_bin_ephys_time + self.bin_size_samples
            self.first_call = False

        # 1. Check if bin needs to roll
        time_elapsed = current_real_time - self.last_update_time
        if time_elapsed >= (self.bin_size + self.t_grace):
            self._roll_bin(current_real_time)

        # 2. Process incoming batch
        if time_stamps.size > 0:
            # Single pass boolean indexing
            late_mask = time_stamps < self.current_bin_ephys_time
            early_mask = time_stamps >= self.next_bin_ephys_start_time
            # In-bin is everything that is neither late nor early
            in_bin_mask = ~(late_mask | early_mask)

            # Update current bin counts
            if np.any(in_bin_mask):
                np.add.at(self.count_vec[-1, :], unique_ids[in_bin_mask], 1)

            # Buffer early spikes (using list extend for speed)
            if np.any(early_mask):
                self.early_ts_buffer.extend(time_stamps[early_mask])
                self.early_ids_buffer.extend(unique_ids[early_mask])

            # Metadata reporting (internal tracking)
            if np.any(late_mask):
                self.report_bin.append([
                    self.current_bin_ephys_time, 
                    np.sum(early_mask), 
                    np.sum(late_mask), 
                    np.sum(in_bin_mask)
                ])

    def _roll_bin(self, current_real_time):
        """Internal helper to handle bin rotation and buffer processing"""
        # Save history before rolling
        self.update_t_save.append(self.current_bin_ephys_time)
        self.scv_save.append(self.count_vec[-1].copy())
        
        # Emit current state to decoder
        self.emit('decode', X=self.output)

        # Shift bins (Rolling window)
        self.count_vec = np.roll(self.count_vec, -1, axis=0)
        self.count_vec[-1, :] = 0  # Clear the new bin

        # Process Buffered Early Spikes
        if self.early_ts_buffer:
            ts_arr = np.array(self.early_ts_buffer)
            id_arr = np.array(self.early_ids_buffer)
            
            # Determine what fits in the new bin
            new_next_start = self.current_bin_ephys_time + (2 * self.bin_size_samples) # Approximation
            in_new_bin = (ts_arr >= self.next_bin_ephys_start_time) & (ts_arr < new_next_start)
            still_early = ts_arr >= new_next_start
            
            if np.any(in_new_bin):
                np.add.at(self.count_vec[-1, :], id_arr[in_new_bin], 1)

            # Cleanup buffer: keep only what is still early
            self.early_ts_buffer = ts_arr[still_early].tolist()
            self.early_ids_buffer = id_arr[still_early].tolist()
            
            # Track missed spikes (those neither in new bin nor early)
            missed = len(ts_arr) - np.sum(in_new_bin) - np.sum(still_early)
            if missed > 0:
                self.missed_spikes_save.append(missed)

        # Update time pointers
        self.last_update_time = current_real_time
        self.current_bin_ephys_time = self.next_bin_ephys_start_time
        self.next_bin_ephys_start_time += self.bin_size_samples

    @property
    def output(self):
        # Efficient reshaping without unnecessary copies
        out = self.count_vec.copy().reshape(1, self.B, self.N)
        if self.exclude_first_unit:
            return out[:, :, 1:]
        return out