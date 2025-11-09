import threading
import time

class NonBlockingPoller:
    def __init__(self, poll_function, wait_for_first=True, always_poll=True):
        self.poll_function = poll_function
        self.latest_result = None
        self.lock = threading.Lock()
        self.stop_thread = False
        self.always_poll = always_poll
        self.poll_requested = False
        
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        
        if wait_for_first:
            self.poll_requested = True
            while self.latest_result is None:
                time.sleep(0.01)
    
    def _poll_loop(self):
        while not self.stop_thread:
            if self.always_poll or self.poll_requested:
                result = self.poll_function()
                with self.lock:
                    self.latest_result = result
                    self.poll_requested = False
            else:
                time.sleep(0.001)
    
    def get(self):
        if not self.always_poll:
            self.poll_requested = True
        with self.lock:
            return self.latest_result
    
    def stop(self):
        self.stop_thread = True
        self.thread.join(timeout=1.0)
