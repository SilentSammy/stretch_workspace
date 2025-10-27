from target_finder import TargetFinder

class TennisFinder(TargetFinder):
    def __init__(self, persistence_frames=10):
        super().__init__(persistence_frames)
        pass
    
    def _detect_target(self, rgb_frame, drawing_frame=None):
        pass