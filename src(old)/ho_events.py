class A3Event:
    def __init__(self, offset=5):
        self.offset = offset  # dB

    def check(self, serving_sinr, neighbor_sinr, serving_offset=0, neighbor_offset=0):
        #TODO: Add hysterisis
        if neighbor_sinr + neighbor_offset > serving_sinr + serving_offset + self.offset:
            return True

