class Polynomial:

    def __init__(self, a, b, c) -> None:
        
        self.a_ = a
        self.b_ = b
        self.c_ = c

    def getValueAtSpecifiedTime(self, t):
        return self.a_*(t**2) + self.b_*(t) + self.c_
    
    