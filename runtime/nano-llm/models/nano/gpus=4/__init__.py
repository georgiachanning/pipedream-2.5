
from .stage0 import stage0
from .stage1 import stage1
from .stage2 import stage2
from .stage3 import stage3
from .stage4 import stage4

def arch():
    return "nanoGPT"

def model(criterion):
    # stages = [stage0(), stage1(), stage2(), stage3(), stage4()]
    stages = [stage0(), stage1(), stage2(), stage3(), stage4(criterion)]
    return stages
