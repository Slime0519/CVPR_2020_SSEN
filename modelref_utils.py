from Modules.OridinaryModels.Baseline import Baseline,Baseline_show
from Modules.OridinaryModels.Baseline_big import BigBaseline,BigBaseline_show
from Modules.OridinaryModels.Baseline_small import Baseline_small,Baseline_small_show
from Modules.OridinaryModels.lightbaseline import Baseline_light
from Modules.OridinaryModels.Baseline128 import Baseline128,Baseline128_show

from Modules.EDSR_pretrained_baseline.EDSR_baseline import EDSR_baseline, EDSR_baseline_show,EDSR_baseline_X4

def getprefixname(modeltype):
    if modeltype == "normal_concat":
        prefix_resultname = "normalModel_concat"
    elif modeltype == "normal":
        prefix_resultname = "normalModel"
    elif modeltype == "normal_cosine":
        prefix_resultname = "normalModel_cosine"
    elif modeltype == "normal128":
        prefix_resultname = "normalModel_model128"
    elif modeltype == "normal_cosine_concat":
        prefix_resultname = "normalModel_cosine_concat"
    elif modeltype == "normal_light":
        prefix_resultname = "normalModel_light"
    elif modeltype == "big":
        prefix_resultname = "bigModel"
    elif modeltype == "EDSR":
        prefix_resultname = "EDSR"
    elif modeltype == "EDSRx4":
        prefix_resultname = "EDSRx4"
    else:
        prefix_resultname = "smallModel"

    return prefix_resultname

def loadmodel(modeltype):
    if modeltype == "normal_concat" or modeltype == "normal_cosine_concat":
        print("load concat baseline module")
        Model = Baseline(mode="concat")
    elif modeltype == "normal" or modeltype == "normal_cosine":
        print("load original baseline module")
        Model = Baseline()
    elif modeltype == "normal128":
        print("load normal128 model")
        Model = Baseline128(mode="concat")
    elif modeltype == "normal_light":
        print("load light extraction model")
        Model = Baseline_light()
    elif modeltype == "big":
        print("load big baseline module")
        Model = BigBaseline()
    elif modeltype == "EDSR":
        print("load EDSRx2 baseline")
        Model = EDSR_baseline()
        Model.load_pretrained_model()
    elif modeltype == "EDSRx4":
        Model = EDSR_baseline_X4()
        Model.load_pretrained_model()
    else:
        print("load small baseline module")
        Model = Baseline_small()

    return Model

def loadshowmodel(modeltype):
    if modeltype == "normal":
        testmodel = Baseline_show()
    elif modeltype == "normal_concat" or modeltype == "normal_cosine_concat":
        print("load concat baseline module")
        testmodel = Baseline_show(mode="concat")
    elif modeltype == "big":
        print("load big baseline module")
        testmodel = BigBaseline_show()
    elif modeltype == "normal128":
        testmodel = Baseline128_show(mode="concat")
    elif modeltype == "EDSR":
        print("load EDSRx2 baseline")
        print("load EDSR_Show")
        testmodel = EDSR_baseline_show()
    elif modeltype == "EDSRx4":
        print("load EDSRx4 baseline")
        testmodel = EDSR_baseline_X4()
    else:
        print("load small baseline module")
        testmodel = Baseline_small_show()
    return testmodel