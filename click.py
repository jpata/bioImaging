import re
import glob
import traceback

def getBlock(blocks, first):
    f = filter(lambda x: x.startswith(first), blocks)
    if len(f)==0:
        raise Exception("Block starting with '{0}' not found.".format(first))
    if len(f)>1:
        raise Exception("Block choice '{0}' is ambigous".format(first))
    return f[0]

def getMatched(x, s):
    m = re.search(x, s)
    if m is None:
        raise Exception("No match found for pattern '{0}'.".format(x))
    found = (m.string[m.start():m.end()]).strip()
    found = found.split("\t")
    return found

class ReadBiasBlock:
    def __init__(self, block):
        try:
            self.binning = int((getMatched("Binning Factor:\s\d+", block))[1])
        except Exception as e:
            raise Exception("Could not get binning: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.exposure = int((getMatched("Background Exposure \(Seconds\):\s\d+", block))[1])
        except Exception as e:
            raise Exception("Could not get exposure: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.image = str((getMatched("readbiasonly image:\s\w+\.\w+", block))[1])
        except Exception as e:
            raise Exception("Could not get image: \n{0}\nin block: \n{1}".format(e.message, block))

    def __repr__(self):
        return str(self.__dict__)

class Ddict(dict):
    def __init__(self, default=None):
        self.default = default
        self.final = False

    def __getitem__(self, key):
        if not self.has_key(key):
            if self.final:
                raise KeyError("Key '{0} not found".format(key))
            self[key] = self.default()
        return dict.__getitem__(self, key)

class LuminescentBlock:
    def __init__(self, block):
        try:
            self.binning = int((getMatched("Binning Factor:\s\d+", block))[1])
        except Exception as e:
            raise Exception("Could not get binning: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.exposure = int((getMatched("Luminescent Exposure \(Seconds\):\s\d+", block))[1])
        except Exception as e:
            raise Exception("Could not get exposure: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.image = str((getMatched("luminescent image:\s\w+\.\w+", block))[1])
        except Exception as e:
            raise Exception("Could not get image: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.fNumber = int((getMatched("f Number:\s\d+", block))[1])
        except Exception as e:
            raise Exception("Could not get f Number: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.FOV = str((getMatched("Field of View:\t\d+\.\d+", block))[1])
        except Exception as e:
            raise Exception("Could not get FOV: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.emissionFilter = str((getMatched("Emission filter:\t\w+", block))[1])
        except Exception as e:
            raise Exception("Could not get emission filter: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.excitationFilter = str((getMatched("Excitation filter:\t\w+", block))[1])
        except Exception as e:
            raise Exception("Could not get excitation filter: \n{0}\nin block: \n{1}".format(e.message, block))

    def __repr__(self):
        return str(self.__dict__)

class CameraBlock:
    def __init__(self, block):
        try:
            ccdCoefs = map(lambda x: x.strip(), re.findall("\n\w*[^#\r\n]*\w*Coef C-ccd at FOV \d+\.\d+, f\w:\t.+", block))
            self.ccdCoefs = Ddict(dict)
            for c in ccdCoefs:
                c = c.split()
                coef = float(c[-1])
                fnumber = int(c[-2][1:-1])
                fov = c[-3][:-1]
                self.ccdCoefs[fov][fnumber] = coef
            self.ccdCoefs.final = True
        except Exception as e:
            raise Exception("Could not get CCD coefs: \n{0}\n".format(e.message))

    def __repr__(self):
        return str(self.__dict__)

class UserInput:
    def __init__(self, block):
        try:
            self.comment = str(getMatched("Comment1:\s.+", block)[1])
        except Exception as e:
            raise Exception("Could not get comment: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.experiment = str(getMatched("Experiment:\s.+", block)[1])
        except Exception as e:
            raise Exception("Could not get experiment: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.cellLine = str(getMatched("Cell Line:\s.+", block)[1])
        except Exception as e:
            raise Exception("Could not get cell line: \n{0}\nin block: \n{1}".format(e.message, block))
        try:
            self.lucInjectionTime = float(getMatched("Luc Injection Time:\s.+", block)[1])
        except Exception as e:
            raise Exception("Could not get luc injection time: \n{0}\nin block: \n{1}".format(e.message, block))

    def __repr__(self):
        return str(self.__dict__)
class ClickInfo:
    def __init__(self, readbias, lumi, cam, dir):
        self.readbias = readbias
        self.lumi = lumi
        self.cam = cam
        self.dir = dir

def readBlocks(fn):
    f = open(fn)
    text = f.read()
    f.close()

    blocks = text.split("***")
    #Remove leading/trailing whitespace from all blocks
    blocks = map(lambda x: x.strip(), blocks)
    return blocks

def readClickInfo(dir):
    blocks = readBlocks(dir+"/ClickInfo.txt")

    rbBlock = ReadBiasBlock(getBlock(blocks, "readbiasonly"))
    lumiBlock = LuminescentBlock(getBlock(blocks, "luminescent"))
    camBlock = CameraBlock(getBlock(blocks, "Camera System Info"))
    return ClickInfo(rbBlock, lumiBlock, camBlock, dir)

def readAnalyzedClickInfo(dir):
    blocks = readBlocks(dir+"/AnalyzedClickInfo.txt")
    userInput = UserInput(getBlock(blocks, "User Label Name Set:	Living Image Universal"))
    return userInput

if __name__=="__main__":
    files = glob.glob("/Users/joosep/Dropbox/hiired/luminestsents/Joosepile luminestsents/mGli2R 20dets2012/PP*/ClickInfo.txt")

    cis = []
    for fn in files:
        dir = fn[:fn.rindex("/")]
        ci = readClickInfo(dir)
        try:
            an = readAnalyzedClickInfo(dir)
        except Exception as e:
            print "Could not get AnalyzedClickInfo for {1}: {0}".format(e.message, dir)
        cis.append(ci)


