import sys
import subprocess
from PIL import Image
import numpy
import logging
import copy
import math
import json

import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib

import os
import xlwt

import glob
from click import *

import traceback
import shutil

import gc

from ConfigParser import RawConfigParser
import pdb

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "encode"):
            return obj.encode()
        return json.JSONEncoder.default(self, obj)



class ImageStats:

    def encode(self):
        return {
            "mean": numpy.asscalar(self.mean),
            "stddev": numpy.asscalar(self.stddev),
            "minVal": numpy.asscalar(self.minVal),
            "maxVal": numpy.asscalar(self.maxVal),
            "total": numpy.asscalar(self.total),
        }

    def __init__(self, src):
        self.mean = numpy.mean(src)
        self.stddev = numpy.std(src)
        self.minVal = numpy.min(src)
        self.maxVal = numpy.max(src)
        self.total = numpy.sum(src)

    def printStats(self, logger):
        logger.info("printStats():Image stats are: mean={0:.4E}, stddev={1:4E}, min={2:4E}, max={3:4E}, total={4:4E}".format(self.mean, self.stddev, self.minVal, self.maxVal, self.total))

class Calibration:
    def __init__(self, ccdCalib):
        self.ccdCountsToRadiance = ccdCalib
        self.totalC = (4*math.pi) * (12.5**2) / float(256*256)

    def encode(self):
        return {"ccdCountsToRadiance": self.ccdCountsToRadiance}

    def __str__(self):
        return "CalibrationCoefficients: " + str(self.encode())

    #return json.JSONEncoder.default(self, obj)

class Measurement:
    outFormat = ["label", "total", "area", "mean", "max", "stddev"]
    def __init__(self, total, area, mean, max, stddev, pos):
        self.total = total
        self.area = area
        self.mean = mean
        self.max = max
        self.stddev = stddev
        self.pos = pos
        self.label = None

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)

    def encode(self):
        return self.__dict__

    def writeToSheet(self, sheet, r, c):

        j = 0
        for o in Measurement.outFormat:
            sheet.write(r, c + j, getattr(self, o))
            j += 1
        return j

class bioImage:
    # def saveToFile(self, fn):
    #     logger = logging.getLogger(self.logger().name + ":saveToFile()")
    # 
    #     logger.info("Saving data to file {0}".format(fn))
    #     s = json.dumps(self, cls=Encoder, sort_keys=True, indent=4)
    #     logger.debug("Encoded to JSON as {0}".format(s))
    #     f = open(fn, "w")
    #     f.write(s)
    #     f.close()
    #     return
    # 
    # def saveImage(self, fn):
    #     logger = logging.getLogger(self.logger().name + ":saveImage()")
    #     logger.info("Saving image to file {0}".format(fn))
    #     img = Image.fromarray(self.arr)
    #     img.save(fn, "TIFF")
    #     return
    # 
    # def encode(self):
    #     out = dict()
    #     out["name"] = self.name
    #     out["stats"] = self.stats
    #     out["binning"] = self.binning
    #     out["biasSubtracted"] = self.biasSubtracted
    #     out["radianceCalibrated"] = self.radianceCalibrated
    #     if hasattr(self, "biasMean"):
    #         out["biasMean"] = self.biasMean
    #     if hasattr(self, "measurements"):
    #         out["measurements"] = self.measurements
    # 
    #     return out

    def calibrate(self, calibration):
        logger = logging.getLogger(self.logger().name + ":calibrate()")
        logger.info("Calibrating image with calibration {0}".format(calibration))

        newArr = self.arr*calibration.ccdCountsToRadiance
        newStats = ImageStats(newArr)
        newStats.printStats(logger)
        self.radianceCalibrated = True

        self.arr = newArr
        del newArr
        self.stats = newStats
        self.calibration = calibration
        self.stats.total *= calibration.totalC

    def logger(self):
        return logging.getLogger("bioImage:{0}".format(self.name))

    def __init__(self, fileName):
        self.name = fileName[fileName.rindex("/")+1:]
        img = Image.open(fileName)
        self.arr = numpy.array(img.getdata())
        self.arr = self.arr.reshape(img.size)
        self.binArr = None
        self.stats = ImageStats(self.arr)
        self.biasSubtracted = False
        self.binning = None
        self.radianceCalibrated = False
        self.exposureTime = None
        logger = logging.getLogger(self.logger().name + ":__init__()")
        logger.info("Loaded image {0}".format(fileName))
        self.printStats()

    def printStats(self):
        self.stats.printStats(self.logger())

    def writePixels(self, jetR, bgR):
        sz = self.arr.shape
        m = self.arr.min()
        #self.rebin(8)
        of = open("pixels.dat", "w")
        of.write("{0} {1}\n".format(jetR, bgR))
        pxarray = []
        for i in range(sz[0]):
            for j in range(sz[1]):
                e = self.arr[i,j]# - m
                phi = math.pi/4 + math.pi/4 * (float(j)/float(sz[1]))
                theta = math.pi/4 + math.pi/8 * (float(i)/float(sz[0]))
                px = e * math.cos(phi) * math.sin(theta)
                py = e * math.sin(phi) * math.sin(theta)
                pz = e * math.cos(theta)
                if (e>5):
                    pxarray += [(px, py, pz, e)]
                    of.write("{0} {1} {2} {3}\n".format(px, py, pz, e))

        of.close()
        return pxarray

    def runJets(self, jetR, bgR):
        self.writePixels(jetR, bgR)
        r = subprocess.check_output("cat pixels.dat | /Users/joosep/Documents/fastjet-test", shell=True)
        jets, cjets = self.parse_jets(r)

        measurements = []
        for i in range(len(jets)):
            m = Measurement(jets[i][2], len(cjets[i]), 0.0, 0.0, 0.0, (jets[i][1], jets[i][0]))
            measurements.append(m)
        measurements = sorted(measurements, key=lambda x: x.total, reverse=True)
        mu = numpy.mean([m.total for m in measurements])
        st = numpy.std([m.total for m in measurements])
        measurements = filter(lambda x: x.total > mu - 3*st, measurements)
        return measurements

    def to_cart(self, l):
        px, py, pz, e = map(float, l)
        pt = math.sqrt(px**2 + py**2)
        sz = self.arr.shape
        if pt>0 and e>0:
            phi = math.acos(px / pt)# + math.pi/2
            theta = math.acos(pz / e) #+ math.pi
            x = (phi-math.pi/4)/(math.pi/4) * sz[0]
            y = (theta-math.pi/4)/(math.pi/8) * sz[1]
            return x, y, e
        else:
            return 0,0,0

    def parse_jets(self, r):
        jets= []
        cjets = {}
        jetidx = -1
        for l in r.split("\n"):
            if l.startswith("#"):
                continue
            if l.startswith("j "):
                jetidx += 1
                jet = self.to_cart(l.split()[1:5])
                jets += [jet]
                cjets[jetidx] = []
            elif l.startswith("cj "):
                cjet = self.to_cart(l.split()[1:5])
                cjets[jetidx] += [cjet]
        return jets, cjets



    def subtractBias(self, biasimg, useMean=False, doPad=False):
        logger = logging.getLogger(self.logger().name + ":subtractBias()")
        if self.arr.shape != biasimg.arr.shape:
            logger.error("This image and supplied bias image have different shape, aborting: {0} != {1}".format(self.arr.shape, biasimg.arr.shape))
            return
        if self.biasSubtracted:
            logger.error("Bias has already been subtracted from this image, aborting.")
            return

        if useMean:
            logger.debug("Using mean bias")
            newArr = self.arr - biasimg.stats.mean
        else:
            logger.debug("Using full bias image")
            newArr = self.arr - biasimg.arr
        self.biasMean = biasimg.stats.mean

        #pdb.set_trace()
        oldStats = self.stats

        self.arr = newArr
        self.stats = ImageStats(self.arr)

        self.biasSubtracted = True
        logger.info("Done subtracting bias, the old and new stats are as follows:")
        oldStats.printStats(self.logger())
        self.stats.printStats(self.logger())

        if doPad and (self.stats.minVal < 0):
            logger.warning("Image contains negative values after bias subtraction: min(lumi)={0}, mean(bias)={1}, min(lumi-bias)={2}".format(oldStats.minVal, biasimg.stats.mean, self.stats.minVal))
            padding = abs(numpy.min(self.arr)) + 1
            self.arr = self.arr + padding
            self.padding = padding
            self.stats = ImageStats(self.arr)
            logger.info("Adding abs(minVal)={0} as padding to make image non-zero.".format(padding))

        return

    """
        Multiplies the image shape by the binning and divides each pixel by the binning squared,
        thus keeping the total fixed. Note that the max, min and mean of the image change accordingly.
    """
    def rebin(self, binning):
        logger = logging.getLogger(self.logger().name + ":rebin()")
        logger.info("Rebinning image from binning {0} to binning {1}.".format(self.binning, binning))

        newArr = self.arr/math.pow(binning,2)
        #newArr = self.arr
        newArr = numpy.kron(newArr, numpy.ones((binning, binning)))

        #Require that the total be the same within numerical precision
        #assert(abs(numpy.sum(self.arr)-numpy.sum(newArr))<0.1)

        self.arr = newArr
        del newArr
        self.stats = ImageStats(self.arr)
        self.binning = binning
        return

    def setExposure(self, expTime):
        self.arr = (1.0/float(expTime))*self.arr
        self.exposureTime = expTime
        self.stats = ImageStats(self.arr)
        return

    def binarize(self, nsigma=2):
        level = self.stats.mean + nsigma*self.stats.stddev
        self.binArr = numpy.greater(self.arr, level)
        return

    def dilate(self, it=3):
        self.binArr = scipy.ndimage.binary_dilation(self.binArr, iterations=it)
        return

    def fixOverflow(self):
        logger = logging.getLogger(self.logger().name + ":fixOverflow()")

        if self.stats.minVal < 0:
            logger.warning("Overflow detected: min(lumi)={0}".format(self.stats.minVal))
            self.arr[self.arr<0] = (2**16)+numpy.abs(self.arr[self.arr<0])
            self.stats = ImageStats(self.arr)
            logger.info("New stats after overflow fixing:")
            self.stats.printStats(logger)
        else:
            logger.info("No overflow")

        return

    def components(self, outfn=None, maxROIs=999):

        #mask = scipy.ndimage.percentile_filter(self.arr, 80, size=(8,8))
        mask = self.binArr
        labels, n = scipy.ndimage.label(mask)
        totals = scipy.ndimage.sum(self.arr, labels, range(1, n + 1))
        means = scipy.ndimage.mean(self.arr, labels, range(1, n + 1))
        sizes = scipy.ndimage.sum(mask, labels, range(1, n + 1))
        maximums = scipy.ndimage.maximum(self.arr, labels, range(1, n + 1))
        stddevs = scipy.ndimage.standard_deviation(self.arr, labels, range(1, n + 1))
        positions = scipy.ndimage.center_of_mass(self.arr, labels, range(1, n + 1))

        measurements = []
        for i in range(len(totals)):
            m = Measurement(totals[i], sizes[i], means[i], maximums[i], stddevs[i], positions[i])
            measurements.append(m)

        measurements = sorted(measurements, key=lambda x: x.total, reverse=True)
        self.measurements = measurements
        return measurements

        # #Remove small components from mask
        # large = map(lambda x: x.area>self.binning*self.binning*100, measurements)
        # largeIndex = zip(range(1,len(large)+1), large)
        # for (n, isLarge) in largeIndex:
        #     if not isLarge:
        #         self.binArr[labels==n] = False

        # #Remove small components from measurements
        # filtered_measurements = []
        # for i in range(len(large)):
        #     if large[i]:
        #         filtered_measurements.append(measurements[i])
        # measurements = filtered_measurements

        # measurements = measurements[0:maxROIs]

        # measurements_sorted = sorted(measurements, key=lambda m: m.pos[1])

    def plot_measurements(self, fig, ax, measurements):
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #pdb.set_trace()

        plt.imshow(numpy.log(self.arr), cmap="YlOrRd")
        #plt.contour(self.binArr, colors="black", alpha=1.0)
        i = 1
        for m in measurements:
            #r = math.sqrt(m.area/math.pi)*4
            r = m.area/10
            m.label = i
            m.total = self.calibration.totalC * m.total
            ax.scatter([m.pos[1]], [m.pos[0]], [r], alpha=0.2)
            ax.text(m.pos[1], m.pos[0], "{0}".format(i), alpha=1.0, color="black", fontsize=16)
            i += 1
        # print outfn
        # if not outfn is None:
        #     fig.savefig(outfn)

        self.analyzedFigure = fig
        #plt.show()
        #plt.close()
        #fig.clf()
        #del ax
        #del fig
        #gc.collect()

        #self.measurements = measurements
        return


def showArr(arr):
    plt.imshow(arr)
    plt.show()
    return

def processLuminescent(dir, outDir, subtractBias=True, nSigma = 0.3, nDilations=2, maxROIs=3):
    logger = logging.getLogger("processLuminescent({0})".format(dir))

    fnLumi = dir + "/luminescent.TIF"
    fnBias = dir + "/readbiasonly.TIF"
    #ciFile = dir + "/ClickInfo.txt"

    ci = readClickInfo(dir)
    lumi = bioImage(fnLumi)
    #lumi.fixOverflow()

    if subtractBias:
        try:
            bias = bioImage(fnBias)
        except IOError as e:
            logger.error("Could not open bias image: {0}".format(str(e)))
            logger.debug(traceback.format_exc())
            raise e
        if ci.readbias.exposure != 0:
            #bias.setExposure(ci.readbias.exposure)
            lumi.subtractBias(bias, useMean=True)
        else:
            logger.warning("Bias exposure time is 0, skipping bias subtraction")

    lumi.setExposure(ci.lumi.exposure)
    lumi.rebin(ci.lumi.binning)

    ccdCoef = ci.cam.ccdCoefs[ci.lumi.FOV][ci.lumi.fNumber]
    #logging.info("using ccdCoef={0}".format(ccdCoef))
    c = Calibration(ccdCoef)
    c.totalC = c.totalC / (ci.lumi.binning**2)
    lumi.calibrate(c)
    lumi.binarize(nSigma)
    lumi.dilate(it=ci.lumi.binning*nDilations)


    lumi.components(outDir, maxROIs)

    if not outDir is None:
        lumi.saveToFile(outDir + "/out.txt")

    return lumi

def convPath(s):
    return s.replace("\\", "/")

def main():
    print "bioImaging loaded"

    configFileName = "settings.txt"
    config = RawConfigParser()
    config.read(configFileName)

    logLevel = config.get("General", "logLevel")
    if logLevel=="INFO":
        level=logging.INFO
    elif logLevel=="WARNING":
        level=logging.WARNING
    elif logLevel=="DEBUG":
        level=logging.DEBUG
    elif logLevel=="ERROR":
        level=logging.ERROR
    else:
        level=logging.INFO

    logging.basicConfig(level=level, filename="log.txt", filemode="w")

    logger = logging.getLogger("main()")
    logger.info("bioImaging program started.")

    inFiles = config.get("Input", "inputFilePattern")
    maxFiles = config.getint("Input", "maximumFiles")

    excelOut = True#config.getboolean("Output", "writeExcelFile")
    excelOutFileName = config.get("Output", "excelFileName")

    tempWorkDir = "temporary"
    logger.info("Removing temporary dir {0}".format(tempWorkDir))
    shutil.rmtree(tempWorkDir, True)
    os.mkdir(tempWorkDir)

    files = glob.glob(inFiles)
    files = map(convPath, files)

    maxROIs = config.getint("Parameters", "maxROIs")
    subtractBias = config.getboolean("Parameters", "subtractDarkChargeBias")

    nSigma = config.getfloat("Parameters", "NumberOfSigmaAboveMean")
    nDilations = config.getint("Parameters", "NumberOfDilations")

    if excelOut:
        wb = xlwt.Workbook()
        sheet = wb.add_sheet("data")

    dataBeginCol = 4
    i = 0
    j = 0
    for nRoi in range(maxROIs):
        for m in Measurement.outFormat:
            sheet.write(0, dataBeginCol+j, m)
            j += 1

    for fn in files[0:maxFiles]:
        sys.stdout.write(".")
        sys.stdout.flush()
        d = fn[:fn.rindex("/")]
        i += 1
        logger.info("Processing directory: {0}".format(d))
        ofdir = tempWorkDir + "/" + str(i)

        if excelOut:
            wb.get_sheet(0).write(i, 0, fn)

        imageName = "Image%d" % i

        try:

            if not ofdir is None:
                os.mkdir(ofdir)
            p = processLuminescent(d, ofdir, subtractBias, nSigma, nDilations, maxROIs)
            if excelOut:
                wb.get_sheet(0).write(i, 1, imageName)
        except IOError as e:
            logger.error("Could not process {0}: {1}".format(d, str(e)))
            #os.rmdir(ofdir)

            if excelOut:
                wb.get_sheet(0).write(i, 1, "FAILED: {0}".format(str(e)))
            continue
        j = dataBeginCol

        if excelOut:
            wb.get_sheet(0).write(i, 2, p.stats.mean)
            wb.get_sheet(0).write(i, 3, p.stats.total)

        for m in p.measurements[0:maxROIs]:
            if excelOut:
                l = m.writeToSheet(wb.get_sheet(0), i, j)
            j += l

        if excelOut and (not ofdir is None):
            sheet = wb.add_sheet(imageName)
            tempImage = Image.open(ofdir + "/out.png").convert("RGB")
            tempImage.save("temp.bmp")
            sheet.insert_bitmap("temp.bmp", 5, 1)
            os.remove("temp.bmp")

        if excelOut:
            wb.save(excelOutFileName)
    print "Done analyzing %d files" % i

class Luminescent:
    def __init__(self, d, outDir):
        self.ci = readClickInfo(d)
        self.name = d[d.rindex("/")+1:]
        self.readBiasImg = None
        self.lumiImg = None
        self.photo = None
        self.outDir = outDir
        self.processedImageFileName = self.outDir + "/" + self.name + "_processed.png"

        self.measurements = None
        self.imgStats = None

    def load(self):
        self.lumiImg = bioImage(self.ci.dir + "/" + self.ci.lumi.image)
        self.readBiasImg = bioImage(self.ci.dir + "/" + self.ci.readbias.image)
        #self.photo = bioImage(self.ci.dir + "/" + self.ci.readbias.image)

    def analyze(self):
        print "analyze", self
        nSigma = 0.3
        nDilations = 2
        maxROIs = 3

        self.lumiImg.subtractBias(self.readBiasImg, useMean=True)
        self.lumiImg.setExposure(self.ci.lumi.exposure)
        self.lumiImg.rebin(self.ci.lumi.binning)

        ccdCoef = self.ci.cam.ccdCoefs[self.ci.lumi.FOV][self.ci.lumi.fNumber]
        c = Calibration(ccdCoef)
        c.totalC = c.totalC / (self.ci.lumi.binning**2)
        self.lumiImg.calibrate(c)
        #self.lumiImg.writePixels()
        self.lumiImg.binarize(nSigma)
        self.lumiImg.dilate(it=self.ci.lumi.binning*nDilations)
        self.lumiImg.components(self.processedImageFileName, maxROIs)

        self.measurements = self.lumiImg.measurements
        self.imgStats = self.lumiImg.stats

    def clean(self):
        #del self.lumiImg
        #del self.readBiasImg
        #gc.collect()
        return

    def all(self):
        try:
            self.load()
            self.analyze()
            self.clean()
            self.errorMessage = None
        except IOError as e:
            self.errorMessage = e.message
            print e
        return

class LuminescentModel:
    def __init__(self, dirs, outDir):
        self.outDir = outDir
        self.lumis = [Luminescent(d, self.outDir) for d in dirs]
        print self.lumis

    def process(self):
        for l in self.lumis:
            print l
            l.all()

class HTMLOut:
    def __init__(self, filename, model):
        self.filename = filename
        self.model = model

    def save(self):
        of = open(self.filename, "w")
        for lumi in self.model.lumis:
            if lumi.errorMessage is None:
                of.write("<img src={0}>\n".format(lumi.processedImageFileName))
                of.write(str(lumi.measurements))
                fig = plt.figure()
                ax = plt.axes()
                print lumi
                #import pdb
                #pdb.set_trace()
                lumi.lumiImg.plot_measurements(fig, ax, lumi.measurements)
        of.close()

if __name__=="__main__":
    dirs = glob.glob("/Users/Joosep/Dropbox/hiired/luminestsents/Joosepile luminestsents/mGli2R 20dets2012/*")
    dirs = filter(lambda x: os.path.exists(x + "/ClickInfo.txt"), dirs)

    dirs = dirs[0:5]
    print dirs

    outDir = "temp"
    if os.path.exists(outDir):
        shutil.rmtree(outDir)
    os.mkdir(outDir)
    lm = LuminescentModel(dirs, outDir)
    out = HTMLOut("index.html", lm)
    lm.process()
    out.save()
