import Image
import numpy
import logging
import ipdb
import copy
import math
import json

import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib

import os
import xlwt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

logger.info("Imported imaging library")

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

    def encode(self):
        return {"ccdCountsToRadiance": self.ccdCountsToRadiance}

    def __str__(self):
        return "CalibrationCoefficients: " + str(self.encode())

    #return json.JSONEncoder.default(self, obj)

class Measurement:
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

class bioImage:

    def saveToFile(self, fn):
        logger = logging.getLogger(self.logger().name + ":saveToFile()")
        
        logger.info("Saving data to file {0}".format(fn))
        s = json.dumps(self, cls=Encoder, sort_keys=True, indent=4)
        logger.debug("Encoded to JSON as {0}".format(s))
        f = open(fn, "w")
        f.write(s)
        f.close()
        return
    
    def saveImage(self, fn):
        logger = logging.getLogger(self.logger().name + ":saveImage()")
        logger.info("Saving image to file {0}".format(fn))
        img = Image.fromarray(self.arr)
        img.save(fn, "TIFF")
        return

    def encode(self):
        out = dict()
        out["name"] = self.name
        out["stats"] = self.stats
        out["binning"] = self.binning
        out["biasSubtracted"] = self.biasSubtracted
        out["radianceCalibrated"] = self.radianceCalibrated
        if hasattr(self, "biasMean"):
            out["biasMean"] = self.biasMean
        if hasattr(self, "measurements"):
            out["measurements"] = self.measurements        
        
        return out
    
    def calibrate(self, calibration):
        logger = logging.getLogger(self.logger().name + ":calibrate()")
        logger.info("Calibrating image with calibration {0}".format(calibration))
                
        newArr = self.arr*calibration.ccdCountsToRadiance
        newStats = ImageStats(newArr)
        newStats.printStats(logger)
        self.radianceCalibrated = True
        
        self.arr = newArr
        self.stats = newStats
        
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
        logger = logging.getLogger(self.logger().name + ":__init__()")
        logger.info("Loaded image {0}".format(fileName))
        self.printStats()

    def printStats(self):
        self.stats.printStats(self.logger())
    
    def subtractBias(self, biasimg, useMean=False):
        logger = logging.getLogger(self.logger().name + ":subtractBias()")
        if self.arr.shape != biasimg.arr.shape:
            logger.error("This image and supplied bias image have different shape, aborting: {0} != {1}".format(elf.arr.shape, biasimg.arr.shape))
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
    
        if(self.stats.minVal < 0):
            logger.error("Image contains negative values after bias subtraction.")
            self.arr = self.arr + abs(numpy.min(self.arr)) + 1
            self.stats = ImageStats(self.arr)
            logger.info("Adding minVal as padding to make image non-zero.")



        return

    """
        Multiplies the image shape by the binning and divides each pixel by the binning squared,
        thus keeping the total fixed. Note that the max, min and mean of the image change accordingly.
    """
    def rebin(self, binning):
        logger = logging.getLogger(self.logger().name + ":rebin()")
        logger.info("Rebinning image from binning {0} to binning {1}.".format(self.binning, binning))

        newArr = self.arr/math.pow(binning,2)
        newArr = numpy.kron(newArr, numpy.ones((binning, binning)))
        
        #Require that the total be the same within numerical precision
        assert(abs(numpy.sum(self.arr)-numpy.sum(newArr))<0.1)

        self.arr = newArr
        self.stats = ImageStats(self.arr)
        self.binning = binning
        return

    def binarize(self, nsigma=2):
        level = self.stats.mean + nsigma*self.stats.stddev
        self.binArr = numpy.greater(self.arr, level)
        return
    
    def dilate(self, it=3):
        self.binArr = scipy.ndimage.binary_dilation(self.binArr, iterations=it)
        return

    def components(self, outfn=None):
        
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
        
        #Remove small components from mask
        large = map(lambda x: x.area>100, measurements)
        remove_pixels = labels[large]
        self.binArr[remove_pixels] = False
        
        #Remove small components from measurements
        filtered_measurements = []
        for i in range(len(large)):
            if large[i]:
                filtered_measurements.append(measurements[i])
        measurements = filtered_measurements
        
        measurements_sorted = sorted(measurements, key=lambda m: m.pos[1])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        #pdb.set_trace()
        
        plt.imshow(numpy.log(self.arr), cmap=matplotlib.cm.gnuplot2_r)
        plt.contour(self.binArr, colors="black", alpha=0.9)
        i = 1
        for m in measurements:
            r = math.sqrt(m.area/math.pi)*4
            m.label = i
            ax.text(m.pos[1], m.pos[0]+r, "%d\n%.2E\n%.2E\n%.2E" % (i, m.total, m.mean, m.area), alpha=0.9, color="black")
            i += 1
        if not outfn is None:
            fig.savefig(outfn)
        
        #plt.show()
    
        self.measurements = measurements
        return measurements
        

def showArr(arr):
    plt.imshow(arr)
    plt.show()
    return

def processLuminescent(dir, outDir):
    fnLumi = dir + "/luminescent.TIF"
    fnBias = dir + "/readbiasonly.TIF"
    lumi = bioImage(fnLumi)
    bias = bioImage(fnBias)    
    lumi.subtractBias(bias, useMean=True)
    
    binning = 1
    lumi.rebin(binning)
    c = Calibration(1)
    lumi.binarize(nsigma=0.3)
    lumi.dilate(it=binning*2)

    measurements = lumi.components(outDir + "/out.png")
    lumi.saveToFile(outDir + "/out.txt")

    return lumi

if __name__=="__main__":
    logger = logging.getLogger("main()")

    base = "/Users/joosep/Dropbox/hiired/luminestsents/Joosepile luminestsents/mGli2R 20dets2012"
    dirs = os.listdir(base)
    dirs = filter(lambda x: x.startswith("PP") and not x.endswith("_SEQ"), dirs)
    dirs = map(lambda x: base + "/" + x, dirs)
    i = 0
    
    wb = xlwt.Workbook()
    sheet = wb.add_sheet("data")
    for d in dirs:
        i += 1        
        logger.info("Processing directory: {0}".format(d))
        ofdir = "testOut/" + str(i)
        try:
            os.mkdir(ofdir)
            p = processLuminescent(d, ofdir)
        except Exception as e:
            logger.error(e)
            os.rmdir(ofdir)
            continue
        sheet = wb.add_sheet("Image%d" % i)
        tempImage = Image.open(ofdir + "/out.png").convert("RGB")
        tempImage.save("temp.bmp")
        sheet.insert_bitmap("temp.bmp", 5, 1)
        os.remove("temp.bmp")

    wb.save("test.xls")
























