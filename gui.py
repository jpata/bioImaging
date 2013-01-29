import wx
import wx.grid

import os
from images import *

# Some classes to use for the notebook pages.  Obviously you would
# want to use something more meaningful for your application, these
# are just for illustration.
model = LuminescentModel()

class PageOne(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        t = wx.StaticText(self, -1, "This is a PageTwo object", (40,40))

class PageTwo(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.grid = wx.grid.Grid(self, -1)
        self.grid.CreateGrid(100000,10)
        self.grid.SetColLabelValue(0, "Input data")
        #self.grid.SetCellValue(3,3, "asdASd")
        bsizer = wx.BoxSizer()
        bsizer.Add(self.grid, 1, wx.EXPAND)
        self.SetSizerAndFit(bsizer)
    
    def setGridRow(self, i, j, items):
        for k in range(len(items)):
            self.grid.SetCellValue(i, j+k, str(items[k]))
        return

    def updateGrid(self, model):
        for i in range(len(model.lumis)):
            self.setGridRow(i, 0, [
                model.lumis[i].ci.dir,
                model.lumis[i].ci.lumi.image,
                model.lumis[i].ci.lumi.binning,
                model.lumis[i].ci.lumi.exposure,
                model.lumis[i].ci.lumi.fNumber,
                model.lumis[i].ci.lumi.FOV,
                model.lumis[i].ci.lumi.excitationFilter,
                model.lumis[i].ci.lumi.emissionFilter
            ])
            if not model.lumis[i].lumiImg is None:
                self.grid.SetCellValue(i, 8, str(model.lumis[i].lumiImg.stats.mean))


class PageThree(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        t = wx.StaticText(self, -1, "This is a PageThree object", (60,60))


def recursiveFind(path, search):
    out = []
    for f in os.walk(path):
        print(f)
        if search in f[2]:
            print f[0]
            out.append(f[0])
    return out

class MainFrame(wx.Frame):

    def OnQuit(self, e):
        self.close()
        print e
        
    def OnOpen(self, e):
        dlg = wx.DirDialog(self, "Choose a directory:")
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            cis = recursiveFind(path, "ClickInfo.txt")
            for i in range(len(cis)):
                dir = cis[i]
                #self.page2.grid.SetCellValue(i,0, dir)
                try:
                    ci = readClickInfo(dir)
                except Exception as e:
                    print e.message
                    continue
                model.addLumi(Luminescent(ci))
                #self.page2.grid.SetCellValue(i,1, ci.lumi.image)
                #self.page2.grid.SetCellValue(i,2, str(ci.lumi.exposure))

        self.page2.updateGrid(model)
        
        dlg.Destroy()
        return
    
    def OnLoad(self, e):
        model.loadAll()
        model.analyzeAll()
        self.page2.updateGrid(model)
        
    def __init__(self):
        wx.Frame.__init__(self, None, title="Simple Notebook Example")
        
        #Menu
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        menuOpen = fileMenu.Append(wx.ID_OPEN, '&Open', 'Open data directory')
        menuLoad = fileMenu.Append(wx.ID_ANY, '&Load', 'Loads images to memory')
        menuQuit = fileMenu.Append(wx.ID_ANY, '&Quit')
        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.OnQuit, menuQuit)
        self.Bind(wx.EVT_MENU, self.OnOpen, menuOpen)
        self.Bind(wx.EVT_MENU, self.OnLoad, menuLoad)

        # Here we create a panel and a notebook on the panel
        p = wx.Panel(self)
        nb = wx.Notebook(p)

        # create the page windows as children of the notebook
        page1 = PageOne(nb)
        self.page2 = PageTwo(nb)
        page3 = PageThree(nb)

        # add the pages to the notebook with the label to show on the tab
        nb.AddPage(page1, "Page 1")
        nb.AddPage(self.page2, "Page 2")
        nb.AddPage(page3, "Page 3")

        # finally, put the notebook in a sizer for the panel to manage
        # the layout
        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        p.SetSizer(sizer)


if __name__ == "__main__":
    app = wx.App()
    mf = MainFrame()
    mf.Show()
    app.MainLoop()