
import unittest
import os
import time
import matplotlib.pyplot as plt

from ..diagrams import Diagram

tests_directory = 'Tests'
os.system('mkdir -p ' + tests_directory)

class TestDiagram(unittest.TestCase):

    #def setUp(self):

    #    self.fig = plt.figure(figsize=(6,6))
    #    self.ax = self.fig.add_subplot(111)
    #    self.ax.set_xlim(0,1)
    #    self.ax.set_ylim(0,1)
    #    self.ax.set_xticks([])
    #    self.ax.set_yticks([])
    #    self.dia = Diagram(self.ax)

    #def plot(self):
    #    self.dia.plot()

    def show(self):
        plt.show()

    def show_pdf(self, fname="tmp.pdf"):
        fname = fname.split('.pdf')[0] + '.pdf'
        fname = os.path.join(tests_directory, fname)
        plt.savefig(fname)
        os.system('open ' + fname)
        #time.sleep(.5)
        #os.system('rm ' + fname)  # quite unsafe



