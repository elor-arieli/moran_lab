__author__ = 'elor'

from PIL import Image
import PIL
import numpy as np
import cv2, serial
import Tkinter,tkFileDialog
import pygtk
import gtk
import gtk.glade
from random import shuffle
import threading
from time import sleep
import gobject
pygtk.require("2.0")
gobject.threads_init()

class GUI:
    def __init__(self):
        self.gladefile = "D:\\pythonGUI\\Set_builder.glade"
        self.wTree = gtk.Builder()
        self.wTree.add_from_file(self.gladefile)
        # Create our dictionay and connect it
        self.dic = {"on_MainWindow_destroy": gtk.main_quit, "create_array": self.create_array}
        self.wTree.connect_signals(self.dic)
        self.get_objects()
        self.set_starting_params()
        self.set_outfile()
        self.window.set_default_size(800,400)
        # style = self.bt1.get_style()

        if (self.window):
            self.window.connect("destroy", gtk.main_quit)
        self.window.show_all()

    def get_objects(self):
        self.window = self.wTree.get_object("window1")
        self.title = self.wTree.get_object("title")
        self.create_array_button = self.wTree.get_object("create array")
        self.laser_entry = self.wTree.get_object("laser")
        self.water_entry = self.wTree.get_object("water")
        self.nacl_entry = self.wTree.get_object("nacl")
        self.quinine_entry = self.wTree.get_object("quinine")
        self.ca_entry = self.wTree.get_object("ca")
        self.sugar_entry = self.wTree.get_object("sugar")
        self.trials_entry = self.wTree.get_object("trials")

    def set_starting_params(self):
        self.trials_per_block = 20
        self.sugar_prob = 0
        self.water_prob = 0
        self.ca_prob = 0
        self.nacl_prob = 0
        self.quinine_prob = 0
        self.laser_prob = 0
        self.taste_list = ['water','nacl','sugar','ca','quinine']

    def create_array(self,button):
        self.set_params()
        if self.laser_prob > 1 or (self.water_prob+self.nacl_prob+self.ca_prob+self.quinine_prob+self.sugar_prob) != 1:
            self.title.set_text("probabilities do not add up to 1!! make sure you entered the correct ratios")
        else:
            self.title.set_text("array created! :)")
            array = self.create_set()
            out = ','.join(array)
            print(out)
            with open(self.outfile,'a+') as f:
                f.write(out + '\n')
        self.set_starting_params()

    def set_params(self):
        if self.trials_entry.get_text() != '':
            self.trials_per_block = float(self.trials_entry.get_text())
        if self.sugar_entry.get_text() != '':
            self.sugar_prob = float(self.sugar_entry.get_text())
        if self.nacl_entry.get_text() != '':
            self.nacl_prob = float(self.nacl_entry.get_text())
        if self.ca_entry.get_text() != '':
            self.ca_prob = float(self.ca_entry.get_text())
        if self.quinine_entry.get_text() != '':
            self.quinine_prob = float(self.quinine_entry.get_text())
        if self.water_entry.get_text() != '':
            self.water_prob = float(self.water_entry.get_text())
        if self.laser_entry.get_text() != '':
            self.laser_prob = float(self.laser_entry.get_text())
    
    def create_set(self):
        tastes = []
        tastes += [1]*int(round(self.trials_per_block*self.water_prob*(1-self.laser_prob)))
        tastes += [2]*int(round(self.trials_per_block*self.water_prob*self.laser_prob))
        tastes += [5]*int(round(self.trials_per_block*self.sugar_prob*(1-self.laser_prob)))
        tastes += [6]*int(round(self.trials_per_block*self.sugar_prob*self.laser_prob))
        tastes += [3]*int(round(self.trials_per_block*self.nacl_prob*(1-self.laser_prob)))
        tastes += [4]*int(round(self.trials_per_block*self.nacl_prob*self.laser_prob))
        tastes += [9]*int(round(self.trials_per_block*self.quinine_prob*(1-self.laser_prob)))
        tastes += [10]*int(round(self.trials_per_block*self.quinine_prob*self.laser_prob))
        tastes += [7]*int(round(self.trials_per_block*self.ca_prob*(1-self.laser_prob)))
        tastes += [8]*int(round(self.trials_per_block*self.ca_prob*self.laser_prob))
        
        # laser_tastes = [i+1 for i in tastes]
        # tastes = tastes + laser_tastes
        output = []
        for i in range(int(400/self.trials_per_block)):
            shuffle(tastes)
            output += tastes
        return [str(a) for a in output]
        
    def set_outfile(self):
        root = Tkinter.Tk()
        root.withdraw()
        myFormats = [
                ('TXT','*.txt'),
                ]

        fileName = tkFileDialog.asksaveasfilename(parent=root,filetypes=myFormats ,title="Save parameter file as...")
        self.outfile = fileName + '.txt'

if __name__ == "__main__":
    GUI_window = GUI()
    gtk.main()
