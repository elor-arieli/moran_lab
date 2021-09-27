#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygtk
import ExtractIntanData
pygtk.require("2.0")
import gtk
import gtk.glade
import matplotlib.pyplot as plt
import serial


class Arduino(object):

    __OUTPUT_PINS = -1

    def __init__(self, port, baudrate=9600):
        self.serial = serial.Serial(port, baudrate)
        self.serial.write(b'99')

    def __str__(self):
        return "Arduino is on port %s at %d baudrate" %(self.serial.port, self.serial.baudrate)

    def output(self, pinArray):
        self.__sendData(len(pinArray))

        if(isinstance(pinArray, list) or isinstance(pinArray, tuple)):
            self.__OUTPUT_PINS = pinArray
            for each_pin in pinArray:
                self.__sendData(each_pin)
        return True

    def setLow(self, pin):
        self.__sendData('0')
        self.__sendData(pin)
        return True

    def setHigh(self, pin):
        self.__sendData('1')
        self.__sendData(pin)
        return True

    def getState(self, pin):
        self.__sendData('2')
        self.__sendData(pin)
        return self.__formatPinState(self.__getData()[0])

    def analogWrite(self, pin, value):
        self.__sendData('3')
        self.__sendData(pin)
        self.__sendData(value)
        return True

    def analogRead(self, pin):
        self.__sendData('4')
        self.__sendData(pin)
        return self.__getData()

    def turnOff(self):
        for each_pin in self.__OUTPUT_PINS:
            self.setLow(each_pin)
        return True

    def __sendData(self, serial_data):
        while(self.__getData()[0] != "w"):
            pass
        serial_data = str(serial_data).encode('utf-8')
        self.serial.write(serial_data)

    def __getData(self):
        input_string = self.serial.readline()
        input_string = input_string.decode('utf-8')
        return input_string.rstrip('\n')

    def __formatPinState(self, pinValue):
        if pinValue == '1':
            return True
        else:
            return False

    def close(self):
        self.serial.close()
        return True


class GUI:
    def __init__(self):
        self.gladefile = "C:\Python27\Lib\Glade\Data Extraction.glade"
        self.wTree = gtk.Builder()
        self.wTree.add_from_file(self.gladefile)
        # Create our dictionay and connect it
        self.dic = {"on_MainWindow_destroy": gtk.main_quit, "flow": self.flow}
        self.wTree.connect_signals(self.dic)
        self.window = self.wTree.get_object("window1")
        self.waterbutton = self.wTree.get_object("water")
        self.naclbutton = self.wTree.get_object("nacl")
        self.cabutton = self.wTree.get_object("ca")
        self.sugarbutton = self.wTree.get_object("sugar")
        self.quininebutton = self.wTree.get_object("quinine")
        self.allbutton = self.wTree.get_object("all")
        self.window.set_default_size(800,400)
        self.arduino = Arduino("Com3")
        Arduino.output([30,31,32,33,34])
        self.transalator = {'Water': 30, 'Sugar': 32, 'NaCl': 31, 'Quinine': 34, 'Citric Acid': 33, 'All tastes': [30,31,32,33,34]}

        if (self.window):
            self.window.connect("destroy", gtk.main_quit)
        self.window.show_all()

    def flow(self,button):
        taste = self.transalator[button.get_label()]
        if isinstance(taste,int):
            if button.get_active() == True :
                Arduino.setHigh(taste)
            else:
                Arduino.setLow(taste)
        else:
            if button.get_active() == True :
                for pin in taste:
                    Arduino.setHigh(pin)
            else:
                for pin in taste:
                    Arduino.setLow(pin)


if __name__ == "__main__":
    a = GUI()
    gtk.main()
