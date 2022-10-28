#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:52:53 2020

@author: emcastro
"""

import os
import sys

path_todo = "/home/emcastro/deepmm/queue.txt"
path_done = "/home/emcastro/deepmm/done.txt"
if __name__ == "__main__":
    if sys.argv[1] == "run":
        while True:
            # read a command
            f = open(path_todo, "r")
            lines = f.readlines()
            if len(lines) == 0:
                break
            f.close()
            command = lines[0]

            # run that command
            os.system(command)
            f = open(path_done, "a")
            f.write(command)
            f.close()
            
            del lines[0]
            # delete command from todo
            f = open(path_todo, "w")
            for line in lines:
                f.write(line)
            f.close()

    if sys.argv[1] == "add":
        f = open(path_todo, "a")
        f.write(sys.argv[2]+"\n")
        f.close()