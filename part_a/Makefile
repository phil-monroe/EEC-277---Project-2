#
# $Id: Makefile-instructional,v 1.1 2010-01-11 16:35:33 wes Exp $
# Version: $Name: V1-2 $
# $Revision: 1.1 $
# $Log: Makefile-instructional,v $
# Revision 1.1  2010-01-11 16:35:33  wes
# Initial entry
#
#

APPS = wesBench-instructional wesBench-visual

#OPT = -g
OPT = -O
CFLAGS = $(OPT) $(ARCH) -I/usr/include


# you might have to change these on your system
OPENGLLIBS =  -lGL -lGLU
GLUTLIBS =  -lglut -lm
MACLIBS =  -framework GLUT -framework OpenGL -framework Cocoa

UNAME = $(shell uname)

ifeq ($(UNAME), Darwin)
	INCLUDES = $(MACLIBS)
else
	INCLUDES = $(OPENGLLIBS) $(GLUTLIBS)
endif


all:	$(APPS)

wesBench-instructional: wesBench-instructional.c 
	$(CC) $(CFLAGS) -o wesBench-instructional $(INCLUDES) wesBench-instructional.c
	
wesBench-visual: wesBench-visual.c 
	$(CC) $(CFLAGS) -o wesBench-visual $(INCLUDES) wesBench-visual.c


clean:
	if (rm *~ $(APPS) TAGS) then :; fi

#EOF