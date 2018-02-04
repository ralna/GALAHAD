#!/usr/bin/python -uO

#  TkGalahad class
#  A python/Tk windowing environment for GALAHAD
#  Nick Gould, January 2004
#  for GALAHAD productions 2002-2004

from Tkinter import *
import os
import tkFont
import re
import sys
import threading

#  set colours

Black     = "#%02x%02x%02x" % (   0,   0,   0 )
White     = "#%02x%02x%02x" % ( 255, 255, 255 )
Red       = "#%02x%02x%02x" % ( 255,   0,   0 )
Green     = "#%02x%02x%02x" % (   0, 255,   0 )
Blue      = "#%02x%02x%02x" % (   0,   0, 255 )
Yellow    = "#%02x%02x%02x" % ( 255, 255,   0 )
Gold      = "#%02x%02x%02x" % ( 255, 215,   0 )
DarkGreen = "#%02x%02x%02x" % (   0, 100,   0 )
Brown     = "#%02x%02x%02x" % ( 165,  42,  42 )
Wheat     = "#%02x%02x%02x" % ( 245, 222, 179 )
Grey80    = "#%02x%02x%02x" % ( 204, 204, 204 )
Grey90    = "#%02x%02x%02x" % ( 229, 229, 229 )
Purple    = "#%02x%02x%02x" % ( 160,  32, 240 )

#  select fixed colours

logobackground         = White
logoforeground         = Blue 
warningbackground      = White
warningforeground      = Red

#  text for help windows

helpcommandstext = "LEFT BUTTONS\n \
  \nCD: change current directory \
  \nREFRESH: refresh list of .SIF files displayed from current directory \
  \nSPEC: set package specification options via a form \
  \nPACKAGE: select which GALAHAD package & architecture to use \
  \nSOLVE: decode current problem and then solve using selected GALAHAD package \
  \nRESOLVE: run selected GALAHAD package on the currently decoded problem \
  \nSOLUTION: display the solution file SOLUTION.d from the last run \
  \nSUMMARY: display the summary file SUMMARY.d from the last run \
  \nSTOP: stop the current galahad run \
  \nEXIT: terminate the session\n \
  \nMOUSE BUTTONS\n \
  \nLEFT SINGLE CLICK: focus on problem under cursor \
  \nLEFT DOUBLE CLICK: edit problem under cursor \
  \nMIDDLE SINGLE CLICK: decode problem under cursor and run selected \
GALAHAD package \
  \nRIGHT CLICK: decode problem under cursor, print problem info, \
and run selected GALAHAD package"
# \nMIDDLE DOUBLE CLICK: decode problem under cursor, print problem info, \
#and run GALAHAD\
#  \nMIDDLE TRIPLE CLICK: decode problem under cursor in debug mode, \
#and run GALAHAD\

helpabouttext = "A GALAHAD windowing environment written in python/Tk \n \
  \n GALAHAD copyright GALAHAD Productions 2002-2004 \
  \n http://galahad.rl.ac.uk/galahad-www"

helptext = helpcommandstext + "\n \n" + helpabouttext

#  text for help window

preftext = "There are no preferences to set at present"

##############################################################################
#                          TkGalahad class                                  #
##############################################################################

#class TkGalahad( Frame ):
class TkGalahad:
  ' TkGalahad class definition (GALAHAD productions 2002-2004)'
  instancesopen = 0
  def __init__( self ):
#   next line should work (but doesn't) 
# def __init__( self, master=None ):
#   Frame.__init__( self, master, class_="TkGalahad" )
    
#  initial values

    self.dir = os.getcwd( )
    self.dotsif = re.compile( r'.SIF$' )
    self.threads =[ ]
    self.specwindowopen = 0
    self.helpwindowopen = 0
    self.advicewindowopen = 0
    self.helpaboutwindowopen = 0
    self.helpcommandswindowopen = 0
    self.prefwindowopen = 0
    self.cdmenuwindowopen = 0
    self.selectwindowopen = 0

    self.lanbspec_used = 'no'
    self.filtranespec_used = 'no'
    self.qpaspec_used = 'no'
    self.qpbspec_used = 'no'
    self.lsqpspec_used = 'no'

#######################
#  Root window setup  #
#######################

    self.root = Tk( )

#  titles and positions

    self.root.title( 'GALAHAD tool' )
    self.root.iconname( 'galahad' )
    self.root.minsize( width = '1', height='1' )
    self.root.geometry( '+68+10' )

#  select default package

    self.var_package = StringVar( )
    self.var_package.set( 'lanb' )
#   self.var_package.set( 'filt' )
#   self.var_package.set( 'qpa' )
#   self.var_package.set( 'qpb' )
    self.var_package_label = StringVar( )

    if self.var_package.get( ) == 'lanb' :
      self.var_package_label.set( "LANCELOT B" )
    elif self.var_package.get( ) == 'filt' :
      self.var_package_label.set( "FILTRANE" )
    elif self.var_package.get( ) == 'qpb' :
      self.var_package_label.set( "QPB / LSQP" )
    elif self.var_package.get( ) == 'qpa' :
      self.var_package_label.set( "QPA" )
    else : self.var_package_label.set( "" )

#  select default fonts

    self.normalfont = tkFont.Font( family="Helvetica",size=9, weight="bold" )
    self.bigfont = tkFont.Font( family="Helvetica",size=24 )
    self.helpfont = tkFont.Font( family="Helvetica",size=14 )
 
#  set default option values

    self.root.option_add( "*Font", self.normalfont )
    self.root.option_add( "*Foreground", Gold )
    self.root.option_add( "*Background", DarkGreen )
    self.root.option_add( "*activeForeground", DarkGreen )
    self.root.option_add( "*activeBackground", Yellow )
    self.root.option_add( "*selectColor", Grey90 )
    self.root.option_add( "*selectForeground", Grey90 )
    self.root.option_add( "*selectBackground", Black )
    self.root.option_add( "*highlightColor", Gold )
    self.root.option_add( "*highlightBackground", Gold )
    self.root.option_add( "*disabledForeground", Grey90 )
    self.root.option_add( "*insertBackground", Grey90 )
    self.root.option_add( "*troughColor", DarkGreen )

#  listbox defaults

    self.root.option_add( "*Listbox*Foreground", Black )
    self.root.option_add( "*Listbox*Background", Grey90 )

#  button defaults

    self.root.option_add( "*Button*Foreground", White )
    self.root.option_add( "*Button*Background", Red )
    self.root.option_add( "*Button*activeForeground", Black )
    self.root.option_add( "*Button*activeBackground", Green )

#  entry defaults

    self.root.option_add( "*Entry*Foreground", Gold )
    self.root.option_add( "*Entry*Background", Brown )
    self.root.option_add( "*Entry*highlightBackground", DarkGreen )
    self.root.option_add( "*Entry*insertBackground", Brown )

#  menu defaults

    self.root.option_add( "*Menu*Foreground", Black )
    self.root.option_add( "*Menu*Background", Grey90 )
    self.root.option_add( "*Menu*activeForeground", Black )
    self.root.option_add( "*Menu*activeBackground", Grey80 )

#  scrollbar defaults

    self.root.option_add( "*Scrollbar*Background", Grey90 )
    self.root.option_add( "*Scrollbar*activeBackground", Grey80 )

#  checkbutton defaults

    self.root.option_add( "*Checkbutton*selectColor", Purple )
    self.root.option_add( "*Checkbutton*selectForeground", DarkGreen )
    self.root.option_add( "*Checkbutton*selectBackground", Gold )
    self.root.option_add( "*Checkbutton*activeForeground", DarkGreen )
    self.root.option_add( "*Checkbutton*activeBackground", Gold )

#  radiobutton defaults

    self.root.option_add( "*Radiobutton*selectColor", Purple )
    self.root.option_add( "*Radiobutton*selectForeground", DarkGreen )
    self.root.option_add( "*Radiobutton*selectBackground", Gold )
    self.root.option_add( "*Radiobutton*activeForeground", DarkGreen )
    self.root.option_add( "*Radiobutton*activeBackground", Gold )

#  cd window defaults

    self.root.option_add( "*cdwindow*Foreground", White )
    self.root.option_add( "*cdwindow*Background", Red )
    self.root.option_add( "*cdwindowEntry*Foreground", Black )
    self.root.option_add( "*cdwindowEntry*Background", White )
    self.root.option_add( "*cdwindowButton*Foreground", White )
    self.root.option_add( "*cdwindowButton*Background", Red )
    self.root.option_add( "*cdwindowButton*activeForeground", Black )
    self.root.option_add( "*cdwindowButton*activeBackground", Green )

#  check for installed galahad

    self.frame = Frame( self.root )
    self.logo = Text( self.frame, borderwidth=1, pady=3, padx=10,
                      width=9, height=2, font=self.bigfont,
#                     background=logoforeground,
#                     foreground=logobackground, 
                      )
    self.logo.insert( END, "GALAHAD\nstarting..." )
    self.logo.pack( side=LEFT )
    self.frame.pack( fill=BOTH, side=TOP, expand=1 )

    self.galahad = os.environ.get("GALAHAD")
    if self.galahad == None :
      print "GALAHAD environment variable not set"
      self.advicetext = "GALAHAD environment variable is not set"
      self.advice( )
      return

    if os.path.exists( self.galahad ) == 1 : 
      self.architecture = os.listdir( self.galahad+"/makefiles" )
      self.architecture.sort( )
      self.var_arch = StringVar( )
      self.var_arch.set( self.architecture[0] )
    else :
      print "GALAHAD environment does not appear to be present"
      self.advicetext = "GALAHAD does not appear to be installed"
      self.advice( )
      return

    self.galahadpython = os.environ.get("GALAHADPYTHON")
    if self.galahadpython == None :
      print "GALAHADPYTHON environment variable not set"
      self.advicetext = "GALAHADPYTHON environment variable is not set"
      self.advice( )
      return

    self.frame.destroy( )

#  read resorce file to override style defaults

    home = os.environ["HOME"]
    if os.path.exists( home+'/.GalahadStyle' ) == 1 : 
      self.root.option_clear()
      self.root.option_readfile( home+"/.GalahadStyle", priority=60 )
      print "Reading style resources from user resorce file ~/.GalahadStyle"
    elif os.path.exists( self.galahadpython+'/GalahadStyle' ) == 1 :
      self.root.option_clear()
      self.root.option_readfile( self.galahadpython+"/GalahadStyle",
                                 priority=100 )
      print "Reading style resources from system resorce file \n "\
        +self.galahadpython+"/GalahadStyle"

#  read resorce file to override package and architecture defaults

    if os.path.exists( home+'/.GalahadDefaults' ) == 1 : 
      self.readdefaults( home+"/.GalahadDefaults" )
      print "Reading package/achitecture resources from user resorce file" \
            +" ~/.GalahadDefaults"
    elif os.path.exists( self.galahadpython+'/GalahadDefaults' ) == 1 :
      self.readdefaults( self.galahadpython+"/GalahadDefaults" )
      print "Reading package/achitecture resources from system resorce file \n "\
        +self.galahadpython+"/GalahadDefaults"

    menubar = Menu( self.root )

#  create pulldown menus, and add them to the menu bar

    filemenu = Menu( menubar, tearoff=0 )
    filemenu.add_command(label="Exit", command = self.root.destroy )
#   filemenu.add_separator( )
#   filemenu.add_command( label="Quit", command = self.root.destroy )
    menubar.add_cascade( label="File", menu=filemenu )

    editmenu = Menu( menubar, tearoff=0 )
    editmenu.add_command( label="Preferences", command = self.prefset )
    menubar.add_cascade( label="Edit", menu=editmenu )

    helpmenu = Menu( menubar, tearoff=0 )
    helpmenu.add_command( label="About", command = self.presshelpabout )
    helpmenu.add_command( label="Commands", command = self.presshelpcommands )
    menubar.add_cascade( label="Help", menu=helpmenu )

#   display the menu

    self.root.config( menu=menubar )

#  construct the frames

    self.frame = Frame( self.root )
    self.frame1 = Frame( self.frame )
    self.frame2 = Frame( self.frame )

#  construct the button box (left-hand side)

    self.buttons = Listbox( self.frame1, relief=SUNKEN,
                            width=5, height=10, setgrid=Y )

#  construct a scrollbar (right-hand side)

    self.scroll = Scrollbar( self.frame2 )
    self.scroll.pack( side=RIGHT, fill=Y )

#  construct a list box (right-hand side)

    self.list = Listbox( self.frame2, relief=SUNKEN,
                         width=15, height=10, setgrid=Y,
                         yscrollcommand=self.scroll.set )
    self.list.pack( side=LEFT, fill=BOTH, expand=1 )

#  Tie the scrollbar to the list

    self.scroll.config( command=self.list.yview )

############# 
#  LHS box  #
############# 

#  assemble buttons for button box

#    Button( self.buttons,
#            width=10, height=1,
#            text="help", 
#            command=self.presshelp
#            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="cd",
            command=self.presscdmenu
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="refresh",
            command=self.runrefresh
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="spec", 
            command=self.pressspec
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="package",
            command=self.selectsolver
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="solve",
            command=self.runsdgal
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="resolve",
            command=self.rungal
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="solution",
            command=self.printsol
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="summary",
            command=self.printsum
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="stop",
            command=self.delalive
            ).pack( )
    Button( self.buttons, 
            width=10, height=1,
            text="exit",
            command=self.root.destroy
            ).pack( )

    self.buttons.pack( side=TOP, fill=Y )

#  Create package/logo box

    self.space( )
    self.createpacklogobox( )

############# 
#  RHS box  #
#############

#  Fill the listbox with a list of all the SIF files in the directory

    self.files = os.listdir( self.dir )
    self.files.sort( )
    for eachfile in self.files:
      if self.dotsif.search( eachfile ) != None:
        self.list.insert( END, eachfile )

###################
#  Assemble boxes #
###################

#  Pack it all together

    self.frame1.pack( fill=BOTH, side=LEFT )
    self.frame2.pack( fill=BOTH, side=LEFT )
    self.frame.pack( fill=BOTH, side=TOP, expand=1 )

####################
#  event bindings  #
####################

# Set up bindings for the mouse buttons

    self.root.bind_all( "<Control-c>", self.cntrlcdestroy )

# left mouse button

    self.list.bind( "<Double-Button-1>", self.editfile )

# middle mouse button (last two don't appear to work here)

    self.list.bind( "<Button-2>", self.runsdgalevent )
    self.list.bind( "<Double-Button-2>", self.runsdgaleventdetails )
    self.list.bind( "<Triple-Button-2>", self.runsdgaleventdebug )

# right mouse button

    self.list.bind( "<Button-3>", self.runsdgaleventdetails )

##############################################################################
#                         function definitions                               #
##############################################################################

#  function to bind Control-c to destroy
#  -------------------------------------

  def cntrlcdestroy( self, event ):
    self.root.destroy( )
    if TkGalahad.instancesopen <= 0:
        import sys
        sys.exit( 0 )

#  function to bind double-mouse-1 to edit            
#  ---------------------------------------

  def editfile( self, event ):
    try:
      editor = os.environ["VISUAL"]
    except KeyError:
      try:
        editor = os.environ["EDITOR"]
      except KeyError:
        editor = emacs
    os.popen( editor+' '+self.list.get( self.list.curselection( )[0], last=None))

#  function to refresh list of files
#  ---------------------------------

  def runrefresh( self ):
    self.files = os.listdir( self.dir )
    self.files.sort( )
    self.list.delete( 0, END )
    for eachfile in self.files:
      if self.dotsif.search( eachfile ) != None:
        self.list.insert( END, eachfile )

#  function to display help
#  ------------------------

  def presshelp( self ):
    if self.helpwindowopen == 1 :
        return
    self.helpwindowopen = 1
    self.helpwindow = Toplevel( self.root )
    self.helpwindow.geometry( '+300+100' )
    self.helpwindow.title( 'About GALAHAD tool' )
    self.helpwindow.menubar = Menu( self.helpwindow )
    self.helpwindow.menubar.add_command( label = "Quit",
                                         command=self.helpwindowdestroy )
    self.helpwindow.config( menu=self.helpwindow.menubar )

    Label( self.helpwindow, anchor=W, justify=LEFT,
           takefocus=1, text=helptext ).pack( side=TOP )
   
  def presshelpabout( self ):
    if self.helpaboutwindowopen == 1 :
        return
    self.helpaboutwindowopen = 1
    self.helpaboutwindow = Toplevel( self.root )
    self.helpaboutwindow.geometry( '+300+100' )
    self.helpaboutwindow.title( 'About GALAHAD tool' )
    self.helpaboutwindow.menubar = Menu( self.helpaboutwindow )
    self.helpaboutwindow.menubar.add_command( label = "Quit",
                                         command=self.helpaboutwindowdestroy )
    self.helpaboutwindow.config( menu=self.helpaboutwindow.menubar )

    Label( self.helpaboutwindow, anchor=W, justify=LEFT,
           takefocus=1, text=helpabouttext ).pack( side=TOP )
   
  def presshelpcommands( self ):
    if self.helpcommandswindowopen == 1 :
        return
    self.helpcommandswindowopen = 1
    self.helpcommandswindow = Toplevel( self.root )
    self.helpcommandswindow.geometry( '+300+100' )
    self.helpcommandswindow.title( 'GALAHAD tool commands' )
    self.helpcommandswindow.menubar = Menu( self.helpcommandswindow )
    self.helpcommandswindow.menubar.add_command( label = "Quit",
                                         command=self.helpcommandswindowdestroy )
    self.helpcommandswindow.config( menu=self.helpcommandswindow.menubar )

    Label( self.helpcommandswindow, anchor=W, justify=LEFT,
           takefocus=1, text=helpcommandstext ).pack( side=TOP )
   
#   function to destroy help windows
#   --------------------------------

  def helpwindowdestroy( self ):
    self.helpwindowopen = 0
    self.helpwindow.destroy( )

  def helpaboutwindowdestroy( self ):
    self.helpaboutwindowopen = 0
    self.helpaboutwindow.destroy( )

  def helpcommandswindowdestroy( self ):
    self.helpcommandswindowopen = 0
    self.helpcommandswindow.destroy( )

#  function to display preferences
#  -------------------------------

  def prefset( self ):
    if self.prefwindowopen == 1 :
        return
    self.prefwindowopen = 1
    self.prefwindow = Toplevel( self.root )
    self.prefwindow.geometry( '+300+100' )
    self.prefwindow.title( 'GALAHAD tool preferences' )
    self.prefwindow.menubar = Menu( self.prefwindow )
    self.prefwindow.menubar.add_command( label = "Quit",
                                         command=self.prefwindowdestroy )
    self.prefwindow.config( menu=self.prefwindow.menubar )

    Label( self.prefwindow, anchor=W, justify=LEFT,
           takefocus=1, text=preftext ).pack( side=TOP )
   
#   function to destroy pref window
#   -------------------------------

  def prefwindowdestroy( self ):
    self.prefwindowopen = 0
    self.prefwindow.destroy( )

#   function to start spec window
#   -----------------------------

  def pressspec( self ):

    if self.specwindowopen == 1 :
        return
    self.specwindowopen = 1

    if self.var_package.get( ) == 'lanb' :
      self.lanbspec( )
    elif self.var_package.get( ) == 'filt' :
      self.filtranespec( )
    elif self.var_package.get( ) == 'qpb' :
      self.qpbspec( )
    elif self.var_package.get( ) == 'qpa' :
      self.qpaspec( )
    else :
      print 'unavailable at the present time, defaults will be used'
      self.specwindowopen = 0
      self.advicetext = "Sorry, no spec window available " \
                       + "for package "+self.var_package_label.get( ) \
                       + " at present. Default values will be used",
      self.advice( )
      return

#  Another function to change directory
#  ------------------------------------

  def presscdmenu( self ):
    if self.cdmenuwindowopen == 1 :
        return
    self.cdmenuwindowopen = 1
    self.cdmenuwindow = Toplevel( self.root )
    self.cdmenuwindow.geometry('+300+100')
    self.cdmenuwindow.title('GALAHAD cd tool')
    self.menubar = Menu( self.cdmenuwindow )
    self.menubar.add_command( label = "Quit", command=self.cdmenuwindowdestroy )
    self.cdmenuwindow.config( menu=self.menubar )

    self.cdmenuwindow.frame = Frame( self.cdmenuwindow,
                                     name="cdmenuwindow" )
    self.cdmenuwindow.frame1 = Frame( self.cdmenuwindow.frame,
                                      name="cdmenuwindow1" )
    self.cdmenuwindow.frame2 = Frame( self.cdmenuwindow.frame,
                                      name="cdmenuwindow2" )
    self.cdmenuwindow.frame3 = Frame( self.cdmenuwindow.frame,
                                      name="cdmenuwindow3" )
    self.cdmenuwindow.frame4 = Frame( self.cdmenuwindow.frame,
                                      name="cdmenuwindow4" )
    self.cdmenuwindow.frame5 = Frame( self.cdmenuwindow.frame,
                                      name="cdmenuwindow5" )

    Label( self.cdmenuwindow.frame1, height=1, text="" ).pack( side=TOP )
    self.cdmenuwindow.label = Label( self.cdmenuwindow.frame1,
                                 text="Current directory is \n " + \
                                 self.dir +"\nChange directory to ...",
                                 anchor=W,
                                 relief=FLAT,
                                 borderwidth=0
                                 )
    self.cdmenuwindow.label.pack( side=TOP )
    Label( self.cdmenuwindow.frame1, height=1, text="" ).pack( side=TOP )
    self.cdmenuwindow.frame1.pack( fill=BOTH, side=TOP, expand=1 )

    self.cdmenuwindow.scroll = Scrollbar( self.cdmenuwindow.frame2 )
    self.cdmenuwindow.scroll.pack( side=RIGHT, fill=Y )

    self.cdmenuwindow.list = Listbox( self.cdmenuwindow.frame2, relief=SUNKEN,
                         width=15, height=10, setgrid=Y,
                         yscrollcommand=self.cdmenuwindow.scroll.set )
    self.cdmenuwindow.list.pack( side=TOP, fill=BOTH, expand=1 )
    self.cdmenuwindow.scroll.config( command=self.cdmenuwindow.list.yview )
    self.cdmenuwindow.list.bind( "<Double-Button-1>", self.cdmenuupdateevent )
    self.cdmenu_listdirs( )
    self.cdmenuwindow.frame2.pack( fill=BOTH, side=TOP, expand=1 )

    Label( self.cdmenuwindow.frame3, height=1, text="" ).pack( side=TOP )
    self.cdmenuwindow.label = Label( self.cdmenuwindow.frame3,
                                 text=" ... or choose it yourself ...",
                                 anchor=W,
                                 relief=FLAT,
                                 borderwidth=0
                                 )
    self.cdmenuwindow.label.pack( side=TOP )
    Label( self.cdmenuwindow.frame3, height=1, text="" ).pack( side=TOP )
    self.cdmenuwindow.frame3.pack( fill=BOTH, side=TOP, expand=1 )

    self.stringnewdir = StringVar( )
    self.cdmenuwindow.entry = Entry( self.cdmenuwindow.frame4,
                                 name="cdwindowEntry",
                                 textvariable=self.stringnewdir,
                                 borderwidth=2, width=49,
                                 relief=SUNKEN,
                                 )
    self.cdmenuwindow.entry.delete( 0, END )
    self.cdmenuwindow.entry.insert( END, self.dir )
    self.cdmenuwindow.entry.bind( "<KeyPress-Return>", self.cdmenuupdateevent )
    self.cdmenuwindow.entry.pack( side=TOP )
    self.cdmenuwindow.frame4.pack( fill=BOTH, side=TOP, expand=1 )

    Label( self.cdmenuwindow.frame5, height=1, text="" ).pack( side=TOP )
    self.cdmenuwindow.buttons = Frame( self.cdmenuwindow.frame5 )
    self.space_cdmenu( )
    Button( self.cdmenuwindow.buttons,
            width=6, pady=2,
            text="ok",
            command=self.cdmenuupdate
            ).pack( side=LEFT, fill=BOTH )
    self.space_cdmenu( )
    Button( self.cdmenuwindow.buttons,
            width=6, pady=2,
            text="quit",
            command=self.cdmenuwindowdestroy
            ).pack( side=LEFT, fill=BOTH )
    self.space_cdmenu( )

    self.cdmenuwindow.buttons.pack( side=TOP, fill=BOTH )
    Label( self.cdmenuwindow.frame5, height=1, text="" ).pack( side=TOP )
    self.cdmenuwindow.frame5.pack( fill=BOTH, side=TOP, expand=1 )
    self.cdmenuwindow.frame.pack( fill=BOTH, side=TOP, expand=1 )

    self.cdmenuwindow.mainloop( )

#  function to build a list of sub-directories of the current directory
#  --------------------------------------------------------------------

  def cdmenu_listdirs( self ):
    self.files = os.listdir( self.dir )
    self.files.sort( )
    self.cdmenuwindow.list.insert( END, '../' )
    self.cdmenuwindow.list.insert( END, './' )
    for eachfile in self.files:
      if os.path.isdir( eachfile ) != 0:
        self.cdmenuwindow.list.insert( END, eachfile )

#  Another function to change directory and refresh list of files
#  --------------------------------------------------------------

  def cdmenuupdateevent( self, event ):
    self.cdmenuupdatemain( )

  def cdmenuupdate( self ):
    self.cdmenuupdatemain( )
    
  def cdmenuupdatemain( self ):

    n = self.cdmenuwindow.list.curselection( )
    if len( n ) == 0 :
      self.newdir = self.stringnewdir.get( )
      if os.path.exists( self.newdir ) :
        if os.path.isdir( self.newdir ) :
          self.dir = self.newdir
          os.chdir( self.dir )
        else:
          self.advicetext = "File '"+self.newdir+"' is not a directory \n " +\
                            "Please try again"
          self.advice( )
          return
      else:
        self.advicetext = "Directory '"+self.newdir+"' not found. \n " +\
                          "Please try again"
        self.advice( )
        return
    else:
      self.dir = self.dir+"/"+self.cdmenuwindow.list.get( n[0], last=None )
      os.chdir( self.dir )
      self.dir = os.getcwd( )
      print "Working directory is now "+self.dir

    self.runrefresh( )
    self.cdmenuwindowopen = 0
    self.cdmenuwindow.destroy( )
    self.presscdmenu( )
    
  def cdmenuwindowdestroy( self ):
    self.cdmenuwindowopen = 0
    self.cdmenuwindow.destroy( )

#  function to select solver to be used
#  ------------------------------------

  def selectsolver( self ):
    if self.selectwindowopen == 1 :
        return
    self.selectwindowopen = 1
    self.selectwindow = Toplevel( self.root )
    self.selectwindow.geometry('+300+100')
    self.selectwindow.title('GALAHAD package selection tool')
#    self.menubar = Menu( self.selectwindow )
#    self.menubar.add_command( label = "Quit", command=self.selectwindowdestroy )
#    self.selectwindow.config( menu=self.menubar )

    self.selectwindow.frame = Frame( self.selectwindow, name="selectwindow2" )
    self.selectwindow.frameb = Frame( self.selectwindow, name="selectwindow2b" )

    self.selectwindow.frame1 = Frame( self.selectwindow.frame,
                                      name="selectwindowf1" )
    self.selectwindow.frame2 = Frame( self.selectwindow.frame,
                                      name="selectwindowf2" )

    Label( self.selectwindow.frame1, width=2, text="" ).pack( side=LEFT )
    Label( self.selectwindow.frame1, 
           anchor=W, relief=FLAT, borderwidth=0, width=14,
           text="\nPackage to use:"
          ).pack( side=TOP )

    self.selectwindow.packagesframe1 = Frame( self.selectwindow.frame1,
                                             name="selectwindow" )

    for packages in [ 'lanb', 'filt', 'qpb', 'qpa' ]:
      if packages == 'lanb' : label = "LANCELOT B"
      elif packages == 'filt' : label = "FILTRANE" 
      elif packages == 'qpb' : label = "QPB / LSQP" 
      elif packages == 'qpa' : label = "QPA" 
      else :label = "" 
      Radiobutton( self.selectwindow.packagesframe1,
                   highlightthickness=0, relief=FLAT,
                   width=14, anchor=W,
                   variable=self.var_package,
                   value=packages,
                   text=label
                   ).pack( side=TOP, fill=NONE )

    self.selectwindow.packagesframe1.pack( side=TOP )
    self.selectwindow.frame1.pack( side=LEFT )

    Label( self.selectwindow.frame2, 
           anchor=W, relief=FLAT, borderwidth=0, width=17,
           text="\nArchitecture to use:"
          ).pack( side=TOP )

    self.selectwindow.packagesframe2 = Frame( self.selectwindow.frame2,
                                             name="selectwindow2" )

    if os.path.exists( self.galahad ) == 1 : 
      self.architecture = os.listdir( self.galahad+"/makefiles" )
      self.architecture.sort( )
      for eachfile in self.architecture :
        Radiobutton( self.selectwindow.packagesframe2,
                     highlightthickness=0, relief=FLAT,
                     width=17, anchor=W,
                     variable=self.var_arch,
                     value=eachfile,
                     text=eachfile
                     ).pack( side=TOP, fill=NONE )

    Label( self.selectwindow.frame2, width=2, text="" ).pack( side=LEFT )
    self.selectwindow.packagesframe2.pack( side=TOP )
    self.selectwindow.frame2.pack( side=TOP )
    self.selectwindow.frame.pack( fill=BOTH, side=TOP, expand=1 )

    Label( self.selectwindow.frameb, height=1, text="" ).pack( side=TOP )
    self.selectwindow.buttons = Frame( self.selectwindow.frameb )
    self.space_select( )
    Button( self.selectwindow.buttons,
            width=6, pady=2,
            text="ok",
            command=self.selectupdate
            ).pack( side=LEFT, fill=BOTH )
    self.space_select( )
    Button( self.selectwindow.buttons,
            width=6, pady=2,
            text="cancel",
            command=self.selectcancel
            ).pack( side=LEFT, fill=BOTH )
    self.space_select( )
    self.selectwindow.buttons.pack( side=TOP, fill=BOTH )

    Label( self.selectwindow.frameb, height=1, text="" ).pack( side=TOP )
    self.selectwindow.frameb.pack( fill=BOTH, side=TOP, expand=1 )

    self.selectwindow.mainloop( )

#  function to display advice message
#  ----------------------------------

  def advice( self ):
    if self.advicewindowopen == 1 :
        return
    self.advicewindowopen = 1
    self.advicewindow = Toplevel( self.root )
    self.advicewindow.geometry('+100+100')

    self.advicewindow.title('GALAHAD advice')
#    self.menubar = Menu( self.advicewindow )
#    self.menubar.add_command( label = "Quit", command=self.advicewindowdestroy )
#    self.advicewindow.config( menu=self.menubar )

    self.advicewindow.frame = Frame( self.advicewindow,
                                     background=warningbackground,
                                     name="advicewindow2" )

    Label( self.advicewindow.frame, height=1, background=warningbackground,
           text="" ).pack( side=TOP )
    Label( self.advicewindow.frame, 
           relief=FLAT, borderwidth=1, 
           foreground=warningforeground,
           background=warningbackground,
           font=self.helpfont,
           text="  "+self.advicetext+"  "
          ).pack( side=TOP )

    Label( self.advicewindow.frame, height=1, background=warningbackground,
           text="" ).pack( side=TOP )
    self.advicewindow.buttons = Frame( self.advicewindow.frame,
                                       background=warningbackground )
    Button( self.advicewindow.buttons,
            width=6, pady=2,
            text="ok",
            command=self.advicecancel
            ).pack( side=TOP, fill=BOTH )
    self.advicewindow.buttons.pack( side=TOP, fill=Y )

    Label( self.advicewindow.frame, height=1, background=warningbackground,
           text="" ).pack( side=TOP )
    self.advicewindow.frame.pack( fill=BOTH, side=TOP, expand=1 )

    self.advicewindow.mainloop( )

#  function to select and record solver to be used
#  -----------------------------------------------

  def selectupdateevent( self, event ):
    self.selectupdatemain( )

  def selectupdate( self ):
    self.selectupdatemain( )
    
  def selectupdatemain( self ):
    self.selectwindowdestroy( )
    
  def selectwindowdestroy( self ):
    self.selectwindowopen = 0
    if self.var_package.get( ) == 'lanb' :
      self.var_package_label.set( "LANCELOT B" )
    elif self.var_package.get( ) == 'filt' :
      self.var_package_label.set( "FILTRANE" )
    elif self.var_package.get( ) == 'qpb' :
      self.var_package_label.set( "QPB / LSQP" )
    elif self.var_package.get( ) == 'qpa' :
      self.var_package_label.set( "QPA" )
    else : self.var_package_label.set( "" )

#  Destroy existing packag/logo box

    self.destroypacklogobox( )

#  Create new package/logo box

    self.createpacklogobox( )
    self.selectwindow.destroy( )

  def selectcancel( self ):
    self.selectwindowopen = 0
    self.selectwindow.destroy( )

  def advicecancel( self ):
    self.advicewindowopen = 0
    self.advicewindow.destroy( )

#  function to run SIF decoder followed by selected GALAHAD package
#  ----------------------------------------------------------------

  def runsdgalevent( self, event ):
    self.sdgalout = ''
    self.runsdgalmain( )
    
  def runsdgaleventdetails( self, event ):
    self.sdgalout = '-o 1'
    self.runsdgalmain( )
    
  def runsdgaleventdebug( self, event ):
    self.sdgalout = '-o -1'
    self.runsdgalmain( )
    
  def runsdgal( self ):
    self.sdgalout = ''
    self.runsdgalmain( )
    
  def runsdgalmain( self ):
    if len( self.threads ) != 0 :
      while self.threads[0].isAlive( ) :
        self.advicetext = "Please wait for run to terminate and try again"
        self.advice( )
        return
    n = self.list.curselection( )
    if len( n ) == 0 :
      self.advicetext = "Please select a test problem"
      self.advice( )
      return
    self.file = self.dotsif.sub( '', self.list.get( n[0], last=None ) )
    self.threads =[ ]
    self.t = threading.Thread( target=self.sdgalthread )
    self.threads.append( self.t )
#   print self.threads
    self.threads[0].start( )

  def sdgalthread( self  ):
    print 'sdgal '+self.var_arch.get( )+' '+self.var_package.get( ) \
                  +' '+self.sdgalout+' '+self.file
    os.system( 'sdgal '+self.var_arch.get( )+' '+self.var_package.get( ) \
                       +' '+self.sdgalout+' '+self.file )

#  function to run selected GALAHAD on previously-decoded SIF problem 
#  ------------------------------------------------------------------

  def rungal( self ):
    if len( self.threads ) != 0 :
      while self.threads[0].isAlive( ) :
        self.advicetext = "Please wait for run to terminate and try again"
        self.advice( )
        return
    if os.path.exists( 'OUTSDIF.d' ) == 0 or \
       os.path.exists( 'ELFUN.f' ) == 0 or  \
       os.path.exists( 'GROUP.f' ) == 0 :
      self.advicetext = "There is no test problem to be"+ \
                        "re-solved in the current directory"
      self.advice( )
      return
    self.threads = [ ]
    self.t = threading.Thread( target=self.galthread )
    self.threads.append( self.t )
    self.threads[0].start( )

  def galthread( self  ):
    print 'gal '+self.var_arch.get( )+' '+self.var_package.get( )
    os.system( 'gal '+self.var_arch.get( )+' '+self.var_package.get( ) )
    
#  function to run selected GALAHAD package on previously-decoded 
#  SIF problem using current spec values
#  -------------------------------------

  def rungaloncurrent( self ):
    if len( self.threads ) != 0 :
      while self.threads[0].isAlive( ) :
        self.advicetext = "Please wait for run to terminate and try again"
        self.advice( )
        return
    package = self.var_package.get( )
    self.spec = 'RUN'+package.upper( )+'.SPC'
    self.specbak = self.spec+'.bak'
    self.specexists = os.path.exists( self.spec )
    if self.specexists == 1 : os.rename( self.spec, self.specbak )
    if self.var_package.get( ) == 'lanb' :
      self.writelanbspec( )
    elif self.var_package.get( ) == 'filt' :
      self.writefiltranespec( )
    elif self.var_package.get( ) == 'qpb' :
      self.writeqpbspec( )
    elif self.var_package.get( ) == 'qpa' :
      self.writeqpaspec( )
    if os.path.exists( 'OUTSDIF.d' ) == 0 or \
       os.path.exists( 'ELFUN.f' ) == 0 or  \
       os.path.exists( 'GROUP.f' ) == 0 :
      self.advicetext = "There is no test problem to be"+ \
                        "re-solved in the current directory"
      self.advice( )
      return
    self.threads = [ ]
    self.t = threading.Thread( target=self.galthreadoncurrent )
    self.threads.append( self.t )
    self.threads[0].start( )

  def galthreadoncurrent( self  ):
    print 'gal '+self.var_arch.get( )+' '+self.var_package.get( )
    os.system( 'gal '+self.var_arch.get( )+' '+self.var_package.get( ) )
    if self.specexists == 1 : os.rename( self.specbak, self.spec )
    
#  function to print solution
#  --------------------------

  def printsol( self ) :
    if len( self.threads ) != 0 :
      while self.threads[0].isAlive( ) :
        self.advicetext = "Please wait for run to terminate and try again"
        self.advice( )
        return
    if os.path.exists( 'SOLUTION.d' ) == 0 :
      self.advicetext = "No solution file available"
      self.advice( )
      return
    self.sol = file( 'SOLUTION.d' )
    print self.sol.read( )

#  function to print summary
#  -------------------------

  def printsum( self ):
    if len( self.threads ) != 0 :
      while self.threads[0].isAlive( ) :
        self.advicetext = "Please wait for run to terminate and try again"
        self.advice( )
        return
    if os.path.exists( 'SUMMARY.d' ) == 0 :
      self.advicetext = "No summary file available"
      self.advice( )
      return
    self.sum = file( 'SUMMARY.d' )
    print self.sum.read( )

#  function to remove ALIVE file
#  -----------------------------

  def delalive( self ):
    if os.path.exists( 'ALIVE.d' ) == 0 :
      print ' no run currently in progress'
      return
    os.remove( 'ALIVE.d' )

#  function to create package name/logo box
#  ----------------------------------------

  def createpacklogobox( self ):

#  Package name

    self.packname = Text( self.frame1,
#                         borderwidth=1, pady=4, padx=10,
                          width=12, height=4,
#                         font=self.bigfont,
                          background=logobackground, 
                          foreground=logoforeground,
                          )
    self.packname.insert( END,
                          "Package:\n"+self.var_package_label.get( )
                           +"\nArchitecture:\n"+self.var_arch.get( ) )
    self.packname.pack( side=TOP )

#  GALAHAD logo

    self.space1 = Text( self.frame1, borderwidth=0, pady=0,
          width=0, height=0, font=self.bigfont,
          insertwidth=0, highlightthickness=0,
          )
    self.space1.pack( side=TOP )

    self.img = PhotoImage( file=self.galahadpython+'/GalahadLogo.gif' )
    self.lbl = Label( self.frame1, image=self.img, relief=SUNKEN )
    self.lbl.img = self.img
    self.lbl.pack( side=TOP, fill=NONE )

#  text in logo box

    self.space2 = Text( self.frame1, borderwidth=0, pady=0,
          width=0, height=0, font=self.bigfont,
          insertwidth=0, highlightthickness=0,
          )
    self.space2.pack( side=TOP )

    self.logo = Text( self.frame1, borderwidth=1, pady=3, padx=10,
                      width=1, height=7, font=self.bigfont,
                      background=logobackground, 
                      foreground=logoforeground,
                      )
    self.logo.insert( END, "G\nA\nL\nA\nH\nA\nD\n" )
    self.logo.pack( side=TOP )

    self.space3 = Text( self.frame1, borderwidth=0, pady=0,
          width=0, height=0, font=self.bigfont,
          insertwidth=0, highlightthickness=0,
          )
    self.space3.pack( side=TOP )

#  function to destroy existing package name/logo box
#  --------------------------------------------------

  def destroypacklogobox( self ):

    self.packname.destroy( )
    self.space1.destroy( )
    self.lbl.destroy( )
    self.space2.destroy( )
    self.logo.destroy( )
    self.space3.destroy( )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#===============================================================================#
#                                                                               #
#                                 SPEC WINDOWS                                  #
#                                                                               #
#===============================================================================#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                         LANCELOT B                        #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#   function for LANCELOT B spec window
#   -----------------------------------

  def lanbspec( self ):

#  set variables if the first time through

    if self.lanbspec_used == 'no' :
      
#  integer (check-button) variables used with encodings

      self.lanb_check_maximizer = IntVar( )
      self.lanb_string_maximizer = 'maximizer-sought'
      self.lanb_default_maximizer = 0

      self.lanb_string_printformax = 'print-for-maximimization'

      self.lanb_check_printfullsol = IntVar( )
      self.lanb_string_printfullsol = 'print-full-solution'
      self.lanb_default_printfullsol = 0

      self.lanb_check_checkder = IntVar( )
      self.lanb_string_checkder = 'check-derivatives'
      self.lanb_default_checkder = 1

      self.lanb_check_checkall = IntVar( )
      self.lanb_string_checkall = 'check-all-derivatives'
      self.lanb_default_checkall = 0

      self.lanb_check_checkel = IntVar( )
      self.lanb_string_checkel = 'check-element-derivatives'
      self.lanb_default_checkel = 1

      self.lanb_check_checkgr = IntVar( )
      self.lanb_string_checkgr = 'check-group-derivatives'
      self.lanb_default_checkgr = 1

      self.lanb_check_ignoreder = IntVar( )
      self.lanb_string_ignoreder = 'ignore-derivative-bugs'
      self.lanb_default_ignoreder = 0

      self.lanb_check_ignoreel = IntVar( )
      self.lanb_string_ignoreel = 'ignore-element-derivative-bugs'
      self.lanb_default_ignoreel = 0

      self.lanb_check_ignoregr = IntVar( )
      self.lanb_string_ignoregr = 'ignore-group-derivative-bugs'
      self.lanb_default_ignoregr = 0

      self.lanb_check_qp = IntVar( )
      self.lanb_string_qp = 'quadratic-problem'
      self.lanb_default_qp = 0

      self.lanb_check_twonorm = IntVar( )
      self.lanb_string_twonorm = 'two-norm-trust-region-used'
      self.lanb_default_twonorm = 0

      self.lanb_check_strtr = IntVar( )
      self.lanb_string_strtr = 'structured-trust-region-used'
      self.lanb_default_strtr = 0

      self.lanb_check_gcp= IntVar( )
      self.lanb_string_gcp = 'exact-gcp-used'
      self.lanb_default_gcp= 1

      self.lanb_check_accuratebqp = IntVar( )
      self.lanb_string_accuratebqp = 'subproblem-solved-accuractely'
      self.lanb_default_accuratebqp = 0

      self.lanb_check_restart = IntVar( )
      self.lanb_string_restart = 'restart-from-previous-point'
      self.lanb_default_restart = 0

      self.lanb_check_scaling = IntVar( )
      self.lanb_string_scaling = 'use-scaling-factors'
      self.lanb_default_scaling = 0

      self.lanb_check_cscaling = IntVar( )
      self.lanb_string_cscaling = 'use-constraint-scaling-factors'
      self.lanb_default_cscaling = 0

      self.lanb_check_vscaling = IntVar( )
      self.lanb_string_vscaling = 'use-variable-scaling-factors'
      self.lanb_default_vscaling = 0

      self.lanb_check_gscaling = IntVar( )
      self.lanb_string_gscaling = 'get-scaling-factors'
      self.lanb_default_gscaling = 0

      self.lanb_check_wproblem = IntVar( )
      self.lanb_string_wproblem = 'write-problem-data'
      self.lanb_default_wproblem = 0

      self.lanb_check_writeall = IntVar( )
      self.lanb_default_writeall = 0

#  string variables used with encodings and defaults

      self.lanb_var_plevel = StringVar( )
      self.lanb_string_plevel = 'print-level'
      self.lanb_default_plevel = '0'

      self.lanb_var_sprint = StringVar( )
      self.lanb_string_sprint = 'start-print'
      self.lanb_default_sprint = '-1'

      self.lanb_var_fprint = StringVar( )
      self.lanb_string_fprint = 'stop-print'
      self.lanb_default_fprint = '-1'

      self.lanb_var_iprint = StringVar( )
      self.lanb_string_iprint = 'iterations-between-printing'
      self.lanb_default_iprint = '1'

      self.lanb_var_slevel = StringVar( )
      self.lanb_string_slevel = 'scaling-print-level'
      self.lanb_default_slevel = '0'

      self.lanb_var_maxit = StringVar( )
      self.lanb_string_maxit = 'maximum-number-of-iterations'
      self.lanb_default_maxit = '1000'

      self.lanb_var_savedata = StringVar( )
      self.lanb_string_savedata = 'save-data-for-restart-every'
      self.lanb_default_savedata = '0'

      self.lanb_var_cstop = StringVar( )
      self.lanb_string_cstop = 'primal-accuracy-required'
      self.lanb_default_cstop = '0.00001'

      self.lanb_var_gstop = StringVar( )
      self.lanb_string_gstop = 'dual-accuracy-required'
      self.lanb_default_gstop = '0.00001'

      self.lanb_var_mu = StringVar( )
      self.lanb_string_mu = 'initial-penalty-parameter'
      self.lanb_default_mu = '0.1'

      self.lanb_var_decreasemu = StringVar( )
      self.lanb_string_decreasemu = \
        'no-dual-updates-until-penalty-parameter-below'
      self.lanb_default_decreasemu = '0.1'

      self.lanb_var_firstc = StringVar( )
      self.lanb_string_firstc = 'initial-primal-accuracy-required'
      self.lanb_default_firstc = '0.1'

      self.lanb_var_firstg = StringVar( )
      self.lanb_string_firstg = 'initial-dual-accuracy-required'
      self.lanb_default_firstg = '0.1'

      self.lanb_var_pivtol = StringVar( )
      self.lanb_string_pivtol = 'pivot-tolerance-used'
      self.lanb_default_pivtol = '0.1'

      self.lanb_var_firstr = StringVar( )
      self.lanb_string_firstr = 'initial-trust-region-radius'
      self.lanb_default_firstr = '1.0'

      self.lanb_var_gradient = StringVar( )
      self.lanb_string_gradient = 'first-derivative-approximations' 
      self.lanb_default_gradient = 'exact'

      self.lanb_var_hessian = StringVar( )
      self.lanb_string_hessian = 'second-derivative-approximations' 
      self.lanb_default_hessian = 'exact'

      self.lanb_var_solver = StringVar( )
      self.lanb_string_solver = 'linear-solver-used' 
      self.lanb_default_solver = 'band_cg'

      self.lanb_var_bandwidth = StringVar( )
      self.lanb_string_bandwidth = 'semi-bandwidth-for-band-preconditioner' 
      self.lanb_default_bandwidth = '5'
      self.current_bandwidth = self.lanb_default_bandwidth

      self.lanb_var_linmore = StringVar( )
      self.lanb_string_linmore = 'number-of-lin-more-vectors-used' 
      self.lanb_default_linmore = '5'
      self.current_linmore = self.lanb_default_linmore

      self.lanb_var_maxsc = StringVar( )
      self.lanb_string_maxsc = 'maximum-dimension-of-schur-complement'
      self.lanb_default_maxsc = '100'

      self.lanb_var_acccg = StringVar( )
      self.lanb_string_acccg = 'inner-iteration-relative-accuracy-required'
      self.lanb_default_acccg = '0.01'

      self.lanb_var_maxrad = StringVar( )
      self.lanb_string_maxrad = 'maximum-radius'
      self.lanb_default_maxrad = '1.0D+20'

      self.lanb_var_etas = StringVar( )
      self.lanb_string_etas = 'eta-successful'
      self.lanb_default_etas = "0.01"

      self.lanb_var_etavs = StringVar( )
      self.lanb_string_etavs = 'eta-very-successful'
      self.lanb_default_etavs = "0.9"

      self.lanb_var_etaes = StringVar( )
      self.lanb_string_etaes = 'eta-extremely-successful'
      self.lanb_default_etaes = "0.95"

      self.lanb_var_gammasmall = StringVar( )
      self.lanb_string_gammasmall = 'gamma-smallest'
      self.lanb_default_gammasmall = "0.0625"

      self.lanb_var_gammadec = StringVar( )
      self.lanb_string_gammadec = 'gamma-decrease'
      self.lanb_default_gammadec = "0.25"

      self.lanb_var_gammainc = StringVar( )
      self.lanb_string_gammainc = 'gamma-increase'
      self.lanb_default_gammainc = "2.0"

      self.lanb_var_mumodel = StringVar( )
      self.lanb_string_mumodel = 'mu-meaningful-model'
      self.lanb_default_mumodel = "0.01"

      self.lanb_var_mugroup = StringVar( )
      self.lanb_string_mugroup = 'mu-meaningful-group'
      self.lanb_default_mugroup = "0.1"

      self.lanb_var_moretoraldosearch = StringVar( )
      self.lanb_string_moretoraldosearch = 'more-toraldo-search-length'
      self.lanb_default_moretoraldosearch = '0'

      self.lanb_var_nonmonotone = StringVar( )
      self.lanb_string_nonmonotone = 'history-length-for-non-monotone-descent'
      self.lanb_default_nonmonotone  = '1'

#  read default values

      self.restorelanbdefaults( )
      self.lanb_check_writeall.set( self.lanb_default_writeall )
      self.lanbspec_used = 'yes'

#  setup the window frame
    
    self.specwindow = Toplevel( self.root )
    self.specwindow.geometry( '+100+100' )
    self.specwindow.title( 'LANCELOT B spec tool' )
    self.specwindow.menubar = Menu( self.specwindow, tearoff=0  )
    self.specwindow.menubar.add_command( label = "Quit",
                                         command=self.specwindowdestroy )

    self.specwindow.helpmenu = Menu( self.specwindow, tearoff=0 )
    self.specwindow.helpmenu.add_command( label="About",
                                          command = self.lanbpresshelp )
    self.specwindow.menubar.add_cascade( label="Help",
                                         menu=self.specwindow.helpmenu )
    self.specwindow.config( menu=self.specwindow.menubar )

#  asseemble (check-button) variables

    self.check = [ self.lanb_check_maximizer,
                   self.lanb_check_printfullsol,
                   self.lanb_check_checkder,
                   self.lanb_check_checkall,
                   self.lanb_check_checkel,
                   self.lanb_check_checkgr,
                   self.lanb_check_ignoreder,
                   self.lanb_check_ignoreel,
                   self.lanb_check_ignoregr, 
                   self.lanb_check_qp
                   ]

    self.checkstring = [ "Maximizer",
                         "Print full solution",
                         "Check derivatives",
                         "Check all derivatives",
                         "Check element derivatives",
                         "Check group derivatives",
                         "Ignore derivative warnings",
                         "Ignore element derivative warnings",
                         "Ignore group derivative warnings",
                         "Linear/quadratic program"
                         ]

    self.specwindow.varlstart = 0
    self.specwindow.varlstop = len( self.check )

    self.check = self.check+[ self.lanb_check_twonorm,
                              self.lanb_check_strtr,
                              self.lanb_check_gcp,
                              self.lanb_check_accuratebqp,
                              self.lanb_check_restart,
                              self.lanb_check_scaling,
                              self.lanb_check_cscaling, 
                              self.lanb_check_vscaling,
                              self.lanb_check_gscaling,
                              self.lanb_check_wproblem
                              ]

    self.checkstring.extend( [ "Two-norm trust region",
                             "Structured trust region",
                             "Exact Cauchy point required",
                             "Solve subproblem accurately",
                             "Restart from previous point",
                             "Use scaling factors",
                             "Use constraint scaling factors",
                             "Use variable scaling factors",
                             "Get scaling factors",
                             "Write problem data to file"
                             ] )
    
    self.specwindow.varrstart = self.specwindow.varlstop
    self.specwindow.varrstop = len( self.check )

#  assemble string variables

    self.var = [ self.lanb_var_plevel,
                 self.lanb_var_sprint,
                 self.lanb_var_fprint,
                 self.lanb_var_iprint
                 ]

    self.varstring = [ "Print level",
                       "Start print at iteration",
                       "Stop printing at iteration",
                       "Iterations between printing"
                       ]

    self.specwindow.entrytlstart = 0
    self.specwindow.entrytlstop = len( self.var )

    self.var = self.var+[ self.lanb_var_slevel,
                          self.lanb_var_maxit,
                          self.lanb_var_savedata,
                          self.lanb_var_moretoraldosearch,

                         ]

    self.varstring.extend( [ "Scaling print level",
                             "Maximum number of iterations",
                             "Save data frequency",
                             "More-Toraldo search length"
                             ] )

    self.specwindow.entrytrstart = self.specwindow.entrytlstop
    self.specwindow.entrytrstop = len( self.var )

    self.var = self.var+[ self.lanb_var_maxsc,
                          self.lanb_var_nonmonotone,
                          self.lanb_var_cstop,
                          self.lanb_var_gstop,
                          self.lanb_var_mu,
                          self.lanb_var_decreasemu,
                          self.lanb_var_firstg,
                          self.lanb_var_firstc,
                          self.lanb_var_pivtol,
                          self.lanb_var_firstr,
                          self.lanb_var_maxrad,
                          self.lanb_var_acccg
                          ]

    self.varstring.extend( [ "Max Schur complement dimension",
                             "Non-monotone descent history",
                             "Constraint (primal) accuracy",
                             "Gradient (dual) accuracy",
                             "Initial penalty parameter",
                             "Decrease penalty paramter until <",
                             "Initial primal accuracy required",
                             "Initial dual accuracy required",
                             "Pivot tolerance",
                             "1st trust-region radius",
                             "Maximum radius",
                             "CG relative accuracy"
                             ] )

    self.specwindow.entrybrstart = self.specwindow.entrytrstop
    self.specwindow.entrybrstop = len( self.var )


    self.var = self.var+[ self.lanb_var_etas,
                          self.lanb_var_etavs,
                          self.lanb_var_etaes,
                          self.lanb_var_mumodel
                         ]

    self.varstring.extend( [ "eta successful",
                             "eta very-successful",
                             "eta extremely-successful",
                             "mu for meaningful model"
                             ] )

    self.specwindow.entrylblstart = self.specwindow.entrybrstop
    self.specwindow.entrylblstop = len( self.var )

    self.var = self.var+[ self.lanb_var_gammasmall,
                          self.lanb_var_gammadec,
                          self.lanb_var_gammainc,
                          self.lanb_var_mugroup
                         ]

    self.varstring.extend( [ "gamma smallest",
                             "gamma decrease",
                             "gamma increase",
                             "mu for meaningful group"
                             ] )

    self.specwindow.entrylbrstart = self.specwindow.entrylblstop
    self.specwindow.entrylbrstop = len( self.var )

#  Set the name and logo 

    Label( self.specwindow, text="\nLANCELOT B OPTIONS\n"
           ).pack( side=TOP, fill=BOTH )

    Label( self.specwindow, image=self.img, relief=SUNKEN
           ).pack( side=TOP, fill=NONE )

    Label( self.specwindow, text="\n"
           ).pack( side=TOP, fill=BOTH )

#  --- set frames  ---

#  main frame

    self.specwindow.frame = Frame( self.specwindow )

#  left and right sub-frames

    self.specwindow.frame.lhs = Frame( self.specwindow.frame )
    self.specwindow.frame.rhs = Frame( self.specwindow.frame )

#  frame to hold check buttons

    self.specwindow.check = Frame( self.specwindow.frame.lhs )

#  sub-frames for check buttons

    self.specwindow.check.left = Frame( self.specwindow.check )
    self.specwindow.check.right = Frame( self.specwindow.check )

#  frame to hold gradient and Hessian check buttons

    self.specwindow.ghcheck = Frame( self.specwindow.frame.lhs )

#  sub-frames for gradient and Hessian check buttons

    self.specwindow.ghcheck.left = Frame( self.specwindow.ghcheck )
    self.specwindow.ghcheck.right = Frame( self.specwindow.ghcheck )

# frame and sub-frames for expert data slots

    self.specwindow.frame.lhs.bottom = Frame( self.specwindow.frame.lhs )
    self.specwindow.frame.lhs.bottom.left \
      = Frame( self.specwindow.frame.lhs.bottom )
    self.specwindow.frame.lhs.bottom.right \
      = Frame( self.specwindow.frame.lhs.bottom )

# frame and sub-frames to hold data entry slots (top, right)

    self.specwindow.frame.rhs.top = Frame( self.specwindow.frame.rhs )
    self.specwindow.frame.rhs.top.left \
      = Frame( self.specwindow.frame.rhs.top )
    self.specwindow.frame.rhs.top.right \
      = Frame( self.specwindow.frame.rhs.top )

# frame and sub-frames to hold button and data entry slots (bottom, right)

    self.specwindow.frame.rhs.bottom = Frame( self.specwindow.frame.rhs )

# sub-frames to hold selection buttons

    self.specwindow.solver = Frame( self.specwindow.frame.rhs.bottom )

#  frame to hold data entry slots

    self.specwindow.text = Frame( self.specwindow.frame.rhs.bottom )

#   self.specwindow.iprint = Frame( self.specwindow.text )

#  --- set contents of frames ---

#  == Left-hand side of window ==

#  contents of check left frame

    for i in range( self.specwindow.varlstart, self.specwindow.varlstop ) :
      Checkbutton( self.specwindow.check.left,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   command=self.nofdgandexacthessian,
                   variable=self.check[i],
                   text=self.checkstring[i]
                   ).pack( side=TOP, fill=BOTH )
    
    self.specwindow.check.left.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.check, width=4,
           text="" ).pack( side=LEFT, fill=BOTH )

#  contents of check right frame

    for i in range( self.specwindow.varrstart, self.specwindow.varrstop ) :
      Checkbutton( self.specwindow.check.right,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   command=self.nofdgandexacthessian,
                   variable=self.check[i],
                   text=self.checkstring[i]
                 ).pack( side=TOP, fill=BOTH )

    self.specwindow.check.right.pack( side=LEFT, fill=BOTH )

#  pack check box

    self.specwindow.check.pack( side=TOP, fill=BOTH )

#  contents of gradient frame (label and radio buttons)

    Label( self.specwindow.ghcheck.left,
           text="\nGradient" ).pack( side=TOP, fill=BOTH )

    self.specwindow.gradient = Frame( self.specwindow.ghcheck.left )
#    Label( self.specwindow.gradient,
#           width=12, text="" ).pack( side=LEFT, fill=BOTH )

    for gradient in [ 'Exact', 'Forward', 'Central']:
      lower = gradient.lower( )
      Radiobutton( self.specwindow.gradient,
                   highlightthickness=0, relief=FLAT,
                   variable=self.lanb_var_gradient,
                   value=lower,
                   command=self.nofdgandexacthessian,
                   text=gradient
                   ).pack( side=LEFT, fill=BOTH )
                                                 
    self.specwindow.gradient.pack( side=TOP, fill=BOTH )
    self.specwindow.ghcheck.left.pack( side=LEFT, fill=BOTH )

#  contents of hessian frame (label and radio buttons)

    Label( self.specwindow.ghcheck.right,
           text="\nHessian" ).pack( side=TOP, fill=BOTH )

    self.specwindow.hessian = Frame( self.specwindow.ghcheck.right )
    Label( self.specwindow.hessian,
           width=4, text="" ).pack( side=LEFT, fill=BOTH )

    for hessian in [ 'Exact', 'BFGS', 'DFP', 'PSB', 'SR1']:
      lower = hessian.lower( )
      Radiobutton( self.specwindow.hessian,
                   highlightthickness=0, relief=FLAT,
                   variable=self.lanb_var_hessian,
                   value=lower,
                   command=self.nofdgandexacthessian,
                   text=hessian
                   ).pack( side=LEFT, fill=BOTH )
                                                 
    self.specwindow.hessian.pack( side=TOP, fill=BOTH )
    self.specwindow.ghcheck.right.pack( side=LEFT, fill=BOTH )

#  pack check box

    self.specwindow.ghcheck.pack( side=TOP, fill=BOTH )

#  Experts' corner

    Label( self.specwindow.frame.lhs.bottom,
           text="\nExperts' corner\n" ).pack( side=TOP, fill=BOTH )

    for i in range( self.specwindow.entrylblstart,
                    self.specwindow.entrylblstop ):
      self.specwindow.i = Frame( self.specwindow.frame.lhs.bottom.left )
      Label( self.specwindow.i,
             anchor=W, width=22,
             text=self.varstring[ i ]             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.lhs.bottom.left.pack( side=LEFT, fill=BOTH )

#  contents of rhs top right data entry frame

    Label( self.specwindow.frame.lhs.bottom, width=4,
           text="" ).pack( side=LEFT, fill=BOTH )

    for i in range( self.specwindow.entrylbrstart,
                    self.specwindow.entrylbrstop ):
      self.specwindow.i = Frame( self.specwindow.frame.lhs.bottom.right )
      Label( self.specwindow.i,
             anchor=W, width=22,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.lhs.bottom.right.pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.lhs.bottom.pack( side=TOP, fill=BOTH )

#  == Right-hand side of window ==

#  contents of rhs top left data entry frame

    for i in range( self.specwindow.entrytlstart, self.specwindow.entrytlstop ):
      self.specwindow.i = Frame( self.specwindow.frame.rhs.top.left )
      Label( self.specwindow.i,
             anchor=W, width=23,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.rhs.top.left.pack( side=LEFT, fill=BOTH )

#  contents of rhs top right data entry frame

    Label( self.specwindow.frame.rhs.top, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    for i in range( self.specwindow.entrytrstart, self.specwindow.entrytrstop ):
      self.specwindow.i = Frame( self.specwindow.frame.rhs.top.right )
      Label( self.specwindow.i,
             anchor=W, width=30,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.rhs.top.right.pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.rhs.top.pack( side=TOP, fill=BOTH )

#  contents of rhs solver frame

    Label( self.specwindow.solver, width=30, anchor=W,
           text="\nSolver" ).pack( side=TOP, fill=BOTH )

    for solvers in [ 'cg', 'diagonal_cg', 'band_cg', 'lin_more_cg', \
                     'munksgaard_cg', 'expanding_band_cg', 'gmps_cg', \
                     'schnabel_eskow_cg', 'multifrontal', \
                     'modified_multifrontal' ]:
      if solvers == 'cg' : label = "CG"
      elif solvers == 'diagonal_cg' : label = "Diagonal" 
      elif solvers == 'munksgaard_cg' : label = "Munksgaard (+)" 
      elif solvers == 'expanding_band_cg' : label = "Expanding Band (*)" 
      elif solvers == 'gmps_cg' : label = "Gill-Murray-Poncelon-Saunders (*)" 
      elif solvers == 'schnabel_eskow_cg' : label = "Schnabel-Eskow (*)" 
      elif solvers == 'band_cg' : label = "Band: semibandwidth" 
      elif solvers == 'lin_more_cg' : label = "Lin-More (\"): vectors" 
      elif solvers == 'multifrontal' : label = "Direct multifrontal (*)" 
      elif solvers == 'modified_multifrontal' :
        label = "Direct modified multifrontal (*)" 
      else :label = "" 
      if solvers == 'band_cg' :
        self.specwindow.bandsolver = Frame( self.specwindow.solver )
        Radiobutton( self.specwindow.bandsolver,
                     highlightthickness=0,
                     relief=FLAT, anchor=W,
                     variable=self.lanb_var_solver,
                     value=solvers,
                     command=self.lanbsolversonoff,
                     text=label
                     ).pack( side=LEFT, fill=NONE )
        Entry( self.specwindow.bandsolver,
               textvariable=self.lanb_var_bandwidth,
               relief=SUNKEN, width=10
               ).pack( side=RIGHT, fill=BOTH )
        self.specwindow.bandsolver.pack( side=TOP, fill=BOTH )
      elif solvers == 'lin_more_cg' :
        self.specwindow.linmore = Frame( self.specwindow.solver )
        Radiobutton( self.specwindow.linmore,
                     highlightthickness=0,
                     relief=FLAT, anchor=W,
                     variable=self.lanb_var_solver,
                     value=solvers,
                     command=self.lanbsolversonoff,
                     text=label
                     ).pack( side=LEFT, fill=NONE )
        Entry( self.specwindow.linmore,
               textvariable=self.lanb_var_linmore,
               relief=SUNKEN, width=10
               ).pack( side=RIGHT, fill=BOTH )
        self.specwindow.linmore.pack( side=TOP, fill=BOTH )
      else :
        Radiobutton( self.specwindow.solver,
                     highlightthickness=0, relief=FLAT,
                     width=31, anchor=W,
                     variable=self.lanb_var_solver,
                     value=solvers,
                     command=self.lanbsolversonoff,
                     text=label
                     ).pack( side=TOP, fill=NONE )

    Label( self.specwindow.solver, width=30, anchor=W,
           text="     (+)  requires MA39" ).pack( side=TOP, fill=BOTH )
    Label( self.specwindow.solver, width=30, anchor=W,
           text="     (*)  requires MA27" ).pack( side=TOP, fill=BOTH )
    Label( self.specwindow.solver, width=30, anchor=W,
           text="     (\")  requires ICFS" ).pack( side=TOP, fill=BOTH )

    self.specwindow.solver.pack( side=LEFT, fill=BOTH )

#  contents of rhs bottom data entry frame

    Label( self.specwindow.frame.rhs.bottom, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    for i in range( self.specwindow.entrybrstart, self.specwindow.entrybrstop ):
      self.specwindow.i = Frame( self.specwindow.text )
      Label( self.specwindow.i,
             anchor=W, width=30,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )

#  Special check button for writeall

    Label( self.specwindow.text, text=" " ).pack( side=TOP, fill=BOTH )
    Checkbutton( self.specwindow.text,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   variable=self.lanb_check_writeall,
                   text="Even write defaults when saving values"
                 ).pack( side=TOP, fill=BOTH )

    self.specwindow.text.pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.rhs.bottom.pack( side=TOP, fill=BOTH )

    Label( self.specwindow.frame.rhs, text="\n" ).pack( side=TOP, fill=BOTH )

#  --- assemble boxes ---

#  Pack it all together

    Label( self.specwindow.frame, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.lhs.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.frame, width=4,
           text="" ).pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.rhs.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.frame, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.pack( side=TOP, fill=BOTH )

#  Pack buttons along the bottom

    self.specwindow.buttons = Frame( self.specwindow )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Dismiss\nwindow", 
            command=self.specwindowdestroy
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Edit RUNLANB.SPC\ndirectly", 
            command=self.editlanbspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Read existing\nvalues", 
            command=self.readlanbspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Restore default\nvalues", 
            command=self.restorelanbdefaults
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Save current\nvalues", 
            command=self.writelanbspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Run LANCELOT B\nwith current values",
            command=self.rungaloncurrent
            ).pack( side=LEFT, fill=BOTH )
    self.spacer( )

    self.specwindow.buttons.pack( side=TOP, fill=BOTH )

    Label( self.specwindow, height=1,
           text="\n" ).pack( side=TOP, fill=BOTH )

#  function to edit RUNLANB.SPC
#  ----------------------------

  def editlanbspec( self ):
    if os.path.exists( 'RUNLANB.SPC' ) == 0 :
      print ' no file RUNLANB.SPC to read'
      self.nospcfile( 'RUNLANB.SPC', 'edit' )
      return
    try:
      editor = os.environ["VISUAL"]
    except KeyError:
      try:
        editor = os.environ["EDITOR"]
      except KeyError:
        editor = emacs
    os.popen( editor+' RUNLANB.SPC' )

#  function to restore default spec values
#  ----------------------------------------

  def restorelanbdefaults( self ):
    self.lanb_check_maximizer.set( self.lanb_default_maximizer )
    self.lanb_check_printfullsol.set( self.lanb_default_printfullsol )
    self.lanb_check_checkder.set( self.lanb_default_checkder )
    self.lanb_check_checkall.set( self.lanb_default_checkall )
    self.lanb_check_checkel.set( self.lanb_default_checkel )
    self.lanb_check_checkgr.set( self.lanb_default_checkgr )
    self.lanb_check_ignoreder.set( self.lanb_default_ignoreder )
    self.lanb_check_ignoreel.set( self.lanb_default_ignoreel )
    self.lanb_check_ignoregr.set( self.lanb_default_ignoregr )
    self.lanb_check_qp.set( self.lanb_default_qp )
    self.lanb_check_twonorm.set( self.lanb_default_twonorm )
    self.lanb_check_strtr.set( self.lanb_default_strtr )
    self.lanb_check_gcp.set( self.lanb_default_gcp )
    self.lanb_check_accuratebqp.set( self.lanb_default_accuratebqp )
    self.lanb_check_restart.set( self.lanb_default_restart )
    self.lanb_check_scaling.set( self.lanb_default_scaling )
    self.lanb_check_cscaling.set( self.lanb_default_cscaling )
    self.lanb_check_vscaling.set( self.lanb_default_vscaling )
    self.lanb_check_gscaling.set( self.lanb_default_gscaling )
    self.lanb_check_wproblem.set( self.lanb_default_wproblem )

    self.lanb_var_plevel.set( self.lanb_default_plevel )
    self.lanb_var_sprint.set( self.lanb_default_sprint )
    self.lanb_var_fprint.set( self.lanb_default_fprint )
    self.lanb_var_iprint.set( self.lanb_default_iprint )
    self.lanb_var_slevel.set( self.lanb_default_slevel )
    self.lanb_var_maxit.set( self.lanb_default_maxit )
    self.lanb_var_savedata.set( self.lanb_default_savedata )
    self.lanb_var_cstop.set( self.lanb_default_cstop )
    self.lanb_var_gstop.set( self.lanb_default_gstop )
    self.lanb_var_mu.set( self.lanb_default_mu )
    self.lanb_var_decreasemu.set( self.lanb_default_decreasemu )
    self.lanb_var_firstg.set( self.lanb_default_firstg )
    self.lanb_var_firstc.set( self.lanb_default_firstc )
    self.lanb_var_pivtol.set( self.lanb_default_pivtol )
    self.lanb_var_firstr.set( self.lanb_default_firstr )
    self.lanb_var_gradient.set( self.lanb_default_gradient )
    self.lanb_var_hessian.set( self.lanb_default_hessian )
    self.lanb_var_maxsc.set( self.lanb_default_maxsc )
    self.lanb_var_acccg.set( self.lanb_default_acccg )
    self.lanb_var_maxrad.set( self.lanb_default_maxrad )
    self.lanb_var_etas.set( self.lanb_default_etas )
    self.lanb_var_etavs.set( self.lanb_default_etavs )
    self.lanb_var_etaes.set( self.lanb_default_etaes )
    self.lanb_var_gammasmall.set( self.lanb_default_gammasmall )
    self.lanb_var_gammadec.set( self.lanb_default_gammadec )
    self.lanb_var_gammainc.set( self.lanb_default_gammainc )
    self.lanb_var_mumodel.set( self.lanb_default_mumodel )
    self.lanb_var_mugroup.set( self.lanb_default_mugroup )
    self.lanb_var_moretoraldosearch.set( self.lanb_default_moretoraldosearch )
    self.lanb_var_nonmonotone.set( self.lanb_default_nonmonotone )
    self.lanb_var_solver.set( self.lanb_default_solver )
    self.lanb_var_bandwidth.set( self.lanb_default_bandwidth )
    self.lanb_var_linmore.set( self.lanb_default_linmore )

    self.current_bandwidth = self.lanb_default_bandwidth
    self.current_linmore = self.lanb_default_linmore
    self.lanbsolversonoff( )
  
#  function to switch on and off semibandwidth/line-more vectors as appropriate
#  ----------------------------------------------------------------------------

  def lanbsolversonoff( self ): 
    if self.lanb_var_solver.get( ) == 'band_cg' :
      self.lanb_var_bandwidth.set( self.current_bandwidth )
    else:
      if self.lanb_var_bandwidth.get( ) != '' :
        self.current_bandwidth = self.lanb_var_bandwidth.get( )
      self.lanb_var_bandwidth.set( '' )
    if self.lanb_var_solver.get( ) == 'lin_more_cg' :
      self.lanb_var_linmore.set( self.current_linmore )
    else:
      if self.lanb_var_linmore.get( ) != '' :
        self.current_linmore = self.lanb_var_linmore.get( )
      self.lanb_var_linmore.set( '' )

#  function to disallow exact second derivatives at the same time
#  as finite-diffreence gradients
#  ------------------------------

  def nofdgandexacthessian( self ): 
    if self.lanb_var_hessian.get( ) == 'exact' and \
       self.lanb_var_gradient.get( ) != 'exact' :
      self.lanb_var_hessian.set( 'sr1' )

#  function to read the current values to the spec file
#  -----------------------------------------------------

  def readlanbspec( self ): 

#  open file and set header

    if os.path.exists( 'RUNLANB.SPC' ) == 0 :
      print ' no file RUNLANB.SPC to read'
      self.nospcfile( 'RUNLANB.SPC', 'read' )
      return
    self.runlanbspc = open( 'RUNLANB.SPC', 'r' )

#  Restore default values

    self.restorelanbdefaults( )

#  loop over lines of files

    self.readyes = 0
    for line in self.runlanbspc:

#  exclude comments

      if line[0] == '!' : continue

#  convert the line to lower case, and remove leading and trailing blanks

      line = line.lower( ) 
      line = line.strip( )
      blank_start = line.find( ' ' ) 
      
      if blank_start != -1 :
        stringc = line[0:blank_start]
      else :
        stringc = line

#  look for string variables to set

      blank_end = line.rfind( ' ' ) 
      if blank_start == -1 :
        stringd = 'YES'
      else:
        stringd = line[ blank_end + 1 : ].upper( )
#     print stringc+' '+stringd

#  only read those segments concerned with LANCELOT B

      if stringc == 'begin' and line.find( 'lancelot' ) >= 0 : self.readyes = 1
      if stringc == 'end' and line.find( 'lancelot' ) >= 0 : self.readyes = 0
      if self.readyes == 0 : continue

#  exclude begin and end lines

      if stringc == 'begin' or stringc == 'end' : continue

#  look for integer (check-button) variables to set

      if stringc == self.lanb_string_maximizer :
        self.yesno( self.lanb_check_maximizer, stringd )
        continue
      elif stringc == self.lanb_string_printfullsol :
        self.yesno( self.lanb_check_printfullsol, stringd )
        continue
      elif stringc == self.lanb_string_checkder :
        self.yesno( self.lanb_check_checkder, stringd )
        continue
      elif stringc == self.lanb_string_checkall :
        self.yesno( self.lanb_check_checkall, stringd )
        continue
      elif stringc == self.lanb_string_checkel :
        self.yesno( self.lanb_check_checkel, stringd )
        continue
      elif stringc == self.lanb_string_checkgr :
        self.yesno( self.lanb_check_checkgr, stringd )
        continue
      elif stringc == self.lanb_string_ignoreder :
        self.yesno( self.lanb_check_ignoreder, stringd )
        continue
      elif stringc == self.lanb_string_ignoreel :
        self.yesno( self.lanb_check_ignoreel, stringd )
        continue
      elif stringc == self.lanb_string_ignoregr :
        self.yesno( self.lanb_check_ignoregr, stringd )
        continue
      elif stringc == self.lanb_string_qp :
        self.yesno( self.lanb_check_qp, stringd )
        continue
      elif stringc == self.lanb_string_twonorm :
        self.yesno( self.lanb_check_twonorm, stringd )
        continue
      elif stringc == self.lanb_string_strtr :
        self.yesno( self.lanb_check_strtr, stringd )
        continue
      elif stringc == self.lanb_string_gcp :
        self.yesno( self.lanb_check_gcp, stringd )
        continue
      elif stringc == self.lanb_string_accuratebqp :
        self.yesno( self.lanb_check_accuratebqp, stringd )
        continue
      elif stringc == self.lanb_string_restart :
        self.yesno( self.lanb_check_restart, stringd )
        continue
      elif stringc == self.lanb_string_scaling :
        self.yesno( self.lanb_check_scaling, stringd )
        continue
      elif stringc == self.lanb_string_cscaling :
        self.yesno( self.lanb_check_cscaling, stringd )
        continue
      elif stringc == self.lanb_string_vscaling :
        self.yesno( self.lanb_check_vscaling, stringd )
        continue
      elif stringc == self.lanb_string_gscaling :
        self.yesno( self.lanb_check_gscaling, stringd )
        continue
      elif stringc == self.lanb_string_wproblem :
        self.yesno( self.lanb_check_wproblem, stringd )
        continue

      if stringc == self.lanb_string_gradient :
        stringd = stringd.lower( ) 
        if stringd == 'forward' :
          self.lanb_var_gradient.set( 'forward' )
        elif stringd == 'central' :
          self.lanb_var_gradient.set( 'central' )
        else :
          self.lanb_var_gradient.set( 'exact' )
        continue
      elif stringc ==  self.lanb_string_hessian :
        stringd = stringd.lower( ) 
        if stringd == 'bfgs':
          self.lanb_var_hessian.set( 'bfgs' )
        elif stringd == 'dfp':
          self.lanb_var_hessian.set( 'dfp' )
        elif stringd == 'psb':
          self.lanb_var_hessian.set( 'psb' )
        elif stringd == 'sr1':
          self.lanb_var_hessian.set( 'sr1' )
        else:
          self.lanb_var_hessian.set( 'exact' )
        continue
      elif stringc == self.lanb_string_solver :
        stringd = stringd.lower( ) 
        if stringd == 'cg':
          self.lanb_var_solver.set( 'cg' ) 
        elif stringd == 'diagonal_cg':
          self.lanb_var_solver.set( 'diagonal_cg' )
        elif stringd == 'munksgaard_cg':
          self.lanb_var_solver.set( 'munksgaard_cg' )
        elif stringd == 'lin_more_cg':
          self.lanb_var_solver.set( 'lin_more_cg' )
        elif stringd == 'expanding_band_cg':
          self.lanb_var_solver.set( 'expanding_band_cg' )
        elif stringd == 'gmps_cg':
          self.lanb_var_solver.set( 'gmps_cg' )
        elif stringd == 'schnabel_eskow_cg':
          self.lanb_var_solver.set( 'schnabel_eskow_cg' )
        elif stringd == 'multifrontal':
          self.lanb_var_solver.set( 'multifrontal' )
        elif stringd == 'modified_multifrontal':
          self.lanb_var_solver.set( 'modified_multifrontal' )
        else:
          self.lanb_var_solver.set( 'band_cg' )
        continue
      elif stringc == self.lanb_string_bandwidth :
        self.lanb_var_bandwidth.set( stringd )
        self.current_bandwidth = stringd
        continue
      elif stringc == self.lanb_string_linmore :
        self.lanb_var_linmore.set( stringd )
        self.current_linmore = stringd
        continue
      elif stringc == self.lanb_string_plevel :
        self.lanb_var_plevel.set( stringd )
        continue
      elif stringc == self.lanb_string_sprint :
        self.lanb_var_sprint.set( stringd )
        continue
      elif stringc == self.lanb_string_fprint :
        self.lanb_var_fprint.set( stringd )
        continue
      elif stringc == self.lanb_string_iprint :
        self.lanb_var_iprint.set( stringd )
        continue
      elif stringc == self.lanb_string_slevel :
        self.lanb_var_slevel.set( stringd )
        continue
      elif stringc == self.lanb_string_maxit :
        self.lanb_var_maxit.set( stringd )
        continue
      elif stringc == self.lanb_string_savedata :
        self.lanb_var_savedata.set( stringd )
        continue
      elif stringc == self.lanb_string_cstop :
        self.lanb_var_cstop.set( stringd )
        continue
      elif stringc == self.lanb_string_gstop :
        self.lanb_var_gstop.set( stringd )
        continue
      elif stringc == self.lanb_string_mu :
        self.lanb_var_mu.set( stringd )
        continue
      elif stringc == self.lanb_string_decreasemu :
        self.lanb_var_decreasemu.set( stringd )
        continue
      elif stringc == self.lanb_string_firstc :
        self.lanb_var_firstc.set( stringd )
        continue
      elif stringc == self.lanb_string_firstg :
        self.lanb_var_firstg.set( stringd )
        continue
      elif stringc == self.lanb_string_pivtol :
        self.lanb_var_pivtol.set( stringd )
        continue
      elif stringc == self.lanb_string_firstr :
        self.lanb_var_firstr.set( stringd )
        continue
      elif stringc == self.lanb_string_etas :
        self.lanb_var_etas.set( stringd )
        continue
      elif stringc == self.lanb_string_etavs :
        self.lanb_var_etavs.set( stringd )
        continue
      elif stringc == self.lanb_string_etaes :
        self.lanb_var_etaes.set( stringd )
        continue
      elif stringc == self.lanb_string_gammasmall :
        self.lanb_var_gammasmall.set( stringd )
        continue
      elif stringc == self.lanb_string_gammadec :
        self.lanb_var_gammadec.set( stringd )
        continue
      elif stringc == self.lanb_string_gammainc :
        self.lanb_var_gammainc.set( stringd )
        continue
      elif stringc == self.lanb_string_mumodel :
        self.lanb_var_mumodel.set( stringd )
        continue
      elif stringc == self.lanb_string_mugroup :
        self.lanb_var_mugroup.set( stringd )
        continue
      elif stringc == self.lanb_string_maxsc :
        self.lanb_var_maxsc.set( stringd )
        continue
      elif stringc == self.lanb_string_acccg :
        self.lanb_var_acccg.set( stringd )
        continue
      elif stringc == self.lanb_string_maxrad :
        self.lanb_var_maxrad.set( stringd )
        continue
      elif stringc == self.lanb_string_moretoraldosearch :
        self.lanb_var_moretoraldosearch.set( stringd )
        continue
      elif stringc == self.lanb_string_nonmonotone :
        self.lanb_var_nonmonotone.set( stringd )
        continue

    self.lanbsolversonoff( )
    self.runlanbspc.close( )
    if self.lanb_var_hessian.get( ) == 'exact' and \
       self.lanb_var_gradient.get( ) != 'exact' :
      self.lanb_var_hessian.set( 'sr1' )

#  function to write the current values to the spec file
#  -----------------------------------------------------

  def writelanbspec( self ): 

#  open file and set header

    self.runlanbspc = open( 'RUNLANB.SPC', 'w' )

#  record RUNLANB options

    self.runlanbspc.write( "BEGIN RUNLANB SPECIFICATIONS\n" )

    self.writelanbspecline_int( self.lanb_check_wproblem,
                            self.lanb_default_wproblem, 
                            self.lanb_string_wproblem )
    self.writelanbspecdummy( 'problem-data-file-name', 'LANB.data' )
    self.writelanbspecdummy( 'problem-data-file-device', '26' )
    self.writelanbspecline_int( self.lanb_check_printfullsol,
                            self.lanb_default_printfullsol, 
                            self.lanb_string_printfullsol )
    self.writelanbspecdummy( 'write-solution', 'YES' )
    self.writelanbspecdummy( 'solution-file-name', 'LANBSOL.d' )
    self.writelanbspecdummy( 'solution-file-device', '62' )
    self.writelanbspecdummy( 'write-result-summary', 'YES' )
    self.writelanbspecdummy( 'result-summary-file-name', 'LANBRES.d' )
    self.writelanbspecdummy( 'result-summary-file-device', '47' )
    self.writelanbspecline_int( self.lanb_check_checkall,
                            self.lanb_default_checkall, 
                            self.lanb_string_checkall )
    self.writelanbspecline_int( self.lanb_check_checkder,
                            self.lanb_default_checkder,
                            self.lanb_string_checkder )
    self.writelanbspecline_int( self.lanb_check_checkel,
                            self.lanb_default_checkel,
                            self.lanb_string_checkel )
    self.writelanbspecline_int( self.lanb_check_checkgr,
                            self.lanb_default_checkgr,
                            self.lanb_string_checkgr )
    self.writelanbspecline_int( self.lanb_check_ignoreder,
                            self.lanb_default_ignoreder, 
                            self.lanb_string_ignoreder )
    self.writelanbspecline_int( self.lanb_check_ignoreel,
                            self.lanb_default_ignoreel, 
                            self.lanb_string_ignoreel )
    self.writelanbspecline_int( self.lanb_check_ignoregr,
                            self.lanb_default_ignoregr, 
                            self.lanb_string_ignoregr )
    self.writelanbspecline_int( self.lanb_check_gscaling,
                            self.lanb_default_gscaling, 
                            self.lanb_string_gscaling )
    self.writelanbspecline_stringval( self.lanb_var_slevel,
                                  self.lanb_default_plevel,
                                  self.lanb_string_slevel )
    self.writelanbspecline_int( self.lanb_check_scaling,
                            self.lanb_default_scaling, 
                            self.lanb_string_scaling )
    self.writelanbspecline_int( self.lanb_check_cscaling,
                            self.lanb_default_cscaling, 
                            self.lanb_string_cscaling )
    self.writelanbspecline_int( self.lanb_check_vscaling,
                            self.lanb_default_vscaling, 
                            self.lanb_string_vscaling )
    self.writelanbspecline_int( self.lanb_check_maximizer,
                            self.lanb_default_maximizer, 
                            self.lanb_string_maximizer )
    self.writelanbspecline_int( self.lanb_check_restart,
                            self.lanb_default_restart, 
                            self.lanb_string_restart )
    self.writelanbspecdummy( 'restart-data-file-name', 'LANBSAVE.d' )
    self.writelanbspecdummy( 'restart-data-file-device', '59' )
    self.writelanbspecline_stringval( self.lanb_var_savedata,
                                  self.lanb_default_savedata,
                                  self.lanb_string_savedata )

    self.runlanbspc.write( "END RUNLANB SPECIFICATIONS\n\n" )

#  record LANCELOT B options

    self.runlanbspc.write( "BEGIN LANCELOT SPECIFICATIONS\n" )

    self.writelanbspecdummy( 'error-printout-device', '6' )
    self.writelanbspecdummy( 'printout-device', '6' )
    self.writelanbspecdummy( 'alive-device', '60' )
    self.writelanbspecline_stringval( self.lanb_var_plevel,
                                  self.lanb_default_plevel,
                                  self.lanb_string_plevel )
    self.writelanbspecline_stringval( self.lanb_var_maxit,
                                  self.lanb_default_maxit,
                                  self.lanb_string_maxit )
    self.writelanbspecline_stringval( self.lanb_var_sprint,
                                  self.lanb_default_sprint,
                                  self.lanb_string_sprint )
    self.writelanbspecline_stringval( self.lanb_var_fprint,
                                  self.lanb_default_fprint,
                                  self.lanb_string_fprint )
    self.writelanbspecline_stringval( self.lanb_var_iprint,
                                  self.lanb_default_iprint,
                                  self.lanb_string_iprint )

#  record solver chosen

    if self.lanb_var_solver.get( ) == "cg" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "cg",
                                 self.lanb_string_solver )
    elif self.lanb_var_solver.get( ) == "diagonal_cg" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "diagonal_cg",
                                 self.lanb_string_solver )
    elif self.lanb_var_solver.get( ) == "expanding_band_cg" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "expanding_band_cg",
                                 self.lanb_string_solver )
    elif self.lanb_var_solver.get( ) == "munksgaard_cg" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "munksgaard_cg",
                                 self.lanb_string_solver )
    elif self.lanb_var_solver.get( ) == "schnabel_eskow_cg" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "schnabel_eskow_cg",
                                 self.lanb_string_solver )
    elif self.lanb_var_solver.get( ) == "gmps_cg" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "gmps_cg",
                                 self.lanb_string_solver )
    elif self.lanb_var_solver.get( ) == "lin_more_cg" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "lin_more_cg",
                                 self.lanb_string_solver )
    elif self.lanb_var_solver.get( ) == "multifrontal" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "multifrontal",
                                 self.lanb_string_solver )
    elif self.lanb_var_solver.get( ) == "modified_multifrontal" :
      self.writelanbspecline_string( self.lanb_var_solver,
                                 "modified_multifrontal",
                                 self.lanb_string_solver )
    else :
      if self.lanb_check_writeall.get( ) == 1 :
        self.runlanbspc.write( "!  "+self.lanb_string_solver.ljust( 50 ) \
                               +"BAND_CG\n" )

#  record further options

    self.writelanbspecline_stringval( self.lanb_var_bandwidth,
                                  self.lanb_default_bandwidth,
                                  self.lanb_string_bandwidth )
    self.writelanbspecline_stringval( self.lanb_var_linmore,
                                      self.lanb_default_linmore,
                                      self.lanb_string_linmore )
    self.writelanbspecline_stringval( self.lanb_var_maxsc,
                                      self.lanb_default_maxsc,
                                      self.lanb_string_maxsc )
    self.writelanbspecdummy( 'unit-number-for-temporary-io', '75' )
    self.writelanbspecline_stringval( self.lanb_var_moretoraldosearch,
                                  self.lanb_default_moretoraldosearch,
                                  self.lanb_string_moretoraldosearch )
    self.writelanbspecline_stringval( self.lanb_var_nonmonotone,
                                  self.lanb_default_nonmonotone,
                                  self.lanb_string_nonmonotone )

#  record gradient chosen

    if self.lanb_var_gradient.get( ) == "forward" :
      self.writelanbspecline_string( self.lanb_var_gradient, "forward",
                                 self.lanb_string_gradient )
    elif self.lanb_var_gradient.get( ) == "central" :
      self.writelanbspecline_string( self.lanb_var_gradient, "central",
                                 self.lanb_string_gradient )
    else : 
      if self.lanb_check_writeall.get( ) == 1 :
        self.runlanbspc.write( "!  "+self.lanb_string_gradient.ljust( 50 ) \
                               +"EXACT\n" )

#  record Hessian chosen

    if self.lanb_var_hessian.get( ) == "bfgs" :
      self.writelanbspecline_string( self.lanb_var_hessian, "bfgs",
                                 self.lanb_string_hessian )
    elif self.lanb_var_hessian.get( ) == "dfp" :
      self.writelanbspecline_string( self.lanb_var_hessian, "dfp",
                                 self.lanb_string_hessian )
    elif self.lanb_var_hessian.get( ) == "psb" :
      self.writelanbspecline_string( self.lanb_var_hessian, "psb",
                                 self.lanb_string_hessian )
    elif self.lanb_var_hessian.get( ) == "sr1" :
      self.writelanbspecline_string( self.lanb_var_hessian, "sr1",
                                 self.lanb_string_hessian )
    else :
      if self.lanb_check_writeall.get( ) == 1 :
        self.runlanbspc.write( "!  "+self.lanb_string_hessian.ljust( 50 ) \
                               +"EXACT\n" )

#  record remaining options

    self.writelanbspecline_stringval( self.lanb_var_cstop,
                                  self.lanb_default_cstop,
                                  self.lanb_string_cstop )
    self.writelanbspecline_stringval( self.lanb_var_gstop,
                                  self.lanb_default_gstop,
                                  self.lanb_string_gstop )
    self.writelanbspecline_stringval( self.lanb_var_acccg,
                                  self.lanb_default_acccg,
                                  self.lanb_string_acccg )
    self.writelanbspecline_stringval( self.lanb_var_firstr,
                                  self.lanb_default_firstr,
                                  self.lanb_string_firstr )
    self.writelanbspecline_stringval( self.lanb_var_maxrad,
                                  self.lanb_default_maxrad,
                                  self.lanb_string_maxrad )
    self.writelanbspecline_stringval( self.lanb_var_etas,
                                  self.lanb_default_etas,
                                  self.lanb_string_etas )
    self.writelanbspecline_stringval( self.lanb_var_etavs,
                                  self.lanb_default_etavs,
                                  self.lanb_string_etavs )
    self.writelanbspecline_stringval( self.lanb_var_etaes,
                                  self.lanb_default_etaes,
                                  self.lanb_string_etaes )
    self.writelanbspecline_stringval( self.lanb_var_gammasmall,
                                  self.lanb_default_gammasmall,
                                  self.lanb_string_gammasmall )
    self.writelanbspecline_stringval( self.lanb_var_gammadec,
                                  self.lanb_default_gammadec,
                                  self.lanb_string_gammadec )
    self.writelanbspecline_stringval( self.lanb_var_gammainc,
                                  self.lanb_default_gammainc,
                                  self.lanb_string_gammainc )
    self.writelanbspecline_stringval( self.lanb_var_mumodel,
                                  self.lanb_default_mumodel,
                                  self.lanb_string_mumodel )
    self.writelanbspecline_stringval( self.lanb_var_mugroup,
                                  self.lanb_default_mugroup,
                                  self.lanb_string_mugroup )
    self.writelanbspecline_stringval( self.lanb_var_mu,
                                  self.lanb_default_mu,
                                  self.lanb_string_mu )
    self.writelanbspecline_stringval( self.lanb_var_decreasemu,
                                  self.lanb_default_decreasemu,
                                  self.lanb_string_decreasemu )
    self.writelanbspecline_stringval( self.lanb_var_firstg,
                                  self.lanb_default_firstg,
                                  self.lanb_string_firstg )
    self.writelanbspecline_stringval( self.lanb_var_firstc,
                                  self.lanb_default_firstc,
                                  self.lanb_string_firstc )
    self.writelanbspecline_stringval( self.lanb_var_pivtol,
                                  self.lanb_default_pivtol,
                                  self.lanb_string_pivtol )
    self.writelanbspecline_int( self.lanb_check_qp,
                            self.lanb_default_qp, 
                            self.lanb_string_qp )
    self.writelanbspecline_int( self.lanb_check_twonorm,
                            self.lanb_default_twonorm, 
                            self.lanb_string_twonorm )
    self.writelanbspecline_int( self.lanb_check_gcp,
                            self.lanb_default_gcp,
                            self.lanb_string_gcp )
    self.writelanbspecdummy( 'magical-steps-allowed', 'NO' )
    self.writelanbspecline_int( self.lanb_check_accuratebqp,
                            self.lanb_default_accuratebqp, 
                            self.lanb_string_accuratebqp )
    self.writelanbspecline_int( self.lanb_check_strtr,
                            self.lanb_default_strtr, 
                            self.lanb_string_strtr )
    if self.lanb_check_maximizer.get( ) == 0 :
      if self.lanb_check_writeall.get( ) == 1 :
        self.runlanbspc.write( "!  "+self.lanb_string_printformax.ljust( 50 ) \
                               +"NO\n" )
    else :
      self.runlanbspc.write( "   "+self.lanb_string_printformax.ljust( 50 ) \
                             +"YES\n" )
    self.writelanbspecdummy( 'alive-filename', 'ALIVE.d' )

#  set footer and close file

    self.runlanbspc.write( "END LANCELOT SPECIFICATIONS\n" )
    self.runlanbspc.close( )
    print "new RUNLANB.SPC saved"

#  functions to produce various output lines

  def writelanbspecline_int( self, var, default, line ): 
    if var.get( ) == default :
      if self.lanb_check_writeall.get( ) == 1 :
        if default == 0 :
          self.runlanbspc.write( "!  "+line.ljust( 50 )+"NO\n" )
        else :
          self.runlanbspc.write( "!  "+line.ljust( 50 )+"YES\n" )
    else :
      if default == 0 :
        self.runlanbspc.write( "   "+line.ljust( 50 )+"YES\n" )
      else :
        self.runlanbspc.write( "   "+line.ljust( 50 )+"NO\n" )
    
  def writelanbspecline_string( self, var, string, line ): 
    self.varget = var.get( )
    stringupper = string.upper( )
    if self.varget == string :
      self.runlanbspc.write( "   "+line.ljust( 50 )+stringupper+"\n" )
    else :
      if self.lanb_check_writeall.get( ) == 1 :
        self.runlanbspc.write( "!  "+line.ljust( 50 )+stringupper+"\n" )

  def writelanbspecline_stringval( self, var, default, line ): 
    self.varget = var.get( )
    if self.varget == default or self.varget == "" :
      if self.lanb_check_writeall.get( ) == 1 :
        self.runlanbspc.write( "!  "+line.ljust( 50 )+default+"\n" )
    else :
      self.runlanbspc.write( "   "+line.ljust( 50 )+self.varget+"\n" )

  def writelanbspecdummy( self, line1, line2 ): 
    if self.lanb_check_writeall.get( ) == 1 :
      self.runlanbspc.write( "!  "+line1.ljust( 50 )+line2+"\n" )

#  function to display help
#  ------------------------

  def lanbpresshelp( self ):
    if os.system( 'which xpdf > /dev/null' ) == 0 :
      self.pdfread = 'xpdf'
    elif os.system( 'which acroread > /dev/null' ) == 0 :
      self.pdfread = 'acroread'
    else:
      print 'error: no known pdf file reader' 
      return
    
    self.threads =[ ]
    self.t = threading.Thread( target=self.pdfreadlanbthread )
    self.threads.append( self.t )
#   print self.threads
    self.threads[0].start( )

# display package documentation by opening an external PDF viewer

  def pdfreadlanbthread( self ) :
    os.system( self.pdfread+' $GALAHAD/doc/lancelot.pdf' )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#              General functions for Spec windows           #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# create a warning meessage when selected file does not exist.

  def nospcfile( self, filename, dowhat ) :
    self.nospcfilewarn = Menu( self.root, tearoff=0, font=self.helpfont )
    self.nospcfilewarnlabel = "No file "+filename+" to "+dowhat
    self.nospcfilewarn.add_command( label=self.nospcfilewarnlabel,
                               activebackground=warningbackground,
                               activeforeground=warningforeground,
                               background=warningbackground,
                               foreground=warningforeground
                               )
    try:
      self.nospcfilewarn.tk_popup( 100, 100, 0 )
    finally:
      self.nospcfilewarn.grab_release( )

#  functions to provide space

  def spacer( self ):
    Label( self.specwindow.buttons, width=7,
           text="" ).pack( side=LEFT, fill=BOTH )

  def space( self ):
    Text( self.frame1, borderwidth=0, pady=0,
          width=0, height=0, font=self.bigfont,
          insertwidth=0, highlightthickness=0,
          ).pack( side=TOP )

  def space_select( self ):
    Label( self.selectwindow.buttons, width=7,
           text="" ).pack( side=LEFT, fill=BOTH )

  def space_cdmenu( self ):
    Label( self.cdmenuwindow.buttons, width=9,
           text="" ).pack( side=LEFT, fill=BOTH )

#  function to set string to 1 if auxiliary string is 'yes', or 0 if it is 'no',
#  but otherwise leave it alone

  def yesno( self, string, stringyesno ) :
    if stringyesno.lower( ) == 'yes' or stringyesno.lower( ) == 't' or \
       stringyesno.lower( ) == 'on' or stringyesno.lower( ) == 'true' or \
       stringyesno.lower( ) == '.true.' or stringyesno.lower( ) == 'y' :
      string.set( 1 )
    elif stringyesno.lower( ) == 'no' or stringyesno.lower( ) == 'f' or \
         stringyesno.lower( ) == 'off' or stringyesno.lower( ) == 'false' or \
         stringyesno.lower( ) == '.false.' or stringyesno.lower( ) == 'n' :
      string.set( 0 )
      
#  function to destroy spec window
#  -------------------------------

  def specwindowdestroy( self ):
    self.specwindowopen = 0
    self.specwindow.destroy( )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                         QPA                               #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#   function for QPA spec window
#   -----------------------------------

  def qpaspec( self ):

#  set variables if the first time through

    if self.qpaspec_used == 'no' :
      
#  integer (check-button) variables used with encodings

      self.qpa_check_write_prob = IntVar( )
      self.qpa_string_write_prob = 'write-problem-data'
      self.qpa_default_write_prob = 0

      self.qpa_check_write_initial_sif = IntVar( )
      self.qpa_string_write_initial_sif = 'write-initial-sif'
      self.qpa_default_write_initial_sif = 0

      self.qpa_check_presolve_prob = IntVar( )
      self.qpa_string_presolve_prob = 'pre-solve-problem'
      self.qpa_default_presolve_prob = 1

      self.qpa_check_write_presolve_sif = IntVar( )
      self.qpa_string_write_presolve_sif = 'write-presolved-sif'
      self.qpa_default_write_presolve_sif = 0

      self.qpa_check_fullsol = IntVar( )
      self.qpa_string_fullsol = 'print-full-solution'
      self.qpa_default_fullsol = 0

      self.qpa_check_treat_zero_bnds = IntVar( )
      self.qpa_string_treat_zero_bnds = 'treat-zero-bounds-as-general'
      self.qpa_default_treat_zero_bnds = 0

      self.qpa_check_solve_qp = IntVar( )
      self.qpa_string_solve_qp = 'solve-qp'
      self.qpa_default_solve_qp = 0

      self.qpa_check_randomize = IntVar( )
      self.qpa_string_randomize = 'solve-within-bounds'
      self.qpa_default_randomize = 0

      self.qpa_check_randomize = IntVar( )
      self.qpa_string_randomize = 'temporarily-perturb-constraint-bounds'
      self.qpa_default_randomize = 1

      self.qpa_check_writeall = IntVar( )
      self.qpa_default_writeall = 0

#  string variables used with encodings and defaults

      self.qpa_var_initial_rho_g = StringVar( )
      self.qpa_string_initial_rho_g = 'initial-rho-g'
      self.qpa_default_initial_rho_g = '-1.0'

      self.qpa_var_initial_rho_b = StringVar( )
      self.qpa_string_initial_rho_b = 'initial-rho-b'
      self.qpa_default_initial_rho_b = '-1.0'

      self.qpa_var_scale = StringVar( )
      self.qpa_string_scale = 'scale-problem'
      self.qpa_default_scale = '0'

      self.qpa_var_print_level = StringVar( )
      self.qpa_string_print_level = 'print-level'
      self.qpa_default_print_level = '0'

      self.qpa_var_maxit = StringVar( )
      self.qpa_string_maxit = 'maximum-number-of-iterations'
      self.qpa_default_maxit = '1000'

      self.qpa_var_start_print = StringVar( )
      self.qpa_string_start_print = 'start-print'
      self.qpa_default_start_print = '-1'

      self.qpa_var_stop_print = StringVar( )
      self.qpa_string_stop_print = 'stop-print'
      self.qpa_default_stop_print = '-1'

      self.qpa_var_max_col = StringVar( )
      self.qpa_string_max_col = 'maximum-column-nonzeros-in-schur-complement'
      self.qpa_default_max_col = '35'

      self.qpa_var_max_sc = StringVar( )
      self.qpa_string_max_sc = 'maximum-dimension-of-schur-complement'
      self.qpa_default_max_sc = '75'

      self.qpa_var_intmin = StringVar( )
      self.qpa_string_intmin = 'initial-integer-workspace'
      self.qpa_default_intmin = '1000'

      self.qpa_var_valmin = StringVar( )
      self.qpa_string_valmin = 'initial-real-workspace'
      self.qpa_default_valmin = '1000'

      self.qpa_var_itref_max = StringVar( )
      self.qpa_string_itref_max = 'maximum-refinements'
      self.qpa_default_itref_max = '1'

      self.qpa_var_infeas_check_interval = StringVar( )
      self.qpa_string_infeas_check_interval = \
        'maximum-infeasible-iterations-before-rho-increase'
      self.qpa_default_infeas_check_interval = '100'

      self.qpa_var_cg_maxit = StringVar( )
      self.qpa_string_cg_maxit = 'maximum-number-of-cg-iterations'
      self.qpa_default_cg_maxit = '-1'

      self.qpa_var_full_max_fill = StringVar( )
      self.qpa_string_full_max_fill = 'full-max-fill-ratio'
      self.qpa_default_full_max_fill = '10'

      self.qpa_var_deletion_strategy = StringVar( )
      self.qpa_string_deletion_strategy = 'deletion-strategy'
      self.qpa_default_deletion_strategy = '0'

      self.qpa_var_reestore_prob = StringVar( )
      self.qpa_string_reestore_prob = 'restore-problem-on-output'
      self.qpa_default_reestore_prob = '0'

      self.qpa_var_monitor_resid = StringVar( )
      self.qpa_string_monitor_resid = 'residual-monitor-interval'
      self.qpa_default_monitor_resid = '1'

      self.qpa_var_cold_start = StringVar( )
      self.qpa_string_cold_start = 'cold-start-strategy'
      self.qpa_default_cold_start = '3'

      self.qpa_var_infinity = StringVar( )
      self.qpa_string_infinity = 'infinity-value'
      self.qpa_default_infinity = '1.0D+19'

      self.qpa_var_feas_tol = StringVar( )
      self.qpa_string_feas_tol = 'feasibility-tolerance'
      self.qpa_default_feas_tol = '1.0D-12'

      self.qpa_var_obj_unbounded = StringVar( )
      self.qpa_string_obj_unbounded = 'minimum-objective-before-unbounded'
      self.qpa_default_obj_unbounded = '-1.0D+32'

      self.qpa_var_inc_rho_g_fac = StringVar( )
      self.qpa_string_inc_rho_g_fac = 'increase-rho-g-factor'
      self.qpa_default_inc_rho_g_fac = '2.0'

      self.qpa_var_inc_rho_b_fac = StringVar( )
      self.qpa_string_inc_rho_b_fac = 'increase-rho-b-factor'
      self.qpa_default_inc_rho_b_fac = '2.0'

      self.qpa_var_infeas_g_impfac = StringVar( )
      self.qpa_string_infeas_g_impfac = \
        'infeasible-g-required-improvement-factor'
      self.qpa_default_infeas_g_impfac = '0.75'

      self.qpa_var_infeas_b_impfac = StringVar( )
      self.qpa_string_infeas_b_impfac = \
        'infeasible-b-required-improvement-factor'
      self.qpa_default_infeas_b_impfac = '0.75'

      self.qpa_var_pivtol = StringVar( )
      self.qpa_string_pivtol = 'pivot-tolerance-used'
      self.qpa_default_pivtol = '1.0D-8'

      self.qpa_var_pivtol_dep = StringVar( )
      self.qpa_string_pivtol_dep = 'pivot-tolerance-used-for-dependencies'
      self.qpa_default_pivtol_dep = '0.5'

      self.qpa_var_zero_piv = StringVar( )
      self.qpa_string_zero_piv = 'zero-pivot-tolerance'
      self.qpa_default_zero_piv = '1.0D-12'

      self.qpa_var_multiplier_tol = StringVar( )
      self.qpa_string_multiplier_tol = 'multiplier-tolerance'
      self.qpa_default_multiplier_tol = '1.0D-8'

      self.qpa_var_inner_stop_rel = StringVar( )
      self.qpa_string_inner_stop_rel = \
        'inner-iteration-relative-accuracy-required'
      self.qpa_default_inner_stop_rel = '0.0'

      self.qpa_var_inner_stop_abs = StringVar( )
      self.qpa_string_inner_stop_abs = \
        'inner-iteration-absolute-accuracy-required'
      self.qpa_default_inner_stop_abs = '1.0D-8'

      self.qpa_var_factor = StringVar( )
      self.qpa_string_factor = 'factorization-used'
      self.qpa_default_factor = '0'

      self.qpa_var_precon = StringVar( )
      self.qpa_string_precon = 'preconditioner-used'
      self.qpa_default_precon = '0'

      self.qpa_var_nsemiba = StringVar( )
      self.qpa_var_nsemibb = StringVar( )
      self.qpa_string_nsemib = 'semi-bandwidth-for-band-preconditioner'
      self.qpa_default_nsemiba = '5'
      self.current_nsemiba = self.qpa_default_nsemiba
      self.qpa_default_nsemibb = '5'
      self.current_nsemibb = self.qpa_default_nsemibb

#  read default values

      self.restoreqpadefaults( )
      self.qpa_check_writeall.set( self.qpa_default_writeall )
      self.qpaspec_used = 'yes'

#  setup the window frame
    
    self.specwindow = Toplevel( self.root )
    self.specwindow.geometry( '+100+100' )
    self.specwindow.title( 'QPA spec tool' )
    self.specwindow.menubar = Menu( self.specwindow, tearoff=0  )
    self.specwindow.menubar.add_command( label = "Quit",
                                         command=self.specwindowdestroy )

    self.specwindow.helpmenu = Menu( self.specwindow, tearoff=0 )
    self.specwindow.helpmenu.add_command( label="About",
                                          command = self.qpapresshelp )
    self.specwindow.menubar.add_cascade( label="Help",
                                         menu=self.specwindow.helpmenu )
    self.specwindow.config( menu=self.specwindow.menubar )

#  asseemble (check-button) variables

    self.check = [ self.qpa_check_write_prob,
                   self.qpa_check_write_initial_sif,
                   self.qpa_check_presolve_prob,
                   self.qpa_check_write_presolve_sif,
                   self.qpa_check_fullsol,
                   ]

    self.checkstring = [ "Write problem data",
                         "Write initial SIF file",
                         "Presolve problem",
                         "Write presolved SIF file",
                         "Print full solution"
                         ]

    self.specwindow.varlstart = 0
    self.specwindow.varlstop = len( self.check )

    self.check = self.check+[ self.qpa_check_treat_zero_bnds,
                              self.qpa_check_solve_qp,
                              self.qpa_check_randomize,
                              self.qpa_check_randomize
                              ]

    self.checkstring.extend( [ "Treat zero bounds as general",
                               "Solve qp",
                               "Solve within bounds",
                               "Perturb constraint bounds"
                               ] )
    
    self.specwindow.varrstart = self.specwindow.varlstop
    self.specwindow.varrstop = len( self.check )

#  assemble string variables

    self.var = [ self.qpa_var_print_level,
                 self.qpa_var_start_print,
                 self.qpa_var_stop_print,
                 self.qpa_var_maxit,
                 self.qpa_var_scale,
                 self.qpa_var_initial_rho_g,
                 self.qpa_var_initial_rho_b,
                 self.qpa_var_max_col,
                 self.qpa_var_max_sc,
                 self.qpa_var_intmin,
                 self.qpa_var_valmin,
                 self.qpa_var_itref_max,
                 self.qpa_var_infeas_check_interval,
                 self.qpa_var_cg_maxit,
                 self.qpa_var_full_max_fill,
                 self.qpa_var_deletion_strategy
                ]

    self.varstring = [ "Print level",
                       "Start print at iteration",
                       "Stop printing at iteration",
                       "Maximum number of iterations",
                       "Problem scaling strategy",
                       "Initial constraint infeasibility weight",
                       "Initial bound infeasibility weight",
                       "Max col nonzeros in Schur complement",
                       "Max Schur complement dimension",
                       "Initial integer workspace",
                       "Initial real workspace",
                       "Maximum number of refinements",
                       "Max iterations before weight increase",
                       "Maximum number of CG iterations",
                       "Full max-fill ratio",
                       "Deletion strategy"
                      ]

    self.specwindow.entrytlstart = 0
    self.specwindow.entrytlstop = len( self.var )

    self.var = self.var+[ self.qpa_var_reestore_prob,
                          self.qpa_var_monitor_resid,
                          self.qpa_var_cold_start,
                          self.qpa_var_infinity,
                          self.qpa_var_feas_tol,
                          self.qpa_var_obj_unbounded,
                          self.qpa_var_inc_rho_g_fac,
                          self.qpa_var_inc_rho_b_fac,
                          self.qpa_var_infeas_g_impfac,
                          self.qpa_var_infeas_b_impfac,
                          self.qpa_var_pivtol,
                          self.qpa_var_pivtol_dep,
                          self.qpa_var_zero_piv,
                          self.qpa_var_multiplier_tol,
                          self.qpa_var_inner_stop_rel,
                          self.qpa_var_inner_stop_abs
                         ]

    self.varstring.extend( [ "Restore problem on output",
                             "Residual monitor interval",
                             "Cold start strategy",
                             "Infinite bound value",
                             "Feasibility tolerance",
                             "Min objective before unbounded",
                             "Constraint weight increase factor",
                             "Bound weight increase factor",
                             "Constraint improvement factor",
                             "Bound improvement factor",
                             "Pivot tolerance",
                             "Pivot tolerance for dependencies",
                             "Zero pivot tolerance",
                             "Multiplier tolerance",
                             "Inner-it. relative accuracy required",
                             "Inner-it. absolute accuracy required"
                             ] )

    self.specwindow.entrytrstart = self.specwindow.entrytlstop
    self.specwindow.entrytrstop = len( self.var )

#  Set the name and logo 

    Label( self.specwindow, text="\nQPA OPTIONS\n"
           ).pack( side=TOP, fill=BOTH )

    Label( self.specwindow, image=self.img, relief=SUNKEN
           ).pack( side=TOP, fill=NONE )

    Label( self.specwindow, text="\n"
           ).pack( side=TOP, fill=BOTH )

#  --- set frames  ---

#  main frame

    self.specwindow.frame = Frame( self.specwindow )

#  left and right sub-frames

    self.specwindow.frame.lhs = Frame( self.specwindow.frame )
    self.specwindow.frame.rhs = Frame( self.specwindow.frame )

#  frame to hold check buttons

    self.specwindow.check = Frame( self.specwindow.frame.lhs )

#  sub-frames for check buttons

    self.specwindow.check.left = Frame( self.specwindow.check )
    self.specwindow.check.right = Frame( self.specwindow.check )

#  frame to hold factors and preconditioners check buttons

    self.specwindow.facprec = Frame( self.specwindow.frame.lhs )

#  frame to hold data entry slots

    self.specwindow.text = Frame( self.specwindow.facprec )

# frame and sub-frames to hold data entry slots (top, right)

    self.specwindow.frame.rhs.top = Frame( self.specwindow.frame.rhs )
    self.specwindow.frame.rhs.top.left \
      = Frame( self.specwindow.frame.rhs.top )
    self.specwindow.frame.rhs.top.right \
      = Frame( self.specwindow.frame.rhs.top )

# frame and sub-frames to hold button and data entry slots (bottom, right)

    self.specwindow.frame.rhs.bottom = Frame( self.specwindow.frame.rhs )

# sub-frames to hold selection buttons

    self.specwindow.precon = Frame( self.specwindow.frame.rhs.bottom )

#  --- set contents of frames ---

#  == Left-hand side of window ==

#  contents of check left frame

    for i in range( self.specwindow.varlstart, self.specwindow.varlstop ) :
      Checkbutton( self.specwindow.check.left,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   variable=self.check[i],
                   text=self.checkstring[i]
                   ).pack( side=TOP, fill=BOTH )
    
    self.specwindow.check.left.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.check, width=4,
           text="" ).pack( side=LEFT, fill=BOTH )

#  contents of check right frame

    for i in range( self.specwindow.varrstart, self.specwindow.varrstop ) :
      Checkbutton( self.specwindow.check.right,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   variable=self.check[i],
                   text=self.checkstring[i]
                 ).pack( side=TOP, fill=BOTH )

    self.specwindow.check.right.pack( side=LEFT, fill=BOTH )

#  pack check box

    self.specwindow.check.pack( side=TOP, fill=BOTH )

#  contents of factors and preconditioners frame (label and radio buttons)

    Label( self.specwindow.facprec, width=1, anchor=W,
           text="" ).pack( side=LEFT, fill=NONE )

    Label( self.specwindow.facprec, width=18, anchor=W,
           text="\nFactorization" ).pack( side=TOP, fill=NONE )

    self.specwindow.factors = Frame( self.specwindow.facprec )

    for factors in [ '0', '1', '2' ]:
      if factors == '0' : label = "Automatic"
      elif factors == '1' : label = "Schur complement"
      elif factors == '2' : label = "Augmented system"
      Radiobutton( self.specwindow.factors,
                   highlightthickness=0, relief=FLAT,
                   variable=self.qpa_var_factor,
                   value=factors,
                   text=label
                   ).pack( side=LEFT, fill=BOTH )

    self.specwindow.factors.pack( side=TOP, fill=BOTH )


    Label( self.specwindow.facprec, width=45, anchor=W,
           text="\n      Preconditioner:  Hessian approximation"
           ).pack( side=TOP, fill=BOTH )

    for precons in [ '0', '1', '2', '3', '4', '5' ]:
      if precons == '0' : label = "Automatic"
      elif precons == '1' : label = "Full"
      elif precons == '2' : label = "Identity"
      elif precons == '3' : label = "Band: semibandwidth"
      elif precons == '4' : label = "Reference identity"
      elif precons == '5' : label = "Reference band: semibandwidth"
      if precons == '3' :
        self.specwindow.bandprecon = Frame( self.specwindow.facprec )
        Radiobutton( self.specwindow.bandprecon,
                     highlightthickness=0,
                     relief=FLAT, width=29, anchor=W,
                     variable=self.qpa_var_precon,
                     value=precons,
                     command=self.qpapreconsonoff,
                     text=label
                     ).pack( side=LEFT, fill=NONE )
        Entry( self.specwindow.bandprecon,
               textvariable=self.qpa_var_nsemiba,
               relief=SUNKEN, width=10
               ).pack( side=LEFT, fill=BOTH )
        self.specwindow.bandprecon.pack( side=TOP, fill=BOTH )
      elif precons == '5' :
        self.specwindow.refbandprecon = Frame( self.specwindow.facprec )
        Radiobutton( self.specwindow.refbandprecon,
                     highlightthickness=0,
                     relief=FLAT, width=29, anchor=W,
                     variable=self.qpa_var_precon,
                     value=precons,
                     command=self.qpapreconsonoff,
                     text=label
                     ).pack( side=LEFT, fill=NONE )
        Entry( self.specwindow.refbandprecon,
               textvariable=self.qpa_var_nsemibb,
               relief=SUNKEN, width=10
               ).pack( side=LEFT, fill=BOTH )
        self.specwindow.refbandprecon.pack( side=TOP, fill=BOTH )
      else :
        Radiobutton( self.specwindow.facprec,
                     highlightthickness=0, relief=FLAT,
                     width=29, anchor=W,
                     variable=self.qpa_var_precon,
                     value=precons,
                     command=self.qpapreconsonoff,
                     text=label
                     ).pack( side=TOP, fill=BOTH )

#  pack check boxes

    self.specwindow.facprec.pack( side=TOP, fill=BOTH )

#  Special check button for writeall

    Label( self.specwindow.text, text=" " ).pack( side=TOP, fill=BOTH )
    Checkbutton( self.specwindow.text,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   variable=self.qpa_check_writeall,
                   text="Even write defaults when saving values"
                 ).pack( side=TOP, fill=BOTH )

    self.specwindow.text.pack( side=LEFT, fill=BOTH )

#  == Right-hand side of window ==

#  contents of rhs top left data entry frame

    for i in range( self.specwindow.entrytlstart, self.specwindow.entrytlstop ):
      self.specwindow.i = Frame( self.specwindow.frame.rhs.top.left )
      Label( self.specwindow.i,
             anchor=W, width=35,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.rhs.top.left.pack( side=LEFT, fill=BOTH )
    
#  contents of rhs top right data entry frame

    Label( self.specwindow.frame.rhs.top, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    for i in range( self.specwindow.entrytrstart, self.specwindow.entrytrstop ):
      self.specwindow.i = Frame( self.specwindow.frame.rhs.top.right )
      Label( self.specwindow.i,
             anchor=W, width=32,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.rhs.top.right.pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.rhs.top.pack( side=TOP, fill=BOTH )

#  contents of rhs bottom data entry frame

    Label( self.specwindow.frame.rhs.bottom, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.rhs.bottom.pack( side=TOP, fill=BOTH )

    Label( self.specwindow.frame.rhs, text="\n" ).pack( side=TOP, fill=BOTH )

#  --- assemble boxes ---

#  Pack it all together

    Label( self.specwindow.frame, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.lhs.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.frame, width=4,
           text="" ).pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.rhs.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.frame, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.pack( side=TOP, fill=BOTH )

#  Pack buttons along the bottom

    self.specwindow.buttons = Frame( self.specwindow )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Dismiss\nwindow", 
            command=self.specwindowdestroy
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Edit RUNQPA.SPC\ndirectly", 
            command=self.editqpaspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Read existing\nvalues", 
            command=self.readqpaspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Restore default\nvalues", 
            command=self.restoreqpadefaults
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Save current\nvalues", 
            command=self.writeqpaspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Run QPA\nwith current values",
            command=self.rungaloncurrent
            ).pack( side=LEFT, fill=BOTH )
    self.spacer( )

    self.specwindow.buttons.pack( side=TOP, fill=BOTH )

    Label( self.specwindow, height=1,
           text="\n" ).pack( side=TOP, fill=BOTH )

#  function to edit RUNQPA.SPC
#  ----------------------------

  def editqpaspec( self ):
    if os.path.exists( 'RUNQPA.SPC' ) == 0 :
      print ' no file RUNQPA.SPC to read'
      self.nospcfile( 'RUNQPA.SPC', 'edit' )
      return
    try:
      editor = os.environ["VISUAL"]
    except KeyError:
      try:
        editor = os.environ["EDITOR"]
      except KeyError:
        editor = emacs
    os.popen( editor+' RUNQPA.SPC' )

#  function to restore default spec values
#  ----------------------------------------

  def restoreqpadefaults( self ):
    self.qpa_check_write_prob.set( self.qpa_default_write_prob )
    self.qpa_check_write_initial_sif.set( self.qpa_default_write_initial_sif )
    self.qpa_check_presolve_prob.set( self.qpa_default_presolve_prob )
    self.qpa_check_write_presolve_sif.set( self.qpa_default_write_presolve_sif )
    self.qpa_check_fullsol.set( self.qpa_default_fullsol )
    self.qpa_check_treat_zero_bnds.set( self.qpa_default_treat_zero_bnds )
    self.qpa_check_solve_qp.set( self.qpa_default_solve_qp )
    self.qpa_check_randomize.set( self.qpa_default_randomize )
    self.qpa_check_randomize.set( self.qpa_default_randomize )

    self.qpa_var_initial_rho_g.set( self.qpa_default_initial_rho_g )
    self.qpa_var_initial_rho_b.set( self.qpa_default_initial_rho_b )
    self.qpa_var_scale.set( self.qpa_default_scale )
    self.qpa_var_print_level.set( self.qpa_default_print_level )
    self.qpa_var_maxit.set( self.qpa_default_maxit )
    self.qpa_var_start_print.set( self.qpa_default_start_print )
    self.qpa_var_stop_print.set( self.qpa_default_stop_print )
    self.qpa_var_max_col.set( self.qpa_default_max_col )
    self.qpa_var_max_sc.set( self.qpa_default_max_sc )
    self.qpa_var_intmin.set( self.qpa_default_intmin )
    self.qpa_var_valmin.set( self.qpa_default_valmin )
    self.qpa_var_itref_max.set( self.qpa_default_itref_max )
    self.qpa_var_infeas_check_interval.set( 
      self.qpa_default_infeas_check_interval )
    self.qpa_var_cg_maxit.set( self.qpa_default_cg_maxit )
    self.qpa_var_full_max_fill.set( self.qpa_default_full_max_fill )
    self.qpa_var_deletion_strategy.set( self.qpa_default_deletion_strategy )
    self.qpa_var_reestore_prob.set( self.qpa_default_reestore_prob )
    self.qpa_var_monitor_resid.set( self.qpa_default_monitor_resid )
    self.qpa_var_cold_start.set( self.qpa_default_cold_start )
    self.qpa_var_infinity.set( self.qpa_default_infinity )
    self.qpa_var_feas_tol.set( self.qpa_default_feas_tol )
    self.qpa_var_obj_unbounded.set( self.qpa_default_obj_unbounded )
    self.qpa_var_inc_rho_g_fac.set( self.qpa_default_inc_rho_g_fac )
    self.qpa_var_inc_rho_b_fac.set( self.qpa_default_inc_rho_b_fac )
    self.qpa_var_infeas_g_impfac.set( self.qpa_default_infeas_g_impfac )
    self.qpa_var_infeas_b_impfac.set( self.qpa_default_infeas_b_impfac )
    self.qpa_var_pivtol.set( self.qpa_default_pivtol )
    self.qpa_var_pivtol_dep.set( self.qpa_default_pivtol_dep )
    self.qpa_var_zero_piv.set( self.qpa_default_zero_piv )
    self.qpa_var_multiplier_tol.set( self.qpa_default_multiplier_tol )
    self.qpa_var_inner_stop_rel.set( self.qpa_default_inner_stop_rel )
    self.qpa_var_inner_stop_abs.set( self.qpa_default_inner_stop_abs )
    self.qpa_var_factor.set( self.qpa_default_factor )
    self.qpa_var_precon.set( self.qpa_default_precon )
    self.qpa_var_nsemiba.set( self.qpa_default_nsemiba )
    self.qpa_var_nsemibb.set( self.qpa_default_nsemibb )

    self.current_nsemiba = self.qpa_default_nsemiba
    self.current_nsemibb = self.qpa_default_nsemibb
    self.qpapreconsonoff( )
  
#  function to switch on and off semibandwidth/line-more vectors as appropriate
#  ----------------------------------------------------------------------------

  def qpapreconsonoff( self ): 
    if self.qpa_var_precon.get( ) == '3' :
      self.qpa_var_nsemiba.set( self.current_nsemiba )
    else:
      if self.qpa_var_nsemiba.get( ) != '' :
        self.current_nsemiba = self.qpa_var_nsemiba.get( )
      self.qpa_var_nsemiba.set( '' )
    if self.qpa_var_precon.get( ) == '5' :
      self.qpa_var_nsemibb.set( self.current_nsemibb )
    else:
      if self.qpa_var_nsemibb.get( ) != '' :
        self.current_nsemibb = self.qpa_var_nsemibb.get( )
      self.qpa_var_nsemibb.set( '' )

#  function to read the current values to the spec file
#  -----------------------------------------------------

  def readqpaspec( self ): 

#  open file and set header

    if os.path.exists( 'RUNQPA.SPC' ) == 0 :
      print ' no file RUNQPA.SPC to read'
      self.nospcfile( 'RUNQPA.SPC', 'read' )
      return
    self.runqpaspc = open( 'RUNQPA.SPC', 'r' )

#  Restore default values

    self.restoreqpadefaults( )

#  loop over lines of files

    self.readyes = 0
    for line in self.runqpaspc:

#  exclude comments

      if line[0] == '!' : continue

#  convert the line to lower case, and remove leading and trailing blanks

      line = line.lower( ) 
      line = line.strip( )
      blank_start = line.find( ' ' ) 
      
      if blank_start != -1 :
        stringc = line[0:blank_start]
      else :
        stringc = line

#  look for string variables to set

      blank_end = line.rfind( ' ' ) 
      if blank_start == -1 :
        stringd = 'YES'
      else:
        stringd = line[ blank_end + 1 : ].upper( )
#     print stringc+' '+stringd

#  only read those segments concerned with QPA

      if stringc == 'begin' and line.find( 'qpa' ) >= 0 : self.readyes = 1
      if stringc == 'end' and line.find( 'qpa' ) >= 0 : self.readyes = 0
      if self.readyes == 0 : continue

#  exclude begin and end lines

      if stringc == 'begin' or stringc == 'end' : continue

#  look for integer (check-button) variables to set

      if stringc == self.qpa_string_write_prob :
        self.yesno( self.qpa_check_write_prob, stringd )
        continue
      elif stringc == self.qpa_string_write_initial_sif :
        self.yesno( self.qpa_check_write_initial_sif, stringd )
        continue
      elif stringc == self.qpa_string_presolve_prob :
        self.yesno( self.qpa_check_presolve_prob, stringd )
        continue
      elif stringc == self.qpa_string_write_presolve_sif :
        self.yesno( self.qpa_check_write_presolve_sif, stringd )
        continue
      elif stringc == self.qpa_string_fullsol :
        self.yesno( self.qpa_check_fullsol, stringd )
        continue
      elif stringc == self.qpa_string_treat_zero_bnds :
        self.yesno( self.qpa_check_treat_zero_bnds, stringd )
        continue
      elif stringc == self.qpa_string_solve_qp :
        self.yesno( self.qpa_check_solve_qp, stringd )
        continue
      elif stringc == self.qpa_string_randomize :
        self.yesno( self.qpa_check_randomize, stringd )
        continue
      elif stringc == self.qpa_string_randomize :
        self.yesno( self.qpa_check_randomize, stringd )
        continue

      if stringc == self.qpa_string_factor :
        stringd = stringd.lower( ) 
        if stringd == '1' :
          self.qpa_var_factor.set( '1' )
        elif stringd == '2' :
          self.qpa_var_factor.set( '2' )
        elif stringd == '3' :
          self.qpa_var_factor.set( '3' )
        continue
      elif stringc == self.qpa_string_precon :
        stringd = stringd.lower( ) 
        if stringd == '0':
          self.qpa_var_precon.set( '0' )
        elif stringd == '1':
          self.qpa_var_precon.set( '1' )
        elif stringd == '2':
          self.qpa_var_precon.set( '2' ) 
        elif stringd == '3':
          self.qpa_var_precon.set( '3' )
        elif stringd == '4':
          self.qpa_var_precon.set( '4' )
        elif stringd == '5':
          self.qpa_var_precon.set( '5' )
        continue
      elif stringc == self.qpa_string_nsemib :
        self.qpa_var_nsemiba.set( stringd )
        self.current_nsemiba = stringd
        self.qpa_var_nsemibb.set( stringd )
        self.current_nsemibb = stringd
        continue
      elif stringc == self.qpa_string_initial_rho_g :
        self.qpa_var_initial_rho_g.set( stringd )
        continue
      elif stringc == self.qpa_string_initial_rho_b :
        self.qpa_var_initial_rho_b.set( stringd )
        continue
      elif stringc == self.qpa_string_scale :
        self.qpa_var_scale.set( stringd )
        continue
      elif stringc == self.qpa_string_print_level :
        self.qpa_var_print_level.set( stringd )
        continue
      elif stringc == self.qpa_string_maxit :
        self.qpa_var_maxit.set( stringd )
        continue
      elif stringc == self.qpa_string_start_print :
        self.qpa_var_start_print.set( stringd )
        continue
      elif stringc == self.qpa_string_stop_print :
        self.qpa_var_stop_print.set( stringd )
        continue
      elif stringc == self.qpa_string_max_col :
        self.qpa_var_max_col.set( stringd )
        continue
      elif stringc == self.qpa_string_max_sc :
        self.qpa_var_max_sc.set( stringd )
        continue
      elif stringc == self.qpa_string_intmin :
        self.qpa_var_intmin.set( stringd )
        continue
      elif stringc == self.qpa_string_valmin :
        self.qpa_var_valmin.set( stringd )
        continue
      elif stringc == self.qpa_string_itref_max :
        self.qpa_var_itref_max.set( stringd )
        continue
      elif stringc == self.qpa_string_infeas_check_interval :
        self.qpa_var_infeas_check_interval.set( stringd )
        continue
      elif stringc == self.qpa_string_cg_maxit :
        self.qpa_var_cg_maxit.set( stringd )
        continue
      elif stringc == self.qpa_string_full_max_fill :
        self.qpa_var_full_max_fill.set( stringd )
        continue
      elif stringc == self.qpa_string_deletion_strategy :
        self.qpa_var_deletion_strategy.set( stringd )
        continue
      elif stringc == self.qpa_string_reestore_prob :
        self.qpa_var_reestore_prob.set( stringd )
        continue
      elif stringc == self.qpa_string_monitor_resid :
        self.qpa_var_monitor_resid.set( stringd )
        continue
      elif stringc == self.qpa_string_cold_start :
        self.qpa_var_cold_start.set( stringd )
        continue
      elif stringc == self.qpa_string_infinity :
        self.qpa_var_infinity.set( stringd )
        continue
      elif stringc == self.qpa_string_feas_tol :
        self.qpa_var_feas_tol.set( stringd )
        continue
      elif stringc == self.qpa_string_obj_unbounded :
        self.qpa_var_obj_unbounded.set( stringd )
        continue
      elif stringc == self.qpa_string_inc_rho_g_fac :
        self.qpa_var_inc_rho_g_fac.set( stringd )
        continue
      elif stringc == self.qpa_string_inc_rho_b_fac :
        self.qpa_var_inc_rho_b_fac.set( stringd )
        continue
      elif stringc == self.qpa_string_infeas_g_impfac :
        self.qpa_var_infeas_g_impfac.set( stringd )
        continue
      elif stringc == self.qpa_string_infeas_b_impfac :
        self.qpa_var_infeas_b_impfac.set( stringd )
        continue
      elif stringc == self.qpa_string_pivtol :
        self.qpa_var_pivtol.set( stringd )
        continue
      elif stringc == self.qpa_string_pivtol_dep :
        self.qpa_var_pivtol_dep.set( stringd )
        continue
      elif stringc == self.qpa_string_zero_piv :
        self.qpa_var_zero_piv.set( stringd )
        continue
      elif stringc == self.qpa_string_multiplier_tol :
        self.qpa_var_multiplier_tol.set( stringd )
        continue
      elif stringc == self.qpa_string_inner_stop_rel :
        self.qpa_var_inner_stop_rel.set( stringd )
        continue
      elif stringc == self.qpa_string_inner_stop_abs :
        self.qpa_var_inner_stop_abs.set( stringd )
        continue

    self.qpapreconsonoff( )
    self.runqpaspc.close( )

#  function to write the current values to the spec file
#  -----------------------------------------------------

  def writeqpaspec( self ): 

#  open file and set header

    self.runqpaspc = open( 'RUNQPA.SPC', 'w' )

#  record RUNQPA options

    self.runqpaspc.write( "BEGIN RUNQPA SPECIFICATIONS\n" )
    self.writeqpaspecline_int( self.qpa_check_write_prob,
                            self.qpa_default_write_prob, 
                            self.qpa_string_write_prob )
    self.writeqpaspecdummy( 'problem-data-file-name', 'QPA.data' )
    self.writeqpaspecdummy( 'problem-data-file-device', '26' )
    self.writeqpaspecline_int( self.qpa_check_write_initial_sif,
                            self.qpa_default_write_initial_sif, 
                            self.qpa_string_write_initial_sif )
    self.writeqpaspecdummy( 'initial-sif-file-name', 'INITIAL.SIF' )
    self.writeqpaspecdummy( 'initial-sif-file-device', '51' )
    self.writeqpaspecline_stringval( self.qpa_var_initial_rho_g,
                                  self.qpa_default_initial_rho_g,
                                  self.qpa_string_initial_rho_g )
    self.writeqpaspecline_stringval( self.qpa_var_initial_rho_b,
                                  self.qpa_default_initial_rho_b,
                                  self.qpa_string_initial_rho_b )
    self.writeqpaspecline_stringval( self.qpa_var_scale,
                                  self.qpa_default_scale,
                                  self.qpa_string_scale )
    self.writeqpaspecline_int( self.qpa_check_presolve_prob,
                            self.qpa_default_presolve_prob,
                            self.qpa_string_presolve_prob )
    self.writeqpaspecline_int( self.qpa_check_write_presolve_sif,
                            self.qpa_default_write_presolve_sif, 
                            self.qpa_string_write_presolve_sif )
    self.writeqpaspecdummy( 'presolved-sif-file-name', 'PRESOLVE.SIF' )
    self.writeqpaspecdummy( 'presolved-sif-file-device', '52' )
    self.writeqpaspecdummy( 'solve-problem', 'YES' )
    self.writeqpaspecline_int( self.qpa_check_fullsol,
                            self.qpa_default_fullsol,
                            self.qpa_string_fullsol )
    self.writeqpaspecdummy( 'write-solution', 'YES' )
    self.writeqpaspecdummy( 'solution-file-name', 'QPASOL.d' )
    self.writeqpaspecdummy( 'solution-file-device', '62' )
    self.writeqpaspecdummy( 'write-result-summary', 'YES' )
    self.writeqpaspecdummy( 'result-summary-file-name', 'QPARES.d' )
    self.writeqpaspecdummy( 'result-summary-file-device', '47' )
    self.runqpaspc.write( "END RUNQPA SPECIFICATIONS\n\n" )

#  record QPA options

    self.runqpaspc.write( "BEGIN QPA SPECIFICATIONS\n" )

    self.writeqpaspecdummy( 'error-printout-device', '6' )
    self.writeqpaspecdummy( 'printout-device', '6' )
    self.writeqpaspecline_stringval( self.qpa_var_print_level,
                                  self.qpa_default_print_level,
                                  self.qpa_string_print_level )
    self.writeqpaspecline_stringval( self.qpa_var_maxit,
                                  self.qpa_default_maxit,
                                  self.qpa_string_maxit )
    self.writeqpaspecline_stringval( self.qpa_var_start_print,
                                  self.qpa_default_start_print,
                                  self.qpa_string_start_print )
    self.writeqpaspecline_stringval( self.qpa_var_stop_print,
                                  self.qpa_default_stop_print,
                                  self.qpa_string_stop_print )

#  record factorization chosen

    if self.qpa_var_factor.get( ) == "0" :
      if self.qpa_check_writeall.get( ) == 1 :
        self.runqpaspc.write( "!  "+self.qpa_string_factor.ljust( 50 ) \
                               +"0\n" )
    elif self.qpa_var_factor.get( ) == "1" :
      self.writeqpaspecline_string( self.qpa_var_factor, "1",
                                 self.qpa_string_factor )
    elif self.qpa_var_factor.get( ) == "2" :
      self.writeqpaspecline_string( self.qpa_var_factor, "2",
                                 self.qpa_string_factor )

#  record further options

    self.writeqpaspecline_stringval( self.qpa_var_max_col,
                                  self.qpa_default_max_col,
                                  self.qpa_string_max_col )
    self.writeqpaspecline_stringval( self.qpa_var_max_sc,
                                  self.qpa_default_max_sc,
                                  self.qpa_string_max_sc )
    self.writeqpaspecline_stringval( self.qpa_var_intmin,
                                  self.qpa_default_intmin,
                                  self.qpa_string_intmin )
    self.writeqpaspecline_stringval( self.qpa_var_valmin,
                                  self.qpa_default_valmin,
                                  self.qpa_string_valmin )
    self.writeqpaspecline_stringval( self.qpa_var_itref_max,
                                  self.qpa_default_itref_max,
                                  self.qpa_string_itref_max )
    self.writeqpaspecline_stringval( self.qpa_var_infeas_check_interval,
                                  self.qpa_default_infeas_check_interval,
                                  self.qpa_string_infeas_check_interval )
    self.writeqpaspecline_stringval( self.qpa_var_cg_maxit,
                                  self.qpa_default_cg_maxit,
                                  self.qpa_string_cg_maxit )

#  record preconditioner chosen

    if self.qpa_var_precon.get( ) == "0" :
      if self.qpa_check_writeall.get( ) == 1 :
        self.runqpaspc.write( "!  "+self.qpa_string_precon.ljust( 50 ) \
                               +"0\n" )
        self.runqpaspc.write( "!  "+self.qpa_string_nsemib.ljust( 50 ) \
                               +self.qpa_default_nsemiba+"\n" )
    elif self.qpa_var_precon.get( ) == "1" :
      self.writeqpaspecline_string( self.qpa_var_precon,
                                 "1",
                                 self.qpa_string_precon )
      if self.qpa_check_writeall.get( ) == 1 :
        self.runqpaspc.write( "!  "+self.qpa_string_nsemib.ljust( 50 ) \
                               +self.qpa_default_nsemiba+"\n" )
    elif self.qpa_var_precon.get( ) == "2" :
      self.writeqpaspecline_string( self.qpa_var_precon,
                                 "2",
                                 self.qpa_string_precon )
      if self.qpa_check_writeall.get( ) == 1 :
        self.runqpaspc.write( "!  "+self.qpa_string_nsemib.ljust( 50 ) \
                               +self.qpa_default_nsemiba+"\n" )
    elif self.qpa_var_precon.get( ) == "3" :
      self.writeqpaspecline_string( self.qpa_var_precon,
                                 "3",
                                 self.qpa_string_precon )
      self.writeqpaspecline_stringval( self.qpa_var_nsemiba,
                                    self.qpa_default_nsemiba,
                                    self.qpa_string_nsemib )
    elif self.qpa_var_precon.get( ) == "4" :
      self.writeqpaspecline_string( self.qpa_var_precon,
                                 "4",
                                 self.qpa_string_precon )
      if self.qpa_check_writeall.get( ) == 1 :
        self.runqpaspc.write( "!  "+self.qpa_string_nsemib.ljust( 50 ) \
                               +self.qpa_default_nsemiba+"\n" )
    elif self.qpa_var_precon.get( ) == "5" :
      self.writeqpaspecline_string( self.qpa_var_precon,
                                 "5",
                                 self.qpa_string_precon )
      self.writeqpaspecline_stringval( self.qpa_var_nsemibb, 
                                    self.qpa_default_nsemibb,
                                    self.qpa_string_nsemib )

#  record remaining options

    self.writeqpaspecline_stringval( self.qpa_var_full_max_fill, 
                                  self.qpa_default_full_max_fill,
                                  self.qpa_string_full_max_fill )
    self.writeqpaspecline_stringval( self.qpa_var_deletion_strategy,
                                  self.qpa_default_deletion_strategy,
                                  self.qpa_string_deletion_strategy )
    self.writeqpaspecline_stringval( self.qpa_var_reestore_prob,
                                  self.qpa_default_reestore_prob,
                                  self.qpa_string_reestore_prob )
    self.writeqpaspecline_stringval( self.qpa_var_monitor_resid,
                                  self.qpa_default_monitor_resid,
                                  self.qpa_string_monitor_resid )
    self.writeqpaspecline_stringval( self.qpa_var_cold_start,
                                  self.qpa_default_cold_start,
                                  self.qpa_string_cold_start )
    self.writeqpaspecline_stringval( self.qpa_var_infinity,
                                  self.qpa_default_infinity,
                                  self.qpa_string_infinity )
    self.writeqpaspecline_stringval( self.qpa_var_feas_tol,
                                  self.qpa_default_feas_tol,
                                  self.qpa_string_feas_tol )
    self.writeqpaspecline_stringval( self.qpa_var_obj_unbounded,
                                  self.qpa_default_obj_unbounded,
                                  self.qpa_string_obj_unbounded )
    self.writeqpaspecline_stringval( self.qpa_var_inc_rho_g_fac,
                                  self.qpa_default_inc_rho_g_fac,
                                  self.qpa_string_inc_rho_g_fac )
    self.writeqpaspecline_stringval( self.qpa_var_inc_rho_b_fac,
                                  self.qpa_default_inc_rho_b_fac,
                                  self.qpa_string_inc_rho_b_fac )
    self.writeqpaspecline_stringval( self.qpa_var_infeas_g_impfac,
                                  self.qpa_default_infeas_g_impfac,
                                  self.qpa_string_infeas_g_impfac )
    self.writeqpaspecline_stringval( self.qpa_var_infeas_b_impfac,
                                  self.qpa_default_infeas_b_impfac,
                                  self.qpa_string_infeas_b_impfac )
    self.writeqpaspecline_stringval( self.qpa_var_pivtol,
                                  self.qpa_default_pivtol,
                                  self.qpa_string_pivtol )
    self.writeqpaspecline_stringval( self.qpa_var_pivtol_dep,
                                  self.qpa_default_pivtol_dep,
                                  self.qpa_string_pivtol_dep )
    self.writeqpaspecline_stringval( self.qpa_var_zero_piv,
                                  self.qpa_default_zero_piv,
                                  self.qpa_string_zero_piv )
    self.writeqpaspecline_stringval( self.qpa_var_multiplier_tol,
                                  self.qpa_default_multiplier_tol,
                                  self.qpa_string_multiplier_tol )
    self.writeqpaspecline_stringval( self.qpa_var_inner_stop_rel,
                                  self.qpa_default_inner_stop_rel,
                                  self.qpa_string_inner_stop_rel )
    self.writeqpaspecline_stringval( self.qpa_var_inner_stop_abs,
                                  self.qpa_default_inner_stop_abs,
                                  self.qpa_string_inner_stop_abs )
    self.writeqpaspecline_int( self.qpa_check_treat_zero_bnds,
                            self.qpa_default_treat_zero_bnds,
                            self.qpa_string_treat_zero_bnds )
    self.writeqpaspecline_int( self.qpa_check_solve_qp,
                            self.qpa_default_solve_qp, 
                            self.qpa_string_solve_qp )
    self.writeqpaspecline_int( self.qpa_check_randomize,
                            self.qpa_default_randomize, 
                            self.qpa_string_randomize )
    self.writeqpaspecline_int( self.qpa_check_randomize,
                            self.qpa_default_randomize, 
                            self.qpa_string_randomize )
    self.writeqpaspecdummy( 'array-syntax-worse-than-do-loop', 'NO' )
    self.runqpaspc.write( "END QPA SPECIFICATIONS\n\n" )

#  If required, record PRESOLVE options

    self.runqpaspc.write( "BEGIN PRESOLVE SPECIFICATIONS\n" )
    self.writeqpaspecdummy( 'printout-device', '6' )
    self.writeqpaspecdummy( 'error-printout-device', '6' )
    self.writeqpaspecdummy( 'print-level', 'TRACE' )
    self.writeqpaspecdummy( 'presolve-termination-strategy', 'REDUCED_SIZE' )
    self.writeqpaspecdummy( 'maximum-number-of-transformations', '1000000' )
    self.writeqpaspecdummy( 'maximum-number-of-passes', '25' )
    self.writeqpaspecdummy( 'constraints-accuracy', '1.0D-6' )
    self.writeqpaspecdummy( 'dual-variables-accuracy', '1.0D-6' )
    self.writeqpaspecdummy( 'allow-dual-transformations', 'YES' )
    self.writeqpaspecdummy( 'remove-redundant-variables-constraints', 'YES' )
    self.writeqpaspecdummy( 'primal-constraints-analysis-frequency', '1' )
    self.writeqpaspecdummy( 'dual-constraints-analysis-frequency', '1' )
    self.writeqpaspecdummy( 'singleton-columns-analysis-frequency', '1' )
    self.writeqpaspecdummy( 'doubleton-columns-analysis-frequency', '1' )
    self.writeqpaspecdummy( 'unconstrained-variables-analysis-frequency', '1' )
    self.writeqpaspecdummy( 'dependent-variables-analysis-frequency', '1' )
    self.writeqpaspecdummy( 'row-sparsification-frequency', '1' )
    self.writeqpaspecdummy( 'maximum-percentage-row-fill', '-1' )
    self.writeqpaspecdummy( 'transformations-buffer-size', '50000' )
    self.writeqpaspecdummy( 'transformations-file-device', '52' )
    self.writeqpaspecdummy( 'transformations-file-status', 'KEEP' )
    self.writeqpaspecdummy( 'transformations-file-name', 'transf.sav' )
    self.writeqpaspecdummy( 'primal-feasibility-check', 'NONE' )
    self.writeqpaspecdummy( 'dual-feasibility-check', 'NONE' )
    self.writeqpaspecdummy( 'active-multipliers-sign', 'POSITIVE' )
    self.writeqpaspecdummy( 'inactive-multipliers-value', 'LEAVE_AS_IS' )
    self.writeqpaspecdummy( 'active-dual-variables-sign', 'POSITIVE' )
    self.writeqpaspecdummy( 'inactive-dual-variables-value', 'LEAVE_AS_IS' )
    self.writeqpaspecdummy( 'primal-variables-bound-status', 'TIGHTEST' )
    self.writeqpaspecdummy( 'dual-variables-bound-status', 'TIGHTEST' )
    self.writeqpaspecdummy( 'constraints-bound-status', 'TIGHTEST' )
    self.writeqpaspecdummy( 'multipliers-bound-status', 'TIGHTEST' )
    self.writeqpaspecdummy( 'infinity-value', '1.0D19' )
    self.writeqpaspecdummy( 'pivoting-threshold', '1.10D-10' )
    self.writeqpaspecdummy( 'minimum-relative-bound-improvement', '1.0D-10' )
    self.writeqpaspecdummy( 'maximum-growth-factor', '1.0D8' )
    self.writeqpaspecdummy( 'compute-quadratic-value', 'YES' )
    self.writeqpaspecdummy( 'compute-objective-constant', 'YES' )
    self.writeqpaspecdummy( 'compute-gradient', 'YES' )
    self.writeqpaspecdummy( 'compute-Hessian', 'YES' )
    self.writeqpaspecdummy( 'compute-constraints-matrix', 'YES' )
    self.writeqpaspecdummy( 'compute-primal-variables-values', 'YES' )
    self.writeqpaspecdummy( 'compute-primal-variables-bounds', 'YES' )
    self.writeqpaspecdummy( 'compute-dual-variables-values', 'YES' )
    self.writeqpaspecdummy( 'compute-dual-variables-bounds', 'YES' )
    self.writeqpaspecdummy( 'compute-constraints-values', 'YES' )
    self.writeqpaspecdummy( 'compute-constraints-bounds', 'YES' )
    self.writeqpaspecdummy( 'compute-multipliers-values', 'YES' )
    self.writeqpaspecdummy( 'compute-multipliers-bounds', 'YES' )
    self.runqpaspc.write( "END PRESOLVE SPECIFICATIONS\n" )

#  close file
   
    self.runqpaspc.close( )
    print "new RUNQPA.SPC saved"

#  functions to produce various output lines

  def writeqpaspecline_int( self, var, default, line ): 
    if var.get( ) == default :
      if self.qpa_check_writeall.get( ) == 1 :
        if default == 0 :
          self.runqpaspc.write( "!  "+line.ljust( 50 )+"NO\n" )
        else :
          self.runqpaspc.write( "!  "+line.ljust( 50 )+"YES\n" )
    else :
      if default == 0 :
        self.runqpaspc.write( "   "+line.ljust( 50 )+"YES\n" )
      else :
        self.runqpaspc.write( "   "+line.ljust( 50 )+"NO\n" )
    
  def writeqpaspecline_string( self, var, string, line ): 
    self.varget = var.get( )
    stringupper = string.upper( )
    if self.varget == string :
      self.runqpaspc.write( "   "+line.ljust( 50 )+stringupper+"\n" )
    else :
      if self.qpa_check_writeall.get( ) == 1 :
        self.runqpaspc.write( "!  "+line.ljust( 50 )+stringupper+"\n" )

  def writeqpaspecline_stringval( self, var, default, line ): 
    self.varget = var.get( )
    if self.varget == default or self.varget == "" :
      if self.qpa_check_writeall.get( ) == 1 :
        self.runqpaspc.write( "!  "+line.ljust( 50 )+default+"\n" )
    else :
      self.runqpaspc.write( "   "+line.ljust( 50 )+self.varget+"\n" )

  def writeqpaspecdummy( self, line1, line2 ): 
    if self.qpa_check_writeall.get( ) == 1 :
      self.runqpaspc.write( "!  "+line1.ljust( 50 )+line2+"\n" )

#  function to display help
#  ------------------------

  def qpapresshelp( self ):
    if os.system( 'which xpdf > /dev/null' ) == 0 :
      self.pdfread = 'xpdf'
    elif os.system( 'which acroread > /dev/null' ) == 0 :
      self.pdfread = 'acroread'
    else:
      print 'error: no known pdf file reader' 
      return
    
    self.threads =[ ]
    self.t = threading.Thread( target=self.pdfreadqpathread )
    self.threads.append( self.t )
#   print self.threads
    self.threads[0].start( )

# display package documentation by opening an external PDF viewer

  def pdfreadqpathread( self ) :
    os.system( self.pdfread+' $GALAHAD/doc/qpa.pdf' )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                         QPB                               #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#   function for QPB spec window
#   -----------------------------------

  def qpbspec( self ):

#  set variables if the first time through

    if self.qpbspec_used == 'no' :
      
#  integer (check-button) variables used with encodings

      self.qpb_check_write_prob = IntVar( )
      self.qpb_string_write_prob = 'write-problem-data'
      self.qpb_default_write_prob = 0

      self.qpb_check_write_initial_sif = IntVar( )
      self.qpb_string_write_initial_sif = 'write-initial-sif'
      self.qpb_default_write_initial_sif = 0

      self.qpb_check_presolve_problem = IntVar( )
      self.qpb_string_presolve_problem = 'pre-solve-problem'
      self.qpb_default_presolve_problem = 1

      self.qpb_check_write_presolve_sif = IntVar( )
      self.qpb_string_write_presolve_sif = 'write-presolved-sif'
      self.qpb_default_write_presolve_sif = 0

      self.qpb_check_fullsol = IntVar( )
      self.qpb_string_fullsol = 'print-full-solution'
      self.qpb_default_fullsol = 0

      self.qpb_check_treat_zero_as_gen = IntVar( )
      self.qpb_string_treat_zero_as_gen = 'treat-zero-bounds-as-general'
      self.qpb_default_treat_zero_as_gen = 0

      self.qpb_check_lsqp = IntVar( )
      self.qpb_string_lsqp = 'least-squares-qp'
      self.qpb_default_lsqp = 0

      self.qpb_check_remove_deps = IntVar( )
      self.qpb_string_remove_deps = 'remove-linear-dependencies'
      self.qpb_default_remove_deps = 1

      self.qpb_check_center = IntVar( )
      self.qpb_string_center = 'start-at-analytic-center'
      self.qpb_default_center = 1

      self.qpb_check_primal = IntVar( )
      self.qpb_string_primal = 'primal-barrier-used'
      self.qpb_default_primal = 0

      self.qpb_check_feasol = IntVar( )
      self.qpb_string_feasol = 'move-final-solution-onto-bound'
      self.qpb_default_feasol = 0

      self.qpb_check_writeall = IntVar( )
      self.qpb_default_writeall = 0

#  string variables used with encodings and defaults

      self.qpb_var_scale = StringVar( )
      self.qpb_string_scale = 'scale-problem'
      self.qpb_default_scale = '0'

      self.qpb_var_print_level = StringVar( )
      self.qpb_string_print_level = 'print-level'
      self.qpb_default_print_level = '0'

      self.qpb_var_maxit = StringVar( )
      self.qpb_string_maxit = 'maximum-number-of-iterations'
      self.qpb_default_maxit = '1000'

      self.qpb_var_start_print = StringVar( )
      self.qpb_string_start_print = 'start-print'
      self.qpb_default_start_print = '-1'

      self.qpb_var_stop_print = StringVar( )
      self.qpb_string_stop_print = 'stop-print'
      self.qpb_default_stop_print = '-1'

      self.qpb_var_max_col = StringVar( )
      self.qpb_string_max_col = 'maximum-column-nonzeros-in-schur-complement'
      self.qpb_default_max_col = '35'

      self.qpb_var_infeas_max = StringVar( )
      self.qpb_string_infeas_max = 'maximum-poor-iterations-before-infeasible'
      self.qpb_default_infeas_max = '200'

      self.qpb_var_sindmin = StringVar( )
      self.qpb_string_sindmin = 'initial-integer-workspace'
      self.qpb_default_sindmin = '1000'

      self.qpb_var_valmin = StringVar( )
      self.qpb_string_valmin = 'initial-real-workspace'
      self.qpb_default_valmin = '1000'

      self.qpb_var_itref_max = StringVar( )
      self.qpb_string_itref_max = 'maximum-refinements'
      self.qpb_default_itref_max = '1'

      self.qpb_var_reduce_infeas = StringVar( )
      self.qpb_string_reduce_infeas = 'poor-iteration-tolerance'
      self.qpb_default_reduce_infeas = '0.98'

      self.qpb_var_cg_maxit = StringVar( )
      self.qpb_string_cg_maxit = 'maximum-number-of-cg-iterations'
      self.qpb_default_cg_maxit = '-1'

      self.qpb_var_identical_bnds = StringVar( )
      self.qpb_string_identical_bnds = 'identical-bounds-tolerance'
      self.qpb_default_identical_bnds = '1.0D-15'

      self.qpb_var_initial_radius = StringVar( )
      self.qpb_string_initial_radius = 'initial-trust-region-radius'
      self.qpb_default_initial_radius = '-1.0'

      self.qpb_var_restore_prob = StringVar( )
      self.qpb_string_restore_prob = 'restore-problem-on-output'
      self.qpb_default_restore_prob = '0'

      self.qpb_var_prfeas = StringVar( )
      self.qpb_string_prfeas = 'mininum-initial-primal-feasibility'
      self.qpb_default_prfeas = '1.0'

      self.qpb_var_dufeas = StringVar( )
      self.qpb_string_dufeas = 'mininum-initial-dual-feasibility'
      self.qpb_default_dufeas = '1.0'

      self.qpb_var_infinity = StringVar( )
      self.qpb_string_infinity = 'infinity-value'
      self.qpb_default_infinity = '1.0D+19'

      self.qpb_var_inner_frac_opt = StringVar( )
      self.qpb_string_inner_frac_opt = \
        'inner-iteration-fraction-optimality-required'
      self.qpb_default_inner_frac_opt = '0.1'

      self.qpb_var_obj_unbounded = StringVar( )
      self.qpb_string_obj_unbounded = 'minimum-objective-before-unbounded'
      self.qpb_default_obj_unbounded = '-1.0D+32'

      self.qpb_var_stop_p = StringVar( )
      self.qpb_string_stop_p = 'primal-accuracy-required'
      self.qpb_default_stop_p = '1.0D-4'

      self.qpb_var_stop_d = StringVar( )
      self.qpb_string_stop_d = 'dual-accuracy-required'
      self.qpb_default_stop_d = '1.0D-4'

      self.qpb_var_stop_c = StringVar( )
      self.qpb_string_stop_c = 'complementary-slackness-accuracy-required'
      self.qpb_default_stop_c = '1.0D-4'

      self.qpb_var_muzero = StringVar( )
      self.qpb_string_muzero = 'initial-barrier-parameter'
      self.qpb_default_muzero = '-1.0'

      self.qpb_var_pivtol = StringVar( )
      self.qpb_string_pivtol = 'pivot-tolerance-used'
      self.qpb_default_pivtol = '1.0D-8'

      self.qpb_var_pivtol_deps = StringVar( )
      self.qpb_string_pivtol_deps = 'pivot-tolerance-used-for-dependencies'
      self.qpb_default_pivtol_deps = '0.5'

      self.qpb_var_zero_pivot = StringVar( )
      self.qpb_string_zero_pivot = 'zero-pivot-tolerance'
      self.qpb_default_zero_pivot = '1.0D-12'

      self.qpb_var_inner_stop_rel = StringVar( )
      self.qpb_string_inner_stop_rel = \
        'inner-iteration-relative-accuracy-required'
      self.qpb_default_inner_stop_rel = '0.0'

      self.qpb_var_inner_stop_abs = StringVar( )
      self.qpb_string_inner_stop_abs = \
        'inner-iteration-absolute-accuracy-required'
      self.qpb_default_inner_stop_abs = '1.0D-8'

      self.qpb_var_factor = StringVar( )
      self.qpb_string_factor = 'factorization-used'
      self.qpb_default_factor = '0'

      self.qpb_var_precon = StringVar( )
      self.qpb_string_precon = 'preconditioner-used'
      self.qpb_default_precon = '0'

      self.qpb_var_nsemiba = StringVar( )
      self.qpb_var_nsemibb = StringVar( )
      self.qpb_string_nsemib = 'semi-bandwidth-for-band-preconditioner'
      self.qpb_default_nsemiba = '5'
      self.current_nsemiba = self.qpb_default_nsemiba
      self.qpb_default_nsemibb = '5'
      self.current_nsemibb = self.qpb_default_nsemibb

#  read default values

      self.restoreqpbdefaults( )
      self.qpb_check_writeall.set( self.qpb_default_writeall )
      self.qpbspec_used = 'yes'

#  setup the window frame
    
    self.specwindow = Toplevel( self.root )
    self.specwindow.geometry( '+100+100' )
    self.specwindow.title( 'QPB / LSQP spec tool' )
    self.specwindow.menubar = Menu( self.specwindow, tearoff=0  )
    self.specwindow.menubar.add_command( label = "Quit",
                                         command=self.specwindowdestroy )

    self.specwindow.helpmenu = Menu( self.specwindow, tearoff=0 )
    self.specwindow.helpmenu.add_command( label="About",
                                          command = self.qpbpresshelp )
    self.specwindow.menubar.add_cascade( label="Help",
                                         menu=self.specwindow.helpmenu )
    self.specwindow.config( menu=self.specwindow.menubar )

#  asseemble (check-button) variables

    self.check = [ self.qpb_check_write_prob,
                   self.qpb_check_write_initial_sif,
                   self.qpb_check_presolve_problem,
                   self.qpb_check_write_presolve_sif,
                   self.qpb_check_fullsol,
                   self.qpb_check_treat_zero_as_gen
                   ]

    self.checkstring = [ "Write problem data",
                         "Write initial SIF file",
                         "Presolve problem",
                         "Write presolved SIF file",
                         "Print full solution",
                         "Treat zero bounds as general"
                         ]

    self.specwindow.varlstart = 0
    self.specwindow.varlstop = len( self.check )

    self.check = self.check+[ self.qpb_check_lsqp,
                              self.qpb_check_remove_deps,
                              self.qpb_check_center,
                              self.qpb_check_primal,
                              self.qpb_check_feasol
                              ]

    self.checkstring.extend( [ "Separable QP (use LSQP)",
                               "Remove linear dependencies",
                               "Start at analytic center",
                               "Use primal barrier",
                               "Move solution onto bounds"
                               ] )
    
    self.specwindow.varrstart = self.specwindow.varlstop
    self.specwindow.varrstop = len( self.check )

#  assemble string variables

    self.var = [ self.qpb_var_print_level,
                 self.qpb_var_start_print,
                 self.qpb_var_stop_print,
                 self.qpb_var_maxit,
                 self.qpb_var_scale,
                 self.qpb_var_stop_p,
                 self.qpb_var_stop_d,
                 self.qpb_var_stop_c,
                 self.qpb_var_initial_radius,
                 self.qpb_var_sindmin,
                 self.qpb_var_valmin,
                 self.qpb_var_itref_max,
                 self.qpb_var_cg_maxit,
                 self.qpb_var_infeas_max,
                 self.qpb_var_reduce_infeas
                ]

    self.varstring = [ "Print level",
                       "Start print at iteration",
                       "Stop printing at iteration",
                       "Maximum number of iterations",
                       "Problem scaling strategy",
                       "Primal accuracy required",
                       "Dual accuracy required",
                       "Comp slackness accuracy required",
                       "Initial trust-region radius",
                       "Initial integer workspace",
                       "Initial real workspace",
                       "Maximum number of refinements",
                       "Maximum number of CG iterations",
                       "Max poor iterations before infeasible",
                       "Poor-iteration tolerance"
                      ]

    self.specwindow.entrytlstart = 0
    self.specwindow.entrytlstop = len( self.var )

    self.var = self.var+[ self.qpb_var_max_col,
                          self.qpb_var_restore_prob,
                          self.qpb_var_prfeas,
                          self.qpb_var_dufeas,
                          self.qpb_var_infinity,
                          self.qpb_var_obj_unbounded,
                          self.qpb_var_muzero,
                          self.qpb_var_pivtol,
                          self.qpb_var_pivtol_deps,
                          self.qpb_var_zero_pivot,
                          self.qpb_var_identical_bnds,
                          self.qpb_var_inner_stop_rel,
                          self.qpb_var_inner_stop_abs,
                          self.qpb_var_inner_frac_opt,
                         ]

    self.varstring.extend( [ "Max col nonzeros in Schur complement",
                             "Restore problem on output",
                             "Min initial primal feasibility",
                             "Min initial dual feasibility",
                             "Infinite bound value",
                             "Min objective before unbounded",
                             "Initial barrier parameter",
                             "Pivot tolerance",
                             "Pivot tolerance for dependencies",
                             "Zero pivot tolerance",
                             "Identical bounds tolerance",
                             "Inner-it. relative accuracy required",
                             "Inner-it. absolute accuracy required",
                             "Inner-it. fraction optimality required"
                             ] )

    self.specwindow.entrytrstart = self.specwindow.entrytlstop
    self.specwindow.entrytrstop = len( self.var )

#  Set the name and logo 

    Label( self.specwindow, text="\nQPB / LSQP OPTIONS\n"
           ).pack( side=TOP, fill=BOTH )

    Label( self.specwindow, image=self.img, relief=SUNKEN
           ).pack( side=TOP, fill=NONE )

    Label( self.specwindow, text="\n"
           ).pack( side=TOP, fill=BOTH )

#  --- set frames  ---

#  main frame

    self.specwindow.frame = Frame( self.specwindow )

#  left and right sub-frames

    self.specwindow.frame.lhs = Frame( self.specwindow.frame )
    self.specwindow.frame.rhs = Frame( self.specwindow.frame )

#  frame to hold check buttons

    self.specwindow.check = Frame( self.specwindow.frame.lhs )

#  sub-frames for check buttons

    self.specwindow.check.left = Frame( self.specwindow.check )
    self.specwindow.check.right = Frame( self.specwindow.check )

#  frame to hold factors and preconditioners check buttons

    self.specwindow.facprec = Frame( self.specwindow.frame.lhs )

# frame and sub-frames to hold data entry slots (top, right)

    self.specwindow.frame.rhs.top = Frame( self.specwindow.frame.rhs )
    self.specwindow.frame.rhs.top.left \
      = Frame( self.specwindow.frame.rhs.top )
    self.specwindow.frame.rhs.top.right \
      = Frame( self.specwindow.frame.rhs.top )

#  frame to hold data entry slots

    self.specwindow.text = Frame( self.specwindow.frame.rhs.top.right )

# frame and sub-frames to hold button and data entry slots (bottom, right)

    self.specwindow.frame.rhs.bottom = Frame( self.specwindow.frame.rhs )

# sub-frames to hold selection buttons

    self.specwindow.precon = Frame( self.specwindow.frame.rhs.bottom )

#  --- set contents of frames ---

#  == Left-hand side of window ==

#  contents of check left frame

    for i in range( self.specwindow.varlstart, self.specwindow.varlstop ) :
      Checkbutton( self.specwindow.check.left,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   variable=self.check[i],
                   text=self.checkstring[i]
                   ).pack( side=TOP, fill=BOTH )
    
    self.specwindow.check.left.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.check, width=2,
           text="" ).pack( side=LEFT, fill=BOTH )

#  contents of check right frame

    for i in range( self.specwindow.varrstart, self.specwindow.varrstop ) :
      Checkbutton( self.specwindow.check.right,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   variable=self.check[i],
                   text=self.checkstring[i]
                 ).pack( side=TOP, fill=BOTH )

    self.specwindow.check.right.pack( side=LEFT, fill=BOTH )

#  pack check box

    self.specwindow.check.pack( side=TOP, fill=BOTH )

#  contents of factors and preconditioners frame (label and radio buttons)

    Label( self.specwindow.facprec, width=2, anchor=W,
           text="" ).pack( side=LEFT, fill=NONE )

    Label( self.specwindow.facprec, width=22, anchor=W,
           text="\nFactorization" ).pack( side=TOP, fill=NONE )

    self.specwindow.factors = Frame( self.specwindow.facprec )

    for factors in [ '0', '1', '2' ]:
      if factors == '0' : label = "Automatic"
      elif factors == '1' : label = "Schur complement"
      elif factors == '2' : label = "Augmented system"
      Radiobutton( self.specwindow.factors,
                   highlightthickness=0, relief=FLAT,
                   variable=self.qpb_var_factor,
                   value=factors,
                   text=label
                   ).pack( side=LEFT, fill=BOTH )

    self.specwindow.factors.pack( side=TOP, fill=BOTH )


    Label( self.specwindow.facprec, width=45, anchor=W,
           text="\n      Preconditioner:  Hessian approximation"
           ).pack( side=TOP, fill=BOTH )

    for precons in [ '0', '1', '2', '3', '4' ]:
      if precons == '0' : label = "Automatic"
      elif precons == '1' : label = "Identity"
      elif precons == '2' : label = "Full"
      elif precons == '3' : label = "Band: semibandwidth"
      elif precons == '4' : label = "Barrier terms"
      if precons == '3' :
        self.specwindow.bandprecon = Frame( self.specwindow.facprec )
        Radiobutton( self.specwindow.bandprecon,
                     highlightthickness=0,
                     relief=FLAT, width=19, anchor=W,
                     variable=self.qpb_var_precon,
                     value=precons,
                     command=self.qpbpreconsonoff,
                     text=label
                     ).pack( side=LEFT, fill=NONE )
        Entry( self.specwindow.bandprecon,
               textvariable=self.qpb_var_nsemiba,
               relief=SUNKEN, width=10
               ).pack( side=LEFT, fill=BOTH )
        self.specwindow.bandprecon.pack( side=TOP, fill=BOTH )
      else :
        Radiobutton( self.specwindow.facprec,
                     highlightthickness=0, relief=FLAT,
                     width=19, anchor=W,
                     variable=self.qpb_var_precon,
                     value=precons,
                     command=self.qpbpreconsonoff,
                     text=label
                     ).pack( side=TOP, fill=BOTH )

#  pack check boxes

    self.specwindow.facprec.pack( side=TOP, fill=BOTH )

#  == Right-hand side of window ==

#  contents of rhs top left data entry frame

    for i in range( self.specwindow.entrytlstart, self.specwindow.entrytlstop ):
      self.specwindow.i = Frame( self.specwindow.frame.rhs.top.left )
      Label( self.specwindow.i,
             anchor=W, width=32,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.rhs.top.left.pack( side=LEFT, fill=BOTH )
    
#  contents of rhs top right data entry frame

    Label( self.specwindow.frame.rhs.top, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    for i in range( self.specwindow.entrytrstart, self.specwindow.entrytrstop ):
      self.specwindow.i = Frame( self.specwindow.frame.rhs.top.right )
      Label( self.specwindow.i,
             anchor=W, width=35,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.rhs.top.right.pack( side=LEFT, fill=BOTH )

#  Special check button for writeall

    Label( self.specwindow.text, text="\n " ).pack( side=TOP, fill=BOTH )
    Checkbutton( self.specwindow.text,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   variable=self.qpb_check_writeall,
                   text="Even write defaults when saving values"
                 ).pack( side=TOP, fill=BOTH )

    self.specwindow.text.pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.rhs.top.pack( side=TOP, fill=BOTH )

#  contents of rhs bottom data entry frame

    Label( self.specwindow.frame.rhs.bottom, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.rhs.bottom.pack( side=TOP, fill=BOTH )

#   Label( self.specwindow.frame.rhs, text="\n" ).pack( side=TOP, fill=BOTH )

#  --- assemble boxes ---

#  Pack it all together

    Label( self.specwindow.frame, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.lhs.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.frame, width=4,
           text="" ).pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.rhs.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.frame, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.pack( side=TOP, fill=BOTH )

#  Pack buttons along the bottom

    self.specwindow.buttons = Frame( self.specwindow )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Dismiss\nwindow", 
            command=self.specwindowdestroy
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Edit RUNQPB.SPC\ndirectly", 
            command=self.editqpbspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Read existing\nvalues", 
            command=self.readqpbspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Restore default\nvalues", 
            command=self.restoreqpbdefaults
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Save current\nvalues", 
            command=self.writeqpbspec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Run QPB\nwith current values",
            command=self.rungaloncurrent
            ).pack( side=LEFT, fill=BOTH )
    self.spacer( )

    self.specwindow.buttons.pack( side=TOP, fill=BOTH )

    Label( self.specwindow, height=1,
           text="\n" ).pack( side=TOP, fill=BOTH )

#  function to edit RUNQPB.SPC
#  ----------------------------

  def editqpbspec( self ):
    if os.path.exists( 'RUNQPB.SPC' ) == 0 :
      print ' no file RUNQPB.SPC to read'
      self.nospcfile( 'RUNQPB.SPC', 'edit' )
      return
    try:
      editor = os.environ["VISUAL"]
    except KeyError:
      try:
        editor = os.environ["EDITOR"]
      except KeyError:
        editor = emacs
    os.popen( editor+' RUNQPB.SPC' )

#  function to restore default spec values
#  ----------------------------------------

  def restoreqpbdefaults( self ):
    self.qpb_check_write_prob.set( self.qpb_default_write_prob )
    self.qpb_check_write_initial_sif.set( self.qpb_default_write_initial_sif )
    self.qpb_check_presolve_problem.set( self.qpb_default_presolve_problem )
    self.qpb_check_write_presolve_sif.set( self.qpb_default_write_presolve_sif )
    self.qpb_check_fullsol.set( self.qpb_default_fullsol )
    self.qpb_check_treat_zero_as_gen.set( self.qpb_default_treat_zero_as_gen )
    self.qpb_check_lsqp.set( self.qpb_default_lsqp )
    self.qpb_check_remove_deps.set( self.qpb_default_remove_deps )
    self.qpb_check_center.set( self.qpb_default_center )
    self.qpb_check_primal.set( self.qpb_default_primal )
    self.qpb_check_feasol.set( self.qpb_default_feasol )

    self.qpb_var_scale.set( self.qpb_default_scale )
    self.qpb_var_print_level.set( self.qpb_default_print_level )
    self.qpb_var_maxit.set( self.qpb_default_maxit )
    self.qpb_var_start_print.set( self.qpb_default_start_print )
    self.qpb_var_stop_print.set( self.qpb_default_stop_print )
    self.qpb_var_max_col.set( self.qpb_default_max_col )
    self.qpb_var_infeas_max.set( self.qpb_default_infeas_max )
    self.qpb_var_sindmin.set( self.qpb_default_sindmin )
    self.qpb_var_valmin.set( self.qpb_default_valmin )
    self.qpb_var_itref_max.set( self.qpb_default_itref_max )
    self.qpb_var_reduce_infeas.set( self.qpb_default_reduce_infeas )
    self.qpb_var_cg_maxit.set( self.qpb_default_cg_maxit )
    self.qpb_var_identical_bnds.set( self.qpb_default_identical_bnds )
    self.qpb_var_initial_radius.set( self.qpb_default_initial_radius )
    self.qpb_var_restore_prob.set( self.qpb_default_restore_prob )
    self.qpb_var_prfeas.set( self.qpb_default_prfeas )
    self.qpb_var_dufeas.set( self.qpb_default_dufeas )
    self.qpb_var_infinity.set( self.qpb_default_infinity )
    self.qpb_var_inner_frac_opt.set( self.qpb_default_inner_frac_opt )
    self.qpb_var_obj_unbounded.set( self.qpb_default_obj_unbounded )
    self.qpb_var_stop_p.set( self.qpb_default_stop_p )
    self.qpb_var_stop_d.set( self.qpb_default_stop_d )
    self.qpb_var_stop_c.set( self.qpb_default_stop_c )
    self.qpb_var_muzero.set( self.qpb_default_muzero )
    self.qpb_var_pivtol.set( self.qpb_default_pivtol )
    self.qpb_var_pivtol_deps.set( self.qpb_default_pivtol_deps )
    self.qpb_var_zero_pivot.set( self.qpb_default_zero_pivot )
    self.qpb_var_inner_stop_rel.set( self.qpb_default_inner_stop_rel )
    self.qpb_var_inner_stop_abs.set( self.qpb_default_inner_stop_abs )
    self.qpb_var_factor.set( self.qpb_default_factor )
    self.qpb_var_precon.set( self.qpb_default_precon )
    self.qpb_var_nsemiba.set( self.qpb_default_nsemiba )
    self.qpb_var_nsemibb.set( self.qpb_default_nsemibb )

    self.current_nsemiba = self.qpb_default_nsemiba
    self.current_nsemibb = self.qpb_default_nsemibb
    self.qpbpreconsonoff( )
  
#  function to switch on and off semibandwidth/line-more vectors as appropriate
#  ----------------------------------------------------------------------------

  def qpbpreconsonoff( self ): 
    if self.qpb_var_precon.get( ) == '3' :
      self.qpb_var_nsemiba.set( self.current_nsemiba )
    else:
      if self.qpb_var_nsemiba.get( ) != '' :
        self.current_nsemiba = self.qpb_var_nsemiba.get( )
      self.qpb_var_nsemiba.set( '' )

#  function to read the current values to the spec file
#  -----------------------------------------------------

  def readqpbspec( self ): 

#  open file and set header

    if os.path.exists( 'RUNQPB.SPC' ) == 0 :
      print ' no file RUNQPB.SPC to read'
      self.nospcfile( 'RUNQPB.SPC', 'read' )
      return
    self.runqpbspc = open( 'RUNQPB.SPC', 'r' )

#  Restore default values

    self.restoreqpbdefaults( )

#  loop over lines of files

    self.readyes = 0
    for line in self.runqpbspc:

#  exclude comments

      if line[0] == '!' : continue

#  convert the line to lower case, and remove leading and trailing blanks

      line = line.lower( ) 
      line = line.strip( )
      blank_start = line.find( ' ' ) 
      
      if blank_start != -1 :
        stringc = line[0:blank_start]
      else :
        stringc = line

#  look for string variables to set

      blank_end = line.rfind( ' ' ) 
      if blank_start == -1 :
        stringd = 'YES'
      else:
        stringd = line[ blank_end + 1 : ].upper( )

#  only read those segments concerned with QPB

      if stringc == 'begin' and line.find( 'qpb' ) >= 0 : self.readyes = 1
      if stringc == 'end' and line.find( 'qpb' ) >= 0 : self.readyes = 0
      if self.readyes == 0 : continue

#  exclude begin and end lines

      if stringc == 'begin' or stringc == 'end' : continue

#  look for integer (check-button) variables to set

      if stringc == self.qpb_string_write_prob :
        self.yesno( self.qpb_check_write_prob, stringd )
        continue
      elif stringc == self.qpb_string_write_initial_sif :
        self.yesno( self.qpb_check_write_initial_sif, stringd )
        continue
      elif stringc == self.qpb_string_presolve_problem :
        self.yesno( self.qpb_check_presolve_problem, stringd )
        continue
      elif stringc == self.qpb_string_write_presolve_sif :
        self.yesno( self.qpb_check_write_presolve_sif, stringd )
        continue
      elif stringc == self.qpb_string_fullsol :
        self.yesno( self.qpb_check_fullsol, stringd )
        continue
      elif stringc == self.qpb_string_treat_zero_as_gen :
        self.yesno( self.qpb_check_treat_zero_as_gen, stringd )
        continue
      elif stringc == self.qpb_string_lsqp :
        self.yesno( self.qpb_check_lsqp, stringd )
        continue
      elif stringc == self.qpb_string_remove_deps :
        self.yesno( self.qpb_check_remove_deps, stringd )
        continue
      elif stringc == self.qpb_string_center :
        self.yesno( self.qpb_check_center, stringd )
        continue
      elif stringc == self.qpb_string_primal :
        self.yesno( self.qpb_check_primal, stringd )
        continue
      elif stringc == self.qpb_string_feasol :
        self.yesno( self.qpb_check_feasol, stringd )
        continue

      if stringc == self.qpb_string_factor :
        stringd = stringd.lower( ) 
        if stringd == '1' :
          self.qpb_var_factor.set( '1' )
        elif stringd == '2' :
          self.qpb_var_factor.set( '2' )
        elif stringd == '3' :
          self.qpb_var_factor.set( '3' )
        continue
      elif stringc == self.qpb_string_precon :
        stringd = stringd.lower( ) 
        if stringd == '0':
          self.qpb_var_precon.set( '0' )
        elif stringd == '1':
          self.qpb_var_precon.set( '1' )
        elif stringd == '2':
          self.qpb_var_precon.set( '2' ) 
        elif stringd == '3':
          self.qpb_var_precon.set( '3' )
        elif stringd == '4':
          self.qpb_var_precon.set( '4' )
        elif stringd == '5':
          self.qpb_var_precon.set( '5' )
        continue
      elif stringc == self.qpb_string_nsemib :
        self.qpb_var_nsemiba.set( stringd )
        self.current_nsemiba = stringd
        self.qpb_var_nsemibb.set( stringd )
        self.current_nsemibb = stringd
        continue
      elif stringc == self.qpb_string_scale :
        self.qpb_var_scale.set( stringd )
        continue
      elif stringc == self.qpb_string_print_level :
        self.qpb_var_print_level.set( stringd )
        continue
      elif stringc == self.qpb_string_maxit :
        self.qpb_var_maxit.set( stringd )
        continue
      elif stringc == self.qpb_string_start_print :
        self.qpb_var_start_print.set( stringd )
        continue
      elif stringc == self.qpb_string_stop_print :
        self.qpb_var_stop_print.set( stringd )
        continue
      elif stringc == self.qpb_string_max_col :
        self.qpb_var_max_col.set( stringd )
        continue
      elif stringc == self.qpb_string_infeas_max :
        self.qpb_var_infeas_max.set( stringd )
        continue
      elif stringc == self.qpb_string_sindmin :
        self.qpb_var_sindmin.set( stringd )
        continue
      elif stringc == self.qpb_string_valmin :
        self.qpb_var_valmin.set( stringd )
        continue
      elif stringc == self.qpb_string_itref_max :
        self.qpb_var_itref_max.set( stringd )
        continue
      elif stringc == self.qpb_string_reduce_infeas :
        self.qpb_var_reduce_infeas.set( stringd )
        continue
      elif stringc == self.qpb_string_cg_maxit :
        self.qpb_var_cg_maxit.set( stringd )
        continue
      elif stringc == self.qpb_string_identical_bnds :
        self.qpb_var_identical_bnds.set( stringd )
        continue
      elif stringc == self.qpb_string_initial_radius :
        self.qpb_var_initial_radius.set( stringd )
        continue
      elif stringc == self.qpb_string_restore_prob :
        self.qpb_var_restore_prob.set( stringd )
        continue
      elif stringc == self.qpb_string_prfeas :
        self.qpb_var_prfeas.set( stringd )
        continue
      elif stringc == self.qpb_string_dufeas :
        self.qpb_var_dufeas.set( stringd )
        continue
      elif stringc == self.qpb_string_infinity :
        self.qpb_var_infinity.set( stringd )
        continue
      elif stringc == self.qpb_string_inner_frac_opt :
        self.qpb_var_inner_frac_opt.set( stringd )
        continue
      elif stringc == self.qpb_string_obj_unbounded :
        self.qpb_var_obj_unbounded.set( stringd )
        continue
      elif stringc == self.qpb_string_stop_p :
        self.qpb_var_stop_p.set( stringd )
        continue
      elif stringc == self.qpb_string_stop_d :
        self.qpb_var_stop_d.set( stringd )
        continue
      elif stringc == self.qpb_string_stop_c :
        self.qpb_var_stop_c.set( stringd )
        continue
      elif stringc == self.qpb_string_muzero :
        self.qpb_var_muzero.set( stringd )
        continue
      elif stringc == self.qpb_string_pivtol :
        self.qpb_var_pivtol.set( stringd )
        continue
      elif stringc == self.qpb_string_pivtol_deps :
        self.qpb_var_pivtol_deps.set( stringd )
        continue
      elif stringc == self.qpb_string_zero_pivot :
        self.qpb_var_zero_pivot.set( stringd )
        continue
      elif stringc == self.qpb_string_inner_stop_rel :
        self.qpb_var_inner_stop_rel.set( stringd )
        continue
      elif stringc == self.qpb_string_inner_stop_abs :
        self.qpb_var_inner_stop_abs.set( stringd )
        continue

    self.qpbpreconsonoff( )
    self.runqpbspc.close( )

#  function to write the current values to the spec file
#  -----------------------------------------------------

  def writeqpbspec( self ): 

#  open file and set header

    self.runqpbspc = open( 'RUNQPB.SPC', 'w' )

#  record RUNQPB options

    self.runqpbspc.write( "BEGIN RUNQPB SPECIFICATIONS\n" )
    self.writeqpbspecline_int( self.qpb_check_write_prob,
                            self.qpb_default_write_prob, 
                            self.qpb_string_write_prob )
    self.writeqpbspecdummy( 'problem-data-file-name', 'QPB.data' )
    self.writeqpbspecdummy( 'problem-data-file-device', '26' )
    self.writeqpbspecline_int( self.qpb_check_write_initial_sif,
                            self.qpb_default_write_initial_sif, 
                            self.qpb_string_write_initial_sif )
    self.writeqpbspecdummy( 'initial-sif-file-name', 'INITIAL.SIF' )
    self.writeqpbspecdummy( 'initial-sif-file-device', '51' )
    self.writeqpbspecline_int( self.qpb_check_lsqp,
                            self.qpb_default_lsqp, 
                            self.qpb_string_lsqp )
    self.writeqpbspecline_stringval( self.qpb_var_scale,
                                  self.qpb_default_scale,
                                  self.qpb_string_scale )
    self.writeqpbspecline_int( self.qpb_check_presolve_problem,
                            self.qpb_default_presolve_problem,
                            self.qpb_string_presolve_problem )
    self.writeqpbspecline_int( self.qpb_check_write_presolve_sif,
                            self.qpb_default_write_presolve_sif, 
                            self.qpb_string_write_presolve_sif )
    self.writeqpbspecdummy( 'presolved-sif-file-name', 'PRESOLVE.SIF' )
    self.writeqpbspecdummy( 'presolved-sif-file-device', '52' )
    self.writeqpbspecdummy( 'solve-problem', 'YES' )
    self.writeqpbspecline_int( self.qpb_check_fullsol,
                            self.qpb_default_fullsol,
                            self.qpb_string_fullsol )
    self.writeqpbspecdummy( 'write-solution', 'YES' )
    self.writeqpbspecdummy( 'solution-file-name', 'QPBSOL.d' )
    self.writeqpbspecdummy( 'solution-file-device', '62' )
    self.writeqpbspecdummy( 'write-result-summary', 'YES' )
    self.writeqpbspecdummy( 'result-summary-file-name', 'QPBRES.d' )
    self.writeqpbspecdummy( 'result-summary-file-device', '47' )
    self.runqpbspc.write( "END RUNQPB SPECIFICATIONS\n\n" )

#  record QPB options

    self.runqpbspc.write( "BEGIN QPB SPECIFICATIONS\n" )

    self.writeqpbspecdummy( 'error-printout-device', '6' )
    self.writeqpbspecdummy( 'printout-device', '6' )
    self.writeqpbspecline_stringval( self.qpb_var_print_level,
                                  self.qpb_default_print_level,
                                  self.qpb_string_print_level )
    self.writeqpbspecline_stringval( self.qpb_var_maxit,
                                  self.qpb_default_maxit,
                                  self.qpb_string_maxit )
    self.writeqpbspecline_stringval( self.qpb_var_start_print,
                                  self.qpb_default_start_print,
                                  self.qpb_string_start_print )
    self.writeqpbspecline_stringval( self.qpb_var_stop_print,
                                  self.qpb_default_stop_print,
                                  self.qpb_string_stop_print )

#  record factorization chosen

    if self.qpb_var_factor.get( ) == "0" :
      if self.qpb_check_writeall.get( ) == 1 :
        self.runqpbspc.write( "!  "+self.qpb_string_factor.ljust( 50 ) \
                               +"0\n" )
    elif self.qpb_var_factor.get( ) == "1" :
      self.writeqpbspecline_string( self.qpb_var_factor, "1",
                                 self.qpb_string_factor )
    elif self.qpb_var_factor.get( ) == "2" :
      self.writeqpbspecline_string( self.qpb_var_factor, "2",
                                 self.qpb_string_factor )

#  record further options

    self.writeqpbspecline_stringval( self.qpb_var_max_col,
                                  self.qpb_default_max_col,
                                  self.qpb_string_max_col )
    self.writeqpbspecline_stringval( self.qpb_var_sindmin,
                                  self.qpb_default_sindmin,
                                  self.qpb_string_sindmin )
    self.writeqpbspecline_stringval( self.qpb_var_valmin,
                                  self.qpb_default_valmin,
                                  self.qpb_string_valmin )
    self.writeqpbspecline_stringval( self.qpb_var_itref_max,
                                  self.qpb_default_itref_max,
                                  self.qpb_string_itref_max )
    self.writeqpbspecline_stringval( self.qpb_var_infeas_max,
                                  self.qpb_default_infeas_max,
                                  self.qpb_string_infeas_max )
    self.writeqpbspecline_stringval( self.qpb_var_cg_maxit,
                                  self.qpb_default_cg_maxit,
                                  self.qpb_string_cg_maxit )

#  record preconditioner chosen

    if self.qpb_var_precon.get( ) == "0" :
      if self.qpb_check_writeall.get( ) == 1 :
        self.runqpbspc.write( "!  "+self.qpb_string_precon.ljust( 50 ) \
                               +"0\n" )
        self.runqpbspc.write( "!  "+self.qpb_string_nsemib.ljust( 50 ) \
                               +self.qpb_default_nsemiba+"\n" )
    elif self.qpb_var_precon.get( ) == "1" :
      self.writeqpbspecline_string( self.qpb_var_precon,
                                 "1",
                                 self.qpb_string_precon )
      if self.qpb_check_writeall.get( ) == 1 :
        self.runqpbspc.write( "!  "+self.qpb_string_nsemib.ljust( 50 ) \
                               +self.qpb_default_nsemiba+"\n" )
    elif self.qpb_var_precon.get( ) == "2" :
      self.writeqpbspecline_string( self.qpb_var_precon,
                                 "2",
                                 self.qpb_string_precon )
      if self.qpb_check_writeall.get( ) == 1 :
        self.runqpbspc.write( "!  "+self.qpb_string_nsemib.ljust( 50 ) \
                               +self.qpb_default_nsemiba+"\n" )
    elif self.qpb_var_precon.get( ) == "3" :
      self.writeqpbspecline_string( self.qpb_var_precon,
                                 "3",
                                 self.qpb_string_precon )
      self.writeqpbspecline_stringval( self.qpb_var_nsemiba,
                                    self.qpb_default_nsemiba,
                                    self.qpb_string_nsemib )
    elif self.qpb_var_precon.get( ) == "4" :
      self.writeqpbspecline_string( self.qpb_var_precon,
                                 "4",
                                 self.qpb_string_precon )
      if self.qpb_check_writeall.get( ) == 1 :
        self.runqpbspc.write( "!  "+self.qpb_string_nsemib.ljust( 50 ) \
                               +self.qpb_default_nsemiba+"\n" )

#  record remaining options

    self.writeqpbspecline_stringval( self.qpb_var_restore_prob,
                                  self.qpb_default_restore_prob,
                                  self.qpb_string_restore_prob )
    self.writeqpbspecline_stringval( self.qpb_var_infinity,
                                  self.qpb_default_infinity,
                                  self.qpb_string_infinity )
    self.writeqpbspecline_stringval( self.qpb_var_stop_p,
                                  self.qpb_default_stop_p,
                                  self.qpb_string_stop_p )
    self.writeqpbspecline_stringval( self.qpb_var_stop_d,
                                  self.qpb_default_stop_d,
                                  self.qpb_string_stop_d )
    self.writeqpbspecline_stringval( self.qpb_var_stop_c,
                                  self.qpb_default_stop_c,
                                  self.qpb_string_stop_c )
    self.writeqpbspecline_stringval( self.qpb_var_prfeas,
                                  self.qpb_default_prfeas,
                                  self.qpb_string_prfeas )
    self.writeqpbspecline_stringval( self.qpb_var_dufeas,
                                  self.qpb_default_dufeas,
                                  self.qpb_string_dufeas )
    self.writeqpbspecline_stringval( self.qpb_var_muzero,
                                  self.qpb_default_muzero,
                                  self.qpb_string_muzero )
    self.writeqpbspecline_stringval( self.qpb_var_reduce_infeas,
                                  self.qpb_default_reduce_infeas,
                                  self.qpb_string_reduce_infeas )
    self.writeqpbspecline_stringval( self.qpb_var_obj_unbounded,
                                  self.qpb_default_obj_unbounded,
                                  self.qpb_string_obj_unbounded )
    self.writeqpbspecline_stringval( self.qpb_var_pivtol,
                                  self.qpb_default_pivtol,
                                  self.qpb_string_pivtol )
    self.writeqpbspecline_stringval( self.qpb_var_pivtol_deps,
                                  self.qpb_default_pivtol_deps,
                                  self.qpb_string_pivtol_deps )
    self.writeqpbspecline_stringval( self.qpb_var_zero_pivot,
                                  self.qpb_default_zero_pivot,
                                  self.qpb_string_zero_pivot )
    self.writeqpbspecline_stringval( self.qpb_var_identical_bnds, 
                                  self.qpb_default_identical_bnds,
                                  self.qpb_string_identical_bnds )
    self.writeqpbspecline_stringval( self.qpb_var_initial_radius,
                                  self.qpb_default_initial_radius,
                                  self.qpb_string_initial_radius )
    self.writeqpbspecline_stringval( self.qpb_var_inner_frac_opt,
                                  self.qpb_default_inner_frac_opt,
                                  self.qpb_string_inner_frac_opt )
    self.writeqpbspecline_stringval( self.qpb_var_inner_stop_rel,
                                  self.qpb_default_inner_stop_rel,
                                  self.qpb_string_inner_stop_rel )
    self.writeqpbspecline_stringval( self.qpb_var_inner_stop_abs,
                                  self.qpb_default_inner_stop_abs,
                                  self.qpb_string_inner_stop_abs )
    self.writeqpbspecline_int( self.qpb_check_treat_zero_as_gen,
                            self.qpb_default_treat_zero_as_gen,
                            self.qpb_string_treat_zero_as_gen )
    self.writeqpbspecline_int( self.qpb_check_remove_deps,
                            self.qpb_default_remove_deps, 
                            self.qpb_string_remove_deps )
    self.writeqpbspecline_int( self.qpb_check_center,
                            self.qpb_default_center, 
                            self.qpb_string_center )
    self.writeqpbspecline_int( self.qpb_check_primal,
                            self.qpb_default_primal, 
                            self.qpb_string_primal )
    self.writeqpbspecline_int( self.qpb_check_feasol,
                            self.qpb_default_feasol, 
                            self.qpb_string_feasol )
    self.writeqpbspecdummy( 'array-syntax-worse-than-do-loop', 'NO' )
    self.runqpbspc.write( "END QPB SPECIFICATIONS\n\n" )

#  If required, record LSQP options

    self.runqpbspc.write( "BEGIN LSQP SPECIFICATIONS\n" )
    self.writeqpbspecdummy( 'error-printout-device', '6' )
    self.writeqpbspecdummy( 'printout-device', '6' )
    self.writeqpbspecdummy( 'print-level', '1' )
    self.writeqpbspecdummy( 'maximum-number-of-iterations', '1000' )
    self.writeqpbspecdummy( 'start-print', '-1' )
    self.writeqpbspecdummy( 'stop-print', '-1' )
    self.writeqpbspecdummy( 'factorization-used', '0' )
    self.writeqpbspecdummy( 'maximum-column-nonzeros-in-schur-complement', '35' )
    self.writeqpbspecdummy( 'initial-integer-workspace', '10000' )
    self.writeqpbspecdummy( 'initial-real-workspace', '10000' )
    self.writeqpbspecdummy( 'maximum-refinements', '1' )
    self.writeqpbspecdummy( 'maximum-poor-iterations-before-infeasible', '200' )
    self.writeqpbspecdummy( 'restore-problem-on-output', '0' )
    self.writeqpbspecdummy( 'infinity-value', '1.0D+10' )
    self.writeqpbspecdummy( 'primal-accuracy-required', '1.0D-4' )
    self.writeqpbspecdummy( 'dual-accuracy-required', '1.0D-4' )
    self.writeqpbspecdummy( 'complementary-slackness-accuracy-required','1.0D-4')
    self.writeqpbspecdummy( 'mininum-initial-primal-feasibility', '1000.0' )
    self.writeqpbspecdummy( 'mininum-initial-dual-feasibility', '1000.0' )
    self.writeqpbspecdummy( 'initial-barrier-parameter', '-1.0' )
    self.writeqpbspecdummy( 'poor-iteration-tolerance', '0.98' )
    self.writeqpbspecdummy( 'minimum-potential-before-unbounded', '-10.0' )
    self.writeqpbspecdummy( 'pivot-tolerance-used', '1.0D-12' )
    self.writeqpbspecdummy( 'pivot-tolerance-used-for-dependencies', '0.5' )
    self.writeqpbspecdummy( 'zero-pivot-tolerance', '1.0D-12' )
    self.writeqpbspecdummy( 'remove-linear-dependencies', 'YES' )
    self.writeqpbspecdummy( 'treat-zero-bounds-as-general', 'NO' )
    self.writeqpbspecdummy( 'just-find-feasible-point', 'NO' )
    self.writeqpbspecdummy( 'get-advanced-dual-variables', 'NO' )
    self.writeqpbspecdummy( 'move-final-solution-onto-bound', 'NO' )
    self.writeqpbspecdummy( 'array-syntax-worse-than-do-loop', 'NO' )
    self.runqpbspc.write( "END LSQP SPECIFICATIONS\n\n" )

#  If required, record PRESOLVE options

    self.runqpbspc.write( "BEGIN PRESOLVE SPECIFICATIONS\n" )
    self.writeqpbspecdummy( 'printout-device', '6' )
    self.writeqpbspecdummy( 'error-printout-device', '6' )
    self.writeqpbspecdummy( 'print-level', 'TRACE' )
    self.writeqpbspecdummy( 'presolve-termination-strategy', 'REDUCED_SIZE' )
    self.writeqpbspecdummy( 'maximum-number-of-transformations', '1000000' )
    self.writeqpbspecdummy( 'maximum-number-of-passes', '25' )
    self.writeqpbspecdummy( 'constraints-accuracy', '1.0D-6' )
    self.writeqpbspecdummy( 'dual-variables-accuracy', '1.0D-6' )
    self.writeqpbspecdummy( 'allow-dual-transformations', 'YES' )
    self.writeqpbspecdummy( 'remove-redundant-variables-constraints', 'YES' )
    self.writeqpbspecdummy( 'primal-constraints-analysis-frequency', '1' )
    self.writeqpbspecdummy( 'dual-constraints-analysis-frequency', '1' )
    self.writeqpbspecdummy( 'singleton-columns-analysis-frequency', '1' )
    self.writeqpbspecdummy( 'doubleton-columns-analysis-frequency', '1' )
    self.writeqpbspecdummy( 'unconstrained-variables-analysis-frequency', '1' )
    self.writeqpbspecdummy( 'dependent-variables-analysis-frequency', '1' )
    self.writeqpbspecdummy( 'row-sparsification-frequency', '1' )
    self.writeqpbspecdummy( 'maximum-percentage-row-fill', '-1' )
    self.writeqpbspecdummy( 'transformations-buffer-size', '50000' )
    self.writeqpbspecdummy( 'transformations-file-device', '52' )
    self.writeqpbspecdummy( 'transformations-file-status', 'KEEP' )
    self.writeqpbspecdummy( 'transformations-file-name', 'transf.sav' )
    self.writeqpbspecdummy( 'primal-feasibility-check', 'NONE' )
    self.writeqpbspecdummy( 'dual-feasibility-check', 'NONE' )
    self.writeqpbspecdummy( 'active-multipliers-sign', 'POSITIVE' )
    self.writeqpbspecdummy( 'inactive-multipliers-value', 'LEAVE_AS_IS' )
    self.writeqpbspecdummy( 'active-dual-variables-sign', 'POSITIVE' )
    self.writeqpbspecdummy( 'inactive-dual-variables-value', 'LEAVE_AS_IS' )
    self.writeqpbspecdummy( 'primal-variables-bound-status', 'TIGHTEST' )
    self.writeqpbspecdummy( 'dual-variables-bound-status', 'TIGHTEST' )
    self.writeqpbspecdummy( 'constraints-bound-status', 'TIGHTEST' )
    self.writeqpbspecdummy( 'multipliers-bound-status', 'TIGHTEST' )
    self.writeqpbspecdummy( 'infinity-value', '1.0D19' )
    self.writeqpbspecdummy( 'pivoting-threshold', '1.10D-10' )
    self.writeqpbspecdummy( 'minimum-relative-bound-improvement', '1.0D-10' )
    self.writeqpbspecdummy( 'maximum-growth-factor', '1.0D8' )
    self.writeqpbspecdummy( 'compute-quadratic-value', 'YES' )
    self.writeqpbspecdummy( 'compute-objective-constant', 'YES' )
    self.writeqpbspecdummy( 'compute-gradient', 'YES' )
    self.writeqpbspecdummy( 'compute-Hessian', 'YES' )
    self.writeqpbspecdummy( 'compute-constraints-matrix', 'YES' )
    self.writeqpbspecdummy( 'compute-primal-variables-values', 'YES' )
    self.writeqpbspecdummy( 'compute-primal-variables-bounds', 'YES' )
    self.writeqpbspecdummy( 'compute-dual-variables-values', 'YES' )
    self.writeqpbspecdummy( 'compute-dual-variables-bounds', 'YES' )
    self.writeqpbspecdummy( 'compute-constraints-values', 'YES' )
    self.writeqpbspecdummy( 'compute-constraints-bounds', 'YES' )
    self.writeqpbspecdummy( 'compute-multipliers-values', 'YES' )
    self.writeqpbspecdummy( 'compute-multipliers-bounds', 'YES' )
    self.runqpbspc.write( "END PRESOLVE SPECIFICATIONS\n" )

#  close file
   
    self.runqpbspc.close( )
    print "new RUNQPB.SPC saved"

#  functions to produce various output lines

  def writeqpbspecline_int( self, var, default, line ): 
    if var.get( ) == default :
      if self.qpb_check_writeall.get( ) == 1 :
        if default == 0 :
          self.runqpbspc.write( "!  "+line.ljust( 50 )+"NO\n" )
        else :
          self.runqpbspc.write( "!  "+line.ljust( 50 )+"YES\n" )
    else :
      if default == 0 :
        self.runqpbspc.write( "   "+line.ljust( 50 )+"YES\n" )
      else :
        self.runqpbspc.write( "   "+line.ljust( 50 )+"NO\n" )
    
  def writeqpbspecline_string( self, var, string, line ): 
    self.varget = var.get( )
    stringupper = string.upper( )
    if self.varget == string :
      self.runqpbspc.write( "   "+line.ljust( 50 )+stringupper+"\n" )
    else :
      if self.qpb_check_writeall.get( ) == 1 :
        self.runqpbspc.write( "!  "+line.ljust( 50 )+stringupper+"\n" )

  def writeqpbspecline_stringval( self, var, default, line ): 
    self.varget = var.get( )
    if self.varget == default or self.varget == "" :
      if self.qpb_check_writeall.get( ) == 1 :
        self.runqpbspc.write( "!  "+line.ljust( 50 )+default+"\n" )
    else :
      self.runqpbspc.write( "   "+line.ljust( 50 )+self.varget+"\n" )

  def writeqpbspecdummy( self, line1, line2 ): 
    if self.qpb_check_writeall.get( ) == 1 :
      self.runqpbspc.write( "!  "+line1.ljust( 50 )+line2+"\n" )

#  function to display help
#  ------------------------

  def qpbpresshelp( self ):
    if os.system( 'which xpdf > /dev/null' ) == 0 :
      self.pdfread = 'xpdf'
    elif os.system( 'which acroread > /dev/null' ) == 0 :
      self.pdfread = 'acroread'
    else:
      print 'error: no known pdf file reader' 
      return
    
    self.threads =[ ]
    self.t = threading.Thread( target=self.pdfreadqpbthread )
    self.threads.append( self.t )
#   print self.threads
    self.threads[0].start( )

# display package documentation by opening an external PDF viewer

  def pdfreadqpbthread( self ) :
    os.system( self.pdfread+' $GALAHAD/doc/qpb.pdf' )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                          FILTRANE                         #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#   function for FILTRANE spec window
#   -----------------------------------

  def filtranespec( self ):

#  set variables if the first time through

    if self.filtranespec_used == 'no' :
      
#  integer (check-button) variables used with encodings

      self.filtrane_check_fulsol = IntVar( )
      self.filtrane_string_fulsol = 'print-full-solution'
      self.filtrane_default_fulsol = 0

      self.filtrane_check_stop_on_prec_g = IntVar( )
      self.filtrane_string_stop_on_prec_g = \
         'stop-on-preconditioned-gradient-norm'
      self.filtrane_default_stop_on_prec_g = 1

      self.filtrane_check_stop_on_g_max = IntVar( )
      self.filtrane_string_stop_on_g_max = 'stop-on-maximum-gradient-norm'
      self.filtrane_default_stop_on_g_max = 0

      self.filtrane_check_balance_group_values = IntVar( )
      self.filtrane_string_balance_group_values = 'balance-initial-group-values'
      self.filtrane_default_balance_group_values = 0

      self.filtrane_check_filter_sign_restriction = IntVar( )
      self.filtrane_string_filter_sign_restriction = 'filter-sign-restriction'
      self.filtrane_default_filter_sign_restriction = 0

      self.filtrane_check_remove_dominated = IntVar( )
      self.filtrane_string_remove_dominated = 'remove-dominated-filter-entries'
      self.filtrane_default_remove_dominated = 1

      self.filtrane_check_save_best_point = IntVar( )
      self.filtrane_string_save_best_point = 'save-best-point'
      self.filtrane_default_save_best_point = 0

      self.filtrane_check_restart_from_checkpoint = IntVar( )
      self.filtrane_string_restart_from_checkpoint = 'restart-from-checkpoint'
      self.filtrane_default_restart_from_checkpoint = 0

      self.filtrane_check_writeall = IntVar( )
      self.filtrane_default_writeall = 0

#  string variables used with encodings and defaults

      self.filtrane_var_start_print = StringVar( )
      self.filtrane_string_start_print = 'start-printing-at-iteration'
      self.filtrane_default_start_print = '0'

      self.filtrane_var_stop_print = StringVar( )
      self.filtrane_string_stop_print = 'stop-printing-at-iteration'
      self.filtrane_default_stop_print = '-1'

      self.filtrane_var_c_accuracy = StringVar( )
      self.filtrane_string_c_accuracy = 'residual-accuracy'
      self.filtrane_default_c_accuracy = '1.0D-6'

      self.filtrane_var_g_accuracy = StringVar( )
      self.filtrane_string_g_accuracy = 'gradient-accuracy'
      self.filtrane_default_g_accuracy = '1.0D-6'

      self.filtrane_var_max_iterations = StringVar( )
      self.filtrane_string_max_iterations = 'maximum-number-of-iterations'
      self.filtrane_default_max_iterations = '1000'

      self.filtrane_var_inequality_penalty_type = StringVar( )
      self.filtrane_string_inequality_penalty_type = 'inequality-penalty-type'
      self.filtrane_default_inequality_penalty_type = '2'

      self.filtrane_var_model_inertia = StringVar( )
      self.filtrane_string_model_inertia = 'automatic-model-inertia'
      self.filtrane_default_model_inertia = '3'

      self.filtrane_var_max_cg_iterations = StringVar( )
      self.filtrane_string_max_cg_iterations = 'maximum-number-of-cg-iterations'
      self.filtrane_default_max_cg_iterations = '15'

      self.filtrane_var_min_gltr_accuracy = StringVar( )
      self.filtrane_string_min_gltr_accuracy = \
        'minimum-relative-subproblem-accuracy'
      self.filtrane_default_min_gltr_accuracy = '0.01'

      self.filtrane_var_gltr_accuracy_power = StringVar( )
      self.filtrane_string_gltr_accuracy_power = \
        'relative-subproblem-accuracy-power'
      self.filtrane_default_gltr_accuracy_power = '1.0'

      self.filtrane_var_maximal_filter_size = StringVar( )
      self.filtrane_string_maximal_filter_size = 'maximum-filter-size'
      self.filtrane_default_maximal_filter_size = '-1'

      self.filtrane_var_filter_size_increment = StringVar( )
      self.filtrane_string_filter_size_increment = 'filter-size-increment'
      self.filtrane_default_filter_size_increment = '50'

      self.filtrane_var_gamma_f = StringVar( )
      self.filtrane_string_gamma_f = 'filter-margin-factor'
      self.filtrane_default_gamma_f = '0.001'

      self.filtrane_var_initial_radius = StringVar( )
      self.filtrane_string_initial_radius = 'initial-radius'
      self.filtrane_default_initial_radius = '1.0'

      self.filtrane_var_min_weak_accept_factor = StringVar( )
      self.filtrane_string_min_weak_accept_factor = \
        'minimum-weak-acceptance-factor'
      self.filtrane_default_min_weak_accept_factor = '0.1'

      self.filtrane_var_weak_accept_power = StringVar( )
      self.filtrane_string_weak_accept_power = 'weak-acceptance-power'
      self.filtrane_default_weak_accept_power = '2.0'

      self.filtrane_var_eta_1 = StringVar( )
      self.filtrane_string_eta_1 = 'minimum-rho-for-successful-iteration'
      self.filtrane_default_eta_1  = '0.01'

      self.filtrane_var_eta_2 = StringVar( )
      self.filtrane_string_eta_2 = 'minimum-rho-for-very-successful-iteration'
      self.filtrane_default_eta_2 = '0.9'

      self.filtrane_var_gamma_1 = StringVar( )
      self.filtrane_string_gamma_1 = 'radius-reduction-factor'
      self.filtrane_default_gamma_1 = '0.25'

      self.filtrane_var_gamma_2 = StringVar( )
      self.filtrane_string_gamma_2 = 'radius-increase-factor'
      self.filtrane_default_gamma_2 = '2.0'

      self.filtrane_var_gamma_0 = StringVar( )
      self.filtrane_string_gamma_0 = 'worst-case-radius-reduction-factor'
      self.filtrane_default_gamma_0 = '0.0625'

      self.filtrane_var_itr_relax = StringVar( )
      self.filtrane_string_itr_relax = 'initial-TR-relaxation-factor'
      self.filtrane_default_itr_relax = '1.0D+20'

      self.filtrane_var_str_relax = StringVar( )
      self.filtrane_string_str_relax = 'secondary-TR-relaxation-factor'
      self.filtrane_default_str_relax = '1000.0'

      self.filtrane_var_model_type = StringVar( )
      self.filtrane_string_model_type = 'model-type'
      self.filtrane_default_model_type = '10'

      self.filtrane_var_prec_used = StringVar( )
      self.filtrane_string_prec_used = 'preconditioner-used'
      self.filtrane_default_prec_used = '0'

      self.filtrane_var_semi_bandwidth = StringVar( )
      self.filtrane_string_semi_bandwidth = \
        'semi-bandwidth-for-band-preconditioner'
      self.filtrane_default_semi_bandwidth = '5'
      self.current_semi_bandwidth = self.filtrane_default_semi_bandwidth

      self.filtrane_var_grouping = StringVar( )
      self.filtrane_string_grouping = 'equations-grouping'
      self.filtrane_default_grouping = '0'

      self.filtrane_var_nbr_groups = StringVar( )
      self.filtrane_string_nbr_groups = 'number-of-groups'
      self.filtrane_default_nbr_groups = '10'
      self.current_nbr_groups = self.filtrane_default_nbr_groups

      self.filtrane_var_print_level = StringVar( )
      self.filtrane_string_print_level = 'print-level'
      self.filtrane_default_print_level = '0'

      self.filtrane_var_model_criterion = StringVar( )
      self.filtrane_string_model_criterion = 'automatic-model-criterion'
      self.filtrane_default_model_criterion = '0'

      self.filtrane_var_subproblem_accuracy = StringVar( )
      self.filtrane_string_subproblem_accuracy = 'subproblem-accuracy'
      self.filtrane_default_subproblem_accuracy = '0'

      self.filtrane_var_use_filter = StringVar( )
      self.filtrane_string_use_filter = 'use-filter'
      self.filtrane_default_use_filter = '5'

      self.filtrane_var_filter_margin_type = StringVar( )
      self.filtrane_string_filter_margin_type = 'filter-margin-type'
      self.filtrane_default_filter_margin_type = '0'

#  read default values

      self.restorefiltranedefaults( )
      self.filtrane_check_writeall.set( self.filtrane_default_writeall )
      self.filtranespec_used = 'yes'

#  setup the window frame
    
    self.specwindow = Toplevel( self.root )
    self.specwindow.geometry( '+100+100' )
    self.specwindow.title( 'FILTRANE spec tool' )
    self.specwindow.menubar = Menu( self.specwindow, tearoff=0  )
    self.specwindow.menubar.add_command( label = "Quit",
                                         command=self.specwindowdestroy )

    self.specwindow.helpmenu = Menu( self.specwindow, tearoff=0 )
    self.specwindow.helpmenu.add_command( label="About",
                                          command = self.filtranepresshelp )
    self.specwindow.menubar.add_cascade( label="Help",
                                         menu=self.specwindow.helpmenu )
    self.specwindow.config( menu=self.specwindow.menubar )

#  asseemble (check-button) variables

    self.check = [ self.filtrane_check_fulsol,
                   self.filtrane_check_filter_sign_restriction,
                   self.filtrane_check_save_best_point,
                   self.filtrane_check_restart_from_checkpoint
    ]

    self.checkstring = [ "Print full solution",
                         "Filter sign restriction",
                         "Save best point found",
                         "Restart from checkpoint"
]

    self.specwindow.varlstart = 0
    self.specwindow.varlstop = len( self.check )

    self.check = self.check+[ self.filtrane_check_stop_on_prec_g,
                              self.filtrane_check_stop_on_g_max,
                              self.filtrane_check_balance_group_values,
                              self.filtrane_check_remove_dominated
                              ]

    self.checkstring.extend( [ "Stop on preconditioned-gradient norm",
                               "Stop on maximum-gradient norm",
                               "Balance initial group values",
                               "Remove dominated filter entries"
                               ] )
    
    self.specwindow.varrstart = self.specwindow.varlstop
    self.specwindow.varrstop = len( self.check )

#  assemble string variables

    self.var = [ self.filtrane_var_start_print,
                 self.filtrane_var_stop_print,
                 self.filtrane_var_c_accuracy,
                 self.filtrane_var_g_accuracy,
                 self.filtrane_var_max_iterations,
                 self.filtrane_var_inequality_penalty_type,
                 self.filtrane_var_model_inertia,
                 self.filtrane_var_max_cg_iterations,
                 ]

    self.varstring = [ "Start printing at iteration",
                       "Stop printing at iteration",
                       "Residual accuracy",
                       "Gradient accuracy",
                       "Maximum number of iterations",
                       "Inequality penalty type",
                       "Automatic model inertia",
                       "Max number of CG iterations"
                        ]

    self.specwindow.entrytlstart = 0
    self.specwindow.entrytlstop = len( self.var )

    self.var = self.var+[ self.filtrane_var_min_gltr_accuracy,
                          self.filtrane_var_gltr_accuracy_power,
                          self.filtrane_var_maximal_filter_size,
                          self.filtrane_var_filter_size_increment,
                          self.filtrane_var_gamma_f,
                          self.filtrane_var_initial_radius,
                          self.filtrane_var_min_weak_accept_factor,
                          self.filtrane_var_weak_accept_power
                         ]

    self.varstring.extend( [ "Min relative subproblem accuracy",
                             "Relative subproblem accuracy power",
                             "Max filter size",
                             "Filter size increment",
                             "Filter margin factor",
                             "Initial radius",
                             "Min weak acceptance factor",
                             "Weak acceptance power"
                            ] )

    self.specwindow.entrytrstart = self.specwindow.entrytlstop
    self.specwindow.entrytrstop = len( self.var )

    self.var = self.var+[ self.filtrane_var_eta_1,
                          self.filtrane_var_eta_2,
                          self.filtrane_var_gamma_1,
                          self.filtrane_var_gamma_2,
                          self.filtrane_var_gamma_0,
                          self.filtrane_var_itr_relax,
                          self.filtrane_var_str_relax
                          ]

    self.varstring.extend( [ "Min rho for successful iteration",
                             "Min rho for very-successful iteration",
                             "Radius reduction factor",
                             "Radius increase factor",
                             "Worst-case radius reduction factor",
                             "Initial TR relaxation factor",
                             "Secondary TR relaxation factor"
                             ] )

    self.specwindow.entrybrstart = self.specwindow.entrytrstop
    self.specwindow.entrybrstop = len( self.var )

#  Set the name and logo 

    Label( self.specwindow, text="\nFILTRANE OPTIONS\n"
           ).pack( side=TOP, fill=BOTH )

    Label( self.specwindow, image=self.img, relief=SUNKEN
           ).pack( side=TOP, fill=NONE )

    Label( self.specwindow, text="\n"
           ).pack( side=TOP, fill=BOTH )

#  --- set frames  ---

#  main frame

    self.specwindow.frame = Frame( self.specwindow )

#  left and right sub-frames

    self.specwindow.frame.lhs = Frame( self.specwindow.frame )
    self.specwindow.frame.rhs = Frame( self.specwindow.frame )

#  frame to hold check buttons

    self.specwindow.check = Frame( self.specwindow.frame.lhs )

#  sub-frames for check buttons

    self.specwindow.check.left = Frame( self.specwindow.check )
    self.specwindow.check.right = Frame( self.specwindow.check )

#  frame to hold gradient and Hessian check buttons

    self.specwindow.ghcheck = Frame( self.specwindow.frame.lhs )

#  sub-frames for gradient and Hessian check buttons

    self.specwindow.ghcheck.left = Frame( self.specwindow.ghcheck )
    self.specwindow.ghcheck.right = Frame( self.specwindow.ghcheck )

# frame and sub-frames for expert data slots

    self.specwindow.frame.lhs.bottom = Frame( self.specwindow.frame.lhs )
    self.specwindow.frame.lhs.bottom.left \
      = Frame( self.specwindow.frame.lhs.bottom )
    self.specwindow.frame.lhs.bottom.right \
      = Frame( self.specwindow.frame.lhs.bottom )

# frame and sub-frames to hold data entry slots (top, right)

    self.specwindow.frame.rhs.top = Frame( self.specwindow.frame.rhs )
    self.specwindow.frame.rhs.top.left \
      = Frame( self.specwindow.frame.rhs.top )
    self.specwindow.frame.rhs.top.right \
      = Frame( self.specwindow.frame.rhs.top )

# frame and sub-frames to hold button and data entry slots (bottom, right)

    self.specwindow.frame.rhs.bottom = Frame( self.specwindow.frame.rhs )

# sub-frames to hold selection buttons

    self.specwindow.solver = Frame( self.specwindow.frame.rhs.bottom )

#  frame to hold data entry slots

    self.specwindow.text = Frame( self.specwindow.frame.rhs.bottom )

#   self.specwindow.iprint = Frame( self.specwindow.text )

#  --- set contents of frames ---

#  == Left-hand side of window ==

#  contents of check left frame

    for i in range( self.specwindow.varlstart, self.specwindow.varlstop ) :
      Checkbutton( self.specwindow.check.left,
                   highlightthickness=0,
                   relief=FLAT, anchor=W, width= "21",
                   command=self.nofdgandexacthessian,
                   variable=self.check[i],
                   text=self.checkstring[i]
                   ).pack( side=TOP, fill=BOTH )
    
    self.specwindow.check.left.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.check, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

#  contents of check right frame

    for i in range( self.specwindow.varrstart, self.specwindow.varrstop ) :
      Checkbutton( self.specwindow.check.right,
                   highlightthickness=0,
                   relief=FLAT, anchor=W,
                   command=self.nofdgandexacthessian,
                   variable=self.check[i],
                   text=self.checkstring[i]
                 ).pack( side=TOP, fill=BOTH )

    self.specwindow.check.right.pack( side=LEFT, fill=BOTH )

#  pack check box

    self.specwindow.check.pack( side=TOP, fill=BOTH )

    Label( self.specwindow.ghcheck.left, 
           text=" " ).pack( side=TOP, fill=BOTH )

#  contents of gradient frame (label and radio buttons)

    self.specwindow.printlevel = Frame( self.specwindow.ghcheck.left )
    Label( self.specwindow.printlevel, width = '15', anchor = 'e',
           text="Print level: " ).pack( side=LEFT, fill=BOTH )
    for printlevel in [ '0', '1', '2', '3' ]:
      if printlevel == '0' :
        lower = 'Silent '
      elif printlevel == '1' :
        lower = 'Trace'
      elif printlevel == '2' :
        lower = 'Action'
      elif printlevel == '3' :
        lower = 'Details'
      Radiobutton( self.specwindow.printlevel,
                   highlightthickness=0, relief=FLAT,
                   variable=self.filtrane_var_print_level,
                   value=printlevel,
                   text=lower
                   ).pack( side=LEFT, fill=BOTH )
    self.specwindow.printlevel.pack( side=TOP, fill=BOTH )

    self.specwindow.printlevelb = Frame( self.specwindow.ghcheck.left )
    Label( self.specwindow.printlevelb, width = '15',
           text="" ).pack( side=LEFT, fill=BOTH )
    for printlevel in [ '4', '5' ]:
      if printlevel == '4' :
        lower = 'Debug'
      elif printlevel == '5' :
        lower = 'Crazy'
      Radiobutton( self.specwindow.printlevelb,
                   highlightthickness=0, relief=FLAT,
                   variable=self.filtrane_var_print_level,
                   value=printlevel,
                   text=lower
                   ).pack( side=LEFT, fill=BOTH )
    self.specwindow.printlevelb.pack( side=TOP, fill=BOTH )

    self.specwindow.model = Frame( self.specwindow.ghcheck.left )
    Label( self.specwindow.model, width = '15', anchor = 'e',
           text="Model type: " ).pack( side=LEFT, fill=BOTH )
    for model in [ '10', '1', '0' ]:
      if model == '0' :
        lower = 'Gauss-Newton'
      elif model == '1' :
        lower = 'Newton'
      else :
        lower = 'Automatic'
      Radiobutton( self.specwindow.model,
                   highlightthickness=0, relief=FLAT,
                   variable=self.filtrane_var_model_type,
                   value=model,
                   text=lower
                   ).pack( side=LEFT, fill=BOTH )
    self.specwindow.model.pack( side=TOP, fill=BOTH )

    self.specwindow.criterion = Frame( self.specwindow.ghcheck.left )
    Label( self.specwindow.criterion, width = '15', anchor = 'e',
           text="Model criterion:" ).pack( side=LEFT, fill=BOTH )
    for criterion in [ '0', '1' ]:
      if criterion == '0' :
        lower = 'Best fit     '
      else :
        lower = 'Best reduction'
      Radiobutton( self.specwindow.criterion,
                   highlightthickness=0, relief=FLAT,
                   variable=self.filtrane_var_model_criterion,
                   value=criterion,
                   text=lower
                   ).pack( side=LEFT, fill=BOTH )
    self.specwindow.criterion.pack( side=TOP, fill=BOTH )


    self.specwindow.subproblem = Frame( self.specwindow.ghcheck.left )
    Label( self.specwindow.subproblem, width = '15', anchor = 'e',
           text="Subproblem:" ).pack( side=LEFT, fill=BOTH )
    for subproblem in [ '0', '1' ]:
      if subproblem == '0' :
        lower = 'Addaptive'
      else :
        lower = 'Full'
      Radiobutton( self.specwindow.subproblem,
                   highlightthickness=0, relief=FLAT,
                   variable=self.filtrane_var_subproblem_accuracy,
                   value=subproblem,
                   text=lower
                   ).pack( side=LEFT, fill=BOTH )
    self.specwindow.subproblem.pack( side=TOP, fill=BOTH )

    self.specwindow.subproblemb = Frame( self.specwindow.ghcheck.left )
    Label( self.specwindow.subproblemb, width = '15', anchor = 'e',
           text="accuracy    " ).pack( side=LEFT, fill=BOTH )
    self.specwindow.subproblemb.pack( side=TOP, fill=BOTH )

    self.specwindow.filter = Frame( self.specwindow.ghcheck.left )
    Label( self.specwindow.filter, width = '15', anchor = 'e',
           text="Use filter: " ).pack( side=LEFT, fill=BOTH )
    for filter in [ '3', '4', '5' ]:
      if filter == '3' :
        lower = 'Never   '
      elif filter == '4' :
        lower = 'Initially  '
      else :
        lower = 'Always   '
      Radiobutton( self.specwindow.filter,
                   highlightthickness=0, relief=FLAT,
                   variable=self.filtrane_var_use_filter,
                   value=filter,
                   text=lower
                   ).pack( side=LEFT, fill=BOTH )
    self.specwindow.filter.pack( side=TOP, fill=BOTH )


    self.specwindow.margin = Frame( self.specwindow.ghcheck.left )
    Label( self.specwindow.margin, width = '15', anchor = 'e',
           text="Margin type: " ).pack( side=LEFT, fill=BOTH )
    for margin in [ '0', '1', '2' ]:
      if margin == '0' :
        lower = 'Current'
      elif margin == '2' :
        lower = 'Fixed'
      else :
        lower = 'Smallest'
      Radiobutton( self.specwindow.margin,
                   highlightthickness=0, relief=FLAT,
                   variable=self.filtrane_var_filter_margin_type,
                   value=margin,
                   text=lower
                   ).pack( side=LEFT, fill=BOTH )
    self.specwindow.margin.pack( side=TOP, fill=BOTH )

    Label( self.specwindow.ghcheck.left, text="\n" ).pack( side=TOP, fill=BOTH )
    self.specwindow.ghcheck.left.pack( side=TOP, fill=BOTH )

#  Special check button for writeall

    self.specwindow.writebox = Frame( self.specwindow.ghcheck )
    Checkbutton( self.specwindow.writebox,
                   highlightthickness=0,
                   relief=FLAT, anchor="center",
                   variable=self.filtrane_check_writeall,
                   text="Even write defaults when saving values"
                 ).pack( side=TOP, fill=BOTH )

    self.specwindow.writebox.pack( side=TOP, fill=BOTH )

#  pack check box

    self.specwindow.ghcheck.pack( side=TOP, fill=BOTH )

#  == Right-hand side of window ==

#  contents of rhs top left data entry frame

    for i in range( self.specwindow.entrytlstart, self.specwindow.entrytlstop ):
      self.specwindow.i = Frame( self.specwindow.frame.rhs.top.left )
      Label( self.specwindow.i,
             anchor=W, width=26,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.rhs.top.left.pack( side=LEFT, fill=BOTH )

#  contents of rhs top right data entry frame

    Label( self.specwindow.frame.rhs.top, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    for i in range( self.specwindow.entrytrstart, self.specwindow.entrytrstop ):
      self.specwindow.i = Frame( self.specwindow.frame.rhs.top.right )
      Label( self.specwindow.i,
             anchor=W, width=32,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )
    self.specwindow.frame.rhs.top.right.pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.rhs.top.pack( side=TOP, fill=BOTH )

#  contents of rhs solver frame

    Label( self.specwindow.solver, width=30, anchor=W,
           text="\nPreconditioner used" ).pack( side=TOP, fill=BOTH )

    for solvers in [ '0', '1' ]:
      if solvers == '0' : label = "None"
      elif solvers == '1' : label = "Band: semibandwidth"
      if solvers == '1' :
        self.specwindow.bandsolver = Frame( self.specwindow.solver )
        Radiobutton( self.specwindow.bandsolver,
                     highlightthickness=0,
                     relief=FLAT, anchor=W,
                     variable=self.filtrane_var_prec_used,
                     value=solvers, width=19,
                     command=self.filtranesolversonoff,
                     text=label
                     ).pack( side=LEFT, fill=NONE )
        Entry( self.specwindow.bandsolver,
               textvariable=self.filtrane_var_semi_bandwidth,
               relief=SUNKEN, width=10
               ).pack( side=RIGHT, fill=BOTH )
        self.specwindow.bandsolver.pack( side=TOP, fill=BOTH )
      else :
        Radiobutton( self.specwindow.solver,
                     highlightthickness=0, relief=FLAT,
                     width=34, anchor=W,
                     variable=self.filtrane_var_prec_used,
                     value=solvers,
                     command=self.filtranesolversonoff,
                     text=label
                     ).pack( side=TOP, fill=NONE )

    Label( self.specwindow.solver, width=30, anchor=W,
           text="\nEquation grouping required" ).pack( side=TOP, fill=BOTH )

    for groups in [ '0', '1' ]:
      if groups == '0' : label = "None"
      elif groups == '1' : label = "Automatic: # groups"
      if groups == '1' :
        self.specwindow.groups = Frame( self.specwindow.solver )
        Radiobutton( self.specwindow.groups,
                     highlightthickness=0,
                     relief=FLAT, anchor=W,
                     variable=self.filtrane_var_grouping,
                     value=groups,  width=19,
                     command=self.filtranegroupsonoff,
                     text=label
                     ).pack( side=LEFT, fill=NONE )
        Entry( self.specwindow.groups,
               textvariable=self.filtrane_var_nbr_groups,
               relief=SUNKEN, width=10
               ).pack( side=RIGHT, fill=BOTH )
        self.specwindow.groups.pack( side=TOP, fill=BOTH )
      else :
        Radiobutton( self.specwindow.solver,
                     highlightthickness=0, relief=FLAT,
                     width=34, anchor=W,
                     variable=self.filtrane_var_grouping,
                     value=groups,
                     command=self.filtranegroupsonoff,
                     text=label
                     ).pack( side=TOP, fill=NONE )

    self.specwindow.solver.pack( side=LEFT, fill=BOTH )

#  contents of rhs bottom data entry frame

    Label( self.specwindow.frame.rhs.bottom, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    for i in range( self.specwindow.entrybrstart, self.specwindow.entrybrstop ):
      self.specwindow.i = Frame( self.specwindow.text )
      Label( self.specwindow.i,
             anchor=W, width=32,
             text=self.varstring[ i ]
             ).pack( side=LEFT, fill=NONE )
      Entry( self.specwindow.i,
             textvariable=self.var[i],
             relief=SUNKEN,
             width=10
             ).pack( side=RIGHT, fill=BOTH )
      self.specwindow.i.pack( side=TOP, fill=BOTH )

    self.specwindow.frame.rhs.bottom.pack( side=TOP, fill=BOTH )

    self.specwindow.text.pack( side=LEFT, fill=BOTH )

    Label( self.specwindow.frame.rhs, text="\n" ).pack( side=TOP, fill=BOTH )

#  --- assemble boxes ---

#  Pack it all together

    Label( self.specwindow.frame, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.lhs.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.frame, width=4,
           text="" ).pack( side=LEFT, fill=BOTH )
    self.specwindow.frame.rhs.pack( side=LEFT, fill=BOTH )
    Label( self.specwindow.frame, width=1,
           text="" ).pack( side=LEFT, fill=BOTH )

    self.specwindow.frame.pack( side=TOP, fill=BOTH )

#  Pack buttons along the bottom

    self.specwindow.buttons = Frame( self.specwindow )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Dismiss\nwindow", 
            command=self.specwindowdestroy
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Edit RUNFILT.SPC\ndirectly", 
            command=self.editfiltranespec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Read existing\nvalues", 
            command=self.readfiltranespec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Restore default\nvalues", 
            command=self.restorefiltranedefaults
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Save current\nvalues", 
            command=self.writefiltranespec
            ).pack( side=LEFT, fill=BOTH  )
    self.spacer( )
    Button( self.specwindow.buttons,
            pady=2,
            text="Run FILTRANE\nwith current values",
            command=self.rungaloncurrent
            ).pack( side=LEFT, fill=BOTH )
    self.spacer( )

    self.specwindow.buttons.pack( side=TOP, fill=BOTH )

    Label( self.specwindow, height=1,
           text="\n" ).pack( side=TOP, fill=BOTH )

#  function to edit RUNFILT.SPC
#  ----------------------------

  def editfiltranespec( self ):
    if os.path.exists( 'RUNFILT.SPC' ) == 0 :
      print ' no file RUNFILT.SPC to read'
      self.nospcfile( 'RUNFILT.SPC', 'edit' )
      return
    try:
      editor = os.environ["VISUAL"]
    except KeyError:
      try:
        editor = os.environ["EDITOR"]
      except KeyError:
        editor = emacs
    os.popen( editor+' RUNFILT.SPC' )

#  function to restore default spec values
#  ----------------------------------------

  def restorefiltranedefaults( self ):
    self.filtrane_check_fulsol.set( self.filtrane_default_fulsol )
    self.filtrane_check_stop_on_prec_g.set( 
       self.filtrane_default_stop_on_prec_g )
    self.filtrane_check_stop_on_g_max.set( 
       self.filtrane_default_stop_on_g_max )
    self.filtrane_check_balance_group_values.set( 
       self.filtrane_default_balance_group_values )
    self.filtrane_check_filter_sign_restriction.set( 
       self.filtrane_default_filter_sign_restriction )
    self.filtrane_check_remove_dominated.set( 
       self.filtrane_default_remove_dominated )
    self.filtrane_check_save_best_point.set( 
       self.filtrane_default_save_best_point )
    self.filtrane_check_restart_from_checkpoint.set( 
       self.filtrane_default_restart_from_checkpoint )

    self.filtrane_var_start_print.set( self.filtrane_default_start_print )
    self.filtrane_var_stop_print.set( self.filtrane_default_stop_print )
    self.filtrane_var_c_accuracy.set( self.filtrane_default_c_accuracy )
    self.filtrane_var_g_accuracy.set( self.filtrane_default_g_accuracy )
    self.filtrane_var_max_iterations.set( self.filtrane_default_max_iterations )
    self.filtrane_var_inequality_penalty_type.set( 
      self.filtrane_default_inequality_penalty_type )
    self.filtrane_var_model_inertia.set( self.filtrane_default_model_inertia )
    self.filtrane_var_max_cg_iterations.set( 
      self.filtrane_default_max_cg_iterations )
    self.filtrane_var_min_gltr_accuracy.set( 
      self.filtrane_default_min_gltr_accuracy )
    self.filtrane_var_gltr_accuracy_power.set( 
      self.filtrane_default_gltr_accuracy_power )
    self.filtrane_var_maximal_filter_size.set( 
      self.filtrane_default_maximal_filter_size )
    self.filtrane_var_filter_size_increment.set( 
      self.filtrane_default_filter_size_increment )
    self.filtrane_var_gamma_f.set( self.filtrane_default_gamma_f )
    self.filtrane_var_initial_radius.set( self.filtrane_default_initial_radius )
    self.filtrane_var_min_weak_accept_factor.set( 
      self.filtrane_default_min_weak_accept_factor )
    self.filtrane_var_weak_accept_power.set( 
      self.filtrane_default_weak_accept_power )
    self.filtrane_var_eta_1.set( self.filtrane_default_eta_1 )
    self.filtrane_var_eta_2.set( self.filtrane_default_eta_2 )
    self.filtrane_var_gamma_1.set( self.filtrane_default_gamma_1 )
    self.filtrane_var_gamma_2.set( self.filtrane_default_gamma_2 )
    self.filtrane_var_gamma_0.set( self.filtrane_default_gamma_0 )
    self.filtrane_var_itr_relax.set( self.filtrane_default_itr_relax )
    self.filtrane_var_str_relax.set( self.filtrane_default_str_relax )
    self.filtrane_var_model_type.set( self.filtrane_default_model_type )
    self.filtrane_var_prec_used.set( self.filtrane_default_prec_used )
    self.filtrane_var_semi_bandwidth.set( self.filtrane_default_semi_bandwidth )
    self.filtrane_var_grouping.set( self.filtrane_default_grouping )
    self.filtrane_var_nbr_groups.set( self.filtrane_default_nbr_groups )
    self.filtrane_var_print_level.set( self.filtrane_default_print_level )
    self.filtrane_var_model_criterion.set( 
      self.filtrane_default_model_criterion )
    self.filtrane_var_subproblem_accuracy.set( 
      self.filtrane_default_subproblem_accuracy )
    self.filtrane_var_use_filter.set( self.filtrane_default_use_filter )
    self.filtrane_var_filter_margin_type.set( 
      self.filtrane_default_filter_margin_type )

    self.current_semi_bandwidth = self.filtrane_default_semi_bandwidth
    self.current_nbr_groups = self.filtrane_default_nbr_groups
    self.filtranesolversonoff( )
    self.filtranegroupsonoff( )
  
#  function to switch on and off semibandwidth as appropriate
#  ----------------------------------------------------------

  def filtranesolversonoff( self ): 
    if self.filtrane_var_prec_used.get( ) == '1' :
      self.filtrane_var_semi_bandwidth.set( self.current_semi_bandwidth )
    else:
      if self.filtrane_var_semi_bandwidth.get( ) != '' :
        self.current_semi_bandwidth = self.filtrane_var_semi_bandwidth.get( )
      self.filtrane_var_semi_bandwidth.set( '' )

#  function to switch on and off number of groups as appropriate
#  -------------------------------------------------------------

  def filtranegroupsonoff( self ): 
    if self.filtrane_var_grouping.get( ) == '1' :
      self.filtrane_var_nbr_groups.set( self.current_nbr_groups )
    else:
      if self.filtrane_var_nbr_groups.get( ) != '' :
        self.current_nbr_groups = self.filtrane_var_nbr_groups.get( )
      self.filtrane_var_nbr_groups.set( '' )

#  function to disallow exact second derivatives at the same time
#  as finite-diffreence gradients
#  ------------------------------

#  def nofdgandexacthessian( self ): 
#    if self.filtrane_var_svarb.get( ) == '1' and \
#       self.filtrane_var_model_type.get( ) != '1' :
#     self.filtrane_var_svarb.set( '5' )

#  function to read the current values to the spec file
#  -----------------------------------------------------

  def readfiltranespec( self ): 

#  open file and set header

    if os.path.exists( 'RUNFILT.SPC' ) == 0 :
      print ' no file RUNFILT.SPC to read'
      self.nospcfile( 'RUNFILT.SPC', 'read' )
      return
    self.runfiltranespc = open( 'RUNFILT.SPC', 'r' )

#  Restore default values

    self.restorefiltranedefaults( )

#  loop over lines of files

    self.readyes = 0
    for line in self.runfiltranespc:

#  exclude comments

      if line[0] == '!' : continue

#  convert the line to lower case, and remove leading and trailing blanks

      line = line.lower( ) 
      line = line.strip( )
      blank_start = line.find( ' ' ) 
      
      if blank_start != -1 :
        stringc = line[0:blank_start]
      else :
        stringc = line

#  look for string variables to set

      blank_end = line.rfind( ' ' ) 
      if blank_start == -1 :
        stringd = 'YES'
      else:
        stringd = line[ blank_end + 1 : ].upper( )
#     print stringc+' '+stringd

#  only read those segments concerned with  FILTRANE

      if stringc == 'begin' and line.find( 'filtrane' ) >= 0 : self.readyes = 1
      if stringc == 'end' and line.find( 'filtrane' ) >= 0 : self.readyes = 0
      if self.readyes == 0 : continue

#  exclude begin and end lines

      if stringc == 'begin' or stringc == 'end' : continue

#  look for integer (check-button) variables to set

      if stringc == self.filtrane_string_fulsol :
        self.yesno( self.filtrane_check_fulsol, stringd )
        continue
      elif stringc == self.filtrane_string_stop_on_prec_g :
        self.yesno( self.filtrane_check_stop_on_prec_g, stringd )
        continue
      elif stringc == self.filtrane_string_stop_on_g_max :
        self.yesno( self.filtrane_check_stop_on_g_max, stringd )
        continue
      elif stringc == self.filtrane_string_balance_group_values :
        self.yesno( self.filtrane_check_balance_group_values, stringd )
        continue
      elif stringc == self.filtrane_string_filter_sign_restriction :
        self.yesno( self.filtrane_check_filter_sign_restriction, stringd )
        continue
      elif stringc == self.filtrane_string_remove_dominated :
        self.yesno( self.filtrane_check_remove_dominated, stringd )
        continue
      elif stringc == self.filtrane_string_save_best_point :
        self.yesno( self.filtrane_check_save_best_point, stringd )
        continue
      elif stringc == self.filtrane_string_restart_from_checkpoint :
        self.yesno( self.filtrane_check_restart_from_checkpoint, stringd )
        continue

      if stringc == self.filtrane_string_model_type :
        stringd = stringd.lower( ) 
        if stringd == '0' or stringd == 'gauss-newton' :
          self.filtrane_var_model_type.set( '0' )
        elif stringd == '1' or stringd == 'newton' :
          self.filtrane_var_model_type.set( '1' )
        elif stringd == '10' or stringd == 'automatic' :
          self.filtrane_var_model_type.set( '10' )
        continue
      elif stringc == self.filtrane_string_prec_used :
        stringd = stringd.lower( ) 
        if stringd == '0' or stringd == 'none' :
          self.filtrane_var_prec_used.set( '0' )
        elif stringd == '1' or stringd == 'banded' :
          self.filtrane_var_prec_used.set( '1' ) 
        continue
      elif stringc == self.filtrane_string_semi_bandwidth :
        self.filtrane_var_semi_bandwidth.set( stringd )
        self.current_semi_bandwidth = stringd
        continue
      elif stringc == self.filtrane_string_grouping :
        stringd = stringd.lower( ) 
        if stringd == '0' or stringd == 'none' :
          self.filtrane_var_grouping.set( '0' )
        elif stringd == '1' or stringd == 'automatic' :
          self.filtrane_var_grouping.set( '1' ) 
        continue
      elif stringc == self.filtrane_string_nbr_groups :
        self.filtrane_var_nbr_groups.set( stringd )
        self.current_nbr_groups = stringd
        continue
      elif stringc ==  self.filtrane_string_print_level :
        stringd = stringd.lower( ) 
        if stringd == '0' or stringd == 'silent' :
          self.filtrane_var_print_level.set( '0' )
        elif stringd == '1' or stringd == 'trace':
          self.filtrane_var_print_level.set( '1' )
        elif stringd == '2' or stringd == 'action':
          self.filtrane_var_print_level.set( '2' )
        elif stringd == '3' or stringd == 'details':
          self.filtrane_var_print_level.set( '3' )
        elif stringd == '4' or stringd == 'debug':
          self.filtrane_var_print_level.set( '4' )
        elif stringd == '5' or stringd == 'czary':
          self.filtrane_var_print_level.set( '5' )
        continue
      elif stringc == self.filtrane_string_model_criterion :
        stringd = stringd.lower( ) 
        if stringd == '0' or stringd == 'best_fit' :
          self.filtrane_var_model_criterion.set( '0' )
        elif stringd == '1' or stringd == 'best_reduction' :
          self.filtrane_var_model_criterion.set( '1' ) 
        continue
      elif stringc == self.filtrane_string_subproblem_accuracy :
        stringd = stringd.lower( ) 
        if stringd == '0' or stringd == 'adaptive' :
          self.filtrane_var_subproblem_accuracy.set( '0' )
        elif stringd == '1' or stringd == 'full' :
          self.filtrane_var_subproblem_accuracy.set( '1' ) 
        continue
      elif stringc == self.filtrane_string_use_filter :
        stringd = stringd.lower( ) 
        if stringd == '3' or stringd == 'never' :
          self.filtrane_var_use_filter.set( '3' )
        elif stringd == '4' or stringd == 'initial' :
          self.filtrane_var_use_filter.set( '4' ) 
        elif stringd == '5' or stringd == 'always' :
          self.filtrane_var_use_filter.set( '5' ) 
        continue
      elif stringc == self.filtrane_string_filter_margin_type :
        stringd = stringd.lower( ) 
        if stringd == '0' or stringd == 'current' :
          self.filtrane_var_filter_margin_type.set( '0' )
        elif stringd == '1' or stringd == 'fixed' :
          self.filtrane_var_filter_margin_type.set( '1' ) 
        elif stringd == '2' or stringd == 'smallest' :
          self.filtrane_var_filter_margin_type.set( '2' ) 
        continue
      elif stringc == self.filtrane_string_start_print :
        self.filtrane_var_start_print.set( stringd )
        continue
      elif stringc == self.filtrane_string_stop_print :
        self.filtrane_var_stop_print.set( stringd )
        continue
      elif stringc == self.filtrane_string_c_accuracy :
        self.filtrane_var_c_accuracy.set( stringd )
        continue
      elif stringc == self.filtrane_string_g_accuracy :
        self.filtrane_var_g_accuracy.set( stringd )
        continue
      elif stringc == self.filtrane_string_max_iterations :
        self.filtrane_var_max_iterations.set( stringd )
        continue
      elif stringc == self.filtrane_string_inequality_penalty_type :
        self.filtrane_var_inequality_penalty_type.set( stringd )
        continue
      elif stringc == self.filtrane_string_model_inertia :
        self.filtrane_var_model_inertia.set( stringd )
        continue
      elif stringc == self.filtrane_string_max_cg_iterations :
        self.filtrane_var_max_cg_iterations.set( stringd )
        continue
      elif stringc == self.filtrane_string_min_gltr_accuracy :
        self.filtrane_var_min_gltr_accuracy.set( stringd )
        continue
      elif stringc == self.filtrane_string_gltr_accuracy_power :
        self.filtrane_var_gltr_accuracy_power.set( stringd )
        continue
      elif stringc == self.filtrane_string_maximal_filter_size :
        self.filtrane_var_maximal_filter_size.set( stringd )
        continue
      elif stringc == self.filtrane_string_filter_size_increment :
        self.filtrane_var_filter_size_increment.set( stringd )
        continue
      elif stringc == self.filtrane_string_gamma_f :
        self.filtrane_var_gamma_f.set( stringd )
        continue
      elif stringc == self.filtrane_string_initial_radius :
        self.filtrane_var_initial_radius.set( stringd )
        continue
      elif stringc == self.filtrane_string_min_weak_accept_factor :
        self.filtrane_var_min_weak_accept_factor.set( stringd )
        continue
      elif stringc == self.filtrane_string_weak_accept_power :
        self.filtrane_var_weak_accept_power.set( stringd )
        continue
      elif stringc == self.filtrane_string_eta_1 :
        self.filtrane_var_eta_1.set( stringd )
        continue
      elif stringc == self.filtrane_string_eta_2 :
        self.filtrane_var_eta_2.set( stringd )
        continue
      elif stringc == self.filtrane_string_gamma_1 :
        self.filtrane_var_gamma_1.set( stringd )
        continue
      elif stringc == self.filtrane_string_gamma_2 :
        self.filtrane_var_gamma_2.set( stringd )
        continue
      elif stringc == self.filtrane_string_gamma_0 :
        self.filtrane_var_gamma_0.set( stringd )
        continue
      elif stringc == self.filtrane_string_itr_relax :
        self.filtrane_var_itr_relax.set( stringd )
        continue
      elif stringc == self.filtrane_string_str_relax :
        self.filtrane_var_str_relax.set( stringd )
        continue

    self.filtranesolversonoff( )
    self.runfiltranespc.close( )
#    if self.filtrane_var_svarb.get( ) == '1' and \
#       self.filtrane_var_model_type.get( ) != '1' :
#      self.filtrane_var_svarb.set( '5' )

#  function to write the current values to the spec file
#  -----------------------------------------------------

  def writefiltranespec( self ): 

#  open file and set header

    self.runfiltranespc = open( 'RUNFILT.SPC', 'w' )

#  record RUNFILT options

    self.runfiltranespc.write( "BEGIN RUNFILT SPECIFICATIONS\n" )
    self.writefiltranespecline_int( self.filtrane_check_fulsol,
                            self.filtrane_default_fulsol, 
                            self.filtrane_string_fulsol )
    self.writefiltranespecdummy( 'write-solution', 'NO' )
    self.writefiltranespecdummy( 'solution-file-name', 'FILTRANE.sol' )
    self.writefiltranespecdummy( 'solution-file-device', '57' )
    self.writefiltranespecdummy( 'write-result-summary', 'NO' )
    self.writefiltranespecdummy( 'result-summary-file-name', 'FILTRANE.sum' )
    self.writefiltranespecdummy( 'result-summary-file-device', '58' )
    self.runfiltranespc.write( "END RUNFILT SPECIFICATIONS\n\n" )

#  record FILTRANE options

    self.runfiltranespc.write( "BEGIN FILTRANE SPECIFICATIONS\n" )
    self.writefiltranespecdummy( 'error-printout-device', '6' )
    self.writefiltranespecdummy( 'printout-device', '6' )

#  record print level

    if self.filtrane_var_print_level.get( ) == "0" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  " \
          +self.filtrane_string_print_level.ljust( 50 )+"SILENT\n" )
    elif self.filtrane_var_print_level.get( ) == "1" :
      self.writefiltranespecline_string( "TRACE",
                                 self.filtrane_string_print_level )
    elif self.filtrane_var_print_level.get( ) == "2" :
      self.writefiltranespecline_string( "ACTION",
                                 self.filtrane_string_print_level )
    elif self.filtrane_var_print_level.get( ) == "3" :
      self.writefiltranespecline_string( "DETAILS",
                                 self.filtrane_string_print_level )
    elif self.filtrane_var_print_level.get( ) == "4" :
      self.writefiltranespecline_string( "DEBUG",
                                 self.filtrane_string_print_level )
    elif self.filtrane_var_print_level.get( ) == "5" :
      self.writefiltranespecline_string( "CRAZY",
                                 self.filtrane_string_print_level )

#  record further options

    self.writefiltranespecline_stringval( self.filtrane_var_start_print,
                                  self.filtrane_default_start_print,
                                  self.filtrane_string_start_print )
    self.writefiltranespecline_stringval( self.filtrane_var_stop_print,
                                  self.filtrane_default_stop_print,
                                  self.filtrane_string_stop_print )
    self.writefiltranespecline_stringval( self.filtrane_var_c_accuracy,
                                  self.filtrane_default_c_accuracy,
                                  self.filtrane_string_c_accuracy )
    self.writefiltranespecline_stringval( self.filtrane_var_g_accuracy,
                                  self.filtrane_default_g_accuracy,
                                  self.filtrane_string_g_accuracy )
    self.writefiltranespecline_int( self.filtrane_check_stop_on_prec_g,
                            self.filtrane_default_stop_on_prec_g, 
                            self.filtrane_string_stop_on_prec_g )
    self.writefiltranespecline_int( self.filtrane_check_stop_on_g_max,
                            self.filtrane_default_stop_on_g_max,
                            self.filtrane_string_stop_on_g_max )
    self.writefiltranespecline_stringval( self.filtrane_var_max_iterations,
                                  self.filtrane_default_max_iterations,
                                  self.filtrane_string_max_iterations )
    self.writefiltranespecline_stringval( 
                                  self.filtrane_var_inequality_penalty_type,
                                  self.filtrane_default_inequality_penalty_type,
                                  self.filtrane_string_inequality_penalty_type )
#  record model chosen

    if self.filtrane_var_model_type.get( ) == "0" :
      self.writefiltranespecline_string( "GAUSS_NEWTON",
                                 self.filtrane_string_model_type )
    elif self.filtrane_var_model_type.get( ) == "1" :
      self.writefiltranespecline_string( "NEWTON",
                                 self.filtrane_string_model_type )
    elif self.filtrane_var_model_type.get( ) == "10" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  " \
             + self.filtrane_string_model_type.ljust( 50 ) +"AUTOMATIC\n" )

#  record further options

    self.writefiltranespecline_stringval( self.filtrane_var_model_inertia,
                                  self.filtrane_default_model_inertia,
                                  self.filtrane_string_model_inertia )

#  record automatic model criterion

    if self.filtrane_var_model_criterion.get( ) == "0" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  " \
          +self.filtrane_string_model_criterion.ljust( 50 )+"BEST_FIT\n" )
    elif self.filtrane_var_model_criterion.get( ) == "1" :
      self.writefiltranespecline_string( "BEST_REDUCTION", 
                                 self.filtrane_string_model_criterion )

#  record further options

    self.writefiltranespecline_stringval( self.filtrane_var_max_cg_iterations,
                                  self.filtrane_default_max_cg_iterations,
                                  self.filtrane_string_max_cg_iterations )

#  record subproblem accuracy

    if self.filtrane_var_subproblem_accuracy.get( ) == "0" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  " \
          +self.filtrane_string_subproblem_accuracy.ljust( 50 )+"ADAPTIVE\n" )
    elif self.filtrane_var_subproblem_accuracy.get( ) == "1" :
      self.writefiltranespecline_string( "FULL",
                                 self.filtrane_string_subproblem_accuracy )

#  record further options

    self.writefiltranespecline_stringval( self.filtrane_var_min_gltr_accuracy,
                                  self.filtrane_default_min_gltr_accuracy,
                                  self.filtrane_string_min_gltr_accuracy )
    self.writefiltranespecline_stringval( self.filtrane_var_gltr_accuracy_power,
                                  self.filtrane_default_gltr_accuracy_power,
                                  self.filtrane_string_gltr_accuracy_power )
#  record preconditioner chosen

    if self.filtrane_var_prec_used.get( ) == "0" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  " \
          +self.filtrane_string_prec_used.ljust( 50 )+"NONE\n" )
    elif self.filtrane_var_prec_used.get( ) == "1" :
      self.writefiltranespecline_string( "BANDED", 
                                 self.filtrane_string_prec_used )

    self.writefiltranespecline_stringval( self.filtrane_var_semi_bandwidth,
                                  self.filtrane_default_semi_bandwidth,
                                  self.filtrane_string_semi_bandwidth )

    self.writefiltranespecdummy( 'external-Jacobian-products', 'NO' )

#  record grouping chosen

    if self.filtrane_var_grouping.get( ) == "0" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  " \
          +self.filtrane_string_grouping.ljust( 50 )+"NONE\n" )
    elif self.filtrane_var_grouping.get( ) == "1" :
      self.writefiltranespecline_string( "AUTOMATIC",
                                 self.filtrane_string_grouping )

    self.writefiltranespecline_stringval( self.filtrane_var_nbr_groups,
                                  self.filtrane_default_nbr_groups,
                                  self.filtrane_string_nbr_groups )

#  record further options

    self.writefiltranespecline_int( self.filtrane_check_balance_group_values,
                            self.filtrane_default_balance_group_values, 
                            self.filtrane_string_balance_group_values )

#  record whether to use filter

    if self.filtrane_var_use_filter.get( ) == "5" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  " \
           +self.filtrane_string_use_filter.ljust( 50 )+"ALWAYS\n" )
    elif self.filtrane_var_use_filter.get( ) == "3" :
      self.writefiltranespecline_string( "NEVER", 
                                 self.filtrane_string_use_filter )
    elif self.filtrane_var_use_filter.get( ) == "4" :
      self.writefiltranespecline_string( "INITIAL", 
                                 self.filtrane_string_use_filter )

    self.writefiltranespecline_int( self.filtrane_check_filter_sign_restriction,
                            self.filtrane_default_filter_sign_restriction,
                            self.filtrane_string_filter_sign_restriction )

    self.writefiltranespecline_stringval( self.filtrane_var_maximal_filter_size,
                                  self.filtrane_default_maximal_filter_size,
                                  self.filtrane_string_maximal_filter_size )

    self.writefiltranespecline_stringval( 
                                  self.filtrane_var_filter_size_increment,
                                  self.filtrane_default_filter_size_increment,
                                  self.filtrane_string_filter_size_increment )

#  record filter margin type

    if self.filtrane_var_filter_margin_type.get( ) == "0" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  " \
            +self.filtrane_string_filter_margin_type.ljust( 50 )+"CURRENT\n" )
    elif self.filtrane_var_filter_margin_type.get( ) == "1" :
      self.writefiltranespecline_string( "FIXED", 
                                      self.filtrane_string_filter_margin_type )
    elif self.filtrane_var_filter_margin_type.get( ) == "2" :
      self.writefiltranespecline_string( "SMALLEST", 
                                      self.filtrane_string_filter_margin_type )

#  record remaining options

    self.writefiltranespecline_stringval( self.filtrane_var_gamma_f,
                                  self.filtrane_default_gamma_f,
                                  self.filtrane_string_gamma_f )
    self.writefiltranespecline_int( self.filtrane_check_remove_dominated,
                                    self.filtrane_default_remove_dominated,
                                    self.filtrane_string_remove_dominated )
    self.writefiltranespecline_stringval( 
                                  self.filtrane_var_min_weak_accept_factor,
                                  self.filtrane_default_min_weak_accept_factor,
                                  self.filtrane_string_min_weak_accept_factor )
    self.writefiltranespecline_stringval( self.filtrane_var_weak_accept_power,
                                  self.filtrane_default_weak_accept_power,
                                  self.filtrane_string_weak_accept_power )
    self.writefiltranespecline_stringval( self.filtrane_var_initial_radius,
                                  self.filtrane_default_initial_radius,
                                  self.filtrane_string_initial_radius )
    self.writefiltranespecline_stringval( self.filtrane_var_eta_1,
                                  self.filtrane_default_eta_1,
                                  self.filtrane_string_eta_1 )
    self.writefiltranespecline_stringval( self.filtrane_var_eta_2, 
                                  self.filtrane_default_eta_2,
                                  self.filtrane_string_eta_2 )
    self.writefiltranespecline_stringval( self.filtrane_var_gamma_1,
                                  self.filtrane_default_gamma_1,
                                  self.filtrane_string_gamma_1 )
    self.writefiltranespecline_stringval( self.filtrane_var_gamma_2,
                                  self.filtrane_default_gamma_2,
                                  self.filtrane_string_gamma_2 )
    self.writefiltranespecline_stringval( self.filtrane_var_gamma_0,
                                  self.filtrane_default_gamma_0,
                                  self.filtrane_string_gamma_0 )
    self.writefiltranespecline_stringval( self.filtrane_var_itr_relax,
                                  self.filtrane_default_itr_relax,
                                  self.filtrane_string_itr_relax )
    self.writefiltranespecline_stringval( self.filtrane_var_str_relax,
                                  self.filtrane_default_str_relax,
                                  self.filtrane_string_str_relax )
    self.writefiltranespecline_int( self.filtrane_check_save_best_point,
                            self.filtrane_default_save_best_point, 
                            self.filtrane_string_save_best_point )
    self.writefiltranespecdummy( 'checkpointing-frequency', '0' )
    self.writefiltranespecdummy( 'checkpointing-device', '55' )
    self.writefiltranespecdummy( 'checkpointing-file', 'FILTRANE.chk' )
    self.writefiltranespecline_int( self.filtrane_check_restart_from_checkpoint,
                            self.filtrane_default_restart_from_checkpoint, 
                            self.filtrane_string_restart_from_checkpoint )

#  set footer and close file

    self.runfiltranespc.write( "END FILTRANE SPECIFICATIONS\n" )
    self.runfiltranespc.close( )
    print "new RUNFILT.SPC saved"

#  functions to produce various output lines

  def writefiltranespecline_int( self, var, default, line ): 
    if var.get( ) == default :
      if self.filtrane_check_writeall.get( ) == 1 :
        if default == 0 :
          self.runfiltranespc.write( "!  "+line.ljust( 50 )+"NO\n" )
        else :
          self.runfiltranespc.write( "!  "+line.ljust( 50 )+"YES\n" )
    else :
      if default == 0 :
        self.runfiltranespc.write( "   "+line.ljust( 50 )+"YES\n" )
      else :
        self.runfiltranespc.write( "   "+line.ljust( 50 )+"NO\n" )
    
  def writefiltranespecline_string( self, string, line ): 
    stringupper = string.upper( )
    self.runfiltranespc.write( "   "+line.ljust( 50 )+stringupper+"\n" )

  def writefiltranespecline_stringval( self, var, default, line ): 
    self.varget = var.get( )
    if self.varget == default or self.varget == "" :
      if self.filtrane_check_writeall.get( ) == 1 :
        self.runfiltranespc.write( "!  "+line.ljust( 50 )+default+"\n" )
    else :
      self.runfiltranespc.write( "   "+line.ljust( 50 )+self.varget+"\n" )

  def writefiltranespecdummy( self, line1, line2 ): 
    if self.filtrane_check_writeall.get( ) == 1 :
      self.runfiltranespc.write( "!  "+line1.ljust( 50 )+line2+"\n" )

#  function to display help
#  ------------------------

  def filtranepresshelp( self ):
    if os.system( 'which xpdf > /dev/null' ) == 0 :
      self.pdfread = 'xpdf'
    elif os.system( 'which acroread > /dev/null' ) == 0 :
      self.pdfread = 'acroread'
    else:
      print 'error: no known pdf file reader' 
      return
    
    self.threads =[ ]
    self.t = threading.Thread( target=self.pdfreadfiltranethread )
    self.threads.append( self.t )
#   print self.threads
    self.threads[0].start( )

# display package documentation by opening an external PDF viewer

  def pdfreadfiltranethread( self ) :
    os.system( self.pdfread+' $GALAHAD/doc/filtrane.pdf' )


#  function to read the current values to the spec file
#  -----------------------------------------------------

  def readdefaults( self, file ): 

#  open file and set header

    self.defaultsfile = open( file, 'r' )

#  loop over lines of files

    self.readyes = 0
    for line in self.defaultsfile:

#  exclude comments

      if line[0] == '!' : continue

#  convert the line to lower case, and remove leading and trailing blanks

      line = line.lower( ) 
      line = line.strip( )
      blank_start = line.find( ':' ) 
      
      if blank_start != -1 :
        stringc = line[0:blank_start]
      else :
        continue

#  look for string variables to set

      blank_end = line.rfind( ' ' ) 
      if blank_start == -1 :
        stringd = 'none'
      else:
        stringd = line[ blank_end + 1 : ]

#  Set the default package

      if stringc == "defaultpackage" :
        if stringd == 'filt' :
          self.var_package.set( stringd )
          self.var_package_label.set( "FILTRANE" )
          print "package reset to "+stringd
        elif stringd == 'qpb' :
          self.var_package.set( stringd )
          self.var_package_label.set( "QPB / LSQP" )
          print "package reset to "+stringd
        elif stringd == 'qpa' :
          self.var_package.set( stringd )
          self.var_package_label.set( "QPA" )
          print "package reset to "+stringd
        else:
          self.var_package.set( "lanb" )
          self.var_package_label.set( "LANCELOT B" )

#  Set the default arcitecture

      elif stringc == "defaultarchitecture" :
        for eachfile in self.architecture :
          if stringd == eachfile :
            self.var_arch.set( stringd )
            print "architecture reset to "+stringd
