#!/usr/bin/python3
# -*- coding : utf-8 -*-
import traceback
import sys, os, time
import numpy as np
from ... import Data as M
from timeit import default_timer as toc

import matplotlib.pyplot as plt
plt.ion()


class SnappingCursor:
    """
    A cross hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """
    def __init__(self, ax, line):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='silver', lw=0.8, ls='-')
        self.vertical_line = ax.axvline(color='silver', lw=0.8, ls='-')
        self.line = line
        self.x, self.y = self.line.get_data()
        self._last_index = None
        # text location in axes coords
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", color=self.get_line_color()),
                            arrowprops=dict(arrowstyle="->", color=self.get_line_color()),
                            color=self.get_line_color())
        self.annot.set_zorder(999)
        self.annot.set_visible(False)
        self.user_annotations = []

    def get_line_ylabel(self): return self.line.get_label()


    def get_line_color(self): return self.line.get_color()

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.annot.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[index]
            y = self.y[index]

            self.annot.xy = [x,y]
            self.annot.set_text('%s=%g\n%s=%g'%(self.ax.xaxis.get_label().get_text(),x,
                                                self.line.get_label(),y))

            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.ax.figure.canvas.draw()

    def on_double_click(self, event):
        if event.dblclick:
            user_annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w", color=self.get_line_color()),
                                arrowprops=dict(arrowstyle="->", color=self.get_line_color()),
                                color=self.get_line_color())
            x = self.x[self._last_index]
            y = self.y[self._last_index]
            user_annot.xy = [x,y]
            user_annot.set_text('%s=%g\n%s=%g'%(self.ax.xaxis.get_label().get_text(),x,
                                                self.line.get_label(),y))
            user_annot.set_zorder(999)
            user_annot.set_visible(True)
            self.ax.plot(x,y,'x',color='red')
            self.user_annotations += [ user_annot ]
            self.ax.figure.canvas.draw()

try:
    from PyQt5 import Qt, QtGui, QtWidgets, QtCore
except:
    print(('import of PyQt5 failed. Please try reinstalling:\n'
          'pip3 install --user --upgrade --force-reinstall PyQt5'))

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QToolBar,
    QAction,
    QStatusBar,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QDockWidget,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QShortcut,
    QAbstractItemView,
    QAbstractButton,
    QComboBox,
    QSpinBox,
    QLineEdit,
    QTreeView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QRadioButton,
    QMessageBox,
)


GUIpath = os.path.dirname(os.path.realpath(__file__))

style="""QTreeView {{
    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #e7effd, stop: 1 #cbdaf1);
}}

QTreeView::branch:has-siblings:!adjoins-item {{
    border-image: url({GUIpath}{sep}style{sep}stylesheet-vline.png) 0;
}}

QTreeView::branch:has-siblings:adjoins-item {{
    border-image: url({GUIpath}{sep}style{sep}stylesheet-branch-more.png) 0;
}}

QTreeView::branch:!has-children:!has-siblings:adjoins-item {{
    border-image: url({GUIpath}{sep}style{sep}stylesheet-branch-end.png) 0;
}}

QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {{
        border-image: none;
        image: url({GUIpath}{sep}style{sep}stylesheet-branch-closed.png);
}}

QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings  {{
        border-image: none;
        image: url({GUIpath}{sep}style{sep}stylesheet-branch-open.png);
}}

""".format(GUIpath=GUIpath, sep=os.path.sep)


class MainWindow(QMainWindow):
    def __init__(self, filename=None, only_skeleton=False, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowIcon(QtGui.QIcon(os.path.join(GUIpath,'..','MOLA.svg')))
        self.fontPointSize = 12.
        self.only_skeleton = only_skeleton

        self.dock = QDockWidget('Please select a node...')
        self.dock.setFeatures(QDockWidget.DockWidgetFloatable |
                              QDockWidget.DockWidgetMovable)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock)
        self.dock.VLayout = QWidget(self)
        self.dock.setWidget(self.dock.VLayout)
        self.dock.VLayout.setLayout(QVBoxLayout())

        self.dock.node_toolbar = QToolBar("Node toolbar")
        self.dock.VLayout.layout().addWidget(self.dock.node_toolbar)


        self.dock.node_toolbar.button_update_node_data = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/arrow-circle-double.png") ,"load or update node(s) data from file (F5)", self)
        self.dock.node_toolbar.button_update_node_data.setStatusTip("load or update selected node(s) data and their children of type DataArray_t from file (F5)")
        self.dock.node_toolbar.addAction(self.dock.node_toolbar.button_update_node_data)
        key_update_node_data = QShortcut(QtGui.QKeySequence('F5'), self)
        key_update_node_data.activated.connect(self.update_node_data)
        self.dock.node_toolbar.button_update_node_data.triggered.connect(self.update_node_data)

        self.dock.node_toolbar.button_unload_node_data_recursively = QAction(QtGui.QIcon(GUIpath+"/icons/icons8/icons8-squelette-16.png") ,"unload data of node(s) from memory (F6)", self)
        self.dock.node_toolbar.button_unload_node_data_recursively.setStatusTip("unload data of selected node(s) and their children of type DataArray_t (F6)")
        self.dock.node_toolbar.addAction(self.dock.node_toolbar.button_unload_node_data_recursively)
        key_unload_node_data = QShortcut(QtGui.QKeySequence('F6'), self)
        key_unload_node_data.activated.connect(self.unload_node_data_recursively)
        self.dock.node_toolbar.button_unload_node_data_recursively.triggered.connect(self.unload_node_data_recursively)

        pixmap = QtGui.QPixmap(GUIpath+"/icons/fugue-icons-3.5.6/external.png")
        tr = QtGui.QTransform()
        tr.rotate(180)
        pixmap = pixmap.transformed(tr)
        icon = QtGui.QIcon()
        icon.addPixmap(pixmap)
        self.dock.node_toolbar.button_replace_link = QAction(icon ,"read link", self)
        self.dock.node_toolbar.button_replace_link.setStatusTip("Read link of selected node(s) from file (must be Link_t)")
        self.dock.node_toolbar.addAction(self.dock.node_toolbar.button_replace_link)
        self.dock.node_toolbar.button_replace_link.triggered.connect(self.replace_link)

        self.dock.node_toolbar.button_modify_node_data = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/disk--arrow.png") ,"write selected node(s) in file (F8)", self)
        self.dock.node_toolbar.button_modify_node_data.setStatusTip("write selected node(s) in file (F8)")
        self.dock.node_toolbar.addAction(self.dock.node_toolbar.button_modify_node_data)
        key_modify_node_data = QShortcut(QtGui.QKeySequence('F8'), self)
        key_modify_node_data.activated.connect(self.modify_node_data)
        self.dock.node_toolbar.button_modify_node_data.triggered.connect(self.modify_node_data)

        self.dock.node_toolbar.addSeparator()

        self.dock.node_toolbar.button_add_plot_x_container = QAction(QtGui.QIcon(GUIpath+"/icons/OwnIcons/x-16.png") ,"add node(s) data to X plotter container", self)
        self.dock.node_toolbar.button_add_plot_x_container.setStatusTip("add node(s) data to X plotter container")
        self.dock.node_toolbar.addAction(self.dock.node_toolbar.button_add_plot_x_container)
        self.dock.node_toolbar.button_add_plot_x_container.triggered.connect(self.add_selected_nodes_to_plot_x_container)

        self.dock.node_toolbar.button_add_plot_y_container = QAction(QtGui.QIcon(GUIpath+"/icons/OwnIcons/y-16.png") ,"add node(s) data to Y plotter container", self)
        self.dock.node_toolbar.button_add_plot_y_container.setStatusTip("add node(s) data to y plotter container")
        self.dock.node_toolbar.addAction(self.dock.node_toolbar.button_add_plot_y_container)
        self.dock.node_toolbar.button_add_plot_y_container.triggered.connect(self.add_selected_nodes_to_plot_y_container)

        self.dock.node_toolbar.button_add_curve = QAction(QtGui.QIcon(GUIpath+"/icons/OwnIcons/add-curve-16.png") ,"add curve to plotter", self)
        self.dock.node_toolbar.button_add_curve.setStatusTip("add curve to plotter")
        self.dock.node_toolbar.addAction(self.dock.node_toolbar.button_add_curve)
        self.dock.node_toolbar.button_add_curve.triggered.connect(self.add_curve)

        self.dock.node_toolbar.button_draw_curves = QAction(QtGui.QIcon(GUIpath+"/icons/OwnIcons/see-curve-16.png") ,"draw all curves", self)
        self.dock.node_toolbar.button_draw_curves.setStatusTip("draw all curves")
        self.dock.node_toolbar.addAction(self.dock.node_toolbar.button_draw_curves)
        self.dock.node_toolbar.button_draw_curves.triggered.connect(self.draw_curves)

        self.dock.node_toolbar.setVisible(False)


        self.dock.plotter = QWidget(self)
        self.dock.plotter.setLayout(QVBoxLayout())
        self.dock.VLayout.layout().addWidget(self.dock.plotter)
        self.dock.plotter.setVisible(False)



        self.dock.typeEditor = QWidget(self)
        self.dock.typeEditor.setLayout(QHBoxLayout())
        self.dock.typeEditor.layout().addWidget(QLabel('CGNS Type:'))
        self.dock.typeEditor.lineEditor = QLineEdit("please select a node...")
        self.dock.typeEditor.lineEditor.editingFinished.connect(self.updateTypeOfNodeCGNS)
        self.dock.typeEditor.layout().addWidget(self.dock.typeEditor.lineEditor)
        self.dock.VLayout.layout().addWidget(self.dock.typeEditor)
        self.dock.typeEditor.setVisible(False)

        self.dock.dataDimensionsLabel = QLabel('Class: toto | Dims: tata')
        self.dock.VLayout.layout().addWidget(self.dock.dataDimensionsLabel)
        self.dock.dataDimensionsLabel.setVisible(False)


        self.dock.dataSlicer = QWidget(self)
        self.dock.dataSlicer.setLayout(QHBoxLayout())
        self.dock.dataSlicer.ijkSelector = QComboBox()
        self.dock.dataSlicer.ijkSelector.insertItems(0,['k','j','i'])
        self.dock.dataSlicer.ijkSelector.currentTextChanged.connect(self.updateTable)
        self.dock.dataSlicer.layout().addWidget(QLabel('Showing data at index:'))
        self.dock.dataSlicer.layout().addWidget(self.dock.dataSlicer.ijkSelector)
        self.dock.dataSlicer.sliceSelector = QSpinBox()
        self.dock.dataSlicer.sliceSelector.setValue(0)
        self.dock.dataSlicer.sliceSelector.valueChanged.connect(self.updateTable)
        self.dock.dataSlicer.layout().addWidget(self.dock.dataSlicer.sliceSelector)
        self.dock.VLayout.layout().addWidget(self.dock.dataSlicer)
        self.dock.dataSlicer.setVisible(False)

        self.createTable()
        self.dock.VLayout.layout().addWidget(self.table)


        self.tree = QTreeView(self)
        self.tree.model = QtGui.QStandardItemModel()
        # self.tree.setMouseTracking(True)
        self.tree.setHeaderHidden(True)
        self.tree.setModel(self.tree.model)
        self.tree.setStyleSheet(style)
        self.tree.setSelectionMode(self.tree.ExtendedSelection)
        self.tree.setDragDropMode(QAbstractItemView.DragDrop)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selectedNodesCGNS = []
        self.tree.selectionModel().selectionChanged.connect(self.selectionInfo)
        self.tree.model.itemChanged.connect(self.updateNameOfNodeCGNS)

        self.setCentralWidget(self.tree)

        if filename:
            onlyFileName = filename.split(os.sep)[-1]
            self.setWindowTitle("TreeLab - "+onlyFileName)
            self.t = M.Tree()
            self.t.file = filename
        else:
            self.t = M.Tree()
            self.t.file = None
            self.setWindowTitle("TreeLab - untitled ")
        
        


        toolbar = QToolBar("Main toolbar")
        self.addToolBar(toolbar)

        button_open = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/folder-horizontal-open.png") ,"open (Ctrl+O)", self)
        button_open.setStatusTip("Open a tree from a file")
        button_open.triggered.connect(self.openTree)
        key_openTree = QShortcut(QtGui.QKeySequence('Ctrl+O'), self)
        key_openTree.activated.connect(self.openTree)
        toolbar.addAction(button_open)

        button_openAdd = QAction(QtGui.QIcon(GUIpath+"/icons/OwnIcons/folder-horizontal-open-plus") ,"add tree (Ctrl+Shift+O)", self)
        button_openAdd.setStatusTip("Open a tree from a file and add it to current tree (Ctrl+Shift+O)")
        button_openAdd.triggered.connect(self.openAddTree)
        key_openAddTree = QShortcut(QtGui.QKeySequence('Ctrl+Shift+O'), self)
        key_openAddTree.activated.connect(self.openAddTree)
        toolbar.addAction(button_openAdd)


        button_reopen = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/arrow-circle.png") ,"Open again (Shift + F5)", self)
        button_reopen.setStatusTip("Open again the current tree from file (Shift + F5)")
        button_reopen.triggered.connect(self.reopenTree)
        key_reopen = QShortcut(QtGui.QKeySequence('Shift+F5'), self)
        key_reopen.activated.connect(self.reopenTree)
        toolbar.addAction(button_reopen)

        button_save = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/disk.png") ,"save (Ctrl+S)", self)
        button_save.setStatusTip("Save the current tree")
        button_save.triggered.connect(self.saveTree)
        key_saveTree = QShortcut(QtGui.QKeySequence('Ctrl+S'), self)
        key_saveTree.activated.connect(self.saveTree)
        toolbar.addAction(button_save)

        button_saveAs = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/disk--plus.png") ,"save as (Ctrl+Shift+S)", self)
        button_saveAs.setStatusTip("Save the current tree as new file")
        button_saveAs.triggered.connect(self.saveTreeAs)
        key_saveTreeAs = QShortcut(QtGui.QKeySequence('Ctrl+Shift+S'), self)
        key_saveTreeAs.activated.connect(self.saveTreeAs)
        toolbar.addAction(button_saveAs)

        toolbar.addSeparator()

        button_zoomIn = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/magnifier-zoom-in.png") ,"zoom in (+)", self)
        button_zoomIn.setStatusTip("Zoom in the tree (+)")
        button_zoomIn.triggered.connect(self.zoomInTree)
        key_zoomInTree = QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Plus), self)
        key_zoomInTree.activated.connect(self.zoomInTree)
        toolbar.addAction(button_zoomIn)

        button_zoomOut = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/magnifier-zoom-out.png") ,"zoom out (-)", self)
        button_zoomOut.setStatusTip("Zoom out the tree (-)")
        button_zoomOut.triggered.connect(self.zoomOutTree)
        key_zoomOutTree = QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Minus), self)
        key_zoomOutTree.activated.connect(self.zoomOutTree)
        toolbar.addAction(button_zoomOut)

        button_expandAll = QAction(QtGui.QIcon(GUIpath+"/icons/icons8/icons8-expand-48") ,"expand all nodes", self)
        button_expandAll.setStatusTip("Expand all the nodes of the tree")
        button_expandAll.triggered.connect(self.tree.expandAll)
        toolbar.addAction(button_expandAll)

        button_expandZones = QAction(QtGui.QIcon(GUIpath+"/icons/icons8/icons8-expand3-48") ,"expand to depth 3", self)
        button_expandZones.setStatusTip("Expand up to three levels of the tree")
        button_expandZones.triggered.connect(self.expandToZones)
        toolbar.addAction(button_expandZones)

        button_collapseAll = QAction(QtGui.QIcon(GUIpath+"/icons/icons8/icons8-collapse-48") ,"collapse all nodes", self)
        button_collapseAll.setStatusTip("Collapse all the nodes of the tree")
        button_collapseAll.triggered.connect(self.tree.collapseAll)
        toolbar.addAction(button_collapseAll)

        toolbar.addSeparator()

        button_findNode = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/node-magnifier") ,"find node (Ctrl+F)", self)
        button_findNode.setStatusTip("Find node using criteria based on Name, Value and Type (Ctrl+F)")
        button_findNode.triggered.connect(self.findNodesTree)
        key_findNodesTree = QShortcut(QtGui.QKeySequence('Ctrl+F'), self)
        key_findNodesTree.activated.connect(self.findNodesTree)
        toolbar.addAction(button_findNode)
        self.NameToBeFound = None
        self.ValueToBeFound = None
        self.TypeToBeFound = None
        self.DepthToBeFound = 100
        self.FoundNodes = []
        self.CurrentFoundNodeIndex = -1

        button_findNextNode = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/magnifier--arrow") ,"find next node (F3)", self)
        button_findNextNode.setStatusTip("Find next node (F3)")
        button_findNextNode.triggered.connect(self.findNextNodeTree)
        key_findNextNodeTree = QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F3), self)
        key_findNextNodeTree.activated.connect(self.findNextNodeTree)
        toolbar.addAction(button_findNextNode)

        toolbar.addSeparator()

        button_newNodeTree = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/plus") ,"New node (Ctrl+N)", self)
        button_newNodeTree.setStatusTip("Create a new node attached to the selected node in tree (Ctrl+N)")
        button_newNodeTree.triggered.connect(self.newNodeTree)
        toolbar.addAction(button_newNodeTree)
        key_newNodeTree = QShortcut(QtGui.QKeySequence('Ctrl+N'), self.tree)
        key_newNodeTree.setContext(QtCore.Qt.WidgetShortcut)
        key_newNodeTree.activated.connect(self.newNodeTree)

        button_deleteNodeTree = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/cross") ,"remove selected nodes (Supr)", self)
        button_deleteNodeTree.setStatusTip("remove selected node (Supr)")
        button_deleteNodeTree.triggered.connect(self.deleteNodeTree)
        toolbar.addAction(button_deleteNodeTree)
        key_deleteNodeTree = QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self.tree)
        key_deleteNodeTree.setContext(QtCore.Qt.WidgetShortcut)
        key_deleteNodeTree.activated.connect(self.deleteNodeTree)


        button_swap = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/arrow-switch.png"), "swap selected nodes", self)
        button_swap.setStatusTip("After choosing two nodes, swap their position in the tree")
        button_swap.triggered.connect(self.swapNodes)
        toolbar.addAction(button_swap)

        button_copyNodeTree = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/blue-document-copy") ,"copy selected nodes (Ctrl+C)", self)
        button_copyNodeTree.setStatusTip("copy selected nodes (Ctrl+C)")
        button_copyNodeTree.triggered.connect(self.copyNodeTree)
        toolbar.addAction(button_copyNodeTree)
        key_copyNodeTree = QShortcut(QtGui.QKeySequence('Ctrl+C'), self.tree)
        key_copyNodeTree.setContext(QtCore.Qt.WidgetShortcut)
        key_copyNodeTree.activated.connect(self.copyNodeTree)
        self.copiedNodes = []

        button_cutNodeTree = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/scissors-blue") ,"cut selected nodes (Ctrl+X)", self)
        button_cutNodeTree.setStatusTip("cut selected nodes (Ctrl+X)")
        button_cutNodeTree.triggered.connect(self.cutNodeTree)
        toolbar.addAction(button_cutNodeTree)
        key_cutNodeTree = QShortcut(QtGui.QKeySequence('Ctrl+X'), self.tree)
        key_cutNodeTree.setContext(QtCore.Qt.WidgetShortcut)
        key_cutNodeTree.activated.connect(self.cutNodeTree)


        button_pasteNodeTree = QAction(QtGui.QIcon(GUIpath+"/icons/fugue-icons-3.5.6/clipboard-paste") ,"paste nodes (Ctrl+V)", self)
        button_pasteNodeTree.setStatusTip("Paste previously copied nodes at currently selected parent nodes (Ctrl+V)")
        button_pasteNodeTree.triggered.connect(self.pasteNodeTree)
        toolbar.addAction(button_pasteNodeTree)
        key_pasteNodeTree = QShortcut(QtGui.QKeySequence('Ctrl+V'), self.tree)
        key_pasteNodeTree.setContext(QtCore.Qt.WidgetShortcut)
        key_pasteNodeTree.activated.connect(self.pasteNodeTree)

        self.plot_x_container = []
        self.plot_y_container = []
        self.curves_container = []

        self.setStatusBar(QStatusBar(self))
        self.tree.setFocus()
        if filename: self.reopenTree()
        else: self.updateModel()


    def get_x_from_curve_item(self, curve):
        path = 'CGNSTree/'+curve.Xchoser.currentText()
        node = self.t.getAtPath(path)
        node_value = node.value()
        if isinstance(node_value, str) and node_value == '_skeleton':
            self.update_node_data_and_children(node)
        return node.value(), node.name()

    def get_y_from_curve_item(self, curve):
        path = 'CGNSTree/'+curve.Ychoser.currentText()
        node = self.t.getAtPath(path)
        node_value = node.value()
        if isinstance(node_value, str) and node_value == '_skeleton':
            self.update_node_data_and_children(node)
        return node.value(), node.name()

    def draw_curves(self):
        if not self.curves_container: return


        self.fig, ax = plt.subplots(1,1,dpi=150)
        self.fig.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.fig.canvas.setFocus()
        xlabels = []
        ylabels = []
        self.snap_cursors = []
        for c in self.curves_container:
            try:
                x, xlabel = self.get_x_from_curve_item(c)
                y, ylabel = self.get_y_from_curve_item(c)
                xlabels += [ xlabel ]
                ylabels += [ ylabel ]
                line, = ax.plot(x,y, label=ylabel)
                snap_cursor = SnappingCursor(ax, line)
                self.snap_cursors += [ snap_cursor ]
                self.fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
                self.fig.canvas.mpl_connect('button_press_event', snap_cursor.on_double_click)


            except BaseException as e:
                err_msg = ''.join(traceback.format_exception(etype=type(e),
                                    value=e, tb=e.__traceback__))

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText(err_msg)
                msg.setWindowTitle("Error")
                msg.exec_()
        ax.set_xlabel(xlabels[0])
        ax.set_ylabel(ylabels[0])
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()



    def add_curve(self):

        curve = QWidget(self)
        curve.setLayout(QHBoxLayout())
        Xlabel = QLabel('X=')
        Xlabel.setSizePolicy(QtWidgets.QSizePolicy.Fixed, 16)
        curve.layout().addWidget(Xlabel)
        curve.Xchoser = QComboBox()
        curve.Xchoser.addItems( self.plot_x_container )
        curve.Xchoser.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        curve.layout().addWidget(curve.Xchoser)


        Ylabel = QLabel('Y=')
        Ylabel.setSizePolicy(QtWidgets.QSizePolicy.Fixed, 16)
        curve.layout().addWidget(Ylabel)
        curve.Ychoser = QComboBox()
        curve.Ychoser.addItems( self.plot_y_container )
        curve.Ychoser.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        curve.layout().addWidget(curve.Ychoser)

        button_remove_curve = QPushButton()
        button_remove_curve.setSizePolicy(QtWidgets.QSizePolicy.Fixed, 16)
        pixmap = QtGui.QPixmap(GUIpath+"/icons/OwnIcons/remove-curve-16.png")
        ButtonIcon = QtGui.QIcon(pixmap)
        button_remove_curve.setIcon(ButtonIcon)
        button_remove_curve.setIconSize(pixmap.rect().size())
        button_remove_curve.clicked.connect(lambda: self.remove_curve(curve) )
        curve.layout().addWidget(button_remove_curve)


        self.dock.plotter.layout().addWidget(curve)

        self.curves_container += [ curve ]

    def remove_curve(self, curve):
        for i, c in enumerate(self.curves_container):
            if c is curve:
                c.setParent(None)
                del self.curves_container[i]
                return


    def add_selected_nodes_to_plot_x_container(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        for node in self.selectedNodesCGNS:
            path = node.path().replace('CGNSTree/','')
            if path not in self.plot_x_container:
                self.plot_x_container += [ path ]
        for c in self.curves_container:
            AllItems = [c.Xchoser.itemText(i) for i in range(c.Xchoser.count())]
            for p in self.plot_x_container:
                if p not in AllItems:
                    c.Xchoser.addItem( p )
        QApplication.restoreOverrideCursor()


    def add_selected_nodes_to_plot_y_container(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        for node in self.selectedNodesCGNS:
            path = node.path().replace('CGNSTree/','')
            if path not in self.plot_y_container:
                self.plot_y_container += [ path ]
        for c in self.curves_container:
            AllItems = [c.Ychoser.itemText(i) for i in range(c.Ychoser.count())]
            for p in self.plot_y_container:
                if p not in AllItems:
                    c.Ychoser.addItem( p )
        QApplication.restoreOverrideCursor()


    def modify_node_data(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        if self.t.file is None or not os.path.exists( self.t.file):
            QApplication.restoreOverrideCursor()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            err_msg = ('Cannot save individual node into file, since there is no existing file.\n'
                       'Maybe you never saved the current tree ?\n')
            msg.setInformativeText(err_msg)
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        for node in self.selectedNodesCGNS:
            try:
                node.saveThisNodeOnly( self.t.file )
            except TypeError as e:
                e_str = str(e)
                if e_str == 'Only chunked datasets can be resized':
                    err_msg = ('cannot save selected nodes since file:\n\n   %s\n\n'
                        'was not generated using chunks.\n\nTo save modifications, '
                        'please save the entire file')%self.t.file
                else:
                    err_msg = e_str
                # except BaseException as e:
                #     err_msg = ''.join(traceback.format_exception(etype=type(e),
                #                           value=e, tb=e.__traceback__))
                # finally:
                QApplication.restoreOverrideCursor()

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText(err_msg)
                msg.setWindowTitle("Error")
                msg.exec_()
                break

        self.selectionInfo( None )
        QApplication.restoreOverrideCursor()

    def update_node_data_and_children(self, node):
        if node.type() == 'DataArray_t':

            try:
                node.reloadNodeData( self.t.file )
            except BaseException as e:
                err_msg = ''.join(traceback.format_exception(etype=type(e),
                                      value=e, tb=e.__traceback__))
                QApplication.restoreOverrideCursor()

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText(err_msg)
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            
            item = node.QStandardItem
            item.isStyleCGNSbeingModified = True
            self.setStyleCGNS( item )
            item.isStyleCGNSbeingModified = False
        for child in node[2]: self.update_node_data_and_children(child)

    def update_node_data(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        for node in self.selectedNodesCGNS:
            self.update_node_data_and_children(node)
        self.selectionInfo( None )
        QApplication.restoreOverrideCursor()

    def unload_data(self, node):
        if node.type() == 'DataArray_t':
            node.setValue('_skeleton')
            item = node.QStandardItem
            item.isStyleCGNSbeingModified = True
            self.setStyleCGNS( item )
            item.isStyleCGNSbeingModified = False

    def unload_data_and_children(self, node):
        self.unload_data(node)
        for child in node[2]:
            self.unload_data_and_children(child)

    def unload_node_data_recursively(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        for node in self.selectedNodesCGNS:
            self.unload_data_and_children(node)
        self.selectionInfo( None )
        QApplication.restoreOverrideCursor()

    def replace_link(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        for node in self.selectedNodesCGNS:
            item = node.QStandardItem
            path = node.path()
            node.replaceLink()
            node = self.t.getAtPath( path )
            item.node_cgns = node
            item.node_cgns.QStandardItem = item
            item.isStyleCGNSbeingModified = True
            self.setStyleCGNS( item )
            item.isStyleCGNSbeingModified = False
            self.addTreeModelChildrenFromCGNSNode(item.node_cgns)
        self.selectionInfo( None )
        QApplication.restoreOverrideCursor()


    def pasteNodeTree(self):
        if not self.copiedNodes: return
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        for node in self.selectedNodesCGNS:
            for paste_node in self.copiedNodes:
                paste_node = paste_node.copy(deep=True)
                node.addChild(paste_node, override_brother_by_name=False)
                paste_node = node.get(paste_node.name(),Depth=1)
                parentitem = node.QStandardItem
                paste_node.QStandardItem = QtGui.QStandardItem(paste_node.name())
                item = paste_node.QStandardItem
                item.node_cgns = paste_node
                item.node_cgns.QStandardItem = item
                item.isStyleCGNSbeingModified = True
                parentitem.appendRow([item])
                self.setStyleCGNS(item)
                item.isStyleCGNSbeingModified = False
                self.addTreeModelChildrenFromCGNSNode(paste_node)
        QApplication.restoreOverrideCursor()

    def addTreeModelChildrenFromCGNSNode(self, node):
        for c in node.children():
            item = QtGui.QStandardItem(c.name())
            item.node_cgns = c
            c.QStandardItem = item
            item.isStyleCGNSbeingModified = True
            node.QStandardItem.appendRow([item])
            self.setStyleCGNS(item)
            item.isStyleCGNSbeingModified = False
            self.addTreeModelChildrenFromCGNSNode(c)



    def copyNodeTree(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.copiedNodes = [n.copy(deep=True) for n in self.selectedNodesCGNS]
        QApplication.restoreOverrideCursor()


    def cutNodeTree(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.copyNodeTree()
        self.deleteNodeTree()
        QApplication.restoreOverrideCursor()


    def newNodeTree(self):
        index = self.tree.selectionModel().selectedIndexes()
        if len(index) < 1: return
        index = index[0]
        item = self.tree.model.itemFromIndex(index)

        dlg = NewNodeDialog(item.node_cgns.Path)
        if dlg.exec():
            NewName = dlg.NameWidget.text()
            if NewName == '': NewName = 'Node'

            NewType = dlg.TypeWidget.text()
            if NewType == '': NewType = 'UserDefinedData_t'

            NewValue = dlg.ValueWidget.text().strip()
            if NewValue in ['',  '{}', '[]', 'None']:
                NewValue = None

            elif (NewValue.startswith('{') and NewValue.endswith('}')) or \
                 (NewValue.startswith('[') and NewValue.endswith(']')):

                if NewValue.startswith('{'): NewValue = NewValue[1:-1]

                try:
                    NewValue = np.array(eval(NewValue,globals(),{}),order='F')
                    if len(NewValue.shape) == 0: NewValue == eval(NewValue,globals(),{})
                except BaseException as e:
                    err_msg = ''.join(traceback.format_exception(etype=type(e),
                                      value=e, tb=e.__traceback__))

                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setText("Error")
                    msg.setInformativeText(err_msg)
                    msg.setWindowTitle("Error")
                    msg.exec_()
                    return



            elif NewValue[0].isdigit():
                if '.' in NewValue:
                    NewValue = np.array([NewValue],dtype=np.float64,order='F')
                else:
                    NewValue = np.array([NewValue],dtype=np.int32,order='F')

            
            indices = self.tree.selectionModel().selectedIndexes()
            self.tree.clearSelection()
            self.tree.setSelectionMode(self.tree.MultiSelection)
            for index in indices:
                item = self.tree.model.itemFromIndex(index)

                parentnode = item.node_cgns
                newnode = M.castNode([NewName, NewValue, [], NewType])
                newnode.attachTo(parentnode, override_brother_by_name=False)

                newitem = newnode.QStandardItem = QtGui.QStandardItem(newnode.name())
                item.isStyleCGNSbeingModified = True
                item.appendRow([newitem])
                newitem.node_cgns = newnode
                newitem.QStandardItem = newitem
                newitem.isStyleCGNSbeingModified = True
                self.setStyleCGNS(newitem)
                newitem.isStyleCGNSbeingModified = False
                item.isStyleCGNSbeingModified = False
                newindex = self.tree.model.indexFromItem(newitem)

                self.tree.setCurrentIndex(newindex)
            self.tree.setSelectionMode(self.tree.ExtendedSelection)


    def deleteNodeTree(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        index = self.tree.selectionModel().selectedIndexes()
        while len(index)>0:
            index=index[0]
            item = self.tree.model.itemFromIndex(index)
            if item.parent():
                item.parent().removeRow(item.row())
            else:
                self.tree.model.invisibleRootItem().removeRow(item.row())
            if item.node_cgns.Parent: item.node_cgns.remove()
            index = self.tree.selectionModel().selectedIndexes()
        QApplication.restoreOverrideCursor()


    def findNodesTree(self):

        dlg = FindNodeDialog(self.NameToBeFound,
                             self.ValueToBeFound,
                             self.TypeToBeFound,
                             self.DepthToBeFound)
        if dlg.exec():
            # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            RequestedName = dlg.NameWidget.text()
            if RequestedName == '':
                RequestedName = None
            self.NameToBeFound = RequestedName

            RequestedValue = dlg.ValueWidget.text()
            if RequestedValue == '':
                RequestedValue = None
            elif RequestedValue[0].isdigit():
                if '.' in RequestedValue or 'e-' in RequestedValue:
                    try: RequestedValue = float(RequestedValue)
                    except: pass
                else:
                    try: RequestedValue = int(RequestedValue)
                    except: pass
            self.ValueToBeFound = RequestedValue

            RequestedType = dlg.TypeWidget.text()
            if RequestedType == '':
                RequestedType = None
            self.TypeToBeFound = RequestedType

            RequestedDepth = dlg.DepthWidget.text()
            if RequestedDepth == '':
                RequestedDepth = 100
            self.DepthToBeFound = int(RequestedDepth)


            if dlg.searchFromSelection.isChecked():
                self.FoundNodes = []
                indices = self.tree.selectionModel().selectedIndexes()
                for index in indices:
                    item = self.tree.model.itemFromIndex(index)
                    self.FoundNodes.extend(item.node_cgns.group(
                                                Name=self.NameToBeFound,
                                                Value=self.ValueToBeFound,
                                                Type=self.TypeToBeFound,
                                                Depth=self.DepthToBeFound))

            else:
                self.FoundNodes = self.t.group(Name=self.NameToBeFound,
                                               Value=self.ValueToBeFound,
                                               Type=self.TypeToBeFound,
                                               Depth=self.DepthToBeFound)

            self.tree.clearSelection()
            self.tree.setSelectionMode(self.tree.MultiSelection)
            for node in self.FoundNodes:
                index = self.tree.model.indexFromItem(node.QStandardItem)
                self.tree.setCurrentIndex(index)
            self.tree.setSelectionMode(self.tree.ExtendedSelection)
            self.selectionInfo(None)
            self.CurrentFoundNodeIndex = -1
            # QApplication.restoreOverrideCursor()

    def findNextNodeTree(self):
        # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.CurrentFoundNodeIndex += 1
        try:
            node = self.FoundNodes[self.CurrentFoundNodeIndex]
        except IndexError:
            self.CurrentFoundNodeIndex = 0

        try:
            node = self.FoundNodes[self.CurrentFoundNodeIndex]
        except IndexError:
            self.CurrentFoundNodeIndex = 0
            return

        index = self.tree.model.indexFromItem(node.QStandardItem)
        self.tree.setCurrentIndex(index)
        # QApplication.restoreOverrideCursor()

    def expandToZones(self):
        self.tree.expandToDepth(1)

    def zoomInTree(self):
        # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        root = self.t.QStandardItem
        self.fontPointSize += 1
        self.setStyleCGNS(root)
        for item in self.iterItems(root):
            self.setStyleCGNS(item)
        # QApplication.restoreOverrideCursor()

    def zoomOutTree(self):
        # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        root = self.t.QStandardItem
        self.fontPointSize -= 1
        self.setStyleCGNS(root)
        for item in self.iterItems(root):
            self.setStyleCGNS(item)
        # QApplication.restoreOverrideCursor()

    def iterItems(self, root):
        stack = [root]
        while stack:
            parent = stack.pop(0)
            for row in range(parent.rowCount()):
                for column in range(parent.columnCount()):
                    child = parent.child(row, column)
                    yield child
                    if child.hasChildren():
                        stack.append(child)

    def openAddTree(self):
        fname = QFileDialog.getOpenFileName(self, 'Add file', '.',"CGNS files (*.cgns)")
        onlyFileName = fname[0].split(os.sep)[-1]
        # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        print('building full CGNS...')
        tic = toc()
        t = M.load(fname[0], only_skeleton=False)
        self.t.merge(t)
        print('done (%g s)'%(toc()-tic))
        # QApplication.restoreOverrideCursor()
        print('updating Qt model...')
        tic = toc()
        self.updateModel()
        print('done (%g s)'%(toc()-tic))


    def openTree(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '.',"CGNS files (*.cgns)")
        # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        onlyFileName = fname[0].split(os.sep)[-1]
        self.setWindowTitle("TreeLab - "+onlyFileName)
        print('building CGNS structure...')
        tic = toc()
        self.t = M.load(fname[0], only_skeleton=self.only_skeleton)
        print('done (%g s)'%(toc()-tic))
        self.t.file = fname[0]
        # QApplication.restoreOverrideCursor()
        print('building Qt model...')
        tic = toc()
        self.updateModel()
        print('done (%g s)'%(toc()-tic))


    def reopenTree(self):
        file = self.t.file
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        print('building CGNS structure...')
        tic = toc()
        self.t = M.load(self.t.file, only_skeleton=self.only_skeleton)
        QApplication.restoreOverrideCursor()
        print('done (%g s)'%(toc()-tic))
        self.t.file = file
        print('building Qt model...')
        tic = toc()
        self.updateModel()
        print('done (%g s)'%(toc()-tic))

    def saveTree(self):
        if self.t.file:
            print('will write: '+self.t.file)
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.t.replaceSkeletonWithDataRecursively(self.t.file)
            self.t.save(self.t.file)
            print('done')
            QApplication.restoreOverrideCursor()
        else:
            self.saveTreeAs()

    def saveTreeAs(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', '.',"CGNS files (*.cgns)")
        onlyFileName = fname[0].split(os.sep)[-1]
        if onlyFileName:
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            print('will write: '+onlyFileName)
            self.t.replaceSkeletonWithDataRecursively(self.t.file)
            self.t.save(fname[0])
            print('done')
            self.t.file = fname[0]
            self.setWindowTitle("TreeLab - "+onlyFileName)
            QApplication.restoreOverrideCursor()


    def updateNameOfNodeCGNS(self, item):
        try:
            if item.isStyleCGNSbeingModified: return
        except AttributeError:
            item.isStyleCGNSbeingModified = False
        if hasattr(item, "node_cgns"):
            item.node_cgns.setName(item.text())
            return

        node = [n for n in self.selectedNodesCGNS if n.name() == item.text()][0]
        item.node_cgns = node
        parentItem = item.parent()
        if parentItem:
            item.node_cgns.moveTo(parentItem.node_cgns, position=item.row())
        elif item.node_cgns.Parent:
            item.node_cgns.dettach()

        item.setText(item.node_cgns.name())
        item.node_cgns.item = item
        self.setStyleCGNS(item)
        self.updateModelChildrenFromItem(item)

    def updateTypeOfNodeCGNS(self):
        for index in self.tree.selectionModel().selectedIndexes():
            item = self.tree.model.itemFromIndex(index)
            newType = self.dock.typeEditor.lineEditor.text()
            item.node_cgns.setType(newType)
            item.isStyleCGNSbeingModified = True
            self.setStyleCGNS(item)
            item.isStyleCGNSbeingModified = False

    def selectionInfo(self, selection):
        self.selectedNodesCGNS = []
        self.selectedNodeCGNS = None
        indexes = self.tree.selectionModel().selectedIndexes()
        if isinstance(indexes, QtCore.QModelIndex): indexes = [indexes]
        MSG = '%d nodes selected'%len(indexes)
        self.setStatusTip( MSG )
        self.statusBar().showMessage( MSG )
        for index in indexes:
            item = self.tree.model.itemFromIndex(index)
            self.selectedNodesCGNS.append( item.node_cgns )
            self.selectedNodeCGNS = item.node_cgns

        if self.selectedNodeCGNS:
            self.dock.node_toolbar.setVisible(True)
            self.dock.typeEditor.setVisible(True)
            self.dock.plotter.setVisible(True)
            self.dock.dataDimensionsLabel.setVisible(True)
            self.dock.dataSlicer.setVisible(True)
            self.table.setVisible(True)
        else:
            self.selectedNodeCGNS = None
            self.dock.node_toolbar.setVisible(False)
            self.dock.typeEditor.setVisible(False)
            self.dock.plotter.setVisible(False)
            self.dock.dataDimensionsLabel.setVisible(False)
            self.dock.dataSlicer.setVisible(False)
            self.table.setVisible(False)

        self.updateTable()


    def updateTable(self):
        # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.table.isBeingUpdated = True
        node = self.selectedNodeCGNS
        if node is None:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0,0, QTableWidgetItem('please select a node'))
            self.table.resizeColumnsToContents()
            self.dock.setWindowTitle('please select a node')
            self.table.isBeingUpdated = False
            return

        self.dock.setWindowTitle(node.name())
        self.dock.typeEditor.lineEditor.setText(node.type())
        value = node.value()
        if isinstance(value, np.ndarray):
            msg = '%s dims %s'%(str(type(value))[1:-1],str(value.shape))
            msg += ' %s'%(str(value.dtype))
            msg += '  F=%s'%str(value.flags['F_CONTIGUOUS'])
            self.dock.dataDimensionsLabel.setText(msg)
            dim = len(value.shape)
            Ni = value.shape[0]
            Nj = value.shape[1] if dim > 1 else 1
            Nk = value.shape[2] if dim > 2 else 1

            if dim == 1:
                self.dock.dataSlicer.setVisible(False)
                self.table.setRowCount(Ni)
                self.table.setColumnCount(1)
                for i in range(Ni):
                    self.table.setItem(i, 0, QTableWidgetItem('{}'.format(value[i])))

            elif dim == 2:
                self.dock.dataSlicer.setVisible(False)
                self.table.setRowCount(Ni)
                self.table.setColumnCount(Nj)

                for i in range(Ni):
                    for j in range(Nj):
                        self.table.setItem(i,j, QTableWidgetItem("%g"%value[i,j]))

            elif dim == 3:
                planeIndex = self.dock.dataSlicer.ijkSelector.currentText()
                planeValue = self.dock.dataSlicer.sliceSelector.value()

                if planeIndex == 'k':
                    self.dock.dataSlicer.sliceSelector.setMaximum(Nk-1)
                    self.dock.dataSlicer.sliceSelector.setMinimum(-Nk)
                    self.table.setRowCount(Ni)
                    self.table.setColumnCount(Nj)

                elif planeIndex == 'j':
                    self.dock.dataSlicer.sliceSelector.setMaximum(Nj-1)
                    self.dock.dataSlicer.sliceSelector.setMinimum(-Nj)
                    self.table.setRowCount(Ni)
                    self.table.setColumnCount(Nk)

                else:
                    self.dock.dataSlicer.sliceSelector.setMaximum(Ni-1)
                    self.dock.dataSlicer.sliceSelector.setMinimum(-Ni)
                    self.table.setRowCount(Nj)
                    self.table.setColumnCount(Nk)


                if planeValue > self.dock.dataSlicer.sliceSelector.maximum() or \
                    planeValue < self.dock.dataSlicer.sliceSelector.minimum():
                    planeValue = 0
                    self.dock.dataSlicer.sliceSelector.setValue(planeValue)

                if planeIndex == 'k':
                    for i in range(Ni):
                        for j in range(Nj):
                            self.table.setItem(i,j, QTableWidgetItem("%g"%value[i,j,planeValue]))
                elif planeIndex == 'j':
                    for i in range(Ni):
                        for k in range(Nk):
                            self.table.setItem(i,k, QTableWidgetItem("%g"%value[i,planeValue,k]))
                else:
                    for j in range(Nj):
                        for k in range(Nk):
                            self.table.setItem(j,k, QTableWidgetItem("%g"%value[planeValue,j,k]))
                self.dock.dataSlicer.setVisible(True)


        elif isinstance(value,str):
            self.dock.dataDimensionsLabel.setText('%s with dims %d'%(str(type(value))[1:-1],len(value)))
            self.dock.dataSlicer.setVisible(False)
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            if value == '_skeleton':
                tableItem = QTableWidgetItem('_skeleton\n(press F5 keytouch to load data)')
                font = tableItem.font()
                font.setItalic(True)
                brush = QtGui.QBrush()
                brush.setColor(QtGui.QColor("#BD3809"))
                tableItem.setForeground(brush)
                tableItem.setFont(font)
                self.table.setItem(0,0, tableItem)
            else:
                self.table.setItem(0,0, QTableWidgetItem(value))

        elif isinstance(value,list):
            self.dock.dataSlicer.setVisible(False)
            if not all([isinstance(v,str) for v in value]):
                raise ValueError('cannot show value of node %s'%node.name())
            Ni = len(value)
            self.dock.dataDimensionsLabel.setText('class list of str with dims %d'%Ni)
            self.table.setRowCount(Ni)
            self.table.setColumnCount(1)
            for i in range(Ni):
                self.table.setItem(i, 0, QTableWidgetItem('{}'.format(value[i])))

        elif value is None:
            self.dock.dataDimensionsLabel.setText('class None')
            self.dock.dataSlicer.setVisible(False)
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            tableItem = QTableWidgetItem('None')
            font = tableItem.font()
            font.setBold(True)
            brush = QtGui.QBrush()
            brush.setColor(QtGui.QColor("#8049d8"))
            tableItem.setForeground(brush)
            tableItem.setFont(font)
            self.table.setItem(0,0, tableItem)

        else:
            self.dock.dataSlicer.setVisible(False)
            self.dock.dataDimensionsLabel.setVisible(False)
            self.table.isBeingUpdated = False
            raise ValueError('type of value of node %s was "%s" and is not supported'%(node.name(),type(value)))

        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.table.isBeingUpdated = False
        # QApplication.restoreOverrideCursor()


    def createTable(self):
        # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.table = TableWithCopy()
        self.table.isBeingUpdated = True
        self.table.setAlternatingRowColors(True)
        self.table.setRowCount(1)
        self.table.setColumnCount(1)
        self.table.setItem(0,0, QTableWidgetItem("... and its data will be shown here"))
        self.table.move(0,0)
        self.table.horizontalHeader().setMinimumSectionSize(20)
        self.table.verticalHeader().setMinimumSectionSize(20)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.table.isBeingUpdated = False
        self.table.corner = self.table.findChild(QAbstractButton)
        self.table.corner.setToolTip('select all (Ctrl+A)')
        self.table.setVisible(False)

        self.table.itemChanged.connect(self.updateValueOfNodeCGNS)
        # QApplication.restoreOverrideCursor()


    def updateValueOfNodeCGNS(self, item):
        if self.table.isBeingUpdated: return
        # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        for index in self.tree.selectionModel().selectedIndexes():
            treeitem = self.tree.model.itemFromIndex(index)

        node_cgns = treeitem.node_cgns
        value = node_cgns.value()
        print('actual value is : '+str(value))
        new_value = item.text().strip()

        i = item.row()
        j = item.column()

        newlocals = {'array':value}

        try:
            if new_value == 'None':
                node_cgns.setValue(None)

            elif new_value in ['{}' , '[]']:
                node_cgns.setValue(value)

            elif new_value.startswith('[') and new_value.endswith(']'):
                newNumpyArray = np.array(eval(new_value,globals(),newlocals), order='F')
                node_cgns.setValue(newNumpyArray)

            elif new_value.startswith('{') and new_value.endswith('}'):
                expr = new_value[1:-1]
                if expr.startswith('this:'):
                    expr = expr.replace('this:','')
                    newNumpyValue = eval(expr,globals(),newlocals)
                    if isinstance(value,np.ndarray):
                        dim = len(value.shape)
                        if dim == 1:
                            value[i] = newNumpyValue
                        elif dim == 2:
                            value[i,j] = newNumpyValue
                        elif dim == 3:
                            planeIndex = self.dock.dataSlicer.ijkSelector.currentText()
                            planeValue = self.dock.dataSlicer.sliceSelector.value()
                            if planeIndex == 'k':
                                value[i,j,planeValue] = newNumpyValue
                            if planeIndex == 'j':
                                value[i,planeValue,j] = newNumpyValue
                            else:
                                value[planeValue,i,j] = newNumpyValue

                    elif isinstance(value, str) or value is None:
                        node_cgns.setValue(newNumpyArray)

                    elif isinstance(value, list):
                        value[i] = newNumpyValue

                else:
                    newNumpyArray = np.array(eval(expr,globals(),newlocals), order='F')
                    if isinstance(value, np.ndarray) and len(newNumpyArray.shape) == 0:
                        value[:] = newNumpyArray
                    else:
                        node_cgns.setValue(newNumpyArray)

            elif isinstance(value, np.ndarray):
                dim = len(value.shape)

                if dim == 1:
                    i = item.row()
                    if new_value == '':
                        value[i] = 0
                    else:
                        try:
                            value[i] = new_value
                        except ValueError:
                            node_cgns.setValue(new_value)

                elif dim == 2:
                    i = item.row()
                    j = item.column()

                    if new_value == '':
                        value[i,j] = 0
                    else:
                        try:
                            value[i,j] = new_value
                        except ValueError:
                            node_cgns.setValue(new_value)

                elif dim == 3:
                    i = item.row()
                    j = item.column()

                    planeIndex = self.dock.dataSlicer.ijkSelector.currentText()
                    planeValue = self.dock.dataSlicer.sliceSelector.value()
                    if planeIndex == 'k':
                        if new_value == '':
                            value[i,j,planeValue] = 0
                        else:
                            try:
                                value[i,j,planeValue] = new_value
                            except ValueError:
                                node_cgns.setValue(new_value)
                    elif planeIndex == 'j':
                        if new_value == '':
                            value[i,planeValue,j] = 0
                        else:
                            try:
                                value[i,planeValue,j] = new_value
                            except ValueError:
                                node_cgns.setValue(new_value)
                    else:
                        if new_value == '':
                            value[planeValue,i,j] = 0
                        else:
                            try:
                                value[planeValue,i,j] = new_value
                            except ValueError:
                                node_cgns.setValue(new_value)


            elif isinstance(value,str) or value is None:
                if new_value == '':
                    node_cgns.setValue(None)
                elif new_value[0].isdigit():
                    if '.' in new_value:
                        new_value = np.array([new_value],dtype=np.float64,order='F')
                    else:
                        new_value = np.array([new_value],dtype=np.int32,order='F')
                node_cgns.setValue(new_value)

            elif isinstance(value,list):
                i = item.row()
                if new_value == '':
                    if len(value) > 1:
                        del value[i]
                    else:
                        value = None
                else:
                    value[i] = new_value
                node_cgns.setValue(value)


        except BaseException as e:
            err_msg = ''.join(traceback.format_exception(etype=type(e),
                                value=e, tb=e.__traceback__))

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText(err_msg)
            msg.setWindowTitle("Error")
            msg.exec_()
            new_value = value
            node_cgns.setValue(new_value)

        self.updateTable()
        if isinstance(new_value, str) and new_value == '_skeleton':
            treeitem.isStyleCGNSbeingModified = True
            self.setStyleCGNS( treeitem )
            treeitem.isStyleCGNSbeingModified = False
        # QApplication.restoreOverrideCursor()


    def swapNodes(self, s):
        indices = self.tree.selectionModel().selectedIndexes()
        if len(indices) != 2:
            print('requires selecting 2 nodes for swapping')
            return
        else:
            # QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            item1 = self.tree.model.itemFromIndex(indices[0])
            item2 = self.tree.model.itemFromIndex(indices[1])

            for item in [item1, item2]:
                item.isStyleCGNSbeingModified = True

            node1 = item1.node_cgns
            node2 = item2.node_cgns


            item1Row = item1.row()
            item1Parent = item1.parent()
            if item1Parent:
                row1 = item1Parent.takeRow(item1Row)

            item2Row = item2.row()
            item2Parent = item2.parent()
            if item2Parent:
                row2 = item2Parent.takeRow(item2Row)

            if item1Parent and item2Parent:
                item2Parent.insertRow(item2Row, row1[0])
                item1Parent.insertRow(item1Row, row2[0])

            node1.QStandardItem = item1
            node2.QStandardItem = item2

            node1.swap(node2)

            self.tree.setSelectionMode(self.tree.MultiSelection)
            for item in [item1, item2]:
                self.setStyleCGNS(item)
                self.updateModelChildrenFromItem(item)
                index = self.tree.model.indexFromItem(item)
                self.tree.setCurrentIndex(index)
            self.tree.setSelectionMode(self.tree.ExtendedSelection)

            for item in [item1, item2]:
                item.isStyleCGNSbeingModified = False
        # QApplication.restoreOverrideCursor()

    def setStyleCGNS(self, MainItem):

        def putIcon(pathToIconImage):
            Icon = Qt.QIcon(pathToIconImage)
            MainItem.setIcon(Icon)

        node = MainItem.node_cgns
        MainItem.isStyleCGNSbeingModified = True
        font = MainItem.font()
        font.setPointSize( int(self.fontPointSize) )
        font.setBold(False)
        font.setItalic(False)
        brush = QtGui.QBrush()
        brush.setColor(QtGui.QColor("black"))
        pointSize = font.pointSize()
        iconSize = int(pointSize*1.333)
        self.tree.setIconSize(QtCore.QSize(iconSize,iconSize))
        MainItem.setSizeHint(QtCore.QSize(int(iconSize*1.5),int(iconSize*1.5)))
        MainItem.setIcon(Qt.QIcon())

        node_value = node.value()
        node_type = node.type()

        if node_type == 'CGNSTree_t':
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/tree")
            font.setBold(True)
        elif not MainItem.parent():
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/tree-red")
            font.setBold(True)
            brush.setColor(QtGui.QColor("red"))
        elif node_type == 'Zone_t':
            putIcon(GUIpath+"/icons/icons8/zone-2D.png")
            font.setBold(True)
            brush.setColor(QtGui.QColor("purple"))
        elif node_type == 'CGNSBase_t':
            font.setBold(True)
            font.setItalic(True)
            brush.setColor(QtGui.QColor("green"))
            putIcon(GUIpath+"/icons/icons8/icons8-box-32.png")
        elif node_type == 'GridCoordinates_t':
            putIcon(GUIpath+"/icons/icons8/icons8-coordinate-system-16.png")
        elif node.name() == 'CoordinateX':
            putIcon(GUIpath+"/icons/icons8/icons8-x-coordinate-16")
        elif node.name() == 'CoordinateY':
            putIcon(GUIpath+"/icons/icons8/icons8-y-coordinate-16")
        elif node.name() == 'CoordinateZ':
            putIcon(GUIpath+"/icons/icons8/icons8-z-coordinate-16")
        elif node_type == 'FlowSolution_t':
            putIcon(GUIpath+"/icons/OwnIcons/field-16")
        elif node_type in ('CGNSLibraryVersion_t','ZoneType_t'):
            font.setItalic(True)
        elif node_type == 'Link_t':
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/external.png")
            font.setBold(True)
            font.setItalic(True)
            brush.setColor(QtGui.QColor("blue"))
        elif node_type in ('Family_t','FamilyName_t','FamilyBC_t','AdditionalFamilyName_t'):
            putIcon(GUIpath+"/icons/icons8/icons8-famille-homme-femme-26.png")
            font.setItalic(True)
            brush = QtGui.QBrush()
            brush.setColor(QtGui.QColor("brown"))
        elif node_type == 'ConvergenceHistory_t':
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/system-monitor.png")
            font.setItalic(True)
        elif node_type == 'ZoneGridConnectivity_t':
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/plug-disconnect.png")
        elif node_type == 'ReferenceState_t':
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/script-attribute-r.png")
            font.setItalic(True)
        elif node_type == 'FlowEquationSet_t':
            putIcon(GUIpath+"/icons/icons8/Sigma.png")
            font.setItalic(True)
        elif node_type == 'UserDefinedData_t':
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/user-silhouette.png")
            font.setItalic(True)
            brush.setColor(QtGui.QColor("gray"))
        elif node_type == 'ZoneBC_t':
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/border-left.png")

        if isinstance(node_value,str) and node_value == '_skeleton':
            putIcon(GUIpath+"/icons/fugue-icons-3.5.6/arrow-circle-double.png")
            font.setBold(True)
            font.setItalic(True)
            brush.setColor(QtGui.QColor("orange"))


        MainItem.setForeground(brush)
        MainItem.setFont(font)
        MainItem.isStyleCGNSbeingModified = False

    def updateModel(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        nodes = self.t.group()

        self.tree.model.setRowCount(0)
        root = self.tree.model.invisibleRootItem()

        self.t.QStandardItem = QtGui.QStandardItem(self.t.name())
        self.t.QStandardItem.node_cgns = self.t
        root.appendRow([self.t.QStandardItem])
        self.setStyleCGNS(self.t.QStandardItem)

        for node in nodes:
            MainItem = node.QStandardItem = QtGui.QStandardItem(node.name())
            MainItem.node_cgns = node
            node.Parent.QStandardItem.appendRow([node.QStandardItem])
            self.setStyleCGNS(MainItem)
        self.tree.expandToDepth(1)
        QApplication.restoreOverrideCursor()

    def updateModelChildrenFromItem(self, item):
        if item.hasChildren():
            node = item.node_cgns
            CGNSchildren = node.children()
            for row in range(item.rowCount()):
                for col in range(item.columnCount()):
                    child = item.child(row, col)
                    try:
                        child.node_cgns = CGNSchildren[row]
                    except IndexError:
                        raise ValueError('could not retrieve child %s (%d) for node %s '%(child.text(),row,node.name()))
                    child.node_cgns.item = child
                    self.updateModelChildrenFromItem(child)

class TableWithCopy(QTableWidget):
    """
    this class extends QTableWidget
    * supports copying multiple cell's text onto the clipboard
    * formatted specifically to work with multiple-cell paste into programs
      like google sheets, excel, or numbers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

        if event.key() == QtCore.Qt.Key_C and (event.modifiers() & QtCore.Qt.ControlModifier):
            copied_cells = self.selectedIndexes()
            self.copied_data = []

            copy_text = ''
            max_column = copied_cells[-1].column()
            for elt, c in enumerate(copied_cells):
                i = c.row()
                j = c.column()
                item = self.item(i,j)
                self.copied_data.append( item.text() )
                copy_text += item.text()
                if c.column() == max_column:
                    copy_text += '\n'
                else:
                    copy_text += '\t'
            Qt.QApplication.clipboard().setText(copy_text)

        elif event.key() == QtCore.Qt.Key_V and (event.modifiers() & QtCore.Qt.ControlModifier):
            pasting_cells = self.selectedIndexes()
            for copy, paste in zip(self.copied_data, pasting_cells):
                self.setItem(paste.row(), paste.column(), QTableWidgetItem(copy))


        elif event.key() == QtCore.Qt.Key_Delete:
            deleting_cells = sorted(self.selectedIndexes())
            for delete in deleting_cells[::-1]:
                self.setItem(delete.row(), delete.column(), QTableWidgetItem(''))

        elif event.key() == QtCore.Qt.Key_A and (event.modifiers() & QtCore.Qt.ControlModifier):
            self.selectAll()


class FindNodeDialog(QDialog):
    def __init__(self, PreviousName, PreviousValue, PreviousType, DepthToBeFound):
        super().__init__()

        self.setWindowTitle("Find Node...")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QFormLayout()
        self.searchFromTop = QRadioButton('Top')
        self.searchFromTop.setChecked(True)
        self.searchFromSelection = QRadioButton('Selection')
        self.searchLayout = QHBoxLayout()
        self.searchLayout.layout().addWidget(self.searchFromTop)
        self.searchLayout.layout().addWidget(self.searchFromSelection)
        self.layout.addRow(QLabel("search from:"), self.searchLayout)
        self.NameWidget = QLineEdit(PreviousName)
        if PreviousValue is not None:
            self.ValueWidget = QLineEdit(str(PreviousValue))
        else:
            self.ValueWidget = QLineEdit()
        self.TypeWidget = QLineEdit(PreviousType)
        self.DepthWidget = QSpinBox()
        self.DepthWidget.setValue(DepthToBeFound)
        self.layout.addRow(QLabel("Name:"),  self.NameWidget)
        self.layout.addRow(QLabel("Value:"), self.ValueWidget)
        self.layout.addRow(QLabel("Type:"),  self.TypeWidget)
        self.layout.addRow(QLabel("Depth:"),  self.DepthWidget)
        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)


class NewNodeDialog(QDialog):
    def __init__(self, NodeParentLabel):
        super().__init__()

        self.setWindowTitle("New Node...")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel


        self.layout = QFormLayout()
        NodeParentQ = QLineEdit(NodeParentLabel)
        NodeParentQ.setReadOnly(True)
        NodeParentQ.setStyleSheet("QLineEdit {background : rgb(220, 224, 230); color : rgb(41, 43, 46);}")
        self.layout.addRow(QLabel("Parent:"), NodeParentQ)
        self.NameWidget = QLineEdit('NodeName')
        self.ValueWidget = QLineEdit('None')
        self.TypeWidget = QLineEdit('DataArray_t')
        self.layout.addRow(QLabel("Name:"),  self.NameWidget)
        self.layout.addRow(QLabel("Value:"), self.ValueWidget)
        self.layout.addRow(QLabel("Type:"),  self.TypeWidget)

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)


def launch( args='' ):
    if isinstance(args, str): args = args.split(' ')

    filename = [f for f in args if f.endswith('.cgns')]
    if filename: filename = filename[0]
    
    only_skeleton = any([f for f in args if f=='-s'])

    app = Qt.QApplication( sys.argv )
    app.setWindowIcon(QtGui.QIcon(os.path.join(GUIpath,'..','MOLA.svg')))
    print('filename=',filename)
    print('only_skeleton=',only_skeleton)
    MW = MainWindow( filename, only_skeleton )
    MW.resize(650, 800)
    MW.show()
    sys.exit( app.exec_() )

if __name__ == "__main__" :
    launch( sys.argv )
