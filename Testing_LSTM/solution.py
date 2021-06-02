import sys
import random
import time
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QBasicTimer, Qt, QDate, QRegExp
from PyQt5.QtGui import QIcon, QFont, QRegExpValidator, QIntValidator, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from test import Test


WIDTH = 850
HEIGHT = 550

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.step = 0
        self.default_stock_item = "종목 선택"
        self.default_algo_item = "알고리즘 선택"
        self.stock_list = ['ExxonMobil(XOM)', 'Chevron(CVX)', 'CimarexEnergy(XEC)', 'HessCorp(HES)']
        self.algo_list = ['DDPG','TD3','AC3','DQN']
        self.selectedStock = ""
        self.selectedAlgo = ""
        boldFont = QtGui.QFont()
        boldFont.setFamily("Times")
        boldFont.setBold(True)

        self.setWindowTitle('Stock price prediction solution')
        self.setWindowIcon(QIcon('stock.png'))
        self.setGeometry(500, 500, WIDTH, HEIGHT)

        labelStock = QLabel(self.default_stock_item, self)
        labelStock.setAlignment(Qt.AlignCenter)
        labelStock.setFont(boldFont)
        
        labelAlgo = QLabel(self.default_algo_item, self)
        labelAlgo.setAlignment(Qt.AlignCenter)
        labelAlgo.setFont(boldFont)

        labelMoney = QLabel("투자 금액 선택", self)
        labelMoney.setAlignment(Qt.AlignCenter)
        labelMoney.setFont(boldFont)

        labelStartDate = QLabel("시작 기간", self)
        labelStartDate.setAlignment(Qt.AlignCenter)
        labelStartDate.setFont(boldFont)

        labelEndDate = QLabel("종료 기간", self)
        labelEndDate.setAlignment(Qt.AlignCenter)
        labelEndDate.setFont(boldFont)

        self.moneyEdit = QLineEdit()
        self.moneyEdit.setPlaceholderText("최소 10만원 최대 1000만원")
        regExp = QRegExp("[0-9]*")
        self.moneyEdit.setValidator(QRegExpValidator(regExp,self))
        self.moneyEdit.setValidator(QIntValidator(100000, 10000000))

        self.startDateEdit = QDateEdit(self)
        self.startDateEdit.setDate(QDate.currentDate())
        self.startDateEdit.setMinimumDate(QDate(2014, 1, 1))
        self.startDateEdit.setMaximumDate(QDate.currentDate())

        self.endDateEdit = QDateEdit(self)
        self.endDateEdit.setDate(QDate.currentDate())
        self.endDateEdit.setMinimumDate(QDate(2014, 1, 2))
        self.endDateEdit.setMaximumDate(QDate.currentDate())

        self.statusLabel = QLabel("Made by 2U2U", self)
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setFont(boldFont)

        self.stockCb = QComboBox(self)
        self.stockCb.setFont(QFont('SansSerit', 10))
        self.stockCb.addItem(self.default_stock_item)
        self.stockCb.addItems(self.stock_list)
        self.stockCb.activated[str].connect(self.onStockActivated)

        self.algoCb = QComboBox(self)
        self.algoCb.setFont(QFont('SansSerit', 10))
        self.algoCb.addItem(self.default_algo_item)
        self.algoCb.addItems(self.algo_list)
        self.algoCb.activated[str].connect(self.onAlgoActivated)

        self.btn = QPushButton('투자 추천받기', self)
        self.btn.setFont(QFont('SansSerit', 10))
        self.btn.setToolTip('AI가 해당 종목에 대한 투자를 추천해줍니다.')
        self.btn.clicked.connect(self.progressUp)
        self.btn.setFont(boldFont)

        self.progressbar = QProgressBar()
        self.progressbar.setValue(self.step)
        self.timer = QBasicTimer()

        # self.fig = plt.Figure()
        # self.canvas = FigureCanvas(self.fig)

        leftInnerLayOut = QVBoxLayout()
        leftInnerLayOut.addWidget(labelStock)
        leftInnerLayOut.addWidget(self.stockCb)
        #leftInnerLayOut.addStretch(1)
        leftInnerLayOut.addWidget(labelAlgo)
        leftInnerLayOut.addWidget(self.algoCb)
        #leftInnerLayOut.addStretch(1)
        leftInnerLayOut.addWidget(labelMoney)
        leftInnerLayOut.addWidget(self.moneyEdit)
        #leftInnerLayOut.addStretch(1)
        leftInnerLayOut.addWidget(labelStartDate)
        leftInnerLayOut.addWidget(self.startDateEdit)
        #leftInnerLayOut.addStretch(1)
        leftInnerLayOut.addWidget(labelEndDate)
        leftInnerLayOut.addWidget(self.endDateEdit)
        leftInnerLayOut.addStretch(2)
        leftInnerLayOut.addWidget(self.btn)
        leftInnerLayOut.addStretch(2)
        leftInnerLayOut.addWidget(self.statusLabel)
        leftInnerLayOut.addWidget(self.progressbar)

        leftLayOut = QVBoxLayout()
        leftLayOut.addLayout(leftInnerLayOut)

        self.rightLayOut = QVBoxLayout()
        #rightLayOut.addWidget(self.canvas)

        layout = QHBoxLayout()
        layout.addLayout(leftLayOut)
        layout.addLayout(self.rightLayOut)
        layout.setStretch(0, 1)
        layout.setStretch(1, 4)

        self.setLayout(layout)


    def onStockActivated(self, text):
        if text == self.default_stock_item:
            QMessageBox.about(self, "종목 선택 오류", "종목을 선택해주세요.")
            self.stockCb.setCurrentText(self.default_stock_item)
        else:
            self.selectedStock = text


    def onAlgoActivated(self, text):
        if text == self.default_algo_item:
            QMessageBox.about(self, "알고리즘 선택 오류", "알고리즘을 선택해주세요.")
            self.algoCb.setCurrentText(self.default_algo_item)
        else:
            self.selectedAlgo = text

    def progressUp(self):
        if str(self.stockCb.currentText()) == self.default_stock_item:
            QMessageBox.about(self, "종목 선택 오류", "종목을 선택해주세요.")
            return 
        if str(self.algoCb.currentText()) == self.default_algo_item:
            QMessageBox.about(self, "알고리즘 선택 오류", "알고리즘을 선택해주세요.")
            return 

        self.startDate = self.startDateEdit.date().toPyDate().strftime("%Y%m%d")[2:]
        self.endDate = self.endDateEdit.date().toPyDate().strftime("%Y%m%d")[2:]

        if self.timer.isActive():
            self.timer.stop()
        else:
            #self.progressbar.setRange(0,0)
            self.timer.start(100, self)
            self.step = 0

    def timerEvent(self, e):
        if self.step >= 100:
            self.showResult()
            self.statusLabel.setText("투자행동 판단 완료.")
            QMessageBox.about(self,"투자행동 판단 결과", self.selectedStock + " 종목에 대해\n" + self.selectedAlgo + " 알고리즘으로 수행한 결과는 " + self.action + "입니다.")
            self.timer.stop()
            return
        if self.step == 0:
            Test(stock_code=self.selectedStock, rl_method=self.selectedAlgo, balance=10000, start_date=self.startDate, end_date=self.endDate)
        self.statusLabel.setText("해당 종목\n 최적의 투자행동 판단 중..")
        #self.progressbar.setRange(0,100)
        self.step += 1
        self.progressbar.setValue(self.step)
        return


    def showResult(self):
        # data = [random.random() for i in range(250)]
        # ax = self.fig.add_subplot(111)
        # ax.plot(data, 'r-', linewidth = 0.5)
        # ax.set_title('PyQt Matplotlib Example')
        # self.canvas.draw()

        for i in reversed(range(self.rightLayOut.count())): 
            self.rightLayOut.itemAt(i).widget().setParent(None)
        
        if self.selectedStock == "Exxon Mobil(XOM)":
            if self.selectedAlgo == "DDPG":
                pixmap = QPixmap('../Stock_Trader/test_xon_None_actorcritic/epoch_summary_xon/xom_result.png')
                self.action = "매도"

        if self.selectedStock == "Hess Corp(HES)":
            if self.selectedAlgo == "DDPG":
                pixmap = QPixmap('../Stock_Trader/test_hes_None_actorcritic/epoch_summary_hes/hes_result.png')
                self.action = "매수"

        result = QLabel(self)
        result.setPixmap(pixmap) # 이미지 세팅
        result.setContentsMargins(10, 10, 10, 10)
        result.resize(pixmap.width(), pixmap.height())
        self.rightLayOut.addWidget(result)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()