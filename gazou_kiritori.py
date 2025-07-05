import sys
import os
import traceback
from io import BytesIO
from PyQt6.QtGui import QActionGroup
from PyQt6.QtGui import QAction
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QShortcut, QKeySequence
from PIL import Image, ImageQt
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6 import QtWidgets, QtCore, QtGui
import sys

class CustomListView(QtWidgets.QListView):
    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            event.ignore()
        else:
            super().keyPressEvent(event)


# --- ズーム倍率表示ラベル ---
class ZoomLabel(QtWidgets.QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(20, 20, 20, 140);
                border-radius: 10px;
                padding: 4px 14px;
                font-weight: bold;
                font-size: 18px;
            }
        """)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()
        self.opacity_anim = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        self.opacity_anim.setDuration(500)
        self.opacity_anim.setStartValue(1.0)
        self.opacity_anim.setEndValue(0.0)
        self._fade_timer = QtCore.QTimer(self)
        self._fade_timer.setSingleShot(True)
        self._fade_timer.timeout.connect(self._start_fade)
        self.opacity_anim.finished.connect(self.hide)

    def show_zoom(self, value: float):
        percent = int(round(value * 100))
        self.setText(f"{percent}%")
        self.adjustSize()
        self.move(12, 12)
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self.opacity_anim.stop()
        self._fade_timer.start(2000)  # 2秒後にフェード開始

    def _start_fade(self):
        self.opacity_anim.setStartValue(1.0)
        self.opacity_anim.setEndValue(0.0)
        self.opacity_anim.setDuration(700)
        self.opacity_anim.start()

def make_fixed_thumbnail(img_path, thumb_size=(80, 120)):
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail(thumb_size, Image.LANCZOS)
        canvas = Image.new("RGB", thumb_size, (240, 240, 240))
        offset = (
            (thumb_size[0] - img.width) // 2,
            (thumb_size[1] - img.height) // 2,
        )
        canvas.paste(img, offset)
        return canvas
    except Exception as e:
        print(f"サムネイル作成エラー: {img_path} {e}")
        return Image.new("RGB", thumb_size, (60, 60, 60))

class SuccessLabel(QtWidgets.QLabel):
    def __init__(self, parent=None, pos=QtCore.QPoint(0, 0), message="✔Success!", timeout=1500):
        super().__init__(parent)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.setFont(font)
        self.setText(message)
        self.setStyleSheet("color: green; background: #ffffffcc; border-radius: 8px; padding: 6px;")
        self.adjustSize()
        self.move(pos)
        self.show()
        QtCore.QTimer.singleShot(timeout, self.close)

class ActionPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, pos=QtCore.QPoint(0, 0), on_save=None, on_cancel=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        layout = QtWidgets.QHBoxLayout(self)
        btn_save = QtWidgets.QPushButton("保存")
        btn_cancel = QtWidgets.QPushButton("キャンセル")
        font = btn_save.font()
        font.setPointSize(11)
        btn_save.setFont(font)
        btn_cancel.setFont(font)
        btn_save.setMinimumHeight(28)
        btn_cancel.setMinimumHeight(28)
        btn_save.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        btn_cancel.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        btn_save.setStyleSheet("""
            QPushButton {background: #4caf50; color: white; font-weight: bold; border-radius: 6px; padding: 4px 16px;}
            QPushButton:hover {background: #388e3c;}
        """)
        btn_cancel.setStyleSheet("""
            QPushButton {background: #e0e0e0; color: #333; font-weight: bold; border-radius: 6px; padding: 4px 16px;}
            QPushButton:hover {background: #bdbdbd;}
        """)
        layout.addWidget(btn_save)
        layout.addWidget(btn_cancel)
        layout.setContentsMargins(4, 4, 4, 4)
        btn_save.clicked.connect(on_save)
        btn_cancel.clicked.connect(on_cancel)
        self.adjustSize()
        self.move(pos)
        self.show()
        self.btn_save = btn_save

class CropLabel(QtWidgets.QLabel):
    selectionMade = QtCore.pyqtSignal(QtCore.QRect, QtCore.QPoint)
    fixedSelectionMade = QtCore.pyqtSignal(QtCore.QRect, QtCore.QPoint)
    movedRect = QtCore.pyqtSignal(QtCore.QRect)

    def __init__(self, mainwin):
        super().__init__(mainwin)
        self.mainwin = mainwin

        self.drag_rect_img = None
        self.fixed_crop_rect_img = None    
        self.fixed_crop_rect_img = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.drag_rect = None
        self.drag_origin = None
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.fixed_crop_mode = False
        self.fixed_crop_rect = None
        self.fixed_crop_size = None
        self.fixed_crop_drag_offset = None
        self.zoom_mode = False
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._gesture_start = None
        self._gesture_in_progress = False
        self._gesture_start = None
        self._gesture_in_progress = False
        self._gesture_current = None
        self.setStyleSheet("""background: #191919;border: 2px solid #f28524; border-radius: 12px;""")
        self._pan_active = False
        self._pan_start_pos = None
        self._pan_offset_x = 0
        self._pan_offset_y = 0

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        # 画像の中央寄せ時のオフセット
        label_w, label_h = self.width(), self.height()
        pm_w, pm_h = pixmap.width(), pixmap.height()
        self.offset_x = (label_w - pm_w) // 2 if label_w > pm_w else 0
        self.offset_y = (label_h - pm_h) // 2 if label_h > pm_h else 0


    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.MiddleButton and self.mainwin.zoom_scale > 1.0:
            self._pan_active = True
            self._pan_start_pos = event.pos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._gesture_start = event.pos()
            self._gesture_current = event.pos()
            return

        if self.fixed_crop_mode:
            if self.fixed_crop_rect_img is not None:
                # 固定枠のドラッグ判定はラベル座標化してチェックする
                x1, y1 = self.image_to_label_coords(self.fixed_crop_rect_img.left(), self.fixed_crop_rect_img.top())
                x2, y2 = self.image_to_label_coords(self.fixed_crop_rect_img.right(), self.fixed_crop_rect_img.bottom())
                rect_x = min(x1, x2)
                rect_y = min(y1, y2)
                rect_w = abs(x2 - x1)
                rect_h = abs(y2 - y1)
                rect = QtCore.QRect(rect_x, rect_y, rect_w, rect_h)
                if rect.contains(event.pos()):
                    self.fixed_crop_drag_offset = event.pos() - rect.topLeft()
        else:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # ドラッグ開始（ピクセル座標で記録）
                img_x, img_y = self.mainwin.label_to_image_coords(event.position().x(), event.position().y())
                self._drag_start_img = (img_x, img_y)
                self.drag_rect_img = None  # ←ドラッグ開始時に初期化
                self.update()

    def mouseMoveEvent(self, event):
        if getattr(self, '_pan_active', False):
            dx = event.pos().x() - self._pan_start_pos.x()
            dy = event.pos().y() - self._pan_start_pos.y()
            self.mainwin.pan_image(-dx, -dy)
            self._pan_start_pos = event.pos()
            return

        if getattr(self, '_gesture_start', None):
            self._gesture_current = event.pos()
            self.update()
            return

        if getattr(self, 'zoom_mode', False):
            return

        if self.fixed_crop_mode:
            # 固定枠のドラッグ移動
            if self.fixed_crop_drag_offset is not None and self.fixed_crop_rect_img is not None:
                new_topleft = event.pos() - self.fixed_crop_drag_offset
                # 新しいラベル座標 → 元画像ピクセル座標に変換
                img_left, img_top = self.mainwin.label_to_image_coords(new_topleft.x(), new_topleft.y())
                crop_w, crop_h = self.fixed_crop_size
                self.fixed_crop_rect_img = QtCore.QRect(img_left, img_top, crop_w, crop_h)
                self.update()
        else:
            # 通常ドラッグ枠
            if hasattr(self, '_drag_start_img') and self._drag_start_img is not None:
                # 現在位置も元画像ピクセル座標に変換
                img_x, img_y = self.mainwin.label_to_image_coords(event.position().x(), event.position().y())
                x1, y1 = self._drag_start_img
                left = min(x1, img_x)
                top = min(y1, img_y)
                right = max(x1, img_x)
                bottom = max(y1, img_y)
                self.drag_rect_img = QtCore.QRect(left, top, right - left, bottom - top)
                self.update()

    def mouseReleaseEvent(self, event):
        if getattr(self, '_pan_active', False):
            self._pan_active = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            return

        if event.button() == QtCore.Qt.MouseButton.RightButton and self._gesture_start:
            dx = event.pos().x() - self._gesture_start.x()
            if dx > 100:
                self.mainwin.show_next_image()
            elif dx < -100:
                self.mainwin.show_prev_image()
            self._gesture_start = None
            self._gesture_current = None
            return
        
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            return

        if self.fixed_crop_mode:
            if self.fixed_crop_drag_offset is not None and self.fixed_crop_rect_img is not None:
                self.fixed_crop_drag_offset = None
                # emitはラベル座標に変換して渡す！
                x1, y1 = self.image_to_label_coords(self.fixed_crop_rect_img.left(), self.fixed_crop_rect_img.top())
                x2, y2 = self.image_to_label_coords(self.fixed_crop_rect_img.right(), self.fixed_crop_rect_img.bottom())
                rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                self.fixedSelectionMade.emit(rect, event.pos())

        # --- ドラッグ選択矩形（ドラッグ開始・終了点が存在し、幅高さが一定以上の場合のみ）---
        if hasattr(self, "_drag_start_img") and self._drag_start_img is not None and self.drag_rect_img is not None:
            gx1, gy1 = self.drag_rect_img.left(), self.drag_rect_img.top()
            gx2, gy2 = self.drag_rect_img.right(), self.drag_rect_img.bottom()
            if abs(gx2-gx1) > 5 and abs(gy2-gy1) > 5:
                # ラベル座標へ変換
                x1, y1 = self.image_to_label_coords(gx1, gy1)
                x2, y2 = self.image_to_label_coords(gx2, gy2)
                rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
                self.selectionMade.emit(rect, event.pos())  # ←コレ！！
            self._drag_start_img = None
            self.update()

    def wheelEvent(self, event):
            if event.angleDelta().y() > 0:
                self.mainwin.zoom_in()
            else:
                self.mainwin.zoom_out()

    def clear_rubberBand(self):
        self.drag_rect = None
        self.drag_origin = None
        self.drag_rect_img = None 
        self.update()
    def clear_fixed_crop(self):
        self.fixed_crop_mode = False
        self.fixed_crop_rect = None
        self.fixed_crop_size = None
        self.fixed_crop_drag_offset = None
        self.fixed_crop_rect_img = None
        self.update()

    def image_to_label_coords(self, gx, gy):
        if self.mainwin.image is None:
        # 画像が読み込まれていない場合は何もしない or (0,0)を返す
            return 0, 0
        lw, lh = self.width(), self.height()
        base_w, base_h = self.mainwin.base_display_width, self.mainwin.base_display_height
        zoom = self.mainwin.zoom_scale
        offset_x = getattr(self, "_pan_offset_x", 0)
        offset_y = getattr(self, "_pan_offset_y", 0)
        img_w, img_h = self.mainwin.image.width, self.mainwin.image.height

        pm_w = int(base_w * zoom)
        pm_h = int(base_h * zoom)
        # 中央基準位置（パンオフセットを逆方向で適用！）
        center_x = pm_w // 2 + offset_x
        center_y = pm_h // 2 + offset_y

        px = gx * (pm_w / img_w)
        py = gy * (pm_h / img_h)
        lx = px - (center_x - lw // 2)
        ly = py - (center_y - lh // 2)
        return int(round(lx)), int(round(ly))
    
    def start_fixed_crop(self, crop_size):
        """
        固定切り出し枠を画像中央に配置し、ズーム・パン状態でも正しくラベル上に表示
        """
        self.fixed_crop_mode = True
        self.fixed_crop_size = crop_size

        if self.mainwin.image is None:
            self.fixed_crop_rect = None
            self.update()
            return

        img_w, img_h = self.mainwin.image.width, self.mainwin.image.height
        crop_w, crop_h = crop_size

        # 1. 元画像中央に crop_w×crop_h の矩形を配置（画像ピクセル座標）
        left = max(0, (img_w - crop_w) // 2)
        top = max(0, (img_h - crop_h) // 2)
        self.fixed_crop_rect_img = QtCore.QRect(left, top, crop_w, crop_h)

        pm = self.pixmap()
        if not pm:
            self.fixed_crop_rect = None
            self.update()
            return

        # 2. 元画像の矩形 → ラベル座標に変換（ズーム・パン考慮）
        x1, y1 = self.image_to_label_coords(left, top)
        x2, y2 = self.image_to_label_coords(left + crop_w, top + crop_h)

        rect_x = min(x1, x2)
        rect_y = min(y1, y2)
        rect_w = abs(x2 - x1)
        rect_h = abs(y2 - y1)

        self.fixed_crop_rect = QtCore.QRect(rect_x, rect_y, rect_w, rect_h)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
            
        if self.mainwin.image is None:
            return

        # 固定切り出し枠を毎回再計算
        if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
            x1, y1 = self.image_to_label_coords(self.fixed_crop_rect_img.left(), self.fixed_crop_rect_img.top())
            x2, y2 = self.image_to_label_coords(self.fixed_crop_rect_img.right(), self.fixed_crop_rect_img.bottom())
            rect_x = min(x1, x2)
            rect_y = min(y1, y2)
            rect_w = abs(x2 - x1)
            rect_h = abs(y2 - y1)
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.blue, 2, QtCore.Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(30, 120, 255, 60))
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)

        # ドラッグでの矩形選択（ラベル上の始点・終点を元画像ピクセル座標に変換して描画）
        elif self.drag_rect_img is not None:
            gx1, gy1 = self.drag_rect_img.left(), self.drag_rect_img.top()
            gx2, gy2 = self.drag_rect_img.right(), self.drag_rect_img.bottom()
            x1, y1 = self.image_to_label_coords(gx1, gy1)
            x2, y2 = self.image_to_label_coords(gx2, gy2)
            rect_x = min(x1, x2)
            rect_y = min(y1, y2)
            rect_w = abs(x2 - x1)
            rect_h = abs(y2 - y1)
            pen = QtGui.QPen(QtGui.QColor("red"), 2, QtCore.Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 60))
            painter.setBrush(brush)
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)

        # ジェスチャー軌跡
        if (
            hasattr(self, "_gesture_in_progress")
            and self._gesture_in_progress
            and self._gesture_start
            and self._gesture_current
        ):
            pen = QtGui.QPen(QtGui.QColor("red"), 2)
            painter.setPen(pen)
            painter.drawLine(self._gesture_start, self._gesture_current)

    def keyPressEvent(self, event):
        key = event.key()
        moved = False
        # --- 固定枠の場合 ---
        if self.fixed_crop_mode and self.fixed_crop_rect_img is not None:
            rect = self.fixed_crop_rect_img
            step = 1  # 移動ピクセル数
            if key == QtCore.Qt.Key.Key_Left:
                rect.moveLeft(rect.left() - step)
                moved = True
            elif key == QtCore.Qt.Key.Key_Right:
                rect.moveLeft(rect.left() + step)
                moved = True
            elif key == QtCore.Qt.Key.Key_Up:
                rect.moveTop(rect.top() - step)
                moved = True
            elif key == QtCore.Qt.Key.Key_Down:
                rect.moveTop(rect.top() + step)
                moved = True
            if moved:
                self.fixed_crop_rect_img = rect
                self.update()
                # ラベル座標に変換してシグナルemit
                x1, y1 = self.image_to_label_coords(rect.left(), rect.top())
                x2, y2 = self.image_to_label_coords(rect.right(), rect.bottom())
                label_rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                self.fixedSelectionMade.emit(label_rect, QtCore.QPoint(x1, y1))
        # --- ドラッグ枠の場合 ---
        elif self.drag_rect_img is not None:
            rect = self.drag_rect_img
            step = 1
            if key == QtCore.Qt.Key.Key_Left:
                rect.translate(-step, 0)
                moved = True
            elif key == QtCore.Qt.Key.Key_Right:
                rect.translate(step, 0)
                moved = True
            elif key == QtCore.Qt.Key.Key_Up:
                rect.translate(0, -step)
                moved = True
            elif key == QtCore.Qt.Key.Key_Down:
                rect.translate(0, step)
                moved = True
            if moved:
                self.drag_rect_img = rect
                self.update()
                # ラベル座標に変換してemit
                x1, y1 = self.image_to_label_coords(rect.left(), rect.top())
                x2, y2 = self.image_to_label_coords(rect.right(), rect.bottom())
                label_rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                self.selectionMade.emit(label_rect, QtCore.QPoint(x1, y1))
        else:
            super().keyPressEvent(event)

class ThumbnailListModel(QtCore.QAbstractListModel):
    def __init__(self, image_list, thumb_size=(80, 120)):
        super().__init__()
        self.image_list = image_list
        self.thumb_size = thumb_size
        self.thumbnails = []
        for path in self.image_list:
            try:
                img = make_fixed_thumbnail(path, self.thumb_size)
                buf = BytesIO()
                img.save(buf, format='PNG')
                qt_thumb = QtGui.QPixmap()
                qt_thumb.loadFromData(buf.getvalue(), "PNG")
            except Exception as e:
                print("サムネイル作成エラー:", path, e)
                qt_thumb = QtGui.QPixmap(self.thumb_size[0], self.thumb_size[1])
                qt_thumb.fill(QtGui.QColor(60, 60, 60))
            self.thumbnails.append(qt_thumb)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.image_list)

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            return QtGui.QIcon(self.thumbnails[index.row()])

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return os.path.basename(self.image_list[index.row()])

        if role == QtCore.Qt.ItemDataRole.UserRole:
            return self.image_list[index.row()]

        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            path = self.image_list[index.row()]
            try:
                from PIL import Image
                img = Image.open(path)
                size_str = f"{img.width} x {img.height}"
            except Exception:
                size_str = "取得失敗"
            try:
                size_kb = os.path.getsize(path) // 1024
            except Exception:
                size_kb = "?"
            return (f"ファイル名: {os.path.basename(path)}\n"
                    f"解像度: {size_str}\n"
                    f"サイズ: {size_kb} KB")
        return None
    
class CropperApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("画像切り取りツール")
        self.resize(1000, 700)
        self.base_display_width = None
        self.base_display_height = None
        self.save_folder = None

        # --- 左カラム：プレビュー用 QLabel 宣言 ---
        self.preview_label = QtWidgets.QLabel(self)
        self.preview_label.setMinimumSize(512, 512)
        self.preview_label.setMaximumSize(512, 512)
        self.preview_label.setStyleSheet("background:#333; border:1px solid #aaa;")

        # --- 追加情報パネル（サブパネル） ---
        self.sub_panel = QtWidgets.QWidget()
        self.sub_panel.setMinimumHeight(370)
        #self.sub_panel.setFixedHeight(370)  # 調整可能
        self.sub_panel.setStyleSheet("background: #353535; border-radius: 10px;")
        self.sub_panel_layout = QtWidgets.QVBoxLayout(self.sub_panel)
        self.sub_panel_layout.setContentsMargins(12, 10, 12, 10)
        self.crop_size_label = QtWidgets.QLabel("0 x 0")
        self.crop_size_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.crop_size_label.setStyleSheet("""
            QLabel {
                color: #222;
                background: #fff;
                border: 1px solid #2c405a;
                border-radius: 8px;
                font-size: 32px;
                font-family: 'Consolas', 'monospace', 'Meiryo';
                font-weight: bold;
                padding: 10px 20px;
            }
        """)
        self.sub_panel_layout.addWidget(self.crop_size_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        main_widget = QtWidgets.QWidget()

        # --- 外側縦レイアウト ---
        outer_layout = QtWidgets.QVBoxLayout(main_widget)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(0)

        # --- 横並びメインレイアウト（左・中央・右カラム） ---
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 左カラム（縦レイアウト） ---
        preview_area = QtWidgets.QWidget()
        preview_area.setMinimumWidth(540)
        preview_area.setMaximumWidth(540)
        preview_layout = QtWidgets.QVBoxLayout(preview_area)
        preview_layout.setContentsMargins(14, 20, 14, 20)
        preview_layout.addWidget(self.preview_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop)
        preview_area.setStyleSheet("border: 2px solid #4b94d9; border-radius: 14px; background: #222;")
        self.preview_label.setStyleSheet("background: #fff; border: 2px solid #aaa; border-radius: 0px;")

        preview_column = QtWidgets.QVBoxLayout()
        preview_column.setContentsMargins(0, 0, 0, 0)
        preview_column.setSpacing(8)
        preview_column.addWidget(preview_area)
        preview_column.addWidget(self.sub_panel)

        main_layout.addLayout(preview_column, stretch=0)

        # --- 中央カラム用パネル ---
        self.central_panel = QtWidgets.QWidget()
        self.central_layout = QtWidgets.QVBoxLayout(self.central_panel)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(8)

        # --- 横並びパネル（戻るボタン、情報ラベル、進むボタン） ---
        self.info_panel = QtWidgets.QWidget()
        info_layout = QtWidgets.QHBoxLayout(self.info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(6)

        self.btn_prev = QtWidgets.QPushButton("◁")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.image_info_label = QtWidgets.QLabel("")
        self.image_info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_info_label.setStyleSheet("color: white; background: #444; font-size: 14px;")

        self.btn_next = QtWidgets.QPushButton("▷")
        self.btn_next.setFixedWidth(30)
        self.btn_next.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        info_layout.addWidget(self.btn_prev)
        info_layout.addWidget(self.image_info_label, stretch=1)
        info_layout.addWidget(self.btn_next)

        # --- 画像表示用 CropLabel ---
        self.label = CropLabel(self)
        self.label.setMinimumSize(1, 1)
        self.central_layout.addWidget(self.label)
        self.central_layout.addWidget(self.info_panel)
        
        main_layout.addWidget(self.central_panel, stretch=1)

        # --- 右カラム：サムネイルリスト ---
        self.listview = CustomListView()
        self.listview.setMaximumWidth(200)
        self.listview.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.listview.setIconSize(QtCore.QSize(160, 240))
        self.listview.setGridSize(QtCore.QSize(180, 260))
        self.listview.setSpacing(16)
        self.listview.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.listview.setMovement(QtWidgets.QListView.Movement.Static)

        main_layout.addWidget(self.listview, stretch=0)

        outer_layout.addLayout(main_layout, stretch=1)

        self.setCentralWidget(main_widget)

        # ボタンのクリックイベント接続
        self.btn_prev.clicked.connect(self.show_prev_image)
        self.btn_next.clicked.connect(self.show_next_image)
        
        menubar = self.menuBar()
        file_menu = menubar.addMenu("ファイル")
        open_action = file_menu.addAction("画像を開く")
        open_action.triggered.connect(self.open_image)
        file_menu.addSeparator()
        save_folder_select = file_menu.addAction("保存先フォルダを設定")
        save_folder_select.triggered.connect(self.set_save_folder)

        # 固定切り出しメニュー
        crop_menu = menubar.addMenu("固定切り出し")
        self.crop_action_group = QActionGroup(self)
        self.crop_action_group.setExclusive(True)
        self.crop_actions = {}
        fixed_sizes = [
            ("1024 × 1024", (1024, 1024)),
            ("1216 × 832", (1216, 832)),
            ("832 × 1216", (832, 1216)),
            ("1536 × 1536", (1536, 1536)),
            ("1536 × 1024", (1536, 1024)),
            ("1024 × 1536", (1024, 1536)),
        ]
        for label, size in fixed_sizes:
            act = act = QAction(label, self, checkable=True)
            act.triggered.connect(lambda checked, s=size: self.fixed_crop_triggered(s))
            crop_menu.addAction(act)
            self.crop_action_group.addAction(act)
            self.crop_actions[size] = act

        self.custom_action = QAction("カスタムサイズ...", self, checkable=True)
        self.custom_action.setChecked(False)
        self.custom_action.triggered.connect(self.show_custom_crop_dialog)
        crop_menu.addAction(self.custom_action)
        #self.crop_action_group.addAction(self.custom_action)

        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setFocus() 
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label.selectionMade.connect(self.on_crop)
        self.label.fixedSelectionMade.connect(self.on_fixed_crop_move)
        self.label.movedRect.connect(self.on_crop_rect_moved)

        # --- ズーム倍率表示ラベルをCropLabel上に追加 ---
        self.zoom_label = ZoomLabel(self.label)
        self.listview.clicked.connect(self.on_thumbnail_clicked)

        self.listview.setStyleSheet("""
        QListView::item:selected {
            border: 2px solid #ff6600;
            background: #223344;;
        }
        """)

        self.setCentralWidget(main_widget)

        self.image = None
        self.image_path = None
        self.img_qt = None
        self.img_pixmap = None

        self.image_list = []
        self.current_index = -1
        self.model = None

        self.zoom_scale = 1.0   # --- 追加: ズーム倍率 ---
        self.shortcut_prev = QShortcut(QKeySequence("Ctrl+Left"), self)
        self.shortcut_next = QShortcut(QKeySequence("Ctrl+Right"), self)
        self.shortcut_prev.activated.connect(self.show_prev_image)
        self.shortcut_next.activated.connect(self.show_next_image)

        self._action_panel = None
        self._crop_rect = None
        self._success_label = None
        self._fixed_crop_rect = None

       # ==== 下部バー（独自実装） ====
        self.bottom_bar = QtWidgets.QWidget(self)
        self.bottom_bar.setFixedHeight(32)
        #self.bottom_bar.setStyleSheet("background: #222; border-top: 2px solid #f28524;")
        bottom_layout = QtWidgets.QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # フルパスラベル
        self.path_label = QtWidgets.QLabel()
        self.path_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred
        )
        try:
            self.path_label.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        except Exception:
            pass  # PyQt5の場合は対応しなくてOK
        bottom_layout.addWidget(self.path_label)
        self.save_folder_label = QtWidgets.QLabel()
        self.save_folder_label.setText("") 
        self.save_folder_label.setStyleSheet("color: #666;")
        self.save_folder_label.setMaximumWidth(550)  # 幅は好みで調整
        bottom_layout.addWidget(self.save_folder_label)

        self.progress_widget = QtWidgets.QWidget(self.bottom_bar)
        progress_layout = QtWidgets.QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        self.status_label = QtWidgets.QLabel()
        self.progress = ClickableProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress.clickedValueChanged.connect(self.on_progress_jump)
        self.progress.setMinimum(1)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress)
        self.progress_widget.setFixedSize(220, 24)

        
        outer_layout.addWidget(self.bottom_bar, stretch=0)

        self.progress_widget.hide()

    def _make_fixed_crop_handler(self, size_tuple):
        def handler(checked):
            self.fixed_crop_triggered(size_tuple)
        return handler
     
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"]:
                        event.acceptProposedAction()
                        self.drag_over = True
                        return
        event.ignore()

    def dropEvent(self, event):
        self.drag_over = False
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"]:
                        self.open_image_from_path(file_path)
                        break

    def open_image_from_path(self, file_path):
        if not file_path:
            return
        folder = os.path.dirname(file_path)
        self.folder = folder
        exts = ("jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp")
        self.image_list = [os.path.normpath(os.path.join(folder, f))
                           for f in os.listdir(folder) if f.lower().endswith(exts)]
        self.image_list.sort()
        file_path = os.path.normpath(file_path)
        self.current_index = self.image_list.index(file_path)
        self.model = ThumbnailListModel(self.image_list, thumb_size=(160, 240))
        self.listview.setModel(self.model)
        self.label.clear_rubberBand()
        self.label.clear_fixed_crop()
        self.load_image_by_index(self.current_index)
        self.listview.setCurrentIndex(self.model.index(self.current_index, 0))

    def set_save_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "保存先フォルダを設定")
        if folder:
            self.save_folder = folder
            text = f"保存先: {folder}"
            fm = self.save_folder_label.fontMetrics()
            max_width = self.save_folder_label.maximumWidth()
            elided = fm.elidedText(text, QtCore.Qt.TextElideMode.ElideMiddle, max_width)
            self.save_folder_label.setText(elided)
            self.save_folder_label.setToolTip(text) 

    def open_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "画像を開く", "", "画像ファイル (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)")
        if not file_path:
            return
        self.open_image_from_path(file_path)
        self.move_progress_widget()
        
    def on_thumbnail_clicked(self, index):
        img_path = index.data(QtCore.Qt.ItemDataRole.UserRole)
        if img_path:
            self.current_index = self.image_list.index(img_path)
            self.load_image_by_index(self.current_index)

    def load_image_by_index(self, idx):
            self.preview_label.clear()
            self.label._pan_offset_x = 0
            self.label._pan_offset_y = 0
            if hasattr(self, "_action_panel") and self._action_panel:
                self._action_panel.close()
                self._action_panel = None
            if not (0 <= idx < len(self.image_list)):
                self.status_label.setText("")
                self.progress.setValue(1)
                self.path_label.setText("")
                self.status_label.hide()
                self.progress.hide()
                self.progress_widget.hide()
                return
        
            file_path = self.image_list[idx]
            self.image_path = file_path
            self.image = Image.open(file_path)
            if self.image.mode in ("P", "PA"):
                self.image = self.image.convert("RGBA")
            self.label.clear_rubberBand()

            base_name = os.path.basename(file_path)
            width, height = self.image.size
            self.image_info_label.setText(f"{base_name}  —  {width} x {height}")

            # ----連続切り出しモード時は固定枠維持＆アクションパネル自動表示----
            if self.label.fixed_crop_mode and self.label.fixed_crop_size:
                self.label.start_fixed_crop(self.label.fixed_crop_size)
                rect = self.label.fixed_crop_rect
                if rect:
                    pos = QtCore.QPoint(rect.x() + rect.width(), rect.y() + rect.height())
                    pos.setX(min(pos.x(), self.label.width() - 60))
                    pos.setY(min(pos.y(), self.label.height() - 40))
                    self._crop_rect = rect
                    self.show_action_panel(rect, pos)
            else:
                # 固定枠・カスタム枠・✔の状態もリセット
                self.label.clear_fixed_crop()
                if hasattr(self, "crop_actions"):
                    for act in self.crop_actions.values():
                        act.setChecked(False)
                if hasattr(self, "custom_action") and self.custom_action.isChecked():
                    self.custom_action.setChecked(False)

            self.zoom_scale = 1.0  # --- 画像切り替え時はズームリセット ---
            self.show_image()
            progress_text = f"{self.current_index + 1} / {len(self.image_list)}"
            self.status_label.setText(progress_text)
            self.progress.setMaximum(len(self.image_list))
            self.progress.setValue(self.current_index + 1)
        
            self.status_label.show()
            self.progress.show()
            self.path_label.setText(self.image_path)
            self.progress_widget.show()
            if self.model:
                self.listview.setCurrentIndex(self.model.index(idx, 0))
            self.zoom_label.show_zoom(self.zoom_scale)
            if (
                self.label.fixed_crop_mode and
                self.label.fixed_crop_rect is not None
            ):
                self.update_preview(self.label.fixed_crop_rect)
                self.move_progress_widget()
            
            if self.save_folder:
                text = f"保存先: {self.save_folder}"
            elif self.image_path:
                text = f"保存先: {os.path.dirname(self.image_path)}"
            else:
                text = "保存先: "

            # --- elideで省略＆ツールチップにフルパス ---
            fm = self.save_folder_label.fontMetrics()
            max_width = self.save_folder_label.maximumWidth()
            elided = fm.elidedText(text, QtCore.Qt.TextElideMode.ElideMiddle, max_width)
            self.save_folder_label.setText(elided)
            self.save_folder_label.setToolTip(text)
     
    def on_progress_jump(self, value):
            if value < 1 or value > len(self.image_list):
                 return
            index = value - 1
            self.current_index = index
            self.load_image_by_index(self.current_index)

    def show_prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image_by_index(self.current_index)

    def show_next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image_by_index(self.current_index)

    def show_image(self):
        if self.image is None:
           return
        img = self.image.convert("RGB")
        self.img_qt = ImageQt.ImageQt(img)
        self.img_pixmap = QtGui.QPixmap.fromImage(self.img_qt)

        label_w, label_h = self.label.width(), self.label.height()
        img_w, img_h = self.img_pixmap.width(), self.img_pixmap.height()

        # 100%時に収まる最大サイズを計算（縦横比を維持）
        if self.zoom_scale == 1.0 or self.base_display_width is None or self.base_display_height is None:
            self.label._pan_offset_x = 0
            self.label._pan_offset_y = 0
            if img_w <= label_w and img_h <= label_h:
                scaled = self.img_pixmap
                self.base_display_width, self.base_display_height = img_w, img_h
                display_pixmap = scaled
            else:
                scaled = self.img_pixmap.scaled(
                    label_w, label_h,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                self.base_display_width = scaled.width()
                self.base_display_height = scaled.height()
                display_pixmap = scaled
        else:
            target_w = int(self.base_display_width * self.zoom_scale)
            target_h = int(self.base_display_height * self.zoom_scale)
            target_w = max(1, target_w)
            target_h = max(1, target_h)
            scaled = self.img_pixmap.scaled(
                target_w, target_h,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            if self.zoom_scale > 1.0:
                # --- パンオフセット（中心からのズレ量）を適用して切り抜き表示 ---
                offset_x = getattr(self.label, "_pan_offset_x", 0)
                offset_y = getattr(self.label, "_pan_offset_y", 0)
                # 画像の中心を基準に表示
                center_x = scaled.width() // 2 + offset_x
                center_y = scaled.height() // 2 + offset_y
                # ラベルの中心座標
                lw2, lh2 = label_w // 2, label_h // 2
                # 描画範囲を決定
                crop_rect = QtCore.QRect(center_x - lw2, center_y - lh2, label_w, label_h)
                # 画像内におさまるよう調整
                crop_rect = crop_rect.intersected(scaled.rect())
                display_pixmap = scaled.copy(crop_rect)
            else:
                display_pixmap = scaled

        self.label.setPixmap(display_pixmap)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.update()
        self.zoom_label.show_zoom(self.zoom_scale)
        self.move_progress_widget()
        
    def update_preview(self, crop_rect=None):
        try:
            if (not hasattr(self, 'label') or not self.label or not hasattr(self, 'preview_label') or not self.preview_label or
                not hasattr(self, 'image') or self.image is None or
                not crop_rect or not hasattr(self.label, 'pixmap') or self.label.pixmap() is None):
                if hasattr(self, "preview_label"):
                    self.preview_label.clear()
                return

            # crop_rectは画像ピクセル座標rectとして使う（変換不要！）
            left = crop_rect.left()
            top = crop_rect.top()
            right = crop_rect.right()
            bottom = crop_rect.bottom()

            # 範囲制限
            left = max(0, left)
            top = max(0, top)
            right = min(self.image.width, right)
            bottom = min(self.image.height, bottom)

            overlap_w = right - left
            overlap_h = bottom - top

            if overlap_w <= 0 or overlap_h <= 0:
                self.preview_label.clear()
                return

            cropped = self.image.crop((left, top, right, bottom))

            PREVIEW_MAX = 512
            if overlap_w > overlap_h:
                preview_w = PREVIEW_MAX
                preview_h = int(overlap_h * PREVIEW_MAX / overlap_w)
            else:
                preview_h = PREVIEW_MAX
                preview_w = int(overlap_w * PREVIEW_MAX / overlap_h)

            resized = cropped.resize((preview_w, preview_h), Image.LANCZOS)
            preview_img = Image.new("RGB", (PREVIEW_MAX, PREVIEW_MAX), (255, 255, 255))
            offset_x = (PREVIEW_MAX - preview_w) // 2
            offset_y = (PREVIEW_MAX - preview_h) // 2
            preview_img.paste(resized, (offset_x, offset_y))

            from PIL.ImageQt import ImageQt
            self.preview_qt_img = ImageQt(preview_img)
            self.preview_pixmap = QtGui.QPixmap.fromImage(self.preview_qt_img)
            self.preview_label.setPixmap(self.preview_pixmap)
            self.preview_label.repaint()

        except Exception as e:
            print("[PREVIEW ERROR]", e)
            import traceback; traceback.print_exc()
            if hasattr(self, "preview_label"):
                self.preview_label.clear()

    def zoom_in(self):
        if abs(self.zoom_scale - 0.10) < 1e-6:  # 10%にほぼ等しいなら
            self.zoom_scale = 0.25  # 25%に直接変更
        else:
            self.zoom_scale = min(self.zoom_scale + 0.25, 8.0)
        self.zoom_scale = round(self.zoom_scale, 2)
        self.show_image()
        self.move_progress_widget()

    def zoom_out(self):
        self.zoom_scale = max(self.zoom_scale - 0.25, 0.1)
        self.zoom_scale = round(self.zoom_scale, 2)
        self.show_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_image()
        if self.label.fixed_crop_mode and self.label.fixed_crop_size:
            self.label.start_fixed_crop(self.label.fixed_crop_size)
            if self.label.fixed_crop_rect:
                rect = self.label.fixed_crop_rect
                pos = QtCore.QPoint(rect.x() + rect.width(), rect.y() + rect.height())
                pos.setX(min(pos.x(), self.label.width() - 60))
                pos.setY(min(pos.y(), self.label.height() - 40))
                self._crop_rect = rect
                self.show_action_panel(rect, pos)
                self.move_progress_widget()

    def on_crop(self, rect, mouse_pos):
        self._crop_rect = rect
        self.show_action_panel(rect, mouse_pos)
        self.update_crop_size_label(rect)
        # --- ラベル座標rect→ピクセル座標へ変換してpreview更新
        gx1, gy1 = self.label_to_image_coords(rect.left(), rect.top())
        gx2, gy2 = self.label_to_image_coords(rect.right(), rect.bottom())
        crop_rect_px = QtCore.QRect(min(gx1,gx2), min(gy1,gy2), abs(gx2-gx1), abs(gy2-gy1))
        #print(f"[ラベル上の矩形] left={rect.left()}, top={rect.top()}, right={rect.right()}, bottom={rect.bottom()}")
        #print(f"[画像上の矩形] left={min(gx1, gx2)}, top={min(gy1, gy2)}, right={max(gx1, gx2)}, bottom={max(gy1, gy2)}")
        QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(crop_rect_px))

    def on_fixed_crop_move(self, rect, mouse_pos):
        self._crop_rect = rect
        self.show_action_panel(rect, mouse_pos)
        self.update_crop_size_label(rect)
        # --- rect（ラベル座標）→ピクセル座標に変換して preview 用に渡す
        x1, y1 = rect.left(), rect.top()
        x2, y2 = rect.right(), rect.bottom()
        gx1, gy1 = self.label_to_image_coords(x1, y1)
        gx2, gy2 = self.label_to_image_coords(x2, y2)
        crop_rect_px = QtCore.QRect(min(gx1, gx2), min(gy1, gy2), abs(gx2-gx1), abs(gy2-gy1))
        QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(crop_rect_px))

    def show_action_panel(self, rect, mouse_pos):
        if not rect:
            return 
        if self.label.fixed_crop_mode and getattr(self.label, "fixed_crop_rect_img", None) is not None:
            x2, y2 = self.label.image_to_label_coords(
                self.label.fixed_crop_rect_img.right(), self.label.fixed_crop_rect_img.bottom()
            )
        # ドラッグ枠は「rect」は既にラベル座標
        else:
            x2, y2 = rect.right(), rect.bottom()
        # ------------------------

        panel_width = 170
        panel_height = 40

        x = min(x2, self.label.width() - panel_width - 4)
        y = min(y2, self.label.height() - panel_height - 4)
        x = max(x, 4)
        y = max(y, 4)
        pos = QtCore.QPoint(x, y)

        if self._action_panel:
            self._action_panel.close()
            self._action_panel = None

        self._action_panel = ActionPanel(
            parent=self.label,
            pos=pos,
            on_save=self.do_crop_save,
            on_cancel=self.cancel_crop
        )

    def do_crop_save(self):
        rect = self._crop_rect
        if self._action_panel:
            btn = self._action_panel.btn_save
            btn_center_local = btn.mapTo(self, btn.rect().center()) + QtCore.QPoint(0, -40)
            self._success_label = SuccessLabel(parent=self, pos=btn_center_local)
        else:
            self._success_label = SuccessLabel(parent=self, pos=QtCore.QPoint(100, 100))
        self.save_cropped(rect)
        # ==== 固定モード時は枠維持 ====
        if not self.label.fixed_crop_mode:
            self.cancel_crop()
        else:
            if self._action_panel:
                self._action_panel.close()
                self._action_panel = None
            # --- アクションパネル再表示 ---
            rect = self.label.fixed_crop_rect
            if rect:
                pos = QtCore.QPoint(rect.x() + rect.width(), rect.y() + rect.height())
                pos.setX(min(pos.x(), self.label.width() - 60))
                pos.setY(min(pos.y(), self.label.height() - 40))
                self._crop_rect = rect
                self.show_action_panel(rect, pos)

    def save_cropped(self, rect):
        if self.image is None:
            return

        img_w, img_h = self.image.width, self.image.height

        # 固定切り出し（fixed_crop_rect_img）優先、なければラベル矩形→変換
        img_rect = getattr(self.label, "fixed_crop_rect_img", None)
        if img_rect is not None:
            x = int(img_rect.left())
            y = int(img_rect.top())
            w = int(img_rect.width())
            h = int(img_rect.height())
            left = max(0, x)
            top = max(0, y)
            right = min(img_w, x + w)
            bottom = min(img_h, y + h)
        else:
            # ドラッグ枠はラベル座標なので変換
            if rect is None:
                print("[SAVE ERROR] 切り出し範囲が指定されていません")
                return
            x1, y1 = rect.left(), rect.top()
            x2, y2 = rect.right(), rect.bottom()
            gx1, gy1 = self.label_to_image_coords(x1, y1)
            gx2, gy2 = self.label_to_image_coords(x2, y2)
            left = max(0, min(gx1, gx2))
            top = max(0, min(gy1, gy2))
            right = min(self.image.width, max(gx1, gx2))
            bottom = min(self.image.height, max(gy1, gy2))

        if right - left <= 0 or bottom - top <= 0:
            print("[SAVE ERROR] 切り出し範囲が画像外です")
            return

        box = (left, top, right, bottom)
        cropped = self.image.crop(box)

        #print(f"--- Crop Debug ---")
        #print(f"選択範囲 元画像ピクセル: left={left}, top={top}, right={right}, bottom={bottom}")
        #print(f"選択範囲サイズ: width={right-left}, height={bottom-top}")
        #print(f"保存画像サイズ: {cropped.size}")
        #print(f"--- End Debug ---")

        # 保存ファイル名生成（これは元のまま）
        folder = self.save_folder if self.save_folder else os.path.dirname(self.image_path)
        base, ext = os.path.splitext(os.path.basename(self.image_path))
        ext = ext.lower().lstrip(".")
        save_ext = ext if ext in ("jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp") else "png"

        for i in range(1, 1000):
            candidate = os.path.join(folder, f"{base}_cropped_{i:03d}.{save_ext}")
            if not os.path.exists(candidate):
                break

        cropped.save(candidate)
        print("保存:", candidate)

    def on_crop_rect_moved(self, rect):
        self._crop_rect = rect
        if self._action_panel:
            pos = QtCore.QPoint(rect.x() + rect.width(), rect.y() + rect.height())
            pos.setX(min(pos.x(), self.label.width() - 60))
            pos.setY(min(pos.y(), self.label.height() - 40))
            self._action_panel.move(pos)
            QtCore.QTimer.singleShot(0, lambda: self.safe_update_preview(self._crop_rect))
        self.update_crop_size_label(rect)

    def cancel_crop(self):
        if self._action_panel:
            self._action_panel.close()
            self._action_panel = None
        self.label.clear_rubberBand()
        # 固定切り出しボタンのチェックも解除
        if hasattr(self, "crop_actions"):
            for act in self.crop_actions.values():
                act.setChecked(False)
                
        self.label.clear_rubberBand()
        self.label.clear_fixed_crop()
        self._crop_rect = None
        #self.preview_label.clear()
        
        if self.custom_action.isChecked():
            self.custom_action.setChecked(False)

        #self.preview_label.clear()

    def fixed_crop_triggered(self, size_tuple, checked=True):
        current = self.label.fixed_crop_mode and self.label.fixed_crop_size == size_tuple
        if current:
            self.label.clear_fixed_crop()
            self._crop_rect = None
            if self._action_panel:
                self._action_panel.close()
                self._action_panel = None
            for act in self.crop_actions.values():
                act.setChecked(False)
            if hasattr(self, "custom_action"):
                self.custom_action.setChecked(False)
            return

        # --- カスタムサイズ選択時 ---
        if size_tuple == "custom":
            ok, custom_size = self.show_custom_size_dialog()
            if not ok:
                # キャンセル時のリセット
                self.custom_action.setChecked(False)
                for act in self.crop_actions.values():
                    act.setChecked(False)
                self.label.clear_fixed_crop()
                self._crop_rect = None
                if self._action_panel:
                    self._action_panel.close()
                    self._action_panel = None
                return
            # OK時: カスタムに✔、他プリセットだけ外す
            for act in self.crop_actions.values():
                act.setChecked(False)
            self.custom_action.setChecked(True)
            size_tuple = custom_size
            # 枠を有効化
            if self.label.pixmap():
                self.label.start_fixed_crop(size_tuple)
                rect = self.label.fixed_crop_rect
                if rect:
                    pos = QtCore.QPoint(rect.x() + rect.width(), rect.y() + rect.height())
                    pos.setX(min(pos.x(), self.label.width() - 60))
                    pos.setY(min(pos.y(), self.label.height() - 40))
                    self._crop_rect = rect
                    self.show_action_panel(rect, pos)
            return

        # --- それ以外は普通にON ---
        if size_tuple not in self.crop_actions:
            print("size_tuple:", size_tuple)
            print("crop_actions.keys():", list(self.crop_actions.keys()))
            print("custom_action:", self.custom_action)
            self.custom_action.setChecked(True)
            print("custom_action.isChecked() after setChecked(True):", self.custom_action.isChecked())
            for act in self.crop_actions.values():
                act.setChecked(False)
        else:
            # 通常のプリセット
            self.custom_action.setChecked(False)
            for k, act in self.crop_actions.items():
                act.setChecked(k == size_tuple)
        # 枠を有効化
        if self.label.pixmap():
            self.label.start_fixed_crop(size_tuple)
            rect = self.label.fixed_crop_rect
            if rect:
                pos = QtCore.QPoint(rect.x() + rect.width(), rect.y() + rect.height())
                pos.setX(min(pos.x(), self.label.width() - 60))
                pos.setY(min(pos.y(), self.label.height() - 40))
                self._crop_rect = rect
                self.show_action_panel(rect, pos)

    def show_custom_crop_dialog(self):
     dialog = QtWidgets.QDialog(self)
     dialog.setWindowTitle("カスタムサイズ入力")
     layout = QtWidgets.QVBoxLayout(dialog)

     width_edit = QtWidgets.QLineEdit()
     width_edit.setPlaceholderText("幅 (例: 1024)")
     height_edit = QtWidgets.QLineEdit()
     height_edit.setPlaceholderText("高さ (例: 1024)")

     layout.addWidget(QtWidgets.QLabel("幅:"))
     layout.addWidget(width_edit)
     layout.addWidget(QtWidgets.QLabel("高さ:"))
     layout.addWidget(height_edit)

     btn_box = QtWidgets.QDialogButtonBox(
         QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
     )
     layout.addWidget(btn_box)

     ok_btn = btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
     ok_btn.setEnabled(False)  # 最初は無効

     # --- 入力値が両方とも空でなければOKボタン有効 ---
     def validate_fields():
         w = width_edit.text().strip()
         h = height_edit.text().strip()
         ok_btn.setEnabled(bool(w) and bool(h))

     width_edit.textChanged.connect(validate_fields)
     height_edit.textChanged.connect(validate_fields)

     btn_box.accepted.connect(dialog.accept)
     btn_box.rejected.connect(dialog.reject)

     if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
         try:
             w = int(width_edit.text())
             h = int(height_edit.text())
             if w > 0 and h > 0:
                 self.custom_action.setChecked(True)
                 self.fixed_crop_triggered((w, h), True)
             else:
                 self.custom_action.setChecked(False)
         except Exception:
             self.custom_action.setChecked(False)
     else:
         self.custom_action.setChecked(False) 

    def safe_update_preview(self, crop_rect):
        try:
            if not self or not hasattr(self, "image") or self.image is None \
               or not hasattr(self, "label") or self.label is None or self.label.pixmap() is None:
                if hasattr(self, "preview_label"):
                    self.preview_label.clear()
                return
            self.update_preview(crop_rect)
        except Exception as e:
            print("[SAFE PREVIEW ERROR]", e)
            import traceback; traceback.print_exc()
            if hasattr(self, "preview_label"):
                self.preview_label.clear()

    def move_progress_widget(self):
        # 画像表示エリアの「グローバル座標」を取得
        label_global = self.label.mapToGlobal(QtCore.QPoint(0, self.label.height()))
        # メインウィンドウのグローバル座標
        win_global = self.mapToGlobal(QtCore.QPoint(0, 0))
        # 下部バーのローカル座標での画像領域の中央
        bar_x = label_global.x() - win_global.x() + (self.label.width() - self.progress_widget.width()) // 2
        bar_y = 4  # bottom_bar内の上から4px（高さ中央にしたい場合は計算する）

        bar_x = max(0, min(bar_x, self.bottom_bar.width() - self.progress_widget.width()))
        self.progress_widget.move(bar_x, bar_y)
        self.progress_widget.raise_()
    
    def update_crop_size_label(self, rect=None):
        if not self.image:
            self.crop_size_label.setText("0 x 0")
            return

        img_w = self.image.width
        img_h = self.image.height

        # --- 固定枠の場合（fixed_crop_rect_img）---
        img_rect = getattr(self.label, "fixed_crop_rect_img", None)
        if img_rect is not None:
            # 完全に保存処理と合わせる
            x = int(img_rect.left())
            y = int(img_rect.top())
            w = int(img_rect.width())
            h = int(img_rect.height())
            left, top, right, bottom = x, y, x + w, y + h
            # 範囲制限（この順序も保存処理と一致させる）
            left = max(0, min(left, img_w))
            top = max(0, min(top, img_h))
            right = max(0, min(right, img_w))
            bottom = max(0, min(bottom, img_h))
            crop_w = max(0, right - left)
            crop_h = max(0, bottom - top)
            self.crop_size_label.setText(f"{crop_w} x {crop_h}")
            return

        # --- ドラッグ枠（drag_rect_img, ラベル座標をピクセル変換）---
        drag_rect = getattr(self.label, "drag_rect_img", None)
        if drag_rect is not None:
            # 枠が正しいQRectの場合のみ
            # rect引数があればそれを使う
            if rect is not None:
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()
            else:
                # 直近のドラッグ範囲
                x1, y1 = drag_rect.left(), drag_rect.top()
                x2, y2 = drag_rect.right(), drag_rect.bottom()
            gx1, gy1 = self.label_to_image_coords(x1, y1)
            gx2, gy2 = self.label_to_image_coords(x2, y2)
            left = max(0, min(gx1, gx2))
            top = max(0, min(gy1, gy2))
            right = min(img_w, max(gx1, gx2))
            bottom = min(img_h, max(gy1, gy2))
            crop_w = max(0, right - left)
            crop_h = max(0, bottom - top)
            self.crop_size_label.setText(f"{crop_w} x {crop_h}")
            return

        # --- それ以外 ---
        self.crop_size_label.setText("0 x 0")

    def pan_image(self, dx, dy):
        # ラベルのオフセット値を変更
        if not hasattr(self.label, "_pan_offset_x"):
            self.label._pan_offset_x = 0
        if not hasattr(self.label, "_pan_offset_y"):
            self.label._pan_offset_y = 0

        # --- 拡大してる場合のみパンを有効 ---
        if self.zoom_scale > 1.0:
            self.label._pan_offset_x += dx
            self.label._pan_offset_y += dy

            # 画像が見切れすぎないように範囲制限
            max_x = int((self.base_display_width * self.zoom_scale - self.label.width()) / 2)
            max_y = int((self.base_display_height * self.zoom_scale - self.label.height()) / 2)
            self.label._pan_offset_x = max(-max_x, min(self.label._pan_offset_x, max_x))
            self.label._pan_offset_y = max(-max_y, min(self.label._pan_offset_y, max_y))

            self.show_image()

    def label_to_image_coords(self, lx, ly):
        if self.image is None:
        # 画像未ロード時は例外防止。0,0を返すなど
            return 0, 0
        """
        QLabel上の座標(lx, ly)→元画像ピクセル座標(gx, gy)に変換（ズーム・パン考慮）
        """
        lw, lh = self.label.width(), self.label.height()
        base_w, base_h = self.base_display_width, self.base_display_height
        zoom = self.zoom_scale
        offset_x = getattr(self.label, "_pan_offset_x", 0)
        offset_y = getattr(self.label, "_pan_offset_y", 0)
        img_w, img_h = self.image.width, self.image.height

        pm_w = int(base_w * zoom)
        pm_h = int(base_h * zoom)
        center_x = pm_w // 2 + offset_x
        center_y = pm_h // 2 + offset_y
        px = center_x - (lw // 2) + lx
        py = center_y - (lh // 2) + ly
        gx = int(round(px * (img_w / pm_w)))
        gy = int(round(py * (img_h / pm_h)))
        return gx, gy
class ClickableProgressBar(QtWidgets.QProgressBar):
    clickedValueChanged = QtCore.pyqtSignal(int)  # 新しい値がクリックで決まった時にemit

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            x = event.position().x() 
            width = self.width()
            minval = self.minimum()
            maxval = self.maximum()
            if maxval > minval:
                percent = min(max(x / width, 0), 1)
                value = int(round(percent * (maxval - minval) + minval))
                self.setValue(value)
                self.clickedValueChanged.emit(value)
        super().mousePressEvent(event)

class ProgressWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        self.count_label = QtWidgets.QLabel("55 / 57")
        layout.addWidget(self.count_label)
        self.progress_bar = ClickableProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(57)
        self.progress_bar.setValue(55)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        layout.addWidget(self.progress_bar, stretch=1)
        self.percent_label = QtWidgets.QLabel("96%")
        layout.addWidget(self.percent_label)

        self.progress_bar.clickedValueChanged.connect(self.on_jump)

    def set_progress(self, current, total):
        self.count_label.setText(f"{current} / {total}")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        percent = int(round(current / total * 100)) if total else 0
        self.percent_label.setText(f"{percent}%")

    def on_jump(self, value):
        self.set_progress(value, self.progress_bar.maximum())
  
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = CropperApp()
    win.show()
    sys.exit(app.exec())
