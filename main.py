import random
import time

import pytesseract
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle
import zlib
import concurrent.futures

# Путь для тессеракта
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import cv2
import numpy as np
import re

from PIL import Image, ImageGrab

import win32gui
import win32con

import pyautogui as pg
import keyboard

import random
from datetime import datetime, timedelta


def word_detect(msg, list_words):
    # Словарь замен (символы, которые часто используют для обхода фильтра)
    char_replacements = {
        'а': ['а', 'a', '1', 'о'],
        'б': ['б', '6', 'b'],
        'в': ['в'],
        'г': ['г', 'g'],
        'д': ['д', 'd'],
        'е': ['е', 'e', 'и', 'э'],
        'ж': ['ж'],
        'з': ['з', '3', 'z'],
        'и': ['и', 'u', 'i', 'ы', 'е'],
        'й': ['й', 'u', 'y', 'i', 'и'],
        'к': ['к', 'k'],
        'л': ['л', 'l'],
        'м': ['м'],
        'н': ['н'],
        'о': ['о', 'o', 'а'],
        'п': ['п', 'n', 'p'],
        'р': ['р', 'p', 'r'],
        'с': ['с', 'c', 's', '$'],
        'т': ['т'],
        'у': ['у', 'y', 'u'],
        'ф': ['ф'],
        'х': ['х', 'x', 'h', ')(', 'k', 'к'],
        'ц': ['ц'],
        'ч': ['ч', '4'],
        'ш': ['ш', 'sh', 'щ'],
        'щ': ['щ'],
        'ъ': ['ъ'],
        'ы': ['ы', 'и'],
        'ь': ['ь'],
        'э': ['э', 'e', 'е'],
        'ю': ['ю', '1', 'у'],
        'я': ['я', 'ya'],
    }

    # Приводим сообщение к нижнему регистру
    msg_lower = msg.lower()

    # Проверяем каждое запрещённое слово
    for banned_word in list_words:
        # Генерируем все возможные варианты замен для каждой буквы в слове
        possible_combinations = ['']
        for char in banned_word:
            new_combinations = []
            replacements = char_replacements.get(char, [char])
            for combo in possible_combinations:
                for replacement in replacements:
                    new_combinations.append(combo + replacement)
            possible_combinations = new_combinations

        # Проверяем все возможные комбинации в сообщении
        for combo in possible_combinations:
            if combo in msg_lower:
                return True  # Найдено запрещённое слово

    # Не найдено
    return False


class Window:
    def __init__(self, close=True):
        self.hwnd = self.find_game_window()
        self.is_open = False if self.hwnd == None else True
        self.executer = Actions()
        self.is_full_window = False

        # если открыто - перезапускаем
        if self.is_open:
            self.hide_problem()
            self.correct_window()
            if self.is_full_window:
                self.correct_setting_game()
            if not close:
                return

            win32gui.PostMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)
            time.sleep(0.5)
            self.hwnd = None

        # Нажимаю на кнопку XP
        game_place = (53, 932)
        self.executer.click(game_place)
        self.executer.click(game_place)

        # Жду открытия XP и жму <играть>
        click_play = [
            ['right', ((557, 236), (16, 40, 57))],
            ['right', ((490, 331), (210, 239, 252))],
            ['left', ((590, 453), (78, 167, 131))],
        ]
        self.executer.executes(click_play, False)

        # жду запуска аватарии
        while self.hwnd == None:
            self.hwnd = self.find_game_window()
            time.sleep(1)

        # Проверяем полноэкранный режим
        self.hide_problem()
        self.correct_window()

        # Если игра открылась в полноэкранном режиме, то меняем настройки игры
        if self.is_full_window == True:
            self.correct_setting_game()

        # Игра перезапущена в оконном режиме

        print(self.hwnd)
        self.hide_problem()

        # Устанавливает окно в правильном месте
        self.correct_window()

    # Убирает полноэкранный режим игры
    def correct_setting_game(self):
        # Окно игры открылось, загрузка. Пару секунд ничего не делаем, чтобы пропустить множественные загрузки запуска
        time.sleep(10)
        # Повторяем до тех пор пока не повезет включить оконный режим
        while self.is_full_window:
            # Нажимаем на крестик или сеттинг 10 раз каждые 1.5 секунды.
            for _ in range(10):
                cross_or_setting = (1886, 32)
                self.executer.click(cross_or_setting)
                time.sleep(1.5)

            # Должен открыться setting
            # шестеренка
            click_setting_gear = (1516, 872)
            self.executer.click(click_setting_gear)
            time.sleep(1)

            # оконный режим
            click_flag_min_win = (817, 769)
            self.executer.click(click_flag_min_win)
            time.sleep(0.2)

            # применить
            click_accept = (979, 902)
            self.executer.click(click_accept)
            time.sleep(1)

            # Подтвердить и перезапустить игру
            click_restart = (1113, 624)
            self.executer.click(click_restart)
            time.sleep(1)

            # проверяем полноэкранный режим
            self.correct_window()

            # если полноэкранный, то закрываем игру и повторяем
            if self.is_full_window:
                win32gui.PostMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)
                self.__init__()

    # Ищем окно игры
    def find_game_window(self):
        def enum_window_callback(hwnd, windows):
            # Получаем заголовок окна
            title = win32gui.GetWindowText(hwnd)
            # Ищем точное название "Avataria"
            if title == "Avataria":
                windows.append(hwnd)
                # print(title)

        windows = []
        # Перебираем все окна
        win32gui.EnumWindows(enum_window_callback, windows)

        if windows:
            print(f"Окно 'Avataria' найдено! hwnd: {windows[0]}")
            return windows[0]  # Возвращаем первый найденный hwnd (идентификатор окна)
        else:
            print("Окно 'Avataria' не найдено.")
            return None

    # Возвращаем скрытое окно
    def hide_problem(self):
        if win32gui.IsIconic(self.hwnd):
            print("Окно свернуто. Восстанавливаю...")
            # Восстанавливаем свернутое окно
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)

    # Фиксируем окно в (0,0), 1280x800
    def correct_window(self):
        # Закрепление окна поверх других
        win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOSIZE)

        # размеры окна
        window_rect = win32gui.GetWindowRect(self.hwnd)
        window_width = window_rect[2] - window_rect[0]
        window_height = window_rect[3] - window_rect[1]

        if (window_width, window_height) in [(1920, 1080), ()]:
            self.is_full_window = True
        else:
            self.is_full_window = False


class Passport:
    def __init__(self):
        # Области для скриншотов
        self.region_myname = (595, 249, 958 - 595, 286 - 249)
        self.region_other_player = (643, 216, 963 - 682, 257 - 216)
        self.region_id = (299, 590, 491 - 299, 620 - 590)

    def read_id(self):
        """
        Распознает последовательность цифр на скриншоте с темно-синими цифрами на голубом фоне

        Параметры:
        screenshot -- скриншот в формате PIL.Image или numpy array

        Возвращает:
        str -- распознанная строка цифр
        numpy array -- обработанное изображение (для отладки)
        """

        screenshot = pg.screenshot(region=self.region_id)
        # Конвертируем в numpy array если нужно
        if isinstance(screenshot, Image.Image):
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            img = screenshot.copy()

        # 1. Улучшение контраста цифр
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Применяем CLAHE к L-каналу
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Объединяем обратно
        lab = cv2.merge((l, a, b))
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Выделение синих цифр
        hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)

        # Диапазон для темно-сине-серых цифр
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 200])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 3. Создание изображения с черными цифрами на белом фоне
        result = cv2.bitwise_and(img_enhanced, img_enhanced, mask=mask)
        result[mask == 0] = [255, 255, 255]  # Белый фон

        # 4. Улучшение для OCR
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

        # Морфологические операции для улучшения цифр
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 5. Распознавание цифр
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        digits = pytesseract.image_to_string(processed, config=custom_config)

        # Очистка результата
        numbers_id = ''.join(c for c in digits if c.isdigit())

        return numbers_id

    def enlarge_image(self, image, target_height=50):
        """Увеличивает изображение до целевой высоты"""
        # Если это PIL Image - конвертируем в numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3:  # Если цветное
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Теперь работаем с numpy array
        h, w = image.shape[:2]
        scale = target_height / h
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def convert_pil_to_cv2(self, pil_image):
        """Конвертация PIL в OpenCV формат"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def preprocess_image(self, img):
        """Улучшенная обработка изображения"""
        # Конвертация в HSV для цветовой фильтрации
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Маска для белого текста
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))

        # Маска для голубого фона
        blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([120, 255, 255]))

        # Удаление фона
        text_only = cv2.bitwise_and(img, img, mask=white_mask)
        text_only = cv2.bitwise_and(text_only, text_only, mask=cv2.bitwise_not(blue_mask))

        # Улучшение контраста
        lab = cv2.cvtColor(text_only, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Бинаризация
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Инверсия если текст белый
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        # Увеличение резкости
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(thresh, -1, kernel)

        return sharpened

    def detect(self):
        # 1. Делаем скриншот
        screenshot = pg.screenshot(region=self.region_other_player)

        # 2. Увеличиваем ДО конвертации в OpenCV формат
        # screenshot = self.enlarge_image(screenshot, target_height=200)  # Увеличиваем до 100px

        # 3. Конвертируем в OpenCV формат
        cv2_image = self.convert_pil_to_cv2(screenshot)

        # 4. Обрабатываем
        processed = self.preprocess_image(cv2_image)

        # 5. Показываем оба изображения
        # cv2.imshow("Original Screenshot", cv2_image)  # Оригинал
        # cv2.imshow("Processed Image", processed)  # Обработанный

        # 6. Распознаём текст
        text = pytesseract.image_to_string(processed, lang='eng+rus')

        return text.replace('\n', '')

    def is_open(self):
        info = [
            ((241, 103), (32, 127, 196)),
            ((1029, 119), (218, 235, 245)),
            ((1050, 679), (18, 116, 165)),
        ]
        for coord, color in info:
            current = pg.pixel(coord[0], coord[1])
            if current != color:
                return False
        return True

    def close(self):
        cross = (1029, 119)
        Actions().click(cross)


class TimeLimit:
    def __init__(self, limit=None):
        self.reset()
        self.limit = limit

    def reset(self, limit=None):
        self.begin = time.time()

        if limit:
            self.limit = limit

    def check(self):
        if self.limit == None or self.limit == True:
            return True
        elif self.limit == False:
            return False

        if time.time() - self.begin < self.limit:
            return True
        else:
            return False

    def show(self):
        return time.time() - self.begin


class Moderation:
    def __init__(self):
        self.cash = {

        }

    def write_global(self, info):
        name, msg = info['nickname'], info['msg']

    def write_private(self):
        pass

    def moderate(self, msg):


        trolling_words = [
            "мать", 'мамк', 'мамаш', "отец", "отчим", "сосешь ", 'сосать', 'хуй', 'член',
            'хуйло', 'нахуй', 'пизду', 'впизду', 'сын ', 'шалава', 'шалавы', 'долбаеб',
            'шлюх', 'шалашовк', 'ебарь', 'ебло', 'уебан', 'пиздюк', 'пиздуй', 'уебок', 'хуевина',
            'соси', 'отсоси', 'сперм', 'раб ', 'бездарь', 'немощь ', 'сосунок', 'выбляд', 'проститутка',
            'шмар', 'прошмандовк', 'хуесос', 'отпизжу'
        ]

        name_words = ['departament', 'депортамент']

        sex_theme = ['секс ', 'sex ', 'сперма ', 'дрочу ', 'рабын', 'сиськ', 'письк', 'трах', 'попк', 'сися ', 'сиси ',
                     'сисе', 'пизденка', 'свою жопу', 'трусики']

        love_theme = ['люблю ', 'помурчу', 'сладкий', 'милый', 'зайка ', 'зай ', 'зая ', 'котик', 'милая', 'любимый',
                      'любимая', 'целую', 'обнимаю']

        war_theme = ['СВО', 'расстрелять', 'ружье', 'автомат', 'расстрел', 'война', 'воен', 'воевать', 'воюют']

        police_theme = ['вор ', 'воровать', 'воруют', 'ворует', 'воруешь', 'нарик', 'наркота', 'веществ', 'доза ',
                       'трава ', 'марихуана', 'кокс ', 'дурь ', 'преступник', 'преступление', 'кража', ]

        out_name = ['бот ', 'ботяра', 'прога']

        if word_detect(msg, trolling_words):
            return 'trolling_words'
        elif word_detect(msg, name_words):
            return 'name_words'
        elif word_detect(msg, sex_theme):
            return 'sex_theme'
        elif word_detect(msg, love_theme):
            return 'love_theme'
        elif word_detect(msg, police_theme):
            return 'police_theme'
        elif word_detect(msg, out_name):
            return 'out_name'
        elif word_detect(msg, war_theme):
            return 'war_theme'

        return False

    def new_msg(self, info_msges, type_chat):
        detect_theme = []
        for info in info_msges:
            name, msg = info['nickname'], info['msg']
            if type_chat == 'private_msg':
                self.write_global(info)
            elif type_chat == 'global_msg':
                self.write_private()


class Visual_Chat:
    # Кешированные данные для оптимизации
    TAG_COLORS_BGR = [
        (204, 204, 0), (255, 153, 0), (0, 102, 255),
        (204, 102, 204), (161, 159, 241), (102, 153, 255),
        (51, 51, 255), (67, 214, 248), (255, 102, 51),
        (0, 204, 255), (204, 51, 153), (51, 153, 0),
        (154, 151, 242)
    ]
    COLOR_RANGES = [
        (np.array([max(0, c - 15) for c in color], dtype=np.uint8),
         np.array([min(255, c + 15) for c in color], dtype=np.uint8))
        for color in TAG_COLORS_BGR
    ]

    @staticmethod
    def mask_clan(image):
        mask_tags = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for lower, upper in Visual_Chat.COLOR_RANGES:
            color_mask = cv2.inRange(image, lower, upper)
            mask_tags = cv2.bitwise_or(mask_tags, color_mask)

        white_mask = cv2.inRange(image, (250, 250, 250), (255, 255, 255))
        final_mask = cv2.bitwise_or(mask_tags, white_mask)

        result = np.zeros_like(image)
        result[final_mask != 0] = (255, 255, 255)
        return result

    @staticmethod
    def mask_other(img, white_threshold=238):
        if len(img.shape) == 3:
            # Цветное изображение
            b, g, r = cv2.split(img)
            # Вычисляем максимальное и минимальное значение по каналам
            max_val = np.maximum(np.maximum(r, g), b)
            min_val = np.minimum(np.minimum(r, g), b)
            diff = max_val - min_val  # Разброс цветов

            # Параметры для определения серого:
            color_deviation = 10  # Макс. отклонение цвета для "серости"
            black_threshold = 60  # Нижний порог яркости

            # Условия для серого:
            # 1. Маленький разброс цветов (нейтральный оттенок)
            # 2. Не слишком яркий (исключаем белый)
            # 3. Не слишком тёмный (исключаем черный)
            mask = np.zeros_like(max_val, dtype=np.uint8)
            mask[
                (diff <= color_deviation) &
                (max_val < white_threshold) &
                (max_val > black_threshold)
                ] = 255
        else:
            # Градации серого
            gray = img.copy()
            black_threshold = 30

            # Создаем черную маску
            mask = np.zeros_like(gray)
            # Серое = не белое и не черное
            mask[
                (gray < white_threshold) &
                (gray > black_threshold)
                ] = 255

        return mask

    @staticmethod
    def crop_and_center_text(binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary_image

        x_min = min(cv2.boundingRect(cnt)[0] for cnt in contours)
        x_max = max(cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt in contours)
        y_min = min(cv2.boundingRect(cnt)[1] for cnt in contours)
        y_max = max(cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in contours)

        text_width = x_max - x_min
        height, width = binary_image.shape

        if text_width >= width:
            return binary_image

        centered = np.zeros((height, text_width + 40), dtype=np.uint8)
        start_x = 20
        centered[y_min:y_max, start_x:start_x + text_width] = binary_image[y_min:y_max, x_min:x_max]
        return centered

    @staticmethod
    def process_image(img, info_type):
        # Проверка на пустое изображение
        if img is None or img.size == 0:
            return ""

        try:
            if info_type == 'nickname':
                img = Visual_Chat.mask_clan(img)
                return pytesseract.image_to_string(img, lang='rus+eng').strip()

            elif info_type == 'msg':
                img = Visual_Chat.mask_other(img)
                img = Visual_Chat.crop_and_center_text(img)

                # Проверка после обрезки
                if img is None or img.size == 0:
                    return ""

                text = pytesseract.image_to_string(
                    img,
                    lang='rus+eng',
                )

                # Обработка переносов
                lines = text.split('\n')
                result = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.endswith('-') and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        result.append(line[:-1] + next_line)
                        i += 2
                    else:
                        result.append(line)
                        i += 1

                return ''.join(result).strip()

            return ""
        except Exception as e:
            print(f"Ошибка обработки изображения: {e}")
            return ""

    def save_buffer(self):
        region_msgs = (56, 171, 744, 499)
        screenshot = pg.screenshot(region=region_msgs)
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area, max_area = 10000, 200000
        filtered_contours = [
            cv2.boundingRect(cnt) for cnt in contours
            if min_area < cv2.contourArea(cnt) < max_area
        ]
        filtered_contours.sort(key=lambda c: c[1])

        messages_crob = [
            screenshot_bgr[y:y + h, x:x + w]
            for x, y, w, h in filtered_contours
        ]

        buffer = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, msg_block in enumerate(messages_crob):
                if msg_block.size == 0:  # Пропускаем пустые блоки
                    continue

                h, w = msg_block.shape[:2]
                is_clan = (h % 28) < 20

                # Безопасное определение зон интереса
                nickname_y_end = 45 if is_clan else 30
                nickname_x_start = 55 if is_clan else 5

                # Проверка валидности координат для ника
                if 5 < nickname_y_end <= h and nickname_x_start < w - 10:
                    nickname_zone = msg_block[5:nickname_y_end, nickname_x_start:w - 10]
                else:
                    nickname_zone = None

                # Проверка валидности координат для текста
                text_y_start = 48 if is_clan else 34
                if text_y_start < h - 20 and 2 < w - 4:
                    main_text = msg_block[text_y_start:h - 20, 2:w - 4]
                else:
                    main_text = None

                # Параллельная обработка только если зоны валидны
                nick_future = executor.submit(
                    Visual_Chat.process_image,
                    nickname_zone,
                    'nickname'
                ) if nickname_zone is not None else None

                msg_future = executor.submit(
                    Visual_Chat.process_image,
                    main_text,
                    'msg'
                ) if main_text is not None else None

                futures.append((nick_future, msg_future, w, h, is_clan))

            for nick_future, msg_future, w, h, is_clan in futures:
                nickname = nick_future.result() if nick_future else ""
                msg_text = msg_future.result() if msg_future else ""

                if nickname or msg_text:
                    time_msg = datetime.now().strftime("%H:%M'%S")
                    meta_block = (w, h, len(msg_text), int(is_clan), None)
                    buffer.append({'nickname': nickname, 'msg': msg_text, 'time': time_msg, 'meta': meta_block})

        return buffer

    def __init__(self):
        pass


class ChatAnalyzer:
    def __init__(self):
        self.buffers = [
            [{'nickname': 'None', 'msg': 'None', 'time': 'None', 'meta': (0, 0, 0, 0, 0)} for _ in range(4)]]
        self.visual_chat = Visual_Chat()
        self.chat_memory = []
        self.new_messages = []
        self.first_buffer = True
        self.miss_msg = 0

    def capture_buffer(self, moder, type_chat):
        """Захватывает новый буфер сообщений"""
        buffer = self.visual_chat.save_buffer()
        self.buffers.append(buffer)
        new_msg = self.merge_message_chains()

        if self.first_buffer:
            self.miss_msg = len(buffer)
            self.first_buffer = False
        else:
            moder.new_msg(new_msg, type_chat)

        return new_msg

    def write(self, msg):
        keyboard.write(msg)
        keyboard.press_and_release('enter')

        buffer = [{'nickname': 'ДЕПАРТАМЕНТ', 'msg': msg, 'time': datetime.now().strftime("%H:%M'%S"),
                   'meta': (0, 0, 0, 0, 0)}]
        self.buffers.append(buffer)
        new_msg = self.merge_message_chains()

    def get(self, first_buffer=False):
        if first_buffer:
            return self.chat_memory[4:]
        else:
            return self.chat_memory[4 + self.miss_msg:]

    def save(self, name_chat, info, file):
        # удаляем сообщения с другой локации
        self.chat_memory = self.chat_memory[4 + self.miss_msg:]
        saver = {'info': info, 'chat': self.chat_memory}

        if file == 'base.json':
            if len(self.chat_memory):
                BaseChatSaver().save_chat(name_chat, saver)
        elif file == 'events.json':
            EventChatSaver().save_chat(name_chat, saver)

    def is_similar(self, msg1, msg2):
        return (abs(msg1['meta'][0] - msg2['meta'][0]) < 3 and
                abs(msg1['meta'][1] - msg2['meta'][1]) < 4 and
                len(msg1['msg']) == len(msg2['msg']) and
                len(msg1['msg']) > 0 and
                msg1['msg'][len(list(msg1['msg'])) // 2] == msg2['msg'][len(list(msg2['msg'])) // 2]
                )

    def merge_message_chains(self):
        if not self.buffers:
            self.chat_memory = self.buffers
            return

        merged = list(self.buffers[0])

        for buffer in self.buffers[1:]:
            max_overlap = min(len(merged), len(buffer))
            best_overlap = 0
            for overlap in range(max_overlap, 0, -1):
                if all(self.is_similar(merged[-i], buffer[overlap - i])
                       for i in range(1, overlap + 1)):
                    best_overlap = overlap
                    break

            for msg in buffer[best_overlap:]:
                if msg['msg'] == '':
                    continue

                if not any(self.is_similar(msg, m) for m in merged):
                    # модерируем и добавляем в чат
                    merged.append(msg)

        new_msg = []
        for i in range(len(merged)):
            if len(self.chat_memory) == 0:
                new_msg = merged
            elif merged[-i - 1]['msg'] == self.chat_memory[-1]['msg']:
                if i == 0:
                    new_msg = []
                else:
                    new_msg = merged[-i:]

                break

        self.chat_memory = merged
        return new_msg


class Chat:
    def __init__(self):
        # Соответствие русских букв английским клавишам
        self.ru_to_en_keys = {
            'а': 'f', 'б': ',', 'в': 'd', 'г': 'u', 'д': 'l',
            'е': 't', 'ё': '`', 'ж': ';', 'з': 'p', 'и': 'b',
            'й': 'q', 'к': 'r', 'л': 'k', 'м': 'v', 'н': 'y',
            'о': 'j', 'п': 'g', 'р': 'h', 'с': 'c', 'т': 'n',
            'у': 'e', 'ф': 'a', 'х': '[', 'ц': 'w', 'ч': 'x',
            'ш': 'i', 'щ': 'o', 'ъ': ']', 'ы': 's', 'ь': 'm',
            'э': '\'', 'ю': '.', 'я': 'z', ' ': ' '
        }

        self.passport_namer = Passport()
        self.visual_chat = Visual_Chat()
        self.EventSave = EventChatSaver()
        self.BaseSave = BaseChatSaver()
        self.event = False
        self.event_info = None
        self.moder = Moderation()
        self.private_chates = {}

    def history(self, location):
        # Загружаем чат локации
        messages = self.base.get_chat(location)

        # показываем цепочку чата
        print()
        for name, msg, time_send, meta_info in messages:
            print(name, ' ' * (25 - len(name)) + ': ', msg, ' ' * (100 - len(msg)) + ': ', time_send,
                  f'  {meta_info[-1]}')

    def reading_gl(self, f_stop, duration_stop=None):
        # открываем чат
        self.open()

        # гарантируем что смотрим в глобальный чат (Пишет пустоту = ничего не пишет)
        self.write_msg_global('')

        executer = Actions()
        duration_time = TimeLimit(duration_stop)

        # ставим метку прочитанного сообщения
        coord_check = (70, 625)

        chat = ChatAnalyzer()

        time_update = 1.2

        def is_new_msg(color):
            if sum(color) // 3 > 205:
                return True
            return False

        while not f_stop(time_update) and duration_time.check():
            # Проверяем наличие нового сообщения
            x, y = coord_check
            current_color = pg.pixel(x, y)

            # проверяем синесть пикселя (видно синий паспорт, значит мы поставили метку прочитанности этого сообщения)
            if not is_new_msg(current_color):
                time_update = min(time_update * 1.1, 8)
                # print('. ', end = ' ')
            else:
                # ускоряем частоту проверок
                time_update = max(0.05, time_update * 0.5)

                # ставим метку прочитанного сообщения
                executer.click(coord_check)

                # Если обнаружили новое сообщение - распознаем новые сообщения
                chat.capture_buffer(self.moder, 'global_msg')

                # print()
                # print('new',end = '')

            time.sleep(time_update)

        # закрываем чат
        self.close()

        return chat

    def reading_private(self, club=True):
        self.open()
        # флаг на глобальном
        self.write_msg_global('')

        begin_x = 160
        if club:
            begin_x = 490

        coord_check = (70, 625)
        action = Actions()
        passport = Passport()

        def is_new_msg(color):
            if sum(color) // 3 > 205:
                return True
            return False

        for x in range(begin_x, 1270, 40):
            click = (x, 140)
            action.click(click)
            color = pg.pixel(coord_check[0], coord_check[1])
            if self.type_msg() == 'private_msg' and is_new_msg(color):

                # заходим в паспорт
                action.click(coord_check)
                time.sleep(0.1)
                action.click(coord_check)
                while not passport.is_open():
                    time.sleep(0.1)

                name, id = passport.detect(), passport.read_id()
                passport.close()
                time.sleep(0.75)

                chat_name = (name, id)
                find = False
                for name, chat in self.private_chates.items():
                    if chat_name[1] == name[1]:
                        find = True
                        chat.capture_buffer(self.moder, 'private_msg')

                if not find:
                    self.private_chates[chat_name] = ChatAnalyzer()
                    self.private_chates[chat_name].capture_buffer(self.moder, 'private_msg')

                if id == '31299433':
                    write = 'Привет, все получилось. Это Дианка)'
                    self.private_chates[chat_name].write(write)


                elif id == '31160467':
                    write = 'Привет, все получилось. Это Мишо)'
                    self.private_chates[chat_name].write(write)

        for name, chat in self.private_chates.items():
            print(name, ':', chat.get(first_buffer=True))

    def type_msg(self):
        # если не открыт чат, то ничего не делаем
        if not self.is_active():
            return

        # Общий чат
        color_global_type = (52, 118, 48)
        coord_global_type = (92, 162)
        x, y = coord_global_type
        current_color = pg.pixel(x, y)
        if current_color == color_global_type:
            return 'global_msg'

        # Чат гильдии
        color_clan_chat = (25, 113, 148)
        coord_clan_chat = (190, 162)
        x, y = coord_global_type
        current_color = pg.pixel(x, y)
        if current_color == coord_clan_chat:
            return 'clan_msg'

        # Обьявления гильдии
        color_clan_event = (25, 113, 148)
        coord_clan_event = (344, 162)
        x, y = coord_global_type
        current_color = pg.pixel(x, y)
        if current_color == color_clan_event:
            return 'clan_event_msg'

        # если ни один из типов не определен значит сообщение личное
        return 'private_msg'

    def is_active(self):
        chat_visual = [
            ((1203, 687), (93, 189, 79)),
            ((1217, 708), (249, 249, 249)),
        ]
        for pos_pix in chat_visual:
            pos, need_color = pos_pix
            x, y = pos
            current_color = pg.pixel(x, y)
            if current_color != need_color:
                return False

        return True

    def open(self):

        situation = Location().situation()
        # Если не можем нажать на кнопки, пропускаем
        if situation == None:
            print('попытка открыть в чат в неизвестной ситуации. Оставляю self.event = ', self.event)
            return

        elif situation == 'guest':
            self.event = True
        else:
            self.event = False

        print('self.event = ', self.event)

        executer = Actions()
        open_chat = [
            ['right', ((51, 575), (254, 254, 254))],
            ['right', ((66, 601), (226, 226, 226))],
            ['left', ((61, 586), (50, 155, 202))],
            ['right', ((1139, 702), (255, 217, 0))],
        ]
        executer.executes(open_chat)

    def close(self):
        executer = Actions()
        if self.is_active():
            cross_pos = (1235, 72)
            executer.click(cross_pos, 'left')

        time.sleep(0.75)

    def write_msg_global(self, msg):
        # если не открыт чат, то ничего не делаем
        if not self.is_active():
            return

        global_button = (98, 140)
        executor = Actions()
        executor.click(global_button)

        keyboard.write(msg)
        keyboard.press_and_release('enter')


is_friend = False


class Location:
    def __init__(self):
        self.index_location = {
            0: 'home',
            1: 'cafe',
            2: 'street',
            3: 'sqwer',
            4: 'school',
            5: 'park',
            6: 'bal',
            7: 'club',
            8: 'beach',
            9: 'movie',
            10: 'beauty',
        }
        self.location_detect_coord = {
            'guest': [(18, 749), (270, 706), (82, 669), (998, 711)],
            'gui': [(52, 668), (354, 702), (1248, 69)],

            'home': [(270, 706), (1257, 68)],

            'cafe': [(12, 439), (11, 195), (1279, 46)],

            'street': [(14, 48), (1283, 44), (1284, 704)],

            'sqwer': [(11, 362), (13, 41), (1100, 41)],

            'school': [(12, 326), (13, 43), (1280, 48)],

            'park': [(11, 41), (12, 265), (342, 41)],

            'bal': [(15, 44), (1283, 44), (1282, 222)],

            'club': [(12, 46), (1284, 43), (1216, 42)],

            'movie': [(12, 47), (917, 44), (1282, 42)],

            'beauty': [(16, 44), (856, 46), (1280, 53)],

        }
        self.location_detect_pixels = {'guest': [(24, 0, 49), (223, 239, 248), (253, 253, 253), (189, 88, 238)],
                                       'gui': [(241, 241, 241), (201, 112, 177), (226, 243, 249)],
                                       'home': [(200, 109, 177), (224, 241, 248)],
                                       'cafe': [(165, 35, 35), (127, 101, 149), (178, 44, 35)],
                                       'street': [(84, 83, 88), (49, 49, 49), (133, 107, 94)],
                                       'sqwer': [(148, 84, 46), (201, 152, 120), (72, 64, 60)],
                                       'school': [(84, 42, 16), (218, 194, 150), (223, 190, 136)],
                                       'park': [(104, 91, 83), (31, 49, 19), (86, 151, 52)],
                                       'bal': [(197, 174, 180), (210, 185, 189), (239, 239, 239)],
                                       'club': [(39, 64, 132), (102, 46, 118), (158, 139, 186)],
                                       'movie': [(20, 20, 22), (47, 47, 45), (0, 8, 15)],
                                       'beauty': [(173, 185, 187), (187, 200, 201), (176, 187, 186)]}

        self.visual_helper_event = Visual_Event()

        self.esc_problem = {
            'guest_close': [
                ((425, 286), (32, 127, 196)),
                ((862, 293), (34, 129, 194)),
                ((708, 473), (68, 167, 54))
            ],
            'exit_esc': [((426, 300), (33, 127, 195)),
                         ((863, 304), (33, 129, 195)),
                         ((696, 446), (77, 176, 63)),
                         ((612, 440), (217, 217, 217)),
                         ],
            'guest_many_people': [
                ((422, 275), (31, 126, 194)),
                ((878, 519), (18, 116, 165)),
                ((618, 482), (71, 169, 57)),
            ],

        }

        self.click_problem = {
            'err_333': [
                ((434, 285), (35, 132, 199)),
                ((864, 287), (33, 129, 196)),
                ((695, 464), (74, 173, 59)),
            ],
            'update_game': [
                ((319, 115), (33, 128, 196)),
                ((969, 648), (18, 116, 165)),
                ((341, 636), (12, 76, 107)),
            ],
            'hide_window': [((436, 263), (36, 132, 198)),
                            ((869, 262), (33, 128, 195)),
                            ((605, 491), (73, 172, 59))
                            ],

            'error_ok': [((439, 297), (37, 133, 201)),
                         ((869, 303), (32, 127, 194)),
                         ((704, 453), (73, 172, 58)),
                         ],

            'passport_open': [
                ((240, 100), (32, 127, 197)),
                ((1053, 684), (18, 116, 165)),
                ((554, 214), (25, 217, 253)),
                ((668, 143), (223, 246, 254)),
                ((1029, 121), (218, 235, 245)),
            ],

            'friend_window': [
                ((312, 115), (34, 129, 196)),
                ((994, 684), (15, 112, 160)),
                ((435, 638), (206, 206, 206)),
            ],
        }

        self.pillow_window = [
            ['right', ((1196, 220), (91, 187, 77))],
            ['right', ((1077, 218), (223, 223, 223))],
        ]

        # если принимаем запрос, то просто изменяется координата клика в конце выполнения
        if is_friend:
            self.click_problem['friend_window'].append(((876, 636), (71, 170, 57)))

    # исправляет непаладки до тех пор, пока ситуация не станет адекватной. return situation
    def wait(self, timeLimit=45):
        executer = Actions()
        situation = self.situation()
        chat = Chat()
        time_break = TimeLimit(timeLimit)
        while situation == None and time_break.check():
            # ищем известные проблемы
            executer.problems()
            time.sleep(0.75)

            # Надеемся что поможет при неизвестных проблемах. Вероятность 10%
            if random.random() < 0.1:
                keyboard.press_and_release('esc')
                time.sleep(1.2)

            # Закрываем чат если активен
            if chat.is_active():
                chat.close()

            # Если вышли из цикла по превышению таймера
            if not time_break.check():
                keyboard.press_and_release('esc')
                Window()
                break
            else:
                print('check window close ', time_break.show(), time_break.limit)

            situation = self.situation()

        return situation

    # определяет текущие правильные цвета пикселей локации
    def set_pixels_location(self):
        # Предполагается что стоим дома с доступом к кнопкам.

        executer = Actions()

        self.location_detect_pixels['home'] = [pg.pixel(x, y) for x, y in self.location_detect_coord['home']]

        # Попадаем в локацию с индексом 1(с гарантией)
        info = [
            ['left', ((121, 698), (45, 155, 208))],
            ['right', ((152, 300), (76, 205, 254))],
            ['right', ((79, 242), (233, 48, 55))],
            ['right', ((802, 70), (105, 215, 251))],
            ['left', ((504, 475), (75, 18, 12))],
            ['right', ((143, 133), (255, 225, 0))],
        ]
        executer.executes(info, False)

        print('Сохранил 1 локацию')
        # отмечаем gui и локацию с индексом 1
        self.location_detect_pixels['gui'] = [pg.pixel(x, y) for x, y in self.location_detect_coord['gui']]
        self.location_detect_pixels[self.index_location[1]] = [pg.pixel(x, y) for x, y in
                                                               self.location_detect_coord[self.index_location[1]]]

        # Смещаем координаты клика по локации вправо
        info[-2] = ['left', ((804, 475), (75, 18, 12))]
        executer.executes(info, False)

        print('Сохранил 2 локацию')
        # отмечаем локацию с индексом 2
        self.location_detect_pixels[self.index_location[2]] = [pg.pixel(x, y) for x, y in
                                                               self.location_detect_coord[self.index_location[2]]]

        # Смещаем координаты клика по локации вправо
        info[-2] = ['left', ((1104, 475), (75, 18, 12))]
        executer.executes(info, False)

        print('Сохранил 3 локацию')
        # отмечаем локацию с индексом 3
        self.location_detect_pixels[self.index_location[3]] = [pg.pixel(x, y) for x, y in
                                                               self.location_detect_coord[self.index_location[3]]]

        info_part1 = info[:-2]
        info_part2 = info[-2:]

        # Потворяем для оставшихся локаций
        for index in range(4, 11):
            # пропускаем пляж
            if self.index_location[index] == 'beach':
                continue

            # Заходим в меню локаций
            executer.executes(info_part1, False)

            scroll = index - 3
            for _ in range(scroll):
                # Крутим локации
                pg.moveTo(1104, 475)
                pg.mouseDown()
                pg.moveTo(804, 475)
                time.sleep(0.75)
                pg.mouseUp()

            # нажимаем на последнюю локацию
            executer.executes(info_part2, False)

            # отмечаем локацию с индексом index
            self.location_detect_pixels[self.index_location[index]] = [pg.pixel(x, y) for x, y in
                                                                       self.location_detect_coord[
                                                                           self.index_location[index]]]

        # Идем на событие ( с гарантией)
        info = [
            ['right', ((122, 652), (50, 165, 207))],
            ['right', ((199, 727), (35, 139, 196))],
            ['left', ((161, 694), (44, 153, 203))],
            ['right', ((206, 317), (45, 191, 253))],
            ['right', ((78, 236), (252, 80, 92))],
            ['right', ((255, 537), (39, 138, 202))],
            ['left', ((227, 544), (33, 134, 198))],
            ['right', ((257, 543), (44, 190, 253))],
            ['right', ((1079, 186), (214, 214, 214))],
            ['right', ((981, 213), (60, 171, 220))],
            ['left', ((724, 660), (231, 240, 244))],
            ['right', ((193, 725), (32, 135, 195))],
            ['right', ((1004, 682), (167, 140, 245))],
        ]
        executer.executes(info, False)

        self.location_detect_pixels['guest'] = [pg.pixel(x, y) for x, y in
                                                self.location_detect_coord['guest']]

        print('self.location_detect_pixels = ', self.location_detect_pixels)

    def goto(self, name):
        # Если не распознали ситуацию, или если уже на локации не выполняем
        situation = self.situation()
        # print('situation: ', situation , ' -> ', name)
        if situation == None or situation == name:
            return

        # реализатор действий
        executer = Actions()

        # Преобразуем имя в индекс локации
        index = None
        for i, Name in self.index_location.items():
            if name == Name:
                index = i
                break

        # print('index: ', index)

        # если домой
        if index == 0:
            # кнопка дома
            info = [
                ['left', ((70, 698), (45, 155, 208))],
                ['right', ((258, 703), (198, 98, 172))],
                ['right', ((909, 720), (193, 71, 235))],
            ]
            executer.executes(info)
            return

        # если на локации
        info_part1 = [
            ['left', ((121, 698), (45, 155, 208))],
            ['right', ((152, 300), (76, 205, 254))],
            ['right', ((79, 242), (233, 48, 55))],
            ['right', ((802, 70), (105, 215, 251))],
        ]

        info_part2 = [
            ['left', ((504, 475), (75, 18, 12))],
            ['right', ((1057, 679), (113, 176, 253))],
        ]

        # Заходим в меню локаций
        executer.executes(info_part1)
        time.sleep(0)

        scroll = index - 3
        for _ in range(scroll):
            # Крутим локации
            pg.moveTo(1104, 475)
            pg.mouseDown()
            pg.moveTo(804, 475)
            time.sleep(0.75)
            pg.mouseUp()

        if index == 2:
            info_part2[0] = ['left', ((804, 475), (75, 18, 12))]
        elif index > 2:
            info_part2[0] = ['left', ((1104, 475), (75, 18, 12))]

        # нажимаем на последнюю локацию
        executer.executes(info_part2)

    def open_guests(self, index):
        # Неизвестная ситуация или Неправильный  индекс события - ничего не делаем
        if self.situation() == None or index < 1:
            return

        executer = Actions()
        open_guest_menu = [
            ['left', ((119, 693), (47, 159, 209))],
            ['right', ((230, 319), (41, 189, 253))],
            ['right', ((84, 234), (226, 69, 79))],
            ['right', ((511, 81), (110, 221, 250))],
            ['left', ((240, 542), (36, 136, 200))],
            ['right', ((240, 542), (46, 191, 253))],
            ['right', ((949, 199), (62, 175, 220))],
            ['right', ((1085, 194), (209, 209, 210))],
        ]
        executer.executes(open_guest_menu)

        # Скролим и определяем координаты нажатия
        if index > 6:
            scroll = index - 6
            delta_click = (623, 220 + 87 * (6 - 1))

            # скролим до гостя нужного индекса
            for _ in range(scroll):
                pg.moveTo(delta_click[0], delta_click[1])
                pg.mouseDown()
                pg.moveTo(delta_click[0], delta_click[1] - 92)
                time.sleep(0.75)
                pg.mouseUp()
        else:
            delta_click = (623, 220 + 87 * (index - 1))

        # координаты инфо события
        coord_blue_info = (943, delta_click[1])

        # если событий под данным индексом нет , ничего не делать(проверка по синиму i события)
        current_pixel = pg.pixel(coord_blue_info[0], coord_blue_info[1])
        if current_pixel[0] > 200 and current_pixel[0] > 200 and current_pixel[0] > 200:
            print('Слишком большой индекс события')
            return

        # гарантировано заходим в инфо события
        info = [
            ['left', (coord_blue_info, (0, 0, 0))],
            ['right', ((350, 198), (33, 128, 198))],
            ['right', ((401, 571), (14, 109, 156))],
            ['right', ((928, 213), (218, 234, 244))],
        ]
        executer.executes(info)
        info_event_out = self.visual_helper_event.read_info_event()

        # выходим из инфо
        keyboard.press_and_release('esc')
        time.sleep(0.75)

        # заходим в событие
        executer.click(delta_click)

        # исправляем возмужную проблему esc

        while True:
            situation = self.wait()
            if situation == 'home':
                return False

            elif situation == 'guest':
                # заходим в инфо комнаты
                info_room_input = [
                    ['left', ((553, 86), (51, 159, 212))],
                    ['right', ((175, 93), (32, 127, 196))],
                    ['right', ((1114, 695), (18, 116, 165))],
                ]
                executer.executes(info_room_input)

                # получаем информацию и обьединяем словари - всю информацию по событию
                info_event_in = self.visual_helper_event.read_info_room()

                # обьединяем всю информацию
                info_event = info_event_out | info_event_in

                # выходим из меню комнаты
                keyboard.press_and_release('esc')

                # ждем окончательного выхода из меню
                while self.situation() != 'guest':
                    time.sleep(0.2)

                return info_event

    def situation(self):
        for name, coords in self.location_detect_coord.items():
            current_pixels = []
            for coord in coords:
                current_pixels.append(pg.pixel(coord[0], coord[1]))

            if current_pixels == self.location_detect_pixels[name]:
                if name != 'gui':
                    return name


class Visual_Event:
    def __init__(self):
        pass

    # меню события
    def read_info_event(self):
        img_name = cv2.cvtColor(np.array(pg.screenshot(region=(372, 323, 910 - 392, 376 - 323))), cv2.COLOR_RGB2BGR)
        img_type = cv2.cvtColor(np.array(pg.screenshot(region=(491, 377, 819 - 491, 402 - 377))), cv2.COLOR_RGB2BGR)
        img_descript = cv2.cvtColor(np.array(pg.screenshot(region=(372, 428, 919 - 372, 490 - 428))), cv2.COLOR_RGB2BGR)

        info = {
            'name': '',
            'type': '',
            'descript': '',

        }

        for i, image in enumerate([img_name, img_type, img_descript]):
            # 1. Конвертируем в HSV и выделяем синий фон
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_blue = (90, 50, 50)  # Диапазон синего
            upper_blue = (130, 255, 255)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # 2. Инвертируем маску (чтобы текст стал черным на белом фоне)
            mask_inv = cv2.bitwise_not(mask)

            # 3. Применяем бинаризацию
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # 4. Объединяем маску и бинаризацию
            final = cv2.bitwise_and(thresh, mask_inv)

            # Сохраняем обработанное изображение (для проверки)
            # cv2.imwrite("processed_text.png", final)

            # 5. Распознаем текст
            text = pytesseract.image_to_string(Image.fromarray(final), lang='rus+eng')

            # убираем переносы строк
            text = text.replace('\n', '')

            # добавляем в словарь по порядку
            key = list(info.keys())[i]
            info[key] = text

        return info

    # меню комнаты
    def read_info_room(self, all_rooms=False):
        def to_text(image, key):

            # 1. Конвертируем в HSV и выделяем синий фон
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 2. Применяем бинаризацию
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 3-4 Адаптируем под синий фон
            if key == 'player':
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

                lower_blue = (90, 50, 50)  # Диапазон синего
                upper_blue = (130, 255, 255)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)

                # 3. Инвертируем маску (чтобы текст стал черным на белом фоне)
                mask_inv = cv2.bitwise_not(mask)

                # 4. Объединяем маску и бинаризацию
                gray = cv2.bitwise_and(thresh, mask_inv)

            # 5. Распознаем текст
            text = pytesseract.image_to_string(Image.fromarray(gray), lang='rus+eng')

            # Сохраняем обработанное изображение (для проверки)
            # cv2.imwrite(f"processed_text{random.random()}.png", gray)
            # cv2.waitKey(0)

            # убираем переносы строк
            text = text.replace('\n', '')

            return text

        img_name_player = cv2.cvtColor(np.array(pg.screenshot(region=(876, 390, 1116 - 876, 445 - 390))),
                                       cv2.COLOR_RGB2BGR)
        img_name_room = cv2.cvtColor(np.array(pg.screenshot(region=(405, 616, 667 - 385, 666 - 616))),
                                     cv2.COLOR_RGB2BGR)

        info = {
            'player': '',
            'rooms': [],

        }

        # распознаем
        for i, image in enumerate([img_name_player, img_name_room]):
            key = list(info.keys())[i]
            # распознаем текст
            text = to_text(image, key)

            # добавляем в словарь по порядку
            if key == 'player':
                info['player'] = text
            elif key == 'rooms':
                # если название комнаты, то добавляем в массив названий комнат
                info['rooms'].append(text)

        # читаем названия всех остальных комнат
        if all_rooms:

            # Если названия по порядку совпали, то обрываем
            while True:
                pg.moveTo(800, 450)
                pg.mouseDown()
                pg.moveTo(270, 450)
                pg.mouseUp()
                time.sleep(0)

                img_name_room = cv2.cvtColor(np.array(pg.screenshot(region=(405, 616, 667 - 385, 666 - 616))),
                                             cv2.COLOR_RGB2BGR)
                room_name = to_text(img_name_room, key='rooms')

                # Если названия одинаковые значит комнаты закончились. Выходим
                if room_name == info['rooms'][-1]:
                    break

                # Добавляем
                info['rooms'].append(room_name)

        return info


class Actions:
    def __init__(self):
        self.finished = None

    def click(self, pos, button='left'):
        x, y = pos
        pg.moveTo(x, y)
        pg.click(button=button)

    # ['left'/'right',(pos),(color)]
    # Использует и постоянно проверяет в self.executes
    def problems(self):
        location = Location()

        self.click((100, 100), button='right')

        for name_problem, info in location.esc_problem.items():
            for pos_pix in info:
                pos, need_color = pos_pix[0], pos_pix[1]
                x, y = pos[0], pos[1]
                current_color = pg.pixel(x, y)
                if current_color != need_color:
                    break
                elif pos_pix == info[-1]:
                    print(f'Обнаружена проблема {name_problem}. Решаю - жму ESC')
                    keyboard.press_and_release('esc')
                    self.finished = False
                    return True

        for name_problem, info in location.click_problem.items():
            for pos_pix in info:
                pos, need_color = pos_pix[0], pos_pix[1]
                x, y = pos[0], pos[1]
                current_color = pg.pixel(x, y)
                if current_color != need_color:
                    break
                elif pos_pix == info[-1]:
                    print(f'Обнаружена проблема {name_problem}. Решаю, кликаю "ОК"')
                    # по идее он проверяет последовательность пикселей ошибки, а последний проверяемый пиксель, является также местом клика для решения проблемы
                    self.click(pos)
                    # Если вылазят окна предложений , то мы нажали галочку(больше не показывать) и нужно еще закрыть само окно с помощью ESC
                    if name_problem == 'update_game':
                        keyboard.press_and_release('esc')

                    self.finished = False
                    return True

        self.finished = True
        return False

    # Если возникает проблема - решает проблему и прерывает выполнение действий
    def executes(self, info, check_problem=True):
        index = 0
        last_type = 'left'
        activing = True
        location = Location()
        time_limit_problem = 60
        while activing:

            # продолжаем выполнение
            for i in range(index, len(info)):
                action = info[i]
                type, pos_pixel = action[0], action[1]
                if type == 'right' and last_type == 'left':
                    index = i

                last_type = type

                pos, need_color = pos_pixel
                x, y = pos
                if type == 'right':
                    current_color = pg.pixel(x, y)

                    # print(pos, current_color) #-------------------------------------------------
                    # pg.moveTo(pos)            #-------------------------------------------------

                    if current_color != need_color:
                        time.sleep(1)
                        break

                elif type == 'left':
                    self.click(pos)

                if i == len(info) - 1:
                    activing = False

            # Проверяем, попали ли мы в проблему.
            # Если попали, то решаем и прерываем выполнение
            if check_problem and self.problems():
                location.wait(time_limit_problem)
                break


class Programms:
    def __init__(self):
        self.window = Window(False)  # False - не перезапускаем окно в начале запуска
        self.location = Location()
        self.action = Actions()
        self.chat = Chat()

        # Сбой определения или изменение пикселей на локациях
        self.crash_pixels_location = False

        # Проверка сбоя
        if not self.crash_pixels_location:
            # устраняем возможные начальные неполадки
            self.location.wait()
        else:
            # сбой пикселей - исправляем, определяем пиксели на локациях
            self.location.goto('home')
            self.location.set_pixels_location()

        # настройка бота
        self.event_control_time = TimeLimit(60 * 10)
        self.free_lama_time = TimeLimit(60 * 30)
        self.presents_time = TimeLimit(60 * 30)

        self.check_bot_event = False
        self.check_lama = True
        self.check_time_presents = True

        self.Event_Control(False)
        self.Free_Lama(True)
        self.Time_Present(True)

        # self.Cake_clear()

        # 'Акаме 9476264'
        # 'Phineas '
        # location.open_quests(index>0)
        # location.goto(name_location)
        # chat.open()
        # chat.write_msg(msg_text)
        # chat.close()

    def time_control(self):

        if self.check_bot_event:
            self.Event_Control(False)

        if self.check_lama:
            self.Free_Lama(False)

        if self.check_time_presents:
            self.Time_Present(False)

    def catch_information(self):
        baseOff = BaseOffers()

        def f_stop(time_update):
            if time_update > 2.5:
                return True
            return False

        def repeat_action(name, event=False):

            if event:
                index = name

                # идем на событие. Сохраняем информацию о последнем событии в классе чата
                self.chat.event_info = self.location.open_guests(index)
                # если не попали на событие(дома)
                if self.chat.event_info == False:
                    print('Не смог попасть на событиее. Пропускаю')
                    return

                info = [self.chat.event_info['name'], datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                name = self.chat.event_info['player']
                file = 'events.json'

            else:
                self.location.goto(name)
                info = []
                file = 'base.json'

            situation = self.location.wait()
            self.Police_animation(situation)

            # читаем глобальный чат
            chat = self.chat.reading_gl(f_stop=f_stop, duration_stop=15 * 60)
            chat.save(name, info, file)

            # дожидаемся распознавания ситуации
            situation = self.location.wait()
            # players = self.find_players(name)
            by_player = None
            if event:
                event_info = self.chat.event_info
                if situation != 'home':
                    name = event_info['name']
                    by_player = event_info['player']

            # baseOff.add_info(name, players_list, by_player)

            # проверяем таймера
            self.Time_Present()
            self.Free_Lama()

        def main():
            for _ in range(1000):

                # идем на событку, если не получилось то пропускаем
                if False:
                    for i in range(1, 20):
                        repeat_action(i, True)

                if True:
                    repeat_action('cafe')
                    self.Cake_clear()
                    repeat_action(1, True)
                    repeat_action('street')
                    repeat_action(2, True)
                    repeat_action('sqwer')
                    repeat_action(3, True)
                    repeat_action('school')
                    repeat_action(4, True)
                    repeat_action('park')
                    repeat_action(5, True)
                    repeat_action('bal')
                    repeat_action(6, True)
                    repeat_action('club')
                    repeat_action(7, True)

        # Запускаем программу
        main()

    def Police_animation(self, location):
        visual = Visual_Player()

        # находим бота
        bot_pos = visual.find_bot(location)
        if bot_pos == None:
            return

        # кликаем на бота
        self.action.click(bot_pos)
        self.action.click(bot_pos)
        time.sleep(0.75)

        # ищем кнопку анимаций
        find_button = visual.find_image_on_screen('button_animation.png')
        if find_button:
            self.action.click(find_button)
            time.sleep(0.75)
        else:
            # выходим из кнопок
            self.action.click((17, 47))
            return

        # ищем кнопку анимаций
        find_button = visual.find_image_on_screen('anim_police.png')
        if find_button:
            self.action.click(find_button)
            time.sleep(0.75)
        else:
            # выходим из кнопок
            self.action.click((17, 47))
            return

    def Cafe_walk(self):
        visual = Visual_Player()

        points = [
            ['right', ((638, 726), (0, 0, 0))],
            ['right', ((305, 489), (124, 101, 109))],
            ['right', ((940, 733), (17, 17, 17))],
            ['right', ((957, 262), (180, 180, 180))],
        ]
        point = random.choice(points)[1][0]

        pixel1 = pg.pixel(point[0], point[1] - 100)
        pixel2 = pg.pixel(point[0], point[1] - 200)
        pixel3 = pg.pixel(point[0], point[1] - 50)

        self.action.click(point)

        pixel11 = pg.pixel(point[0], point[1] - 100)
        pixel22 = pg.pixel(point[0], point[1] - 200)
        pixel33 = pg.pixel(point[0], point[1] - 50)

        if [pixel1, pixel2, pixel3] == [pixel11, pixel22, pixel33]:
            bot_pos = (10000, 10000)
            while (bot_pos[0] - point[0]) ** 2 + (bot_pos[1] + 50 - point[1]) ** 2 > 50 ** 2:
                bot_pos = visual.find_bot('cafe')
                if bot_pos == None:
                    bot_pos = (1000, 1000)
                time.sleep(1)
            time.sleep(3)

            self.Police_animation('cafe')





        else:
            print(False)
            self.action.click(point)

    def find_players(self, location, name='+]=-0', id='ab8'):
        def get_points():
            points = []
            if location == 'cafe':
                points = [
                    ['right', ((481, 228), (54, 52, 52))],
                    ['right', ((530, 253), (57, 55, 55))],
                    ['right', ((581, 274), (35, 33, 33))],
                    ['right', ((683, 144), (246, 219, 213))],
                    ['right', ((721, 124), (189, 62, 59))],
                    ['right', ((847, 134), (183, 51, 49))],
                    ['right', ((886, 151), (249, 223, 217))],
                    ['right', ((1086, 321), (242, 87, 71))],
                    ['right', ((1188, 304), (197, 61, 58))],
                    ['right', ((1229, 318), (233, 200, 190))],
                    ['right', ((1029, 493), (242, 89, 71))],
                    ['right', ((1158, 487), (237, 76, 67))],
                    ['right', ((733, 331), (236, 80, 65))],
                    ['right', ((841, 319), (233, 73, 62))],
                    ['right', ((882, 337), (239, 82, 67))],
                    ['right', ((867, 580), (236, 77, 66))],
                    ['right', ((758, 570), (231, 75, 65))],
                    ['right', ((715, 589), (242, 86, 69))],
                    ['right', ((705, 433), (232, 76, 66))],
                    ['right', ((583, 434), (232, 75, 65))],
                    ['right', ((355, 556), (193, 55, 52))],
                    ['right', ((318, 576), (227, 188, 177))],
                    ['right', ((479, 542), (180, 47, 46))],
                    ['right', ((518, 563), (246, 219, 213))],
                    # стулья

                    ['right', ((13, 622), (137, 19, 20))],
                    ['right', ((13, 679), (3, 3, 3))],
                    ['right', ((10, 720), (76, 44, 55))],
                    ['right', ((103, 731), (13, 8, 10))],
                    ['right', ((104, 684), (84, 52, 63))],
                    ['right', ((103, 640), (83, 53, 63))],
                    ['right', ((102, 573), (133, 17, 18))],
                    ['right', ((132, 565), (140, 45, 46))],
                    ['right', ((145, 605), (15, 12, 13))],
                    ['right', ((173, 618), (17, 17, 17))],
                    ['right', ((205, 619), (96, 69, 78))],
                    ['right', ((231, 615), (26, 25, 26))],
                    ['right', ((237, 591), (28, 28, 28))],
                    ['right', ((203, 590), (97, 70, 79))],
                    ['right', ((214, 569), (62, 32, 42))],
                    ['right', ((254, 570), (33, 33, 33))],
                    ['right', ((257, 543), (34, 34, 34))],
                    ['right', ((294, 544), (116, 93, 101))],
                    ['right', ((315, 513), (124, 101, 109))],
                    ['right', ((300, 485), (121, 98, 106))],
                    ['right', ((333, 499), (131, 108, 116))],
                    ['right', ((381, 501), (144, 125, 131))],
                    ['right', ((403, 504), (149, 130, 136))],
                    ['right', ((442, 509), (78, 80, 79))],
                    ['right', ((468, 511), (82, 82, 82))],
                    ['right', ((410, 471), (82, 81, 80))],
                    ['right', ((436, 473), (84, 84, 84))],
                    ['right', ((472, 475), (89, 89, 89))],
                    ['right', ((496, 476), (91, 91, 91))],
                    ['right', ((525, 474), (95, 95, 95))],
                    ['right', ((462, 446), (92, 92, 92))],
                    ['right', ((501, 446), (172, 157, 162))],
                    ['right', ((527, 445), (176, 161, 166))],
                    ['right', ((495, 421), (102, 102, 102))],
                    ['right', ((537, 421), (107, 107, 107))],
                    ['right', ((566, 419), (111, 111, 111))],
                    ['right', ((543, 398), (186, 174, 178))],
                    ['right', ((579, 397), (190, 179, 183))],
                    ['right', ((615, 401), (195, 184, 188))],
                    ['right', ((665, 400), (127, 127, 127))],
                    ['right', ((702, 403), (204, 193, 197))],
                    ['right', ((729, 402), (192, 184, 187))],
                    ['right', ((598, 369), (125, 125, 125))],
                    ['right', ((633, 369), (128, 128, 128))],
                    ['right', ((667, 370), (132, 132, 132))],
                    ['right', ((699, 366), (137, 137, 137))],
                    ['right', ((602, 337), (206, 197, 200))],
                    ['right', ((630, 338), (209, 200, 203))],
                    ['right', ((681, 332), (141, 141, 141))],
                    ['right', ((613, 304), (213, 204, 207))],
                    ['right', ((649, 304), (142, 142, 142))],
                    ['right', ((689, 309), (146, 146, 146))],
                    ['right', ((731, 308), (222, 218, 219))],
                    ['right', ((771, 306), (156, 156, 156))],
                    ['right', ((800, 305), (181, 177, 178))],
                    ['right', ((833, 308), (237, 233, 234))],
                    ['right', ((872, 308), (166, 166, 166))],
                    ['right', ((921, 311), (235, 231, 232))],
                    ['right', ((955, 309), (163, 163, 163))],
                    ['right', ((983, 311), (160, 160, 160))],
                    ['right', ((1024, 313), (181, 178, 179))],
                    ['right', ((1050, 282), (145, 118, 183))],
                    ['right', ((1006, 280), (243, 241, 242))],
                    ['right', ((958, 281), (246, 244, 245))],
                    ['right', ((898, 273), (173, 173, 173))],
                    ['right', ((849, 270), (170, 170, 170))],
                    ['right', ((815, 270), (152, 151, 152))],
                    ['right', ((749, 266), (160, 160, 160))],
                    ['right', ((708, 266), (156, 156, 156))],
                    ['right', ((657, 264), (151, 151, 151))],
                    ['right', ((610, 259), (216, 210, 212))],
                    ['right', ((578, 256), (143, 143, 143))],
                    ['right', ((599, 228), (225, 219, 221))],
                    ['right', ((651, 228), (156, 156, 156))],
                    ['right', ((718, 235), (237, 233, 234))],
                    ['right', ((766, 239), (131, 131, 131))],
                    ['right', ((808, 238), (91, 91, 91))],
                    ['right', ((880, 246), (176, 176, 176))],
                    ['right', ((945, 252), (254, 254, 254))],
                    ['right', ((987, 252), (179, 179, 179))],
                    ['right', ((1027, 255), (121, 66, 54))],
                    ['right', ((1070, 256), (232, 213, 192))],
                    ['right', ((1108, 254), (182, 43, 40))],
                    ['right', ((1145, 250), (159, 190, 107))],
                    # левый тунель и вверх и с левым верхним углом

                    ['right', ((906, 325), (157, 157, 157))],
                    ['right', ((944, 323), (157, 157, 157))],
                    ['right', ((991, 329), (151, 151, 151))],
                    ['right', ((1040, 326), (222, 216, 218))],
                    ['right', ((1058, 354), (163, 156, 159))],
                    ['right', ((1016, 356), (213, 204, 207))],
                    ['right', ((958, 358), (142, 142, 142))],
                    ['right', ((919, 363), (196, 188, 191))],
                    ['right', ((905, 402), (203, 192, 196))],
                    ['right', ((954, 400), (199, 188, 192))],
                    ['right', ((1000, 395), (197, 186, 190))],
                    ['right', ((1048, 398), (190, 178, 182))],
                    ['right', ((1078, 398), (111, 111, 111))],
                    ['right', ((1126, 419), (98, 98, 98))],
                    ['right', ((1076, 426), (100, 100, 100))],
                    ['right', ((1023, 429), (178, 166, 170))],
                    ['right', ((968, 426), (112, 112, 112))],
                    ['right', ((909, 429), (157, 152, 153))],
                    ['right', ((871, 432), (116, 116, 116))],
                    ['right', ((843, 458), (181, 169, 173))],
                    ['right', ((909, 459), (175, 163, 167))],
                    ['right', ((971, 459), (97, 97, 97))],
                    ['right', ((1027, 459), (166, 149, 155))],
                    ['right', ((1075, 460), (85, 85, 85))],
                    ['right', ((1125, 459), (155, 136, 142))],
                    ['right', ((1175, 459), (73, 73, 73))],
                    ['right', ((1247, 456), (138, 117, 124))],
                    ['right', ((1275, 457), (60, 60, 60))],
                    ['right', ((1190, 515), (50, 50, 50))],
                    ['right', ((1086, 466), (82, 82, 82))],
                    ['right', ((1011, 462), (139, 122, 128))],
                    ['right', ((966, 455), (100, 100, 100))],
                    ['right', ((912, 453), (178, 166, 170))],
                    ['right', ((867, 454), (108, 108, 108))],
                    ['right', ((816, 456), (183, 171, 175))],
                    ['right', ((769, 456), (110, 110, 110))],
                    ['right', ((731, 455), (185, 173, 177))],
                    ['right', ((733, 494), (171, 156, 161))],
                    ['right', ((764, 491), (95, 95, 95))],
                    ['right', ((842, 494), (166, 149, 155))],
                    ['right', ((880, 493), (91, 90, 90))],
                    ['right', ((942, 495), (159, 142, 148))],
                    ['right', ((984, 498), (157, 138, 144))],
                    ['right', ((1191, 521), (49, 49, 49))],
                    ['right', ((980, 537), (65, 65, 65))],
                    ['right', ((917, 534), (147, 128, 134))],
                    ['right', ((869, 528), (77, 77, 77))],
                    ['right', ((843, 527), (80, 80, 80))],
                    ['right', ((804, 523), (83, 83, 83))],
                    ['right', ((768, 523), (84, 84, 84))],
                    ['right', ((727, 523), (85, 85, 85))],
                    ['right', ((693, 527), (84, 84, 84))],
                    ['right', ((660, 549), (152, 133, 139))],
                    ['right', ((698, 550), (152, 133, 139))],
                    ['right', ((748, 553), (149, 130, 136))],
                    ['right', ((785, 554), (148, 129, 135))],
                    ['right', ((828, 556), (144, 125, 131))],
                    ['right', ((863, 556), (68, 66, 67))],
                    ['right', ((916, 559), (138, 117, 124))],
                    ['right', ((965, 560), (58, 58, 58))],
                    ['right', ((1017, 563), (128, 105, 113))],
                    ['right', ((1164, 573), (36, 36, 36))],
                    ['right', ((1183, 613), (26, 26, 26))],
                    ['right', ((1108, 606), (107, 81, 90))],
                    ['right', ((1035, 603), (115, 89, 98))],
                    ['right', ((988, 603), (117, 94, 102))],
                    ['right', ((937, 596), (123, 100, 108))],
                    ['right', ((889, 594), (127, 102, 111))],
                    ['right', ((672, 573), (69, 69, 69))],
                    ['right', ((644, 573), (69, 69, 69))],
                    ['right', ((599, 567), (72, 72, 72))],
                    ['right', ((564, 528), (82, 82, 82))],
                    ['right', ((583, 603), (134, 113, 120))],
                    ['right', ((619, 603), (134, 113, 120))],
                    ['right', ((669, 608), (57, 57, 57))],
                    ['right', ((915, 633), (114, 91, 97))],
                    ['right', ((986, 642), (33, 33, 33))],
                    ['right', ((1024, 640), (105, 79, 88))],
                    ['right', ((1088, 643), (26, 26, 26))],
                    ['right', ((1128, 643), (97, 70, 79))],
                    ['right', ((1164, 644), (93, 62, 70))],
                    ['right', ((1162, 691), (14, 12, 13))],
                    ['right', ((1075, 696), (17, 17, 17))],
                    ['right', ((966, 691), (23, 23, 23))],
                    ['right', ((905, 690), (103, 74, 84))],
                    ['right', ((852, 685), (31, 31, 31))],
                    ['right', ((814, 686), (106, 80, 89))],
                    ['right', ((718, 690), (109, 83, 92))],
                    ['right', ((669, 681), (38, 38, 38))],
                    ['right', ((617, 674), (113, 87, 96))],
                    ['right', ((583, 672), (41, 41, 41))],
                    ['right', ((547, 672), (41, 41, 41))],
                    ['right', ((510, 671), (116, 90, 99))],
                    ['right', ((492, 671), (41, 41, 41))],
                    ['right', ((407, 655), (156, 148, 162))],
                    ['right', ((403, 696), (108, 82, 91))],
                    ['right', ((501, 695), (111, 85, 94))],
                    ['right', ((545, 700), (110, 84, 93))],
                    ['right', ((593, 704), (109, 83, 92))],
                    ['right', ((645, 703), (108, 82, 91))],
                    ['right', ((705, 705), (106, 80, 89))],
                    ['right', ((770, 705), (30, 30, 30))],
                    ['right', ((806, 708), (102, 75, 84))],
                    ['right', ((863, 711), (26, 23, 24))],
                    ['right', ((909, 711), (97, 70, 77))],
                    ['right', ((967, 710), (21, 21, 21))],
                    ['right', ((1073, 709), (15, 15, 15))],
                    ['right', ((1166, 706), (87, 55, 66))],
                    ['right', ((1167, 746), (7, 6, 6))],
                    ['right', ((1102, 750), (83, 53, 63))],
                    ['right', ((1056, 750), (87, 55, 66))],
                    ['right', ((999, 751), (87, 57, 67))],
                    ['right', ((949, 750), (90, 60, 70))],
                    ['right', ((888, 747), (19, 15, 17))],
                    ['right', ((840, 743), (96, 66, 76))],
                    ['right', ((805, 743), (96, 69, 78))],
                    ['right', ((749, 739), (23, 23, 23))],
                    ['right', ((692, 741), (25, 26, 26))],
                    ['right', ((653, 741), (26, 26, 26))],
                    ['right', ((597, 740), (102, 75, 84))],
                    ['right', ((547, 738), (27, 27, 27))],
                    ['right', ((489, 743), (100, 74, 83))],
                    ['right', ((433, 748), (100, 73, 82))],
                    ['right', ((389, 751), (98, 71, 80))],
                    ['right', ((331, 748), (98, 71, 80))],
                    ['right', ((269, 751), (95, 65, 75))],
                    ['right', ((222, 753), (90, 60, 70))],
                    ['right', ((169, 751), (86, 56, 66))],
                    ['right', ((126, 751), (84, 52, 63))],
                    ['right', ((82, 751), (81, 49, 60))],
                    # вся правая часть

                ]
            elif location == 'street':
                points = [
                    ['right', ((110, 459), (85, 56, 40))],
                    ['right', ((198, 474), (234, 77, 67))],
                    ['right', ((297, 363), (235, 76, 67))],
                    ['right', ((332, 348), (236, 76, 66))],
                    ['right', ((434, 358), (233, 78, 67))],
                    ['right', ((452, 579), (168, 109, 72))],
                    ['right', ((493, 560), (151, 95, 60))],
                    ['right', ((767, 555), (147, 92, 60))],
                    ['right', ((807, 573), (172, 113, 75))],
                    ['right', ((944, 441), (163, 103, 68))],
                    ['right', ((916, 432), (177, 115, 74))],
                    ['right', ((738, 341), (164, 104, 70))],
                    ['right', ((703, 325), (178, 116, 77))],
                    ['right', ((606, 273), (50, 17, 5))],
                    ['right', ((575, 260), (169, 109, 72))],
                    ['right', ((474, 209), (125, 75, 48))],
                    ['right', ((437, 195), (77, 40, 25))],

                    # лавки
                    ['right', ((105, 274), (161, 162, 167))],
                    ['right', ((132, 267), (148, 149, 153))],
                    ['right', ((166, 250), (149, 150, 154))],
                    ['right', ((193, 233), (151, 152, 157))],
                    ['right', ((228, 218), (141, 142, 146))],
                    ['right', ((268, 202), (137, 138, 142))],
                    ['right', ((302, 185), (117, 118, 122))],
                    ['right', ((337, 174), (109, 108, 114))],
                    ['right', ((336, 209), (196, 182, 164))],
                    ['right', ((301, 217), (206, 22, 2))],
                    ['right', ((252, 240), (206, 25, 0))],
                    ['right', ((256, 285), (223, 215, 192))],
                    ['right', ((279, 298), (97, 112, 119))],
                    ['right', ((319, 279), (87, 90, 87))],
                    ['right', ((339, 292), (195, 180, 161))],
                    ['right', ((305, 322), (0, 0, 0))],
                    ['right', ((264, 343), (207, 194, 175))],
                    ['right', ((252, 366), (212, 202, 203))],
                    ['right', ((290, 346), (1, 0, 0))],
                    ['right', ((338, 326), (4, 0, 0))],
                    ['right', ((360, 353), (110, 68, 44))],
                    ['right', ((159, 391), (0, 1, 0))],
                    ['right', ((131, 404), (0, 0, 0))],
                    ['right', ((93, 416), (216, 209, 185))],
                    ['right', ((55, 431), (218, 207, 185))],
                    ['right', ((16, 446), (191, 176, 157))],
                    ['right', ((127, 458), (62, 39, 28))],
                    ['right', ((174, 439), (92, 59, 38))],
                    ['right', ((218, 422), (106, 68, 45))],
                    ['right', ((260, 397), (106, 68, 46))],
                    ['right', ((284, 436), (138, 84, 55))],
                    ['right', ((254, 444), (127, 80, 50))],
                    ['right', ((221, 455), (141, 92, 60))],
                    ['right', ((229, 506), (142, 87, 57))],
                    ['right', ((258, 490), (145, 90, 60))],
                    ['right', ((287, 480), (141, 88, 56))],
                    ['right', ((319, 471), (170, 112, 73))],
                    ['right', ((487, 419), (153, 90, 57))],
                    ['right', ((465, 417), (161, 98, 63))],
                    ['right', ((383, 461), (141, 88, 57))],
                    ['right', ((357, 479), (154, 95, 62))],
                    ['right', ((319, 492), (140, 87, 56))],
                    ['right', ((290, 504), (144, 91, 60))],
                    ['right', ((253, 517), (122, 72, 45))],
                    ['right', ((221, 537), (133, 83, 56))],
                    ['right', ((128, 589), (126, 74, 50))],
                    ['right', ((242, 582), (203, 169, 131))],
                    ['right', ((272, 569), (211, 181, 145))],
                    ['right', ((309, 552), (213, 179, 144))],
                    ['right', ((349, 532), (214, 179, 146))],
                    ['right', ((377, 519), (216, 180, 148))],
                    ['right', ((400, 505), (209, 174, 144))],
                    ['right', ((572, 427), (209, 200, 189))],
                    ['right', ((557, 434), (198, 186, 170))],
                    ['right', ((533, 441), (61, 96, 27))],
                    ['right', ((508, 453), (98, 150, 32))],
                    ['right', ((483, 462), (225, 125, 7))],
                    ['right', ((455, 472), (129, 146, 114))],
                    ['right', ((433, 485), (55, 84, 17))],
                    ['right', ((408, 499), (175, 143, 114))],
                    ['right', ((385, 511), (206, 171, 141))],
                    ['right', ((352, 526), (202, 167, 136))],
                    ['right', ((308, 549), (206, 173, 140))],
                    ['right', ((270, 574), (226, 207, 188))],
                    ['right', ((234, 593), (64, 99, 27))],
                    ['right', ((198, 606), (83, 127, 35))],
                    ['right', ((163, 625), (72, 124, 38))],
                    ['right', ((196, 629), (227, 194, 149))],
                    ['right', ((247, 612), (228, 193, 151))],
                    ['right', ((275, 592), (226, 191, 149))],
                    ['right', ((322, 573), (227, 189, 149))],
                    ['right', ((362, 551), (227, 189, 152))],
                    ['right', ((413, 525), (226, 185, 153))],
                    ['right', ((454, 503), (225, 184, 152))],
                    ['right', ((510, 478), (227, 187, 161))],
                    ['right', ((553, 456), (223, 180, 148))],
                    ['right', ((589, 440), (223, 179, 150))],
                    ['right', ((613, 456), (222, 181, 151))],
                    ['right', ((582, 471), (224, 183, 155))],
                    ['right', ((535, 495), (233, 204, 195))],
                    ['right', ((483, 514), (226, 185, 155))],
                    ['right', ((425, 544), (227, 188, 157))],
                    ['right', ((395, 560), (227, 189, 152))],
                    ['right', ((350, 571), (237, 211, 195))],
                    ['right', ((309, 596), (230, 194, 160))],
                    ['right', ((278, 625), (231, 195, 159))],
                    ['right', ((228, 646), (230, 197, 156))],
                    ['right', ((262, 649), (227, 193, 148))],
                    ['right', ((302, 630), (230, 196, 159))],
                    ['right', ((344, 610), (235, 212, 195))],
                    ['right', ((380, 581), (236, 211, 198))],
                    ['right', ((430, 565), (228, 191, 162))],
                    ['right', ((491, 538), (226, 185, 157))],
                    ['right', ((521, 527), (227, 186, 158))],
                    ['right', ((567, 512), (232, 203, 195))],
                    ['right', ((597, 494), (223, 184, 155))],
                    ['right', ((626, 486), (224, 188, 165))],
                    ['right', ((655, 471), (228, 195, 185))],
                    ['right', ((673, 486), (228, 204, 199))],
                    ['right', ((640, 501), (221, 184, 161))],
                    ['right', ((599, 519), (215, 178, 152))],
                    ['right', ((550, 539), (226, 196, 189))],
                    ['right', ((519, 553), (222, 185, 159))],
                    ['right', ((415, 605), (221, 183, 147))],
                    ['right', ((376, 630), (221, 185, 152))],
                    ['right', ((326, 649), (224, 190, 153))],
                    ['right', ((286, 661), (226, 193, 154))],
                    ['right', ((312, 733), (202, 186, 166))],
                    ['right', ((224, 734), (229, 196, 150))],
                    ['right', ((104, 726), (228, 195, 150))],
                    ['right', ((17, 716), (201, 168, 136))],
                    # левое четверокрестие

                    ['right', ((408, 665), (197, 187, 177))],
                    ['right', ((437, 658), (203, 195, 184))],
                    ['right', ((479, 642), (208, 200, 189))],
                    ['right', ((517, 626), (210, 202, 191))],
                    ['right', ((554, 605), (208, 200, 189))],
                    ['right', ((583, 588), (189, 182, 172))],
                    ['right', ((608, 567), (31, 54, 2))],
                    ['right', ((652, 568), (46, 25, 152))],
                    ['right', ((683, 568), (252, 175, 12))],
                    ['right', ((723, 569), (235, 223, 212))],
                    ['right', ((733, 599), (196, 186, 176))],
                    ['right', ((708, 599), (207, 199, 188))],
                    ['right', ((666, 599), (214, 206, 195))],
                    ['right', ((591, 607), (216, 208, 197))],
                    ['right', ((540, 605), (204, 194, 184))],
                    ['right', ((486, 632), (145, 140, 132))],
                    ['right', ((539, 633), (216, 208, 197))],
                    ['right', ((564, 632), (217, 209, 198))],
                    ['right', ((702, 630), (217, 209, 198))],
                    ['right', ((738, 633), (215, 207, 196))],
                    ['right', ((776, 634), (208, 200, 189))],
                    ['right', ((844, 660), (202, 192, 182))],
                    ['right', ((795, 665), (216, 208, 197))],
                    ['right', ((748, 665), (217, 209, 198))],
                    ['right', ((725, 665), (217, 209, 198))],
                    ['right', ((544, 660), (217, 209, 198))],
                    ['right', ((491, 663), (217, 209, 198))],
                    ['right', ((432, 659), (203, 195, 184))],
                    ['right', ((505, 683), (217, 209, 198))],
                    ['right', ((531, 684), (217, 209, 198))],
                    ['right', ((499, 701), (217, 209, 198))],
                    ['right', ((540, 702), (217, 209, 198))],
                    ['right', ((502, 721), (217, 209, 198))],
                    ['right', ((548, 723), (217, 209, 198))],
                    ['right', ((582, 725), (217, 209, 198))],
                    ['right', ((639, 726), (173, 166, 158))],
                    ['right', ((683, 729), (217, 209, 198))],
                    ['right', ((738, 730), (217, 209, 198))],
                    ['right', ((773, 727), (217, 209, 198))],
                    ['right', ((831, 725), (216, 208, 197))],
                    ['right', ((862, 723), (115, 109, 102))],
                    ['right', ((889, 747), (215, 202, 193))],
                    ['right', ((842, 747), (204, 194, 184))],
                    ['right', ((782, 747), (100, 60, 16))],
                    ['right', ((716, 750), (216, 208, 197))],
                    ['right', ((672, 745), (217, 209, 198))],
                    ['right', ((604, 743), (217, 209, 198))],
                    ['right', ((551, 741), (217, 209, 198))],
                    ['right', ((518, 744), (174, 158, 144))],
                    ['right', ((881, 747), (183, 170, 161))],
                    ['right', ((944, 745), (201, 160, 130))],
                    ['right', ((987, 747), (221, 177, 149))],
                    ['right', ((978, 702), (204, 188, 174))],
                    ['right', ((1041, 751), (223, 183, 157))],
                    ['right', ((1077, 715), (222, 182, 157))],
                    ['right', ((1076, 740), (221, 184, 158))],
                    ['right', ((1166, 741), (220, 181, 150))],
                    # нижнее четверокрестие

                    ['right', ((538, 44), (88, 47, 33))],
                    ['right', ((582, 44), (89, 63, 43))],
                    ['right', ((621, 45), (133, 102, 90))],
                    ['right', ((623, 81), (179, 140, 111))],
                    ['right', ((596, 72), (163, 137, 134))],
                    ['right', ((534, 70), (112, 79, 53))],
                    ['right', ((482, 74), (129, 63, 41))],
                    ['right', ((436, 71), (82, 85, 74))],
                    ['right', ((412, 96), (67, 41, 35))],
                    ['right', ((459, 94), (117, 92, 84))],
                    ['right', ((518, 97), (175, 146, 134))],
                    ['right', ((578, 97), (208, 182, 178))],
                    ['right', ((616, 98), (211, 172, 145))],
                    ['right', ((615, 131), (222, 185, 166))],
                    ['right', ((574, 130), (229, 204, 200))],
                    ['right', ((530, 126), (220, 183, 156))],
                    ['right', ((465, 130), (224, 182, 157))],
                    ['right', ((428, 130), (215, 170, 139))],
                    ['right', ((395, 132), (186, 153, 136))],
                    ['right', ((368, 149), (227, 192, 168))],
                    ['right', ((406, 150), (230, 207, 194))],
                    ['right', ((460, 153), (231, 203, 196))],
                    ['right', ((516, 152), (229, 205, 199))],
                    ['right', ((566, 154), (221, 181, 161))],
                    ['right', ((610, 157), (212, 18, 141))],
                    ['right', ((649, 190), (168, 167, 165))],
                    ['right', ((572, 184), (221, 183, 160))],
                    ['right', ((519, 185), (222, 184, 161))],
                    ['right', ((480, 187), (223, 182, 152))],
                    ['right', ((398, 185), (237, 209, 201))],
                    ['right', ((362, 190), (228, 189, 160))],
                    ['right', ((361, 236), (226, 183, 150))],
                    ['right', ((400, 233), (229, 187, 163))],
                    ['right', ((449, 235), (238, 203, 197))],
                    ['right', ((573, 229), (218, 181, 149))],
                    ['right', ((638, 227), (222, 185, 166))],
                    ['right', ((691, 231), (215, 179, 153))],
                    ['right', ((738, 231), (212, 121, 140))],
                    ['right', ((797, 258), (167, 167, 169))],
                    ['right', ((734, 263), (215, 183, 162))],
                    ['right', ((665, 263), (221, 184, 166))],
                    ['right', ((482, 266), (223, 182, 151))],
                    ['right', ((404, 274), (224, 183, 153))],
                    ['right', ((365, 274), (227, 188, 158))],
                    ['right', ((401, 295), (223, 182, 150))],
                    ['right', ((441, 296), (224, 179, 148))],
                    ['right', ((497, 296), (224, 183, 155))],
                    ['right', ((551, 294), (221, 181, 155))],
                    ['right', ((682, 293), (218, 182, 158))],
                    ['right', ((747, 293), (224, 203, 202))],
                    ['right', ((804, 291), (222, 202, 203))],
                    ['right', ((853, 290), (107, 172, 70))],
                    ['right', ((868, 334), (211, 177, 150))],
                    ['right', ((817, 339), (211, 181, 156))],
                    ['right', ((774, 339), (215, 181, 156))],
                    ['right', ((637, 342), (218, 181, 154))],
                    ['right', ((568, 350), (222, 182, 157))],
                    ['right', ((523, 347), (218, 180, 152))],
                    ['right', ((649, 319), (226, 199, 191))],
                    ['right', ((544, 392), (227, 185, 163))],
                    ['right', ((600, 391), (223, 183, 160))],
                    ['right', ((667, 390), (221, 183, 161))],
                    ['right', ((729, 390), (217, 182, 161))],
                    ['right', ((794, 385), (218, 185, 166))],
                    ['right', ((855, 379), (215, 184, 164))],
                    ['right', ((901, 378), (212, 180, 159))],
                    ['right', ((972, 379), (213, 184, 168))],
                    ['right', ((948, 362), (210, 179, 158))],
                    ['right', ((1048, 386), (173, 175, 176))],
                    ['right', ((1098, 410), (255, 214, 51))],
                    ['right', ((1026, 417), (213, 184, 168))],
                    ['right', ((958, 418), (215, 186, 170))],
                    ['right', ((881, 413), (215, 184, 164))],
                    ['right', ((800, 408), (215, 183, 161))],
                    ['right', ((747, 411), (217, 180, 154))],
                    ['right', ((667, 404), (220, 182, 159))],
                    ['right', ((604, 397), (220, 181, 150))],
                    ['right', ((609, 429), (221, 178, 150))],
                    ['right', ((694, 430), (219, 179, 152))],
                    ['right', ((751, 429), (227, 206, 203))],
                    ['right', ((821, 428), (224, 204, 199))],
                    ['right', ((880, 425), (218, 186, 168))],
                    ['right', ((980, 421), (211, 181, 155))],
                    ['right', ((1041, 419), (212, 183, 165))],
                    ['right', ((1096, 424), (141, 140, 138))],
                    ['right', ((1166, 442), (146, 146, 143))],
                    ['right', ((1107, 449), (222, 203, 206))],
                    ['right', ((1050, 451), (215, 189, 180))],
                    ['right', ((854, 457), (218, 185, 168))],
                    ['right', ((791, 461), (219, 184, 164))],
                    ['right', ((735, 460), (219, 183, 157))],
                    ['right', ((687, 464), (222, 188, 172))],
                    ['right', ((684, 510), (218, 177, 159))],
                    ['right', ((754, 500), (224, 183, 156))],
                    ['right', ((826, 501), (216, 179, 152))],
                    ['right', ((886, 505), (217, 181, 157))],
                    ['right', ((963, 507), (213, 181, 158))],
                    ['right', ((1046, 507), (221, 203, 203))],
                    ['right', ((1088, 506), (214, 181, 162))],
                    ['right', ((1147, 509), (220, 192, 185))],
                    ['right', ((1186, 549), (218, 187, 169))],
                    ['right', ((1127, 548), (216, 183, 162))],
                    ['right', ((1076, 550), (217, 183, 163))],
                    ['right', ((1023, 548), (95, 95, 95))],
                    ['right', ((954, 551), (220, 190, 175))],
                    ['right', ((883, 549), (217, 181, 157))],
                    ['right', ((823, 552), (220, 183, 157))],
                    ['right', ((858, 591), (220, 184, 162))],
                    ['right', ((910, 587), (220, 185, 165))],
                    ['right', ((956, 587), (219, 184, 162))],
                    ['right', ((1025, 589), (80, 80, 80))],
                    ['right', ((1081, 589), (219, 184, 164))],
                    ['right', ((1151, 591), (218, 183, 163))],
                    ['right', ((1191, 591), (218, 185, 168))],
                    ['right', ((1181, 628), (217, 180, 154))],
                    ['right', ((1117, 625), (225, 197, 187))],
                    ['right', ((1038, 625), (221, 184, 165))],
                    ['right', ((998, 620), (226, 205, 204))],
                    ['right', ((935, 617), (222, 185, 159))],
                    ['right', ((884, 615), (224, 197, 191))],
                    ['right', ((930, 645), (210, 173, 147))],
                    ['right', ((994, 646), (222, 186, 164))],
                    ['right', ((1049, 648), (226, 200, 195))],
                    ['right', ((1113, 647), (217, 180, 155))],
                    ['right', ((1155, 645), (219, 183, 161))],
                    ['right', ((1083, 662), (220, 180, 154))],
                    ['right', ((1078, 685), (219, 179, 153))],
                    ['right', ((985, 685), (223, 194, 187))],
                    ['right', ((1223, 409), (75, 70, 66))],
                    ['right', ((1246, 403), (79, 74, 70))],
                    ['right', ((1271, 396), (79, 74, 70))],
                    ['right', ((1266, 449), (94, 92, 94))],
                    # остальная площадь улицы
                ]
            elif location == 'sqwer':
                points = [
                    ['right', ((246, 464), (115, 79, 57))],
                    ['right', ((290, 451), (121, 77, 55))],
                    ['right', ((296, 585), (142, 129, 62))],
                    ['right', ((334, 569), (99, 137, 54))],
                    ['right', ((709, 374), (116, 144, 56))],
                    ['right', ((751, 390), (78, 50, 37))],
                    ['right', ((950, 576), (168, 142, 78))],
                    ['right', ((995, 583), (180, 117, 82))],
                    ['right', ((899, 231), (81, 59, 49))],
                    ['right', ((930, 240), (230, 98, 161))],
                    ['right', ((743, 168), (116, 57, 26))],
                    ['right', ((544, 169), (189, 182, 173))],
                    ['right', ((522, 177), (164, 140, 111))],
                    ['right', ((134, 388), (112, 74, 42))],
                    # лавки

                    ['right', ((359, 49), (52, 52, 52))],
                    ['right', ((415, 50), (38, 38, 38))],
                    ['right', ((464, 49), (187, 142, 122))],
                    ['right', ((518, 45), (181, 145, 114))],
                    ['right', ((583, 48), (173, 137, 117))],
                    ['right', ((742, 50), (184, 141, 106))],
                    ['right', ((804, 51), (177, 127, 104))],
                    ['right', ((881, 49), (193, 159, 122))],
                    ['right', ((865, 82), (193, 147, 126))],
                    ['right', ((801, 79), (204, 174, 136))],
                    ['right', ((903, 81), (185, 148, 126))],
                    ['right', ((747, 77), (209, 174, 137))],
                    ['right', ((645, 100), (176, 126, 96))],
                    ['right', ((550, 86), (181, 134, 93))],
                    ['right', ((483, 100), (180, 133, 104))],
                    ['right', ((412, 99), (46, 46, 46))],
                    ['right', ((355, 104), (38, 38, 38))],
                    ['right', ((335, 136), (49, 49, 49))],
                    ['right', ((390, 131), (37, 37, 37))],
                    ['right', ((453, 133), (171, 122, 84))],
                    ['right', ((866, 153), (163, 113, 78))],
                    ['right', ((929, 161), (103, 130, 58))],
                    ['right', ((1025, 206), (205, 175, 137))],
                    ['right', ((989, 209), (189, 142, 121))],
                    ['right', ((209, 245), (42, 42, 42))],
                    ['right', ((154, 242), (58, 58, 58))],
                    ['right', ((102, 246), (33, 33, 33))],
                    ['right', ((105, 281), (49, 49, 49))],
                    ['right', ((137, 282), (61, 61, 61))],
                    ['right', ((210, 281), (208, 174, 137))],
                    ['right', ((285, 284), (207, 173, 136))],
                    ['right', ((337, 283), (229, 204, 172))],
                    ['right', ((437, 285), (229, 136, 170))],
                    ['right', ((505, 288), (91, 129, 43))],
                    ['right', ((559, 290), (115, 149, 57))],
                    ['right', ((602, 290), (91, 124, 51))],
                    ['right', ((705, 298), (97, 131, 47))],
                    ['right', ((746, 301), (101, 136, 37))],
                    ['right', ((794, 304), (100, 141, 52))],
                    ['right', ((840, 305), (112, 148, 35))],
                    ['right', ((888, 311), (84, 101, 42))],
                    ['right', ((941, 310), (80, 99, 32))],
                    ['right', ((975, 310), (116, 85, 109))],
                    ['right', ((1026, 310), (34, 34, 33))],
                    ['right', ((1074, 308), (214, 181, 145))],
                    ['right', ((1130, 304), (194, 152, 121))],
                    ['right', ((1188, 306), (20, 61, 0))],
                    ['right', ((1189, 353), (17, 57, 0))],
                    ['right', ((1134, 360), (138, 80, 43))],
                    ['right', ((1055, 358), (212, 173, 139))],
                    ['right', ((979, 358), (112, 71, 39))],
                    ['right', ((926, 354), (114, 148, 54))],
                    ['right', ((860, 348), (135, 152, 58))],
                    ['right', ((802, 346), (130, 225, 62))],
                    ['right', ((759, 343), (101, 138, 51))],
                    ['right', ((699, 341), (135, 157, 57))],
                    ['right', ((655, 344), (51, 51, 51))],
                    ['right', ((583, 339), (92, 122, 44))],
                    ['right', ((547, 340), (114, 145, 52))],
                    ['right', ((489, 339), (136, 223, 28))],
                    ['right', ((432, 340), (90, 128, 55))],
                    ['right', ((387, 340), (109, 150, 55))],
                    ['right', ((338, 341), (140, 166, 59))],
                    ['right', ((306, 341), (84, 112, 36))],
                    ['right', ((269, 341), (194, 152, 132))],
                    ['right', ((217, 343), (210, 175, 137))],
                    ['right', ((195, 367), (170, 186, 146))],
                    ['right', ((265, 373), (208, 174, 137))],
                    ['right', ((310, 371), (214, 188, 145))],
                    ['right', ((366, 373), (61, 89, 29))],
                    ['right', ((414, 375), (71, 111, 36))],
                    ['right', ((455, 375), (84, 151, 21))],
                    ['right', ((515, 375), (217, 97, 166))],
                    ['right', ((603, 372), (89, 123, 49))],
                    ['right', ((658, 374), (149, 173, 61))],
                    ['right', ((712, 377), (168, 117, 78))],
                    ['right', ((764, 377), (114, 154, 66))],
                    ['right', ((812, 378), (143, 181, 80))],
                    ['right', ((849, 378), (52, 120, 33))],
                    ['right', ((913, 381), (117, 142, 50))],
                    ['right', ((960, 378), (193, 153, 134))],
                    ['right', ((1036, 382), (207, 176, 138))],
                    ['right', ((1093, 384), (88, 67, 53))],
                    ['right', ((1143, 380), (62, 62, 62))],
                    ['right', ((1183, 409), (168, 97, 55))],
                    ['right', ((1131, 417), (142, 183, 68))],
                    ['right', ((1060, 419), (84, 84, 86))],
                    ['right', ((1014, 421), (169, 100, 55))],
                    ['right', ((959, 420), (209, 174, 137))],
                    ['right', ((901, 420), (197, 150, 128))],
                    ['right', ((834, 415), (39, 111, 57))],
                    ['right', ((781, 415), (209, 125, 199))],
                    ['right', ((728, 417), (180, 116, 82))],
                    ['right', ((689, 415), (117, 155, 51))],
                    ['right', ((631, 412), (105, 143, 49))],
                    ['right', ((575, 408), (172, 125, 71))],
                    ['right', ((507, 408), (51, 98, 21))],
                    ['right', ((459, 406), (169, 184, 156))],
                    ['right', ((415, 404), (161, 94, 50))],
                    ['right', ((326, 403), (210, 181, 140))],
                    ['right', ((230, 404), (99, 130, 36))],
                    ['right', ((202, 402), (159, 183, 71))],
                    ['right', ((185, 435), (127, 154, 77))],
                    ['right', ((211, 437), (112, 153, 61))],
                    ['right', ((261, 438), (141, 162, 43))],
                    ['right', ((314, 439), (96, 77, 30))],
                    ['right', ((338, 442), (170, 139, 110))],
                    ['right', ((393, 445), (178, 127, 100))],
                    ['right', ((437, 445), (214, 190, 156))],
                    ['right', ((469, 446), (208, 175, 137))],
                    ['right', ((520, 449), (134, 201, 90))],
                    ['right', ((569, 450), (91, 123, 45))],
                    ['right', ((611, 451), (214, 180, 143))],
                    ['right', ((654, 455), (70, 70, 70))],
                    ['right', ((721, 455), (144, 81, 45))],
                    ['right', ((797, 458), (213, 212, 210))],
                    ['right', ((884, 463), (187, 155, 131))],
                    ['right', ((967, 464), (128, 157, 50))],
                    ['right', ((1044, 459), (108, 147, 53))],
                    ['right', ((1101, 449), (109, 149, 54))],
                    ['right', ((1176, 455), (66, 65, 66))],
                    ['right', ((1240, 461), (96, 113, 49))],
                    ['right', ((1189, 536), (125, 145, 48))],
                    ['right', ((1139, 533), (89, 124, 38))],
                    ['right', ((1085, 532), (50, 49, 50))],
                    ['right', ((1035, 530), (51, 71, 20))],
                    ['right', ((974, 528), (119, 156, 65))],
                    ['right', ((923, 527), (94, 131, 48))],
                    ['right', ((863, 523), (167, 96, 53))],
                    ['right', ((808, 518), (51, 97, 16))],
                    ['right', ((726, 515), (199, 152, 115))],
                    ['right', ((659, 513), (156, 156, 156))],
                    ['right', ((600, 510), (198, 165, 132))],
                    ['right', ((527, 506), (208, 174, 137))],
                    ['right', ((460, 504), (82, 158, 10))],
                    ['right', ((417, 505), (146, 144, 100))],
                    ['right', ((381, 510), (109, 146, 51))],
                    ['right', ((342, 510), (101, 127, 38))],
                    ['right', ((284, 510), (92, 110, 37))],
                    ['right', ((215, 510), (220, 165, 215))],
                    ['right', ((146, 511), (252, 87, 110))],
                    ['right', ((156, 551), (120, 151, 50))],
                    ['right', ((199, 555), (137, 164, 54))],
                    ['right', ((255, 555), (139, 162, 42))],
                    ['right', ((321, 556), (111, 150, 55))],
                    ['right', ((378, 552), (78, 107, 24))],
                    ['right', ((427, 554), (176, 128, 101))],
                    ['right', ((490, 558), (190, 144, 123))],
                    ['right', ((539, 557), (160, 160, 160))],
                    ['right', ((598, 555), (210, 210, 210))],
                    ['right', ((665, 558), (168, 162, 157))],
                    ['right', ((721, 559), (202, 202, 202))],
                    ['right', ((771, 558), (106, 134, 95))],
                    ['right', ((823, 560), (214, 184, 149))],
                    ['right', ((906, 557), (181, 152, 120))],
                    ['right', ((996, 555), (133, 168, 65))],
                    ['right', ((1094, 558), (110, 155, 44))],
                    ['right', ((1185, 556), (127, 148, 47))],
                    ['right', ((1190, 599), (90, 128, 43))],
                    ['right', ((1144, 603), (119, 149, 59))],
                    ['right', ((1076, 606), (96, 136, 52))],
                    ['right', ((1006, 607), (157, 154, 171))],
                    ['right', ((848, 609), (57, 122, 51))],
                    ['right', ((765, 611), (215, 189, 152))],
                    ['right', ((685, 615), (146, 107, 84))],
                    ['right', ((636, 615), (172, 172, 172))],
                    ['right', ((565, 614), (210, 177, 141))],
                    ['right', ((472, 613), (191, 160, 129))],
                    ['right', ((391, 606), (112, 153, 54))],
                    ['right', ((215, 599), (124, 154, 59))],
                    ['right', ((163, 597), (119, 148, 40))],
                    ['right', ((134, 623), (107, 147, 53))],
                    ['right', ((218, 625), (120, 156, 60))],
                    ['right', ((340, 637), (69, 96, 32))],
                    ['right', ((429, 641), (202, 170, 134))],
                    ['right', ((496, 643), (209, 174, 137))],
                    ['right', ((571, 645), (193, 144, 116))],
                    ['right', ((618, 648), (208, 174, 138))],
                    ['right', ((669, 646), (196, 156, 132))],
                    ['right', ((777, 645), (51, 122, 55))],
                    ['right', ((869, 645), (212, 178, 141))],
                    ['right', ((957, 647), (111, 155, 55))],
                    ['right', ((1056, 649), (103, 145, 43))],
                    ['right', ((1127, 652), (239, 122, 153))],
                    ['right', ((1179, 653), (56, 173, 207))],
                    ['right', ((1166, 690), (101, 133, 50))],
                    ['right', ((1075, 694), (132, 142, 39))],
                    ['right', ((968, 697), (178, 126, 96))],
                    ['right', ((895, 695), (204, 174, 135))],
                    ['right', ((842, 693), (179, 150, 118))],
                    ['right', ((746, 692), (104, 139, 48))],
                    ['right', ((682, 690), (154, 91, 49))],
                    ['right', ((616, 684), (155, 127, 106))],
                    ['right', ((561, 679), (92, 129, 51))],
                    ['right', ((497, 675), (137, 90, 73))],
                    ['right', ((406, 673), (208, 174, 137))],
                    ['right', ((311, 668), (117, 155, 73))],
                    ['right', ((222, 664), (103, 143, 58))],
                    ['right', ((106, 669), (130, 172, 59))],
                    ['right', ((103, 711), (82, 114, 42))],
                    ['right', ((224, 709), (133, 137, 112))],
                    ['right', ((311, 712), (191, 154, 131))],
                    ['right', ((405, 720), (152, 121, 101))],
                    ['right', ((510, 722), (102, 139, 49))],
                    ['right', ((572, 725), (165, 209, 91))],
                    ['right', ((602, 723), (67, 125, 10))],
                    ['right', ((641, 723), (44, 92, 9))],
                    ['right', ((687, 726), (71, 107, 38))],
                    ['right', ((738, 725), (70, 106, 14))],
                    ['right', ((780, 718), (128, 173, 54))],
                    ['right', ((822, 715), (70, 103, 37))],
                    ['right', ((872, 713), (147, 84, 45))],
                    ['right', ((920, 714), (206, 182, 147))],
                    ['right', ((995, 719), (199, 70, 234))],
                    ['right', ((1075, 720), (114, 141, 42))],
                    ['right', ((1171, 726), (113, 147, 56))],
                    # остальное
                ]

            elif location == 'school':
                points = [
                    ['right', ((704, 163), (196, 162, 121))],
                    ['right', ((762, 197), (190, 159, 119))],
                    ['right', ((944, 284), (192, 160, 119))],
                    ['right', ((1008, 319), (191, 160, 119))],
                    ['right', ((800, 422), (192, 161, 120))],
                    ['right', ((738, 390), (190, 160, 119))],
                    ['right', ((558, 301), (191, 161, 119))],
                    ['right', ((495, 268), (191, 160, 119))],
                    ['right', ((472, 299), (184, 152, 111))],
                    ['right', ((720, 425), (183, 151, 111))],
                    ['right', ((921, 318), (182, 150, 110))],
                    ['right', ((679, 199), (185, 153, 112))],
                    ['right', ((1014, 490), (191, 161, 119))],
                    ['right', ((950, 522), (190, 159, 119))],
                    ['right', ((768, 650), (190, 160, 119))],
                    ['right', ((705, 682), (192, 161, 119))],
                    ['right', ((195, 441), (170, 143, 111))],
                    ['right', ((322, 211), (222, 194, 146))],
                    ['right', ((214, 271), (214, 189, 145))],
                    # лавки

                    ['right', ((482, 200), (90, 120, 128))],
                    ['right', ((511, 205), (92, 125, 132))],
                    ['right', ((535, 206), (94, 124, 132))],
                    ['right', ((802, 222), (94, 125, 130))],
                    ['right', ((863, 230), (94, 125, 130))],
                    ['right', ((800, 267), (95, 125, 133))],
                    ['right', ((831, 267), (98, 129, 134))],
                    ['right', ((889, 270), (95, 126, 131))],
                    ['right', ((1025, 302), (91, 120, 126))],
                    ['right', ((1076, 305), (96, 125, 131))],
                    ['right', ((1131, 320), (97, 126, 132))],
                    ['right', ((1053, 353), (95, 124, 130))],
                    ['right', ((1101, 348), (97, 126, 132))],
                    ['right', ((1133, 351), (97, 126, 132))],
                    ['right', ((1039, 416), (94, 124, 129))],
                    ['right', ((1099, 412), (96, 125, 131))],
                    ['right', ((1165, 467), (95, 124, 130))],
                    ['right', ((1010, 448), (94, 125, 130))],
                    ['right', ((791, 299), (94, 124, 132))],
                    ['right', ((432, 243), (89, 122, 129))],
                    ['right', ((361, 275), (87, 117, 127))],
                    ['right', ((315, 307), (91, 124, 133))],
                    ['right', ((264, 333), (92, 125, 134))],
                    ['right', ((203, 360), (90, 123, 132))],
                    ['right', ((133, 388), (90, 123, 132))],
                    ['right', ((75, 419), (88, 121, 130))],
                    ['right', ((125, 434), (98, 131, 140))],
                    ['right', ((216, 388), (94, 127, 136))],
                    ['right', ((290, 357), (94, 127, 136))],
                    ['right', ((340, 384), (92, 125, 134))],
                    ['right', ((256, 413), (95, 128, 137))],
                    ['right', ((370, 411), (90, 123, 132))],
                    ['right', ((412, 418), (95, 128, 137))],
                    ['right', ((456, 442), (93, 126, 135))],
                    ['right', ((492, 464), (94, 127, 136))],
                    ['right', ((559, 442), (93, 126, 133))],
                    ['right', ((562, 504), (94, 127, 134))],
                    ['right', ((642, 534), (93, 126, 133))],
                    ['right', ((719, 582), (93, 126, 133))],
                    ['right', ((755, 623), (93, 126, 133))],
                    ['right', ((679, 609), (93, 126, 133))],
                    ['right', ((597, 584), (91, 124, 131))],
                    ['right', ((550, 559), (92, 125, 134))],
                    ['right', ((480, 532), (93, 126, 135))],
                    ['right', ((429, 507), (93, 126, 135))],
                    ['right', ((387, 509), (86, 56, 32))],
                    ['right', ((331, 516), (57, 37, 20))],
                    ['right', ((288, 533), (84, 55, 28))],
                    ['right', ((249, 543), (70, 47, 27))],
                    ['right', ((213, 553), (57, 37, 21))],
                    ['right', ((183, 564), (43, 29, 17))],
                    ['right', ((158, 618), (94, 128, 137))],
                    ['right', ((219, 623), (93, 126, 135))],
                    ['right', ((270, 621), (94, 127, 136))],
                    ['right', ((328, 624), (92, 125, 134))],
                    ['right', ((392, 625), (91, 124, 133))],
                    ['right', ((480, 631), (94, 127, 136))],
                    ['right', ((561, 636), (95, 128, 137))],
                    ['right', ((612, 636), (92, 125, 132))],
                    ['right', ((676, 634), (95, 128, 135))],
                    ['right', ((722, 628), (94, 127, 134))],
                    ['right', ((639, 648), (93, 126, 133))],
                    ['right', ((561, 651), (95, 128, 137))],
                    ['right', ((476, 645), (93, 126, 135))],
                    ['right', ((407, 641), (93, 126, 135))],
                    ['right', ((327, 634), (93, 126, 135))],
                    ['right', ((249, 638), (96, 129, 138))],
                    ['right', ((220, 676), (96, 129, 138))],
                    ['right', ((222, 711), (92, 126, 135))],
                    ['right', ((307, 714), (92, 125, 134))],
                    ['right', ((408, 717), (93, 126, 135))],
                    ['right', ((495, 722), (94, 127, 136))],
                    ['right', ((541, 723), (92, 125, 134))],
                    ['right', ((606, 723), (95, 128, 137))],
                    ['right', ((652, 726), (94, 127, 136))],
                    ['right', ((955, 743), (93, 123, 131))],
                    ['right', ((979, 710), (95, 125, 132))],
                    ['right', ((1075, 728), (96, 126, 134))],
                    ['right', ((1081, 646), (96, 127, 132))],
                    ['right', ((1127, 630), (93, 124, 129))],
                    ['right', ((1189, 612), (94, 123, 129))],
                    ['right', ((1157, 663), (95, 126, 131))],
                    ['right', ((1165, 724), (97, 128, 133))],
                    ['right', ((1181, 521), (94, 123, 129))],
                    ['right', ((1146, 493), (94, 123, 129))],
                    ['right', ((1066, 433), (93, 122, 128))],
                    ['right', ((1092, 379), (96, 125, 131))],
                    ['right', ((1161, 432), (94, 123, 129))],
                    ['right', ((1194, 461), (100, 129, 135))],
                    ['right', ((1279, 544), (79, 108, 114))],
                    # остальное
                ]
            elif location == 'park':
                points = [
                    ['right', ((579, 193), (187, 169, 139))],
                    ['right', ((613, 212), (217, 206, 182))],
                    ['right', ((691, 248), (202, 188, 162))],
                    ['right', ((720, 264), (196, 180, 153))],
                    ['right', ((536, 520), (207, 194, 170))],
                    ['right', ((571, 505), (96, 79, 56))],
                    ['right', ((1021, 615), (86, 116, 45))],
                    ['right', ((1113, 617), (105, 133, 53))],
                    ['right', ((1075, 353), (111, 134, 45))],
                    ['right', ((1162, 356), (88, 114, 47))],
                    ['right', ((696, 168), (103, 133, 44))],
                    ['right', ((792, 177), (243, 226, 228))],
                    ['right', ((931, 110), (76, 54, 34))],
                    ['right', ((1018, 149), (78, 53, 34))],
                    ['right', ((961, 59), (88, 116, 39))],
                    ['right', ((106, 192), (81, 153, 193))],
                    ['right', ((136, 189), (91, 168, 207))],
                    ['right', ((186, 189), (84, 157, 204))],
                    ['right', ((233, 183), (74, 152, 194))],
                    ['right', ((227, 229), (89, 161, 201))],
                    ['right', ((172, 222), (92, 164, 204))],
                    ['right', ((154, 284), (108, 136, 151))],
                    ['right', ((195, 261), (95, 157, 198))],
                    ['right', ((189, 422), (61, 160, 199))],
                    ['right', ((226, 385), (41, 152, 197))],
                    ['right', ((287, 367), (45, 167, 208))],
                    ['right', ((320, 364), (60, 141, 168))],
                    ['right', ((302, 400), (35, 144, 185))],
                    ['right', ((244, 424), (39, 167, 202))],
                    ['right', ((239, 468), (69, 161, 200))],
                    ['right', ((301, 454), (56, 154, 199))],
                    ['right', ((367, 458), (37, 164, 199))],
                    ['right', ((450, 456), (50, 156, 198))],
                    ['right', ((501, 437), (57, 151, 197))],
                    ['right', ((421, 430), (39, 159, 196))],
                    ['right', ((383, 432), (34, 135, 179))],
                    ['right', ((320, 425), (30, 137, 179))],
                    ['right', ((266, 422), (34, 143, 184))],
                    ['right', ((382, 373), (73, 164, 189))],
                    ['right', ((429, 362), (62, 147, 168))],
                    ['right', ((478, 351), (59, 160, 184))],
                    ['right', ((537, 341), (98, 138, 148))],
                    ['right', ((458, 377), (47, 158, 204))],
                    ['right', ((374, 413), (35, 136, 180))],
                    ['right', ((457, 411), (38, 149, 194))],
                    ['right', ((484, 392), (48, 155, 195))],
                    ['right', ((539, 413), (70, 159, 202))],
                    ['right', ((333, 168), (100, 117, 46))],
                    ['right', ((373, 159), (120, 140, 55))],
                    ['right', ((434, 118), (80, 112, 38))],
                    ['right', ((486, 146), (163, 145, 129))],
                    ['right', ((549, 164), (157, 127, 105))],
                    ['right', ((498, 177), (138, 130, 61))],
                    ['right', ((430, 180), (170, 153, 138))],
                    ['right', ((315, 216), (75, 104, 38))],
                    ['right', ((372, 218), (184, 163, 142))],
                    ['right', ((435, 218), (126, 115, 61))],
                    ['right', ((471, 217), (102, 125, 38))],
                    ['right', ((531, 218), (83, 116, 26))],
                    ['right', ((559, 256), (163, 152, 122))],
                    ['right', ((483, 262), (201, 176, 164))],
                    ['right', ((428, 265), (129, 124, 87))],
                    ['right', ((316, 272), (152, 133, 124))],
                    ['right', ((353, 308), (184, 163, 145))],
                    ['right', ((416, 291), (181, 168, 153))],
                    ['right', ((477, 291), (188, 166, 145))],
                    ['right', ((553, 286), (137, 115, 100))],
                    ['right', ((612, 279), (120, 101, 82))],
                    ['right', ((591, 306), (193, 173, 162))],
                    ['right', ((647, 306), (119, 105, 76))],
                    ['right', ((687, 312), (111, 117, 60))],
                    ['right', ((628, 345), (148, 145, 108))],
                    ['right', ((686, 343), (167, 161, 135))],
                    ['right', ((734, 345), (75, 100, 9))],
                    ['right', ((632, 388), (121, 133, 79))],
                    ['right', ((657, 386), (83, 103, 50))],
                    ['right', ((699, 386), (120, 192, 17))],
                    ['right', ((757, 381), (110, 130, 37))],
                    ['right', ((804, 382), (145, 133, 81))],
                    ['right', ((864, 388), (93, 72, 55))],
                    ['right', ((736, 343), (111, 136, 43))],
                    ['right', ((803, 345), (56, 45, 49))],
                    ['right', ((876, 352), (120, 117, 83))],
                    ['right', ((943, 353), (106, 86, 75))],
                    ['right', ((1036, 359), (102, 126, 42))],
                    ['right', ((1226, 367), (88, 116, 37))],
                    ['right', ((1212, 309), (21, 37, 14))],
                    ['right', ((1148, 312), (99, 128, 60))],
                    ['right', ((1046, 309), (126, 136, 55))],
                    ['right', ((976, 310), (149, 123, 103))],
                    ['right', ((883, 308), (64, 94, 23))],
                    ['right', ((777, 297), (168, 189, 85))],
                    ['right', ((756, 251), (104, 127, 40))],
                    ['right', ((837, 249), (104, 127, 45))],
                    ['right', ((915, 256), (131, 146, 83))],
                    ['right', ((1007, 257), (160, 142, 111))],
                    ['right', ((1113, 254), (79, 94, 41))],
                    ['right', ((1173, 258), (95, 112, 41))],
                    ['right', ((1219, 213), (91, 116, 33))],
                    ['right', ((1167, 240), (103, 120, 42))],
                    ['right', ((979, 255), (218, 192, 176))],
                    ['right', ((873, 222), (126, 148, 63))],
                    ['right', ((830, 220), (121, 140, 51))],
                    ['right', ((813, 135), (75, 109, 49))],
                    ['right', ((779, 116), (112, 141, 58))],
                    ['right', ((697, 133), (97, 122, 39))],
                    ['right', ((657, 172), (99, 121, 29))],
                    ['right', ((102, 399), (181, 152, 130))],
                    ['right', ((104, 445), (172, 151, 134))],
                    ['right', ((124, 488), (153, 133, 122))],
                    ['right', ((178, 510), (134, 124, 54))],
                    ['right', ((227, 513), (141, 161, 100))],
                    ['right', ((153, 576), (72, 39, 34))],
                    ['right', ((229, 570), (129, 112, 96))],
                    ['right', ((339, 561), (109, 141, 71))],
                    ['right', ((359, 534), (119, 133, 45))],
                    ['right', ((455, 508), (144, 121, 53))],
                    ['right', ((352, 574), (90, 121, 43))],
                    ['right', ((327, 628), (138, 114, 102))],
                    ['right', ((382, 658), (207, 186, 167))],
                    ['right', ((415, 619), (93, 70, 44))],
                    ['right', ((428, 574), (114, 129, 36))],
                    ['right', ((491, 542), (167, 160, 34))],
                    ['right', ((488, 493), (53, 97, 25))],
                    ['right', ((502, 600), (149, 114, 91))],
                    ['right', ((503, 652), (131, 110, 91))],
                    ['right', ((509, 710), (104, 129, 61))],
                    ['right', ((401, 743), (70, 108, 44))],
                    ['right', ((315, 745), (84, 115, 46))],
                    ['right', ((571, 671), (117, 127, 67))],
                    ['right', ((616, 670), (105, 114, 53))],
                    ['right', ((659, 705), (23, 63, 11))],
                    ['right', ((734, 726), (134, 155, 62))],
                    ['right', ((806, 730), (127, 172, 48))],
                    ['right', ((885, 727), (132, 109, 96))],
                    ['right', ((985, 732), (145, 115, 79))],
                    ['right', ((974, 685), (162, 137, 94))],
                    ['right', ((900, 701), (225, 207, 185))],
                    ['right', ((811, 699), (212, 193, 174))],
                    ['right', ((739, 698), (14, 87, 14))],
                    ['right', ((666, 684), (101, 118, 41))],
                    ['right', ((701, 644), (173, 149, 131))],
                    ['right', ((770, 648), (176, 150, 128))],
                    ['right', ((853, 658), (134, 129, 87))],
                    ['right', ((931, 667), (173, 194, 100))],
                    ['right', ((996, 682), (175, 135, 244))],
                    ['right', ((984, 616), (86, 118, 44))],
                    ['right', ((1034, 582), (124, 153, 65))],
                    ['right', ((993, 550), (99, 127, 43))],
                    ['right', ((965, 540), (94, 124, 51))],
                    ['right', ((1030, 517), (111, 130, 47))],
                    ['right', ((1067, 475), (116, 140, 62))],
                    ['right', ((1043, 421), (107, 132, 49))],
                    ['right', ((1015, 361), (87, 88, 95))],
                    ['right', ((943, 334), (189, 166, 154))],
                    ['right', ((890, 355), (175, 156, 139))],
                    ['right', ((850, 393), (219, 200, 184))],
                    ['right', ((976, 281), (173, 152, 135))],
                    ['right', ((968, 228), (117, 100, 80))],
                    ['right', ((974, 187), (108, 93, 83))],
                    ['right', ((1036, 544), (98, 118, 32))],
                    ['right', ((1088, 555), (100, 121, 30))],
                    ['right', ((1119, 611), (100, 129, 47))],
                    ['right', ((1156, 612), (98, 127, 48))],
                    ['right', ((1116, 499), (80, 110, 48))],
                    ['right', ((1053, 487), (104, 128, 43))],
                    ['right', ((1110, 446), (56, 92, 5))],
                    ['right', ((1144, 408), (108, 127, 39))],
                    ['right', ((1195, 406), (100, 100, 70))],
                    ['right', ((1266, 407), (71, 101, 39))],
                    ['right', ((1205, 445), (95, 118, 64))],
                    ['right', ((1194, 266), (94, 111, 40))],
                    ['right', ((1211, 213), (100, 125, 42))],
                ]
            elif location == 'bal':
                points = [
                    ['right', ((506, 114), (211, 171, 181))],
                    ['right', ((612, 115), (213, 171, 184))],
                    ['right', ((47, 395), (198, 155, 170))],
                    ['right', ((180, 386), (213, 174, 184))],
                    ['right', ((217, 403), (209, 171, 182))],
                    ['right', ((158, 611), (214, 168, 183))],
                    ['right', ((263, 611), (220, 182, 193))],
                    ['right', ((363, 656), (218, 185, 194))],
                    ['right', ((422, 655), (210, 167, 180))],
                    ['right', ((560, 652), (216, 183, 192))],
                    ['right', ((750, 609), (255, 228, 233))],
                    ['right', ((852, 607), (255, 228, 233))],
                    ['right', ((954, 658), (255, 228, 233))],
                    ['right', ((1054, 659), (253, 226, 231))],
                    ['right', ((1111, 534), (251, 224, 229))],
                    ['right', ((1201, 550), (255, 215, 223))],
                    ['right', ((1023, 296), (136, 91, 54))],
                    ['right', ((988, 267), (205, 108, 76))],
                    ['right', ((768, 172), (147, 100, 71))],
                    ['right', ((634, 668), (152, 100, 67))],
                    ['right', ((710, 671), (166, 117, 63))],
                    ['right', ((942, 556), (152, 106, 71))],
                    ['right', ((1029, 551), (165, 112, 64))],
                    ['right', ((490, 180), (152, 114, 63))],
                    ['right', ((450, 201), (151, 117, 79))],
                    ['right', ((410, 225), (173, 112, 57))],
                    ['right', ((364, 252), (161, 110, 60))],
                    ['right', ((317, 274), (136, 80, 28))],
                    ['right', ((247, 312), (176, 118, 87))],
                    ['right', ((238, 352), (141, 100, 50))],
                    ['right', ((267, 336), (158, 106, 64))],
                    ['right', ((318, 312), (84, 35, 11))],
                    ['right', ((391, 273), (128, 82, 60))],
                    ['right', ((460, 234), (180, 143, 114))],
                    ['right', ((533, 196), (158, 109, 78))],
                    ['right', ((700, 159), (120, 80, 55))],
                    ['right', ((640, 193), (143, 98, 61))],
                    ['right', ((586, 225), (146, 100, 64))],
                    ['right', ((520, 263), (174, 126, 80))],
                    ['right', ((470, 283), (166, 118, 72))],
                    ['right', ((395, 323), (102, 52, 22))],
                    ['right', ((342, 341), (170, 125, 78))],
                    ['right', ((282, 383), (165, 117, 71))],
                    ['right', ((299, 430), (167, 119, 73))],
                    ['right', ((368, 396), (163, 115, 69))],
                    ['right', ((457, 354), (163, 115, 68))],
                    ['right', ((524, 316), (161, 114, 68))],
                    ['right', ((581, 282), (174, 127, 82))],
                    ['right', ((655, 241), (153, 103, 60))],
                    ['right', ((709, 211), (143, 99, 65))],
                    ['right', ((774, 137), (90, 56, 31))],
                    ['right', ((719, 267), (111, 59, 26))],
                    ['right', ((666, 289), (106, 57, 27))],
                    ['right', ((627, 307), (150, 101, 58))],
                    ['right', ((567, 339), (173, 126, 82))],
                    ['right', ((499, 372), (171, 123, 75))],
                    ['right', ((457, 388), (218, 168, 129))],
                    ['right', ((419, 404), (178, 130, 84))],
                    ['right', ((383, 423), (209, 160, 116))],
                    ['right', ((330, 458), (205, 168, 126))],
                    ['right', ((289, 485), (182, 131, 86))],
                    ['right', ((289, 527), (157, 112, 62))],
                    ['right', ((353, 508), (110, 63, 25))],
                    ['right', ((388, 486), (165, 116, 73))],
                    ['right', ((438, 456), (162, 114, 68))],
                    ['right', ((508, 420), (124, 76, 40))],
                    ['right', ((566, 383), (170, 124, 78))],
                    ['right', ((625, 353), (173, 126, 80))],
                    ['right', ((685, 318), (154, 105, 62))],
                    ['right', ((849, 315), (168, 120, 74))],
                    ['right', ((788, 345), (208, 157, 114))],
                    ['right', ((732, 365), (145, 96, 55))],
                    ['right', ((629, 403), (96, 51, 20))],
                    ['right', ((551, 455), (228, 177, 137))],
                    ['right', ((491, 482), (153, 105, 59))],
                    ['right', ((419, 516), (144, 96, 58))],
                    ['right', ((333, 555), (180, 123, 75))],
                    ['right', ((341, 618), (176, 118, 80))],
                    ['right', ((389, 607), (158, 107, 73))],
                    ['right', ((430, 576), (170, 122, 76))],
                    ['right', ((471, 551), (92, 46, 20))],
                    ['right', ((570, 505), (205, 154, 109))],
                    ['right', ((630, 475), (169, 122, 78))],
                    ['right', ((689, 444), (175, 127, 81))],
                    ['right', ((740, 416), (207, 156, 111))],
                    ['right', ((786, 392), (171, 124, 78))],
                    ['right', ((845, 364), (104, 54, 26))],
                    ['right', ((895, 341), (169, 121, 75))],
                    ['right', ((957, 321), (154, 95, 49))],
                    ['right', ((951, 385), (187, 136, 81))],
                    ['right', ((893, 416), (125, 76, 36))],
                    ['right', ((828, 448), (145, 98, 54))],
                    ['right', ((765, 481), (172, 125, 79))],
                    ['right', ((691, 514), (164, 116, 70))],
                    ['right', ((622, 539), (149, 101, 57))],
                    ['right', ((535, 569), (101, 50, 21))],
                    ['right', ((455, 603), (154, 107, 60))],
                    ['right', ((372, 604), (128, 84, 42))],
                    ['right', ((313, 632), (184, 184, 184))],
                    ['right', ((572, 606), (183, 130, 88))],
                    ['right', ((617, 585), (168, 121, 77))],
                    ['right', ((689, 552), (105, 56, 25))],
                    ['right', ((742, 524), (160, 112, 66))],
                    ['right', ((791, 493), (157, 109, 63))],
                    ['right', ((844, 465), (92, 49, 23))],
                    ['right', ((914, 425), (177, 126, 81))],
                    ['right', ((967, 398), (169, 125, 95))],
                    ['right', ((1026, 363), (94, 40, 27))],
                    ['right', ((1080, 343), (114, 59, 20))],
                    ['right', ((1121, 360), (158, 117, 92))],
                    ['right', ((1065, 399), (164, 109, 78))],
                    ['right', ((1017, 423), (153, 97, 60))],
                    ['right', ((950, 453), (152, 105, 60))],
                    ['right', ((871, 495), (171, 123, 77))],
                    ['right', ((801, 535), (169, 122, 76))],
                    ['right', ((730, 570), (184, 133, 88))],
                    ['right', ((634, 610), (165, 118, 84))],
                    ['right', ((585, 638), (146, 103, 68))],
                    ['right', ((574, 728), (123, 66, 32))],
                    ['right', ((622, 702), (151, 105, 59))],
                    ['right', ((910, 566), (131, 92, 52))],
                    ['right', ((1059, 490), (163, 114, 71))],
                    ['right', ((1109, 460), (153, 103, 55))],
                    ['right', ((1151, 442), (124, 80, 35))],
                    ['right', ((1197, 426), (139, 91, 53))],
                    ['right', ((1108, 393), (147, 100, 57))],
                    ['right', ((1038, 515), (168, 116, 67))],
                    ['right', ((1059, 602), (181, 129, 72))],
                    ['right', ((1084, 640), (142, 98, 61))],
                    ['right', ((1139, 626), (164, 109, 52))],
                    ['right', ((904, 667), (188, 125, 77))],
                    ['right', ((953, 742), (185, 135, 108))],
                    ['right', ((558, 746), (177, 127, 91))],
                    ['right', ((773, 718), (148, 101, 57))],
                    ['right', ((1110, 293), (197, 192, 192))],
                    ['right', ((1266, 356), (241, 243, 243))],
                ]
            elif location == 'club':
                points = [
                    ['right', ((603, 116), (210, 210, 210))],
                    ['right', ((662, 113), (228, 228, 228))],
                    ['right', ((723, 147), (232, 232, 232))],
                    ['right', ((808, 186), (216, 215, 215))],
                    ['right', ((871, 220), (202, 202, 202))],
                    ['right', ((1154, 410), (197, 76, 184))],
                    ['right', ((1195, 529), (202, 73, 185))],
                    ['right', ((849, 595), (90, 20, 75))],
                    ['right', ((617, 598), (183, 53, 165))],
                    ['right', ((799, 482), (148, 43, 132))],
                    ['right', ((839, 458), (55, 16, 50))],
                    ['right', ((881, 436), (42, 5, 20))],
                    ['right', ((925, 413), (201, 75, 189))],
                    ['right', ((12, 335), (7, 1, 11))],
                    ['right', ((107, 286), (11, 1, 14))],
                    ['right', ((155, 258), (6, 1, 7))],
                    ['right', ((292, 192), (11, 2, 14))],
                    ['right', ((340, 171), (17, 3, 24))],
                    ['right', ((470, 166), (5, 23, 61))],
                    ['right', ((439, 190), (5, 23, 61))],
                    ['right', ((374, 234), (4, 21, 57))],
                    ['right', ((307, 265), (4, 22, 59))],
                    ['right', ((246, 303), (6, 29, 75))],
                    ['right', ((163, 346), (19, 32, 77))],
                    ['right', ((83, 392), (31, 41, 94))],
                    ['right', ((35, 424), (35, 44, 101))],
                    ['right', ((106, 415), (18, 51, 127))],
                    ['right', ((169, 377), (35, 52, 109))],
                    ['right', ((249, 336), (30, 44, 97))],
                    ['right', ((335, 297), (108, 103, 161))],
                    ['right', ((410, 258), (6, 32, 80))],
                    ['right', ((477, 214), (6, 32, 80))],
                    ['right', ((533, 186), (8, 33, 79))],
                    ['right', ((657, 216), (20, 49, 113))],
                    ['right', ((603, 244), (136, 66, 123))],
                    ['right', ((541, 271), (107, 65, 131))],
                    ['right', ((486, 307), (90, 76, 145))],
                    ['right', ((438, 336), (70, 84, 157))],
                    ['right', ((370, 368), (60, 89, 167))],
                    ['right', ((337, 394), (168, 167, 219))],
                    ['right', ((269, 438), (59, 127, 189))],
                    ['right', ((224, 474), (49, 190, 221))],
                    ['right', ((189, 501), (7, 101, 179))],
                    ['right', ((126, 539), (10, 53, 98))],
                    ['right', ((100, 575), (101, 84, 149))],
                    ['right', ((18, 634), (59, 11, 83))],
                    ['right', ((103, 631), (69, 45, 113))],
                    ['right', ((150, 601), (100, 83, 159))],
                    ['right', ((221, 561), (82, 92, 184))],
                    ['right', ((289, 521), (44, 142, 195))],
                    ['right', ((354, 474), (40, 137, 192))],
                    ['right', ((404, 444), (72, 125, 188))],
                    ['right', ((476, 413), (76, 137, 192))],
                    ['right', ((576, 357), (142, 118, 185))],
                    ['right', ((670, 318), (150, 139, 194))],
                    ['right', ((749, 274), (68, 80, 157))],
                    ['right', ((836, 288), (71, 74, 158))],
                    ['right', ((774, 326), (134, 130, 178))],
                    ['right', ((717, 364), (132, 123, 177))],
                    ['right', ((641, 399), (133, 132, 182))],
                    ['right', ((587, 422), (131, 145, 188))],
                    ['right', ((518, 459), (116, 125, 190))],
                    ['right', ((459, 491), (112, 126, 190))],
                    ['right', ((409, 523), (69, 111, 186))],
                    ['right', ((351, 561), (74, 96, 180))],
                    ['right', ((274, 610), (135, 134, 196))],
                    ['right', ((228, 641), (109, 93, 165))],
                    ['right', ((140, 616), (91, 66, 142))],
                    ['right', ((220, 698), (57, 45, 102))],
                    ['right', ((311, 705), (59, 59, 109))],
                    ['right', ((348, 651), (14, 38, 87))],
                    ['right', ((415, 615), (180, 166, 202))],
                    ['right', ((480, 580), (112, 91, 170))],
                    ['right', ((540, 542), (110, 120, 185))],
                    ['right', ((607, 495), (135, 130, 189))],
                    ['right', ((664, 459), (138, 106, 161))],
                    ['right', ((709, 432), (138, 106, 160))],
                    ['right', ((789, 382), (146, 113, 177))],
                    ['right', ((837, 355), (223, 184, 224))],
                    ['right', ((885, 331), (99, 81, 167))],
                    ['right', ((938, 299), (65, 58, 142))],
                    ['right', ((957, 343), (86, 67, 160))],
                    ['right', ((891, 383), (175, 138, 171))],
                    ['right', ((834, 399), (140, 83, 139))],
                    ['right', ((763, 436), (137, 80, 134))],
                    ['right', ((715, 463), (196, 171, 199))],
                    ['right', ((637, 504), (139, 106, 162))],
                    ['right', ((547, 550), (166, 159, 206))],
                    ['right', ((489, 605), (98, 91, 157))],
                    ['right', ((424, 641), (45, 66, 133))],
                    ['right', ((405, 685), (104, 69, 151))],
                    ['right', ((406, 725), (37, 34, 85))],
                    ['right', ((495, 684), (68, 61, 114))],
                    ['right', ((550, 659), (89, 77, 127))],
                    ['right', ((586, 694), (55, 49, 102))],
                    ['right', ((536, 711), (48, 44, 95))],
                    ['right', ((530, 738), (22, 23, 65))],
                    ['right', ((610, 740), (21, 22, 68))],
                    ['right', ((695, 744), (21, 17, 67))],
                    ['right', ((773, 748), (19, 12, 60))],
                    ['right', ((853, 741), (22, 10, 62))],
                    ['right', ((965, 738), (20, 8, 56))],
                    ['right', ((978, 701), (22, 11, 63))],
                    ['right', ((929, 703), (24, 13, 68))],
                    ['right', ((876, 706), (25, 14, 69))],
                    ['right', ((915, 661), (33, 20, 86))],
                    ['right', ((961, 649), (30, 13, 74))],
                    ['right', ((987, 605), (34, 19, 85))],
                    ['right', ((1043, 572), (37, 21, 89))],
                    ['right', ((1140, 653), (24, 9, 61))],
                    ['right', ((1025, 411), (80, 47, 144))],
                ]
            elif location == 'beauty':
                points = [
                    ['right', ((234, 319), (0, 76, 101))],
                    ['right', ((205, 573), (216, 216, 216))],
                    ['right', ((152, 600), (218, 218, 218))],
                    ['right', ((1082, 412), (0, 76, 103))],
                    ['right', ((1150, 449), (0, 79, 103))],
                    ['right', ((1195, 462), (0, 72, 95))],
                    ['right', ((754, 482), (28, 68, 79))],
                    ['right', ((474, 348), (55, 85, 91))],
                    ['right', ((561, 296), (120, 132, 134))],
                    ['right', ((826, 431), (120, 132, 134))],
                    ['right', ((371, 231), (171, 182, 183))],
                    ['right', ((401, 308), (48, 70, 75))],
                    ['right', ((341, 292), (45, 64, 68))],
                    ['right', ((386, 396), (65, 77, 80))],
                    ['right', ((305, 391), (73, 82, 85))],
                    ['right', ((340, 511), (77, 86, 89))],
                    ['right', ((408, 411), (59, 73, 77))],
                    ['right', ((455, 516), (69, 101, 106))],
                    ['right', ((221, 447), (59, 71, 73))],
                    ['right', ((267, 504), (71, 81, 84))],
                    ['right', ((133, 470), (49, 70, 75))],
                    ['right', ((567, 467), (79, 119, 123))],
                    ['right', ((621, 571), (73, 110, 114))],
                    ['right', ((491, 546), (72, 109, 113))],
                    ['right', ((564, 658), (68, 80, 83))],
                    ['right', ((568, 593), (74, 96, 99))],
                    ['right', ((682, 691), (72, 83, 86))],
                    ['right', ((651, 583), (79, 111, 114))],
                    ['right', ((795, 701), (62, 81, 84))],
                    ['right', ((698, 560), (91, 131, 133))],
                    ['right', ((847, 659), (70, 93, 96))],
                    ['right', ((840, 607), (70, 97, 102))],
                    ['right', ((936, 672), (85, 119, 122))],
                    ['right', ((635, 721), (69, 80, 82))],
                    ['right', ((510, 665), (86, 123, 126))],
                    ['right', ((930, 739), (56, 79, 84))],
                    ['right', ((1099, 631), (65, 74, 77))],
                    ['right', ((1171, 613), (62, 80, 83))],
                    ['right', ((1165, 709), (32, 60, 68))],
                ]
            elif location == 'guest':
                points = [
                    ['right', ((101, 540), (25, 14, 21))],
                    ['right', ((129, 509), (30, 18, 26))],
                    ['right', ((127, 559), (26, 16, 23))],
                    ['right', ((185, 480), (63, 26, 37))],
                    ['right', ((188, 591), (50, 25, 33))],
                    ['right', ((231, 447), (30, 19, 26))],
                    ['right', ((249, 622), (33, 21, 29))],
                    ['right', ((335, 406), (40, 25, 35))],
                    ['right', ((328, 651), (45, 26, 38))],
                    ['right', ((428, 350), (144, 109, 77))],
                    ['right', ((418, 650), (98, 49, 47))],
                    ['right', ((483, 332), (28, 17, 25))],
                    ['right', ((505, 743), (26, 15, 23))],
                    ['right', ((593, 268), (34, 21, 29))],
                    ['right', ((597, 742), (233, 135, 111))],
                    ['right', ((683, 259), (28, 18, 25))],
                    ['right', ((692, 749), (57, 36, 47))],
                    ['right', ((742, 288), (29, 18, 25))],
                    ['right', ((753, 745), (52, 34, 44))],
                    ['right', ((805, 319), (29, 18, 26))],
                    ['right', ((819, 736), (31, 20, 28))],
                    ['right', ((845, 342), (32, 20, 28))],
                    ['right', ((877, 714), (29, 18, 26))],
                    ['right', ((914, 366), (32, 22, 28))],
                    ['right', ((930, 687), (39, 24, 35))],
                    ['right', ((938, 390), (21, 11, 18))],
                    ['right', ((976, 672), (173, 170, 173))],
                    ['right', ((1002, 434), (29, 15, 22))],
                    ['right', ((1046, 621), (29, 18, 26))],
                    ['right', ((1051, 438), (48, 38, 44))],
                    ['right', ((1094, 615), (173, 170, 173))],
                    ['right', ((1090, 465), (28, 18, 26))],
                    ['right', ((1123, 569), (222, 86, 66))],
                    ['right', ((1124, 477), (25, 16, 22))],
                    ['right', ((1163, 562), (30, 19, 27))],
                    ['right', ((1173, 493), (34, 22, 30))],
                    ['right', ((1191, 548), (28, 18, 24))],
                    ['right', ((580, 295), (51, 32, 42))],
                    ['right', ((613, 280), (51, 32, 42))],
                    ['right', ((642, 322), (252, 137, 105))],
                    ['right', ((668, 278), (52, 33, 43))],
                    ['right', ((697, 339), (54, 35, 45))],
                    ['right', ((729, 285), (30, 19, 27))],
                    ['right', ((760, 343), (242, 98, 75))],
                    ['right', ((330, 472), (76, 47, 64))],
                    ['right', ((397, 375), (182, 182, 182))],
                    ['right', ((403, 481), (69, 43, 59))],
                    ['right', ((463, 348), (201, 86, 66))],
                    ['right', ((478, 458), (63, 40, 52))],
                    ['right', ((518, 320), (211, 153, 116))],
                    ['right', ((528, 479), (153, 81, 161))],
                    ['right', ((578, 322), (241, 101, 78))],
                    ['right', ((591, 474), (188, 184, 187))],
                    ['right', ((632, 326), (56, 36, 47))],
                    ['right', ((647, 458), (189, 185, 188))],
                    ['right', ((670, 327), (56, 35, 47))],
                    ['right', ((702, 458), (129, 126, 128))],
                    ['right', ((713, 332), (56, 36, 47))],
                    ['right', ((743, 478), (191, 186, 190))],
                    ['right', ((781, 348), (54, 33, 44))],
                    ['right', ((821, 487), (194, 188, 192))],
                    ['right', ((853, 362), (51, 33, 42))],
                    ['right', ((903, 488), (198, 190, 195))],
                    ['right', ((913, 372), (28, 17, 25))],
                    ['right', ((942, 438), (73, 45, 62))],
                    ['right', ((964, 404), (26, 15, 23))],
                    ['right', ((182, 519), (78, 42, 50))],
                    ['right', ((228, 458), (26, 16, 23))],
                    ['right', ((248, 536), (57, 36, 48))],
                    ['right', ((322, 410), (38, 24, 33))],
                    ['right', ((327, 550), (62, 38, 52))],
                    ['right', ((382, 380), (52, 51, 52))],
                    ['right', ((386, 546), (57, 36, 47))],
                    ['right', ((470, 379), (76, 47, 64))],
                    ['right', ((466, 543), (206, 226, 249))],
                    ['right', ((507, 383), (78, 49, 66))],
                    ['right', ((533, 551), (240, 206, 124))],
                    ['right', ((585, 391), (67, 41, 57))],
                    ['right', ((592, 547), (35, 20, 30))],
                    ['right', ((647, 400), (44, 20, 24))],
                    ['right', ((678, 600), (34, 20, 30))],
                    ['right', ((739, 436), (56, 35, 46))],
                    ['right', ((773, 600), (100, 51, 50))],
                    ['right', ((852, 429), (73, 45, 61))],
                    ['right', ((873, 584), (75, 47, 64))],
                    ['right', ((916, 425), (72, 45, 61))],
                    ['right', ((958, 586), (77, 49, 66))],
                    ['right', ((1000, 455), (181, 75, 63))],
                    ['right', ((1032, 592), (210, 84, 65))],
                    ['right', ((1056, 452), (22, 11, 19))],
                    ['right', ((1086, 584), (41, 24, 34))],
                    ['right', ((1110, 474), (21, 11, 19))],
                    ['right', ((1144, 583), (29, 18, 26))],
                    ['right', ((169, 580), (29, 17, 25))],
                    ['right', ((219, 531), (54, 34, 45))],
                    ['right', ((236, 626), (30, 19, 27))],
                    ['right', ((292, 482), (64, 37, 48))],
                    ['right', ((270, 644), (97, 89, 95))],
                    ['right', ((354, 478), (77, 48, 65))],
                    ['right', ((363, 654), (51, 32, 43))],
                    ['right', ((448, 498), (108, 61, 57))],
                    ['right', ((430, 653), (75, 47, 63))],
                    ['right', ((527, 492), (45, 28, 40))],
                    ['right', ((513, 637), (216, 177, 159))],
                    ['right', ((593, 502), (40, 25, 35))],
                    ['right', ((592, 708), (77, 48, 65))],
                    ['right', ((654, 541), (38, 24, 34))],
                    ['right', ((655, 705), (122, 111, 78))],
                    ['right', ((717, 547), (34, 20, 30))],
                    ['right', ((731, 741), (92, 52, 50))],
                    ['right', ((829, 543), (46, 28, 41))],
                    ['right', ((853, 705), (225, 104, 80))],
                    ['right', ((927, 535), (52, 33, 43))],
                    ['right', ((947, 663), (20, 10, 16))],
                    ['right', ((1012, 567), (253, 101, 76))],
                    ['right', ((1025, 640), (29, 19, 27))],
                    ['right', ((956, 740), (24, 0, 49))],
                    ['right', ((917, 585), (77, 48, 65))],
                    ['right', ((902, 737), (24, 0, 49))],
                    ['right', ((866, 588), (73, 46, 62))],
                    ['right', ((832, 727), (27, 16, 24))],
                    ['right', ((833, 576), (46, 29, 39))],
                    ['right', ((783, 748), (27, 16, 23))],
                    ['right', ((745, 570), (32, 19, 29))],
                    ['right', ((715, 726), (51, 31, 42))],
                    ['right', ((711, 542), (35, 22, 31))],
                    ['right', ((667, 751), (8, 55, 103))],
                    ['right', ((636, 580), (244, 223, 94))],
                    ['right', ((555, 710), (221, 188, 171))],
                    ['right', ((546, 543), (48, 30, 43))],
                    ['right', ((515, 697), (75, 48, 64))],
                    ['right', ((498, 565), (219, 184, 166))],
                    ['right', ((470, 650), (80, 50, 67))],
                    ['right', ((445, 554), (46, 28, 40))],
                ]

            return points

        visual = Visual_Player()
        namer = Passport()

        points = get_points()
        poses = []

        if location != 'guest':
            for point in points:
                point_pos = point[1][0]
                pixel = pg.pixel(point_pos[0], point_pos[1])

                if pixel != point[1][1]:
                    poses.append(point_pos)
        else:
            poses = [point[1][0] for point in points]

        location_players = []
        random.shuffle(poses)

        # степень раздраженности поиска
        annoyed = 0.7

        # проверка на <усмиири дракошу>
        def dragon_play():
            dragon_window = [
                ((430, 301), (34, 129, 196)),
                ((834, 460), (68, 167, 54)),
                ((597, 459), (205, 205, 205)),
            ]
            for pos, need_color in dragon_window:
                current_color = pg.pixel(pos[0], pos[1])
                if current_color != need_color:
                    return False

            cancel = (597, 459)
            self.action.click(cancel)
            return True

        for point in poses:
            self.action.click(point)
            self.action.click(point)

            if dragon_play():
                continue

            time.sleep(0.75)

            annoyed *= 0.95

            find_passport = visual.find_image_on_screen('button_passport.png')
            if find_passport:
                self.action.click(find_passport)
            else:
                if location == 'guest' and annoyed < 0.32:
                    break

                continue

            time.sleep(0.75)

            player_id = namer.read_id()
            nickname = namer.detect()

            # если указали и нашли, то выходим
            if name == nickname or player_id == id:
                return location_players

            if not any(p['id'] == player_id for p in location_players) and len(player_id) > 0:
                location_players.append({'id': player_id, 'name': nickname})
                annoyed += 0.2

            keyboard.press_and_release('esc')
            time.sleep(0.75)

        return location_players

    def target(self, id, name, duration=10 * 60):
        baseOff = BaseOffers()
        time_limit = TimeLimit(duration)
        finish = False

        def repeat(location, event=False):
            by_player = None
            if event:
                event_info = self.location.open_guests(location)
                location = self.location.situation()
                if location != 'home':
                    location = event_info['name']
                    by_player = event_info['player']
            else:
                self.location.goto(location)

            players_list = self.find_players(location, id=id, name=name)
            baseOff.add_info(location, players_list, by_player)

            for player in players_list:
                if player['id'] == id or player['name'] == name:
                    print('Нашел игрока ', player['name'])
                    return True

            return players_list

        # проверяем на текущей локации
        players_list = self.find_players(self.location.situation(), id=id, name=name)

        for player in players_list:
            if player['id'] == id or player['name'] == name:
                print('Нашел игрока ', player['name'])
                return True

        # идем дальше
        while time_limit.check() and finish != True:
            finish = repeat('cafe')
            finish = repeat('street')
            finish = repeat('sqwer')
            finish = repeat('school')
            finish = repeat('park')
            finish = repeat('bal')
            finish = repeat('club')

            for i in range(1, 10):
                finish = repeat(i, True)

    def Patrol(self):

        msg = [
            'ДЕПАРТАМЕНТ активирован. Готов к миссиям/',
            'Система перезагружена. Штаб на связи',
            'Пепел — удобрение. Мы выросли.',
            'Код 0xRESURRECT. ДЕПАРТАМЕНТ в строю.',
            'Всем - Вперёд.',

            'Мы не просили разрешения. Мы вернулись..',
            'Ваши старые долги — наши новые приказы. .',
            'Кто забыл нас — вспомнит. Кто помнит — испугается. <>',
            'Территория под наблюдением. Не мешайте. ,,.,,',
            'Следующий ход — наш. Готовьтесь. %**}',

            'ДЕПАРТАМЕНТ не умер. Он затаился. :::',
            'Наши глаза открыты. Наши руки свободны. ^^',
            'Порядок — не просьба. Это приказ. ',
            'Кто-то правил в наше отсутствие. Ошибка.',
            'Город не ваш. Он под защитой. ^.',

            'Мы были в тени. Теперь тень — это МЫ',
            'Вы не видели нас. Но мы видели вас. ',
            'Тишина — наше оружие. Ждите. ',
            'Никто не исчезает навсегда. Особенно мы. -.--',
            'Нас стёрли? Мы перезагрузились.-..--.',

            'Мёртвые не уходят. Они перегруппировываются.%',
            'Прах — это память. Мы — её голос*',
            'Кости скелетов крепче, чем кажется.',
            'Нас хоронили. Мы выкопались.##',
            'Кладбище — наша тренировочная база.##/',

            'Тихий час окончен. Время грохота.-..',
            'Кто-то думал, что мы сломаны? Ошиблись.//',
            'Мы не просим. Мы забираем. ,.',
            'Мирный сон Аватарии прерван. Это мы. ',
            'Террор? Нет. Просто ДЕПАРТАМЕНТ.',

            'Ваши секреты — наши архивы. --',
            'Агенты активированы. Ждите сигнала.-/--/-',
            'Мы знаем, что вы делали прошлым летом. ==',
            'Шёпот в темноте — это мы. ',
            'Кто-то говорил, что мы мертвы? Врут.',

            'Система ONLINE. Цели определены.',
            'Протокол «Феникс» завершён. Мы живы.',
            '01000100 01000101 01010000 — это мы.',
            'Роботы? Нет. Мы — киборги мести.',
            'Сервера горят. Мы — нет.',

            'Короли умерли. Мы — их палачи. 0-0-0',
            'Революция? Нет. Реставрация. ==',
            'Кто-то правил без нас. Смешно.',
            'Мы не герои. Мы — необходимость. /.',
            'Ваш рай — наша мишень. &',

            'Падение — не конец. Это разгон.',
            'Игра началась. Мы — новые правила.',
            'История пишется нами. Снова.',

        ]

        def actions_repeat(name_location, time_duration):
            self.location.goto(name_location)
            time.sleep(10)
            chat.open()
            chat.write_msg_global(random.choice(msg))
            chat.close()
            time.sleep(2)

        chat = self.chat

        for _ in range(1):
            actions_repeat('cafe', 10)
            actions_repeat('street', 10)
            actions_repeat('sqwer', 10)
            actions_repeat('school', 10)
            actions_repeat('park', 10)
            actions_repeat('bal', 10)
            actions_repeat('club', 10)
            actions_repeat('beauty', 10)

            for i in range(1, 10):
                print(i)
                self.location.open_guests(i)

                # Если нет ошибок, и не дома, то пишем в чат
                situation = self.location.wait()

                if situation == 'guest':
                    time.sleep(10)
                    chat.open()
                    # chat.write_msg(random.choice(msg))
                    chat.close()
                    time.sleep(2)

    def Cake_clear(self):
        self.location.goto('home')

        face_clear = [
            ['right', ((918, 682), (162, 143, 246))],
            ['right', ((270, 708), (199, 102, 174))],
            ['left', ((604, 179), (49, 135, 191))],
            ['left', ((604, 179), (49, 135, 191))],
        ]
        self.action.executes(face_clear)

        time.sleep(0.75)

        button_clear = (734, 302)
        self.action.click(button_clear)

    def Time_Present(self, admission=False):
        def finished():
            keyboard.press_and_release('esc')
            self.location.wait()

        if not self.presents_time.check() or admission:

            if not self.location.situation():
                return

            # Нажимаем на кнопку подарков
            present_button = (73, 491)
            self.action.click(present_button)
            time.sleep(0.78)

            check_orange_present = [
                ((1083, 597), (176, 105, 0)),
                ((923, 587), (176, 106, 0)),
                ((776, 584), (176, 106, 0)),
                ((624, 582), (176, 106, 0)),
                ((464, 583), (176, 106, 0)),
                ((310, 588), (176, 106, 0)),
            ]

            # проверяем взяли ли мы зеленые подарки
            def is_green(color):
                if color[0] < 50 and color[1] > 50 and color[2] < 100:
                    return True
                return False

            pos, need_color = check_orange_present[0]
            current_color = pg.pixel(pos[0], pos[1])

            # если последний подарок открыт то больше не заходим сюда
            if is_green(current_color):
                self.presents_time.reset(True)
                finished()
                return

            # проверяем оранжевые доступные подарки
            def is_orange(color):
                if color[0] > 160 and color[1] < 130:
                    return True
                return False

            for info in check_orange_present:
                pos, need_color = info
                current_color = pg.pixel(pos[0], pos[1])

                # если не все подарки открыты - выходим
                if not is_orange(current_color):
                    finished()
                    return

            for info in check_orange_present:
                pos, need_color = info
                self.action.click(pos)

                # Ждем окно ok
                while not self.action.problems():
                    pass
                time.sleep(0.75)

            # Больше не проверяем подарки
            self.presents_time.reset(True)

            # выходим из меню подарков
            finished()

    def Event_Control(self, admission=False):
        if not self.event_control_time.check() or admission:
            print('Проверяю активность события: ', end='')
            # открыть список событий
            open_event_list = [
                ['left', ((119, 693), (47, 159, 209))],
                ['right', ((230, 319), (41, 189, 253))],
                ['right', ((84, 234), (226, 69, 79))],
                ['right', ((511, 81), (110, 221, 250))],
                ['left', ((240, 542), (36, 136, 200))],
                ['right', ((240, 542), (46, 191, 253))],
                ['right', ((949, 199), (62, 175, 220))],
                ['right', ((1085, 194), (209, 209, 210))],
            ]
            self.action.executes(open_event_list)

            # проверяем есть ли собственное активное событие
            current_color = pg.pixel(1180, 515)
            not_event_color = (18, 116, 164)

            if current_color == not_event_color:
                print('Не активно')

                event_name = 'ДЕПАРТАМЕНТ ПОЛИЦИИ'
                event_descript = 'Новая эра Аватарии!'
                golden_star = False
                married_top = False

                # открываем настройки события и указываем название события
                open_setting_name_event = [
                    ['right', ((1077, 648), (90, 186, 76))],
                    ['left', ((1098, 652), (83, 180, 68))],
                    ['right', ((626, 542), (162, 168, 171))],
                    ['right', ((633, 593), (162, 168, 171))],
                    ['left', ((497, 257), (231, 240, 244))],
                ]
                self.action.executes(open_setting_name_event)
                keyboard.write(event_name)

                # указываем описание события
                self.action.click((598, 350))
                keyboard.write(event_descript)

                # добавляем золотую звездочку
                if golden_star:
                    self.action.click((624, 546))

                # ставим свадебный тип события
                if married_top:
                    self.action.click((1224, 289))

                # создаем событие
                self.action.click((1127, 668))

            else:
                print('Активно')

            # обновляем таймер обновления события
            self.event_control_time.reset()

            # закрываем список событий
            keyboard.press_and_release('esc')

            # ждем закрытия
            self.location.wait()

    def Free_Lama(self, admission=False):
        if not self.free_lama_time.check() or admission:

            if self.location.situation() == None:
                return

            # зашли в банк ламы
            open_bank_lama = [
                ['left', ((1152, 67), (244, 210, 63))],
                ['right', ((237, 262), (36, 188, 253))],
                ['right', ((244, 339), (30, 132, 197))],
                ['left', ((244, 338), (30, 132, 197))],
                ['right', ((244, 338), (37, 188, 252))],
                ['right', ((256, 260), (32, 133, 197))],
            ]
            self.action.executes(open_bank_lama)

            # прокрутили до щедрой ламы
            pg.scroll(-5000)

            # зашли вглубь ламы
            inside_free_lama = [
                ['left', ((1139, 673), (201, 230, 196))],
                ['right', ((847, 691), (8, 65, 110))],
            ]
            self.action.executes(inside_free_lama)

            color_break = (141, 141, 141)
            coord_check = (742, 685)
            x, y = coord_check
            current_pixel = pg.pixel(x, y)
            # Если кнопка не серая - активируем ламу
            if current_pixel != color_break:
                active_lama_click = coord_check
                self.action.click(active_lama_click)

            close_lama_and_bank = [
                ['right', ((774, 680), (142, 142, 143))],
                ['right', ((1015, 64), (102, 212, 251))],
                ['left', ((1240, 70), (220, 237, 246))],
                ['right', ((251, 263), (29, 130, 195))],
                ['right', ((262, 341), (28, 183, 251))],
                ['left', ((1242, 69), (193, 221, 238))],
            ]
            self.action.executes(close_lama_and_bank)

            self.free_lama_time.reset()

            self.location.wait()


class BaseChatSaver:
    def __init__(self):
        self.data_file = 'base.json'

    def save_chat(self, location: str, new_messages: List[Tuple]) -> None:
        print('yes')
        current_data = self._load_data()

        if location not in current_data:
            current_data[location] = []

        current_data[location].append(new_messages)
        self._save_data(current_data)

    def words_target(self, words_list):
        def print_messages(dialog):
            for message in dialog:
                print(message[0], end=' ' * (25 - len(message[0])))
                if word_detect(message[1], words_list):
                    print('*!*', end=' ' * 5)
                else:
                    print('   ', end=' ' * 5)
                print(message[1], ' ' * (120 - len(message[1])), message[2])

        location_chats = self._load_data()

        for location, chats in location_chats.items():
            for dialog in chats:
                sender_found = any(
                    word_detect(message[1], words_list)
                    for message in dialog
                    if isinstance(message, list) and len(message) > 0
                )

                if sender_found:
                    print('-' * 120)
                    print(location)
                    print('-' * 120)
                    print_messages(dialog)

    def player_target(self, name):
        def print_messages(dialog):
            for message in dialog:
                print(message[0], end=' ' * (25 - len(message[0])))
                if message[0] == name:
                    print('*!*', end=' ' * 5)
                else:
                    print('   ', end=' ' * 5)
                print(message[1], ' ' * (120 - len(message[1])), message[2])

        location_chats = self._load_data()

        for location, chats in location_chats.items():
            for dialog in chats:
                sender_found = any(
                    message[0] == name
                    for message in dialog
                    if isinstance(message, list) and len(message) > 0
                )

                if sender_found:
                    print('-' * 120)
                    print(location)
                    print('-' * 120)
                    print_messages(dialog)

    def get_chat(self, location: str) -> List[Tuple]:
        data = self._load_data()
        return data.get(location, [])

    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_data(self, data: Dict[str, Any]) -> None:
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


class EventChatSaver:
    def __init__(self):
        self.data_file = 'events.json'

    def save_chat(self, location: str, new_messages: List[Tuple]) -> None:
        current_data = self._load_data()

        if location not in current_data:
            current_data[location] = []

        current_data[location].append(new_messages)
        self._save_data(current_data)

    def words_target(self, words_list):
        def print_messages(dialog):
            for message in dialog:
                print(message[0], end=' ' * (25 - len(message[0])))
                if word_detect(message[1], words_list):
                    print('*!*', end=' ' * 5)
                else:
                    print('   ', end=' ' * 5)
                print(message[1], ' ' * (120 - len(message[1])), message[2])

        events_chats = self._load_data()

        for location, events in events_chats.items():
            for event in events:
                if not event:
                    continue

                event_title, event_time = event[0]
                dialog = event[1] if len(event) > 1 else []

                sender_found = any(
                    word_detect(message[1], words_list)
                    for message in dialog
                    if isinstance(message, list) and len(message) > 0
                )

                if sender_found:
                    print('-' * 120)
                    print(event_title, '   ::от ', location)
                    print('-' * 120)
                    print_messages(dialog)

    def player_target(self, name):
        def print_messages(dialog):
            for message in dialog:
                print(message[0], end=' ' * (25 - len(message[0])))
                if message[0] == name:
                    print('*!*', end=' ' * 5)
                else:
                    print('   ', end=' ' * 5)
                print(message[1], ' ' * (120 - len(message[1])), message[2])

        events_chats = self._load_data()

        for location, events in events_chats.items():
            for event in events:
                if not event:
                    continue

                event_title, event_time = event[0]
                dialog = event[1] if len(event) > 1 else []

                sender_found = any(
                    message[0] == name
                    for message in dialog
                    if isinstance(message, list) and len(message) > 0
                )

                if sender_found:
                    print('-' * 120)
                    print(event_title, '   ::от ', location)
                    print('-' * 120)
                    print_messages(dialog)

    def get_chat(self, location: str) -> List[Tuple]:
        data = self._load_data()
        return data.get(location, [])

    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_data(self, data: Dict[str, Any]) -> None:
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


class BaseOffers:
    def __init__(self, file_path: str = 'offers.json'):
        """
        Инициализация менеджера предложений

        :param file_path: путь к JSON-файлу с данными
        """
        self.file_path = file_path
        self.datas: List[List[List[Any], List[Dict]]] = []
        self.load_data()

    def load_data(self) -> None:
        """Загружает данные из файла"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.datas = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.datas = []

    def save_data(self) -> None:
        """Сохраняет данные в файл"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.datas, f, ensure_ascii=False, indent=4)

    def add_info(self, name_location: str, players: List[Dict], event_organizer=None) -> None:
        """
        Добавляет событие с организатором

        :param name_location: название локации/события
        :param event_organizer: организатор {'id': id, 'name': name}
        :param players: список участников [{'id': id, 'name': name}, ...]
        """
        time_in = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = {
            'info': {'location': name_location, 'time': time_in, 'oranizer': event_organizer},  # info с организатором
            'players': players.copy()  # players
        }
        self.datas.append(new_data)
        self.save_data()

    def get_all_offers(self) -> List[Dict[str, Any]]:
        """Возвращает все предложения в удобном формате"""
        result = []
        for data in self.datas:
            info, players = data
            result.append({
                'location': info[0],
                'time': info[1],
                'event_organizer': info[2],
                'players': players
            })
        return result

    def get_events(self) -> List[Dict[str, Any]]:
        """Возвращает только события с организаторами"""
        return [
            {
                'location': data[0][0],
                'time': data[0][1],
                'organizer': data[0][2],
                'players': data[1]
            }
            for data in self.datas
            if data[0][2] is not None  # Только где есть организатор
        ]

    def get_regular_locations(self) -> List[Dict[str, Any]]:
        """Возвращает обычные локации без событий"""
        return [
            {
                'location': data[0][0],
                'time': data[0][1],
                'players': data[1]
            }
            for data in self.datas
            if data[0][2] is None  # Только без организатора
        ]

    def clear_all(self) -> None:
        """Очищает все данные"""
        self.datas = []
        self.save_data()


class Visual_Player:
    def __init__(self):
        pass

    def read(self, img):
        # Конвертируем в OpenCV формат
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 1. Подготовка изображения
        # Увеличиваем контраст текста
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # 2. Удаление цветного префикса
        # Преобразуем в HSV для обнаружения цветных областей
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

        # Создаем маску для цветных префиксов (игнорируем серые/белые)
        lower_color = np.array([0, 50, 50])
        upper_color = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # Инвертируем маску, чтобы оставить только серый текст
        text_mask = cv2.bitwise_not(color_mask)

        # 3. Объединяем маски
        final_mask = cv2.bitwise_and(thresh, text_mask)

        cv2.imshow('wer', final_mask)
        cv2.waitKey(0)

        # 4. Находим контуры текста
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтруем слишком маленькие контуры
        min_text_width = 5
        text_contours = [c for c in contours if cv2.boundingRect(c)[2] > min_text_width]

        if not text_contours:
            print('blya')
            return ""

        # 5. Вырезаем область с ником
        x, y, w, h = cv2.boundingRect(text_contours[0])
        nickname_region = gray[y:y + h, x:x + w]

        # 6. Улучшаем читаемость для Tesseract
        nickname_processed = cv2.threshold(nickname_region, 0, 255,
                                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # 7. Распознаем текст
        # custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_[]()'
        text = pytesseract.image_to_string(nickname_processed, lang='rus+eng')
        print(text)
        return text.strip()

    def get_scale(self, location):
        scale_factor = 1
        if location == 'home' or location == 'guest':
            scale_factor = 0.6

        elif location == 'street' or location == 'sqwer' or location == 'cafe' or location == 'park' or location == 'bal':
            scale_factor = 0.33

        elif location == 'school' or location == 'club':
            scale_factor = 0.4

        elif location == 'movie':
            scale_factor = 0.28

        elif location == 'beauty':
            scale_factor = 0.5

        return scale_factor

    def find_bot(self, location, threshold=0.8):
        # Делаем скриншот экрана
        screenshot = pg.screenshot(region=(100, 100, 1290, 720))

        # Конвертируем в numpy array и меняем цветовую схему
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        # Загружаем шаблон из сохраненного файла
        img_name_face = 'bot_face.png'
        img_name_face_not = 'bot_face_not.png'
        template_face = cv2.imread(img_name_face, cv2.IMREAD_COLOR)
        template_face_not = cv2.imread(img_name_face_not, cv2.IMREAD_COLOR)

        # Увеличение в 2 раза
        scale_factor = self.get_scale(location)

        image_face = cv2.resize(template_face, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        flipped_image_face = cv2.flip(image_face, 1)

        image_face_not = cv2.resize(template_face_not, None, fx=scale_factor, fy=scale_factor,
                                    interpolation=cv2.INTER_CUBIC)
        flipped_image_face_not = cv2.flip(image_face_not, 1)

        # cv2.imshow('image_face', image_face_not)
        # cv2.waitKey(10000)
        # cv2.imshow('image_face', image_face)
        # cv2.waitKey(10000)

        for template in [image_face, flipped_image_face, image_face_not, flipped_image_face_not]:
            if template is None:
                print("Ошибка загрузки шаблона!")
                continue

            # Ищем совпадения
            result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Если совпадение достаточно точное
            if max_val >= threshold:
                # Возвращаем координаты центра найденного изображения
                h, w = template.shape[:-1]
                center_x = max_loc[0] + w // 2 + 100
                center_y = max_loc[1] + h // 2 + 100
                pg.moveTo(center_x, center_y)
                return (center_x, center_y)
            else:
                print(f"Совпадение не найдено (максимальное значение: {max_val})")
                continue

    def find_image_on_screen(self, img_name, threshold=0.8):
        # bot_face.png
        # button_animation.png
        # anim_police.png
        # 'button_passport.png'

        # Делаем скриншот экрана
        screenshot = pg.screenshot(region=(0, 0, 1290, 720))

        # Конвертируем в numpy array и меняем цветовую схему
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        # Загружаем шаблон из сохраненного файла
        template = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # cv2.imshow(img_name,template)
        # cv2.waitKey(0)

        if template is None:
            print("Ошибка загрузки шаблона!")
            return None

        # Ищем совпадения
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Если совпадение достаточно точное
        if max_val >= threshold:
            # Возвращаем координаты центра найденного изображения
            h, w = template.shape[:-1]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            pg.moveTo(center_x, center_y)
            return (center_x, center_y)
        else:
            print(f"Совпадение не найдено (максимальное значение: {max_val})")
            return None

    def bot_position_control(self):
        action = Actions()

        pos = (int(100 + random.random() * 500), int(400 + 200 * random.random()))
        begin_pixel = pg.pixel(pos[0], pos[1] - 50)

        action.click(pos)

        while True:
            change_pixel = pg.pixel(pos[0], pos[1] - 50)
            if change_pixel != begin_pixel:
                break
            time.sleep(1)

        pg.moveTo(pos[0], pos[1] - 50)
        print('обнаружен бот')


program = Programms()
program.catch_information()

# program.chat.reading_private(True)
# chat = ChatAnalyzer()
# chat.capture_buffer()
# print(chat.merge_message_chains())
# def f_stop(update):
#    return False
# program.chat.reading(f_stop)

# time.sleep(2)
# program.find_players(program.location.situation())
# program.target(id='33200779',name='хуйNA', duration=10*60)
# location = 'movie'
# move = Visual_Player().find_bot(location,0.8)
# pg.moveTo(move[0],move[1])

# print(Visual_Player().find_image_on_screen())
# program.find_players()

# img = pg.screenshot(region=(10, 10,1290,720))
# img.save('cafe_1.png')
# img.show()

# list_ban = ['сука','ахуеть','шлюха', 'отсос','соси','отсоси','сын','мать','отчим','мамаша','мамка', 'ебал','шмара','аутист','ебать','завались','ебало','завали','сосал','сосала','хуй','хуесос']
# list_bot = ['департамента','департаменте','тип','деп','депо']
# a = Base().words_target(list_bot)
