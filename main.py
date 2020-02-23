import tkinter as tk
import pickle, os, sys, configparser, time
from os import path
from datetime import datetime
from tkinter import ttk, messagebox, filedialog as fd


def import_modules():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from PIL import Image
    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from skimage import measure
    from sklearn.linear_model import LinearRegression
    from scipy.ndimage.interpolation import zoom
    import pickle as pickle
    from openpyxl import Workbook
    return pd, Workbook, pickle, np, plt, Figure, FigureCanvasTkAgg, Image, KMeans, make_pipeline, PolynomialFeatures, measure, LinearRegression, zoom


class Interface():


    def __init__(self):
        # Функция инициализации GUI + извлечение параметров config-файла
        self.win = tk.Tk()              # Создание окна
        self.win.title("LevelVisio")    # Заголовок программы
        self.open_config()              # Извлечение параметров из config-файла
        self.create_widgets()
        self.win.update()
        self.win.deiconify()
        self.win.mainloop()


    def open_config(self):
        # Открыть и задать параметры из конфигурационного файла "config.conf"
        self.fDir = path.dirname(__file__)
        self.dir = self.fDir + '\\' + 'config.conf'
        self.parser = configparser.SafeConfigParser()
        self.parser.read(self.dir)
        print(self.dir)
        self.params = {}
        for section in self.parser.sections():
            print(section)
            for name, value in self.parser.items(section):
                self.params[name] = value
                print(f'{name} = {value}')
        print(self.params['dead_volume'])
        self.dead_volume = float(self.params['dead_volume'])


    def setup_packages_button(self):
        #	Кнопка установки пакетов
        self.setup_packages_label = ttk.Label(self.tab2, text="Установка модулей").grid(column=2, row=0)
        self.setup_packages = ttk.Button(self.tab2, text="Установить", command=self.setup_packages).grid(column=2, row=1)


    def setup_packages(self):
        # Функционал кнопки установки пакетов (Создание и описание кнопки Установка пакетов)
        packages = 'numpy scipy matplotlib pillow scikit-learn scikit-image python-shell opencv-python'
        try:
            self.setup(packages)
            print('All packages have been installed correctly!')
        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError:
            print("Could not convert data to an integer.")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise


    def setup(self, packages):
        # Функция установки пакетов (Производит установку необходимых в системе пакетов для работы программы)
        pckgs = packages.split()
        for pkg in pckgs:
            global installPythonPackage
            installPythonPackage = 'pip install --save ' + pkg
            os.system('start cmd /c ' + installPythonPackage)


    def checkPhotosTurn(self):
        # Возвращает 1, если стоит галочка для поворота фото, иначе 0
        check_photos_state = self.chVarEn.get()
        return check_photos_state


    def check_save_model(self):
        # Возвращает 1, если стоит галочка для сохранения модели, иначе 0
        check_save_model = self.ch_save_model2.get()
        return check_save_model


    def photos_folder_button(self):
        # Функционал кнопки выбора папки фотографий
        self.photos_folder_label = ttk.Label(self.tab1, text="1. Выберите папку").grid(column=1, row=0)
        self.photos_folder_label = ttk.Label(self.tab1, text="со снимками").grid(column=1, row=1)
        self.photos_folder = ttk.Button(self.tab1, text="Открыть снимки...", command=self.photos_folder).grid(column=1, row=3)
        self.chVarEn = tk.IntVar()
        self.check3 = tk.Checkbutton(self.tab1, text="Поворот фото", variable=self.chVarEn)
        self.check3.deselect()
        self.check3.grid(column=1, row=2)


    def photos_folder(self):
        # Функция кнопки выбора папки фотографий
        self.photos_folder = fd.askdirectory()
        print(self.photos_folder)
        self.files_extract(self.photos_folder)
        # self.checkPhotosTurn()


    def open_model_button(self):
        #	Функционал кнопки выбора модели
        self.open_model_label = ttk.Label(self.tab1, text="Выбрать модель:").grid(column=2, row=0)
        self.open_model = ttk.Button(self.tab1, text="Открыть модель...", command=self.open_model).grid(column=2, row=3)


    def open_model(self):
        # Функция кнопки выбора модели
        self.model_dir =  fd.askopenfilename(initialdir = self.fDir,title = "Select model file",filetypes = (("Pickle models files","*.sav"),("all files","*.*")))  # print(self.model_dir)
        self.loaded_model = pickle.load(open(self.model_dir, 'rb'))
        result = loaded_model.score(X_test, Y_test) #for key, value in self.params.items(): print(f'key = {key}, value = {value}, type = {type(value)}')
        print(result)


    def create_model_button(self):
        # Кнопка выбора модели
        self.train_model_label = ttk.Label(self.tab1, text="Для обучения модели").grid(column=3, row=0)
        self.train_model_label = ttk.Label(self.tab1, text="выберите фото!     ").grid(column=3, row=1) #self.train_model_label = ttk.Label(self.tab1, text="б) Обучить модель.").grid(column=3, row=2)
        self.ch_save_model2 = tk.IntVar()
        self.ch_save_model2_ = tk.Checkbutton(self.tab1, text="Сохранить модель?", variable=self.ch_save_model2).grid(column=3, row=2)
        # self.ch_save_model2.select() не работает "!!"
        self.train_model = ttk.Button(self.tab1, text="Обучить модель...", command=self.create_model).grid(column=3, row=3)
        self.train_model_label_ = ttk.Label(self.tab1, text="Имя модели:").grid(column=2, row=4)
        self.train_model_lbl = tk.StringVar()
        self.train_model_ = ttk.Entry(self.tab1, width=18, textvariable=self.train_model_lbl).grid(column=3, row=4)
        self.train_model_lbl.set('Model_name')


    def photo_cycle_button(self):
        # Кнопка запуска камеры для создания фотоснимков
        self.start_cam_model_label = ttk.Label(self.tab1, text="Запустить камеру").grid(column=5, row=0)
        self.start_cam_model = ttk.Button(self.tab1, text="Запустить", command=self.photo_cycle).grid(column=5, row=3)


    def create_model(self):
        # Функция выделяет кластеры, разбивает на сегменты, извлекает значения коэффициентов из изображений и строит модель
        contours_of_images, dict_of_names, j = {}, {}, 0
        # Перебор всех фото в папке и извлечение кластеров из них
        for image in self.files:
            print(f'file) = {image}')
            sample = str(self.photos_folder + '/' + image) # !!!
            print(f'sample!! = {sample}')
            km, h, w = self.kmeans_clustering(sample, n=2)
            seg = self.segmentation(km, h, w)
            contours_of_images[j] = seg # self.graph_clasters(seg)
            dict_of_names[j] = self.photos_folder + image
            j += float(self.params['photos_step'])
        self.delta_coeffs = {}
        i, j = 0, 1             # print(f'contours_of_images = {contours_of_images}')
        # Перебор всех контуров фото и извлечение из них сегментов, коэффициентов и дельта-коэффициентов
        for contours in contours_of_images.items():
            segment = self.find_segment(contours)
            coeff, delta = self.find_delta(segment, dict_of_names, j)
            self.delta_coeffs[j] = coeff
            j += 1 # float(self.params['photos_step'])
            i += 1
        X, y = [], []
        low_level, high_level, step, j = 1, len(self.files), 1, 0
        # Перебор всех контуров фото и извлечение из них сегментов, коэффициентов и дельта-коэффициентов
        for i in range(low_level, high_level, step):
            X.append(j)
            y.append(self.delta_coeffs[j])
            j += 1 # float(self.params['photos_step'])
        self.X, self.Y = X, y   # print(X); print(y)
        # Строим график X-y
        plot = plt.plot(X, y)
        plt.setp(plot, color='g', linewidth=2.0)
        plt.grid(True)
        plt.xlabel('Level, mm')
        plt.ylabel('Coefficients')
        plt.show()
        print(f'self.delta_coeffs.items() = {self.delta_coeffs.items()}')
        print(f'self.delta_coeffs = {self.delta_coeffs}')
        coeff_avg = 0
        for mm, coeff in self.delta_coeffs.items():
            coeff_avg = coeff_avg + coeff
        coeff_avg = coeff_avg / len(self.delta_coeffs)
        print(f'Average coefficient is {coeff_avg}')
        '''
        print(f'self.files[8] = {self.files[8]}')
        test = f'{self.photos_folder}/{self.files[8]}'
        #test = u"1-18_ml/" + self.files[8]
        km_2, h_2, w_2 = self.kmeans_clustering(test, n=2)
        seg_2 = self.segmentation(km_2, h_2, w_2)
        #self.graph_clasters(seg_2)
        segment_2 = self.find_segment(seg_2)
        coeff_2, delta_2 = self.find_delta(segment_2, test_2, 6)
        print(coeff_2)
        '''
        x_train = np.array([X]).T
        y_train = np.array([y]).T
        x_train, y_train = y_train, x_train
        n_points = 10
        plt.scatter(x_train, y_train, s=80, c ='r', edgecolors='k', linewidths=0.3);
        lr = LinearRegression(fit_intercept=False)  # Строим предсказательную модель линейной и полиномиальной регресси
        lr.fit(x_train, y_train)
        y_regression = lr.predict(x_train)
        self.simple_plot(x_train, y_train, y_regression, prefix='Train')
        poly = make_pipeline(PolynomialFeatures(degree=5),LinearRegression(fit_intercept=False))
        poly.fit(x_train, y_train)
        y_poly_regression = poly.predict(x_train)
        self.simple_plot(x_train, y_train, y_poly_regression, prefix='Poly_Train')
        if self.check_save_model() == 1:
            filename = '/models/' + self.train_model_lbl.get() + '.sav'
            fn = self.train_model_lbl.get() + '_poly_model' + ".sav"
            pickle.dump(poly, open(fn, 'wb'))
            fn2 = self.train_model_lbl.get() + '_linear_model' + ".sav"
            pickle.dump(lr, open(fn2, 'wb'))
        y_regression = poly.predict(x_train)
        self.simple_plot(x_train, y_train, y_regression, prefix='Train')


    def create_graphs_button(self):
        #	Функционал кнопки отображения графиков
        self.create_graphs_label = ttk.Label(self.tab1, text="Построить графики:").grid(column=4, row=0)
        self.create_graphs = ttk.Button(self.tab1, text="Построить", command=self.create_graphs).grid(column=4, row=3)


    def create_graphs(self):
        #	Функция кнопки отображения графиков
        # Функция выделяет кластеры, разбивает на сегменты, извлекает значения коэффициентов из изображений и строит модель
        coeff_avg = 0
        for mm, coeff in self.delta_coeffs.items():
            coeff_avg = coeff_avg + coeff
        coeff_avg = coeff_avg / len(self.delta_coeffs)
        x_train = np.array([self.X]).T
        y_train = np.array([self.Y]).T
        x_train, y_train = y_train, x_train
        n_points = 10
        plt.scatter(x_train, y_train, s=80, c ='r', edgecolors='k', linewidths=0.3);
        lr = LinearRegression(fit_intercept=False)  # Строим предсказательную модель линейной и полиномиальной регресси
        lr.fit(x_train, y_train)
        # self.simple_plot(x_train, y_train, y_regression, prefix='Train')
        poly = make_pipeline(PolynomialFeatures(degree=5),LinearRegression(fit_intercept=False))
        poly.fit(x_train, y_train)
        y_poly_regression = poly.predict(x_train)
        contours_of_images, dict_of_names, j = {}, {}, 0
        # Перебор всех фото в папке и извлечение кластеров из них
        for image in self.files:
            print(f'file) = {image}')
            sample = str(self.photos_folder + '/' + image) # !!!
            print(f'sample!! = {sample}')
            km, h, w = self.kmeans_clustering(sample, n=2)
            seg = self.segmentation(km, h, w)
            contours_of_images[j] = seg # self.graph_clasters(seg)
            dict_of_names[j] = self.photos_folder + image
            j += int(self.params['friquency'])
        self.delta_coeffs = {}
        i, j = 0, 0             # print(f'contours_of_images = {contours_of_images}')
        # Перебор всех контуров фото и извлечение из них сегментов, коэффициентов и дельта-коэффициентов
        for contours in contours_of_images.items():
            segment = self.find_segment(contours)
            coeff, delta = self.find_delta(segment, dict_of_names, 6)
            self.delta_coeffs[j] = coeff
            j += int(self.params['friquency'])
            i += 1
        X, y = [], []
        low_level, high_level, step, j = 0, len(self.files), 1, 0
        # Перебор всех контуров фото и извлечение из них сегментов, коэффициентов и дельта-коэффициентов
        h = self.delta_coeffs[j]
        for i in range(low_level, high_level, step):
            X.append(j)
            x_pred =  np.array([[self.delta_coeffs[j]]]).T
            y_pred = lr.predict(x_pred)     # self.create_graphs(x_train, y_train)
            print(f'x_pred = {x_pred}, [[x_pred]] = {[[x_pred]]}, x_pred[0][0] = {x_pred[0][0]}')
            print(f'y_pred = {y_pred}, [[y_pred]] = {[[y_pred]]}, y_pred[0][0] = {y_pred[0][0]}')
            print(f'coeff_avg = {coeff_avg}')
            y.append(y_pred[0][0] * coeff_avg)   # * self.avg_coeff self.coeff_avg_ x_pred[0][0]
            print(f'x_pred = {x_pred}')
            print(f'Predicted coefficient for {x_pred} value of claster = {y_pred[0][0]}')
            print(f'y_pred = {y_pred}')
            j += int(self.params['friquency'])
        y_regression = lr.predict(x_train)
        print(f'self.delta_coeffs = {self.delta_coeffs}')
        print(f'X = {X}, y = {y}')
        '''
        xValues, yValues, yValues1 = X, [], []
        print(f'self.coeff {self.coeff}, self.delta_coeffs = {self.delta_coeffs}')
        X = 0
        for value in self.Y:
            print(f'self.delta_coeffs[j] = {self.delta_coeffs[X]}')
            yValues.append(value * self.delta_coeffs[X])                     # Задаём значения Y для первого графика
            yValues1.append(60 - (value * self.coeff))    # Задаём значения Y для второго графика
            X += float(self.params['photos_step'])
        '''
        fig = Figure(figsize=(12, 8), facecolor='white')
        axis = fig.add_subplot(211)     # 2 rows, 1 column, Top graph
        axis.plot(X, y)
        axis.set_xlabel('Время, мин.')
        axis.set_ylabel('Объем, мл.')
        axis.grid(linestyle='-')            # solid grid lines
        axis1 = fig.add_subplot(212)        # 2 rows, 1 column, Bottom graph
        axis1.set_xlabel('Время, мин.')
        axis1.set_ylabel('Объем, мл.')
        axis1.plot(X, y)
        axis1.grid()
        self.win.withdraw()
        self.win.protocol('WM_DELETE_WINDOW', self._destroyWindow)
        canvas = FigureCanvasTkAgg(fig, master=self.win)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.win.update()
        self.win.deiconify()
        self.win.mainloop()


    def change_params_button(self):
        # Функционал и описание параметров кнопок для изменения параметров конфигурационного файла
        self.friquency_input_label = ttk.Label(self.tab2, text="Частота фото, сек:").grid(column=0, row=1)
        self.friquency_input = tk.StringVar()
        self.friquency = ttk.Entry(self.tab2, width=12, textvariable=self.friquency_input).grid(column=1, row=1)
        self.friquency_input.set(self.params['friquency'])
        self.sum_volume_label = ttk.Label(self.tab2, text="Суммарный объем, мл:").grid(column=0, row=2)
        self.sum_volume_input = tk.StringVar()
        self.sum_volume = ttk.Entry(self.tab2, width=12, textvariable=self.sum_volume_input).grid(column=1, row=2)
        self.sum_volume_input.set(self.params['sum_volume'])
        self.dead_volume_label = ttk.Label(self.tab2, text="Мертвый объем, мл:").grid(column=0, row=3)
        self.dead_volume_input = tk.StringVar()
        self.dead_volume = ttk.Entry(self.tab2, width=12, textvariable=self.dead_volume_input).grid(column=1, row=3)
        self.dead_volume_input.set(self.params['dead_volume'])
        self.photos_step_label = ttk.Label(self.tab2, text="Шаг снимков, мл.:").grid(column=0, row=4)
        self.photos_step_input = tk.StringVar()
        self.photos_step = ttk.Entry(self.tab2, width=12, textvariable=self.photos_step_input).grid(column=1, row=4)
        self.photos_step_input.set(self.params['photos_step'])
        self.show_message_label = ttk.Label(self.tab2, text="Введите параметры:").grid(column=0, row=0)
        self.show_message = ttk.Button(self.tab2, text="Изменить", command=self.change_params).grid(column=2, row=3)


    def change_params(self):
        # Функция изменения параметров в конфигурационном файле
        self.config_file = open(self.dir,"w")                               # self.parser.read(self.dir)                        # self.parser.set('Photo', 'dir', self.photos_folder)
        self.parser.set('Photo', 'friquency', self.friquency_input.get())   # print(self.photos_folder)
        self.parser.set('Photo', 'photos_step', self.photos_step_input.get())
        self.parser.set('Options', 'sum_Volume', self.sum_volume_input.get())
        self.parser.set('Options', 'dead_volume', self.dead_volume_input.get())
        for section in self.parser.sections():                              # print(self.parser.items('Options'))
            print(section)
            for name, value in self.parser.items(section):
                print(f'{name} = {value}')
        self.parser.write(self.config_file)                                 # print(self.config_file)


    def create_widgets(self):
        # Функционал создания интерфейса
        self.tabControl = ttk.Notebook(self.win)
        self.tab1 = ttk.Frame(self.tabControl) # Create a tab
        self.tabControl.add(self.tab1, text='Настройка параметров системы') # Add the tab #buttons_frame = ttk.Labelframe(self.tab1, text ='Frame').grid(column=6, row=6)
        self.tab2 = ttk.Frame(self.tabControl) # Create second tab
        self.tabControl.add(self.tab2, text='Общие настройки') # Add second tab # buttons_frame2 = ttk.Labelframe(self.tab2, text ='Frame2').grid(column=0, row=0)
        self.tabControl.pack(expand=1, fill="both")         # Pack to make visible
        self.FORMAT_STRING = "%Y-%m-%d_%H-%M-%S"            # print(f'self.DELAY_TIME_SEC = {self.DELAY_TIME_SEC}')
        self.DELAY_TIME_SEC = int(self.params['friquency']) # print(f'self.FORMAT_STRING = {self.FORMAT_STRING}')
        self.photos_folder_button()
        self.open_model_button()
        self.change_params_button()
        self.create_graphs_button()
        self.setup_packages_button()
        self.create_model_button()
        self.photo_cycle_button()


    def files_extract(self, folder_path):
        # Функция извлечения файлов из пути
        files = []
        def images_names(path):
            files = os.listdir(path)    # print(f'!!!path = {path}')
            self.images = filter(lambda x: x.endswith('.png'), files)
            self.list_of_datetime = []
            for image in self.images:
                im = Image.open(f'{folder_path}/{image}')   # print(f'path = {self.photos_folder}/{image}')
                image = image.split('.')[0]
                if self.checkPhotosTurn() == 1:
                    im2 = im.transpose(Image.ROTATE_270)
                    im2.save(folder_path + '/' + image + '.png', "JPEG")
        images_names(folder_path)
        folder = folder_path                 # print(f'folder = {folder}')
        self.files = os.listdir(folder_path) # print(f'os.listdir(self.photos_folder) = {os.listdir(self.photos_folder)}')
        print(f'self.files = {self.files}')
        for image in self.files:
            sample = f'{folder}/{image}'            # print(sample)
            image = plt.imread(sample)              # graph_an_image(image)    print(image.shape)
        self.win.update()
        self.win.deiconify()
        self.win.mainloop()                         # axis1 = fig.add_subplot(212) # 2 rows, 1 column, Bottom graph


    def elbow_method(self, image):
        # НЕ ИСПОЛЬЗУЕТСЯ, ранее уже производился расчёт - количество кластеров, =2, было выявлено оптимальным (нужно было для перовой версии программы
        X, distortions = image[0], []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(X)
            distortions.append(km.inertia_)
        plt.plot(range(1, 11), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')    # image_test = plt.imread(train_images_names[8])
        plt.show()                  # elbow_method(image_test)


    def kmeans_clustering(self, img, n=4):
        # Функция кластеризации изображений - выделяет кластеры из изображений
        image = plt.imread(img)  # !!!
        im_small = zoom(image, (0.2,0.2,1))
        h,w = im_small.shape[:2]
        im_small_long = im_small.reshape((h * w, 3))
        im_small_wide = im_small_long.reshape((h,w,3))
        km = KMeans(n_clusters=n)
        km.fit(im_small_long)
        cc = km.cluster_centers_.astype(np.uint8)
        out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))  # print('K-means_clustering ends!')
        return [km, h, w]


    def segmentation(self, km, h, w):
        # Функция выделения сегментов контуров в изображении
        seg = np.asarray([(1 if i == 1 else 0) for i in km.labels_]).reshape((h,w))
        contours = measure.find_contours(seg, 0.5, fully_connected="high")
        simplified_contours = [measure.approximate_polygon(c, tolerance=4) for c in contours]
        return contours


    def graph_clasters(self, contours):
        # НЕ ИСПОЛЬЗУЕТСЯ, функция отображения кластеров (нужно было для перовой версии программы)
        plt.figure(figsize=(5,10))
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.ylim(h,0)
        plt.axes().set_aspect('equal')


    def find_segment(self, contours):
        # Функция поиска нужного сегмента (электролита) из всех сегметнов контуров в изображении
        segment, max = 0, 0
        for contour in contours:
            try:                        # print(f'!!!!! {contour}')
                if len(contour) > max:  # len(contour) ??
                    max = len(contour)
                    segment = contour   # print('Success!:)')
            except:
                pass                    # print('ERROR!')
        return segment


    def find_delta(self, segment, images, i):
        # Функция поиска максимума в сегменте для выбора наибольшего размера кластера и минимума
        min, max = 1000000, 0
        for seg in segment:             # print(f'segment = {segment}')
            segG = seg[0][0]            # print(f'seg[0][0] = {seg[0][0]}')
            try:
                if segG > max:
                    max = segG
                if segG < min:
                    min = segG          # print('Success!:)')
            except:
                pass                    # print('ERROR!')
        self.delta = max - min          # print('Image is ', images[i])
        self.coeff = i / self.delta
        print(f'Level is {str(self.delta * self.coeff)} mm, \n self.files = {self.files}, \n self.delta = {self.delta}, \n self.coeff = {self.coeff}')
        return self.coeff, self.delta


    def coeff_avg(self, delta_coeffs):
        # Функция усреднения значения высоты кластера в пикселях
        coeff_avg, sum = 0, 0
        for mm, coeff in delta_coeffs.items():
            sum = sum + coeff
        coeff_avg = sum / len(delta_coeffs); print(f'Average coefficient is {coeff_avg}')
        COEFFAVG = coeff_avg
        return coeff_avg


    def simple_plot(self, x, y, y_regression, test_idx=None, prefix=''):
        # Функция строит график по X, Y, y_regression
        plt.scatter(x, y, s=80, c ='r', edgecolors='k', linewidths=0.3);
        plt.plot(x, y_regression);
        if test_idx is not None:
            plt.scatter(x[test_idx], y[test_idx], s=80, c ='b', edgecolors='k', linewidths=0.3);
        plt.title('{} MSE = {}'.format(prefix, self.mean_squared_error(y, y_regression)));


    def mean_squared_error(self, y_true, y_predicted):
        return ((y_true - y_predicted)**2).mean()


    def photo_cycle(self):
        cap = cv2.VideoCapture(1)
        path = '''C:\\Users\\CEE_182\\Documents\\Arduino\\Camera\\image_'''
        while (True):
            _, frame = cap.read()
            t = datetime.now().strftime(self.FORMAT_STRING)
            cv2.imwrite(f'''{path}{t}.png''', frame)
            time.sleep(self.DELAY_TIME_SEC)
        cap.release()                   # When everything done, release the capture
        cv2.destroyAllWindows()


    def _destroyWindow(self):
        self.win.quit()
        self.win.destroy()


if __name__ == "__main__":  # Точка входа в программу
    pd, Workbook, pickle, np, plt, Figure, FigureCanvasTkAgg, Image, KMeans, make_pipeline, PolynomialFeatures, measure, LinearRegression, zoom = import_modules()
    COEFFAVG = 0
    interface = Interface()
