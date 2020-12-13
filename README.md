## Лабораторная работа по дисциплине информационные системы аэрокосмических комплексов.  
### Акимов В.Н. M30-312Б-18.  
  
Главный исполняемый файл - main.py, внутри комментарии ко всем функциям.
Файл city_data.xml генерируется автоматически при выполнении программы.
  
# Задание №1.  
https://gist.github.com/ilyashatalov/08f28665645a8e8709f1ed51fdc00791
  
Преобразование координат происходит в фунции get_pix_from_coord.  
Вырезание изображения по географическим координатам происходит в функции cut_bbox.  
  
Принцип определения координаты пикселя по географическим координатам:
1. На координатной плоскости широты и долготы откладываем два отрезка из COORD_UL к точке COORD_UR и COORD_LL.
2. Откладываем от заданной точки города вертикальную прямую к первому отрезку параллельно ко второму и ко второму парралельно к первому (строим крест).
3. Находим отношение длины отсеченных относительно точки COORD_UL отрезов ко всей длинне этих отрезков.
4. Откладываем эти же отношения на плоскости пиксельного изображения на сторонах заданного прямоугольника изображения снимка.
5. Из точки на верхней стороне прямоугольника откладываем прямую параллельно левой стороне прямоугольника и 
   из точки на левой стороне прямоугольника вторую прямую параллельно верхней стороне прямоугольника (строим крест).
6. На пересечении прямых будет лежать искомая точка пикселя, соответствующая заданной широте и долготе.
  
# Задание №2.  
https://gist.github.com/ilyashatalov/5c6d8d24222c8fb07a7921dda109c8ea
  
Рассчет NDVI происходит в функции get_ndvi. (их 2, с и без трансормациии, одну закомментировал в коде).  
Получение цветного изображения из матрицы NDVI происходит в функции apply_gradient.  
  
Из диапазонов длин волн снимков Band 3 (0.63 - 0.69 мкм) и Band 4 (0.77 - 0.90 мкм) Landsat 7 следует, что они хранят в себе информацию о красном и инфракрасном спектрах, поэтому далее будем использовать их для рассчета NDVI.  
Формула для расчета NDVI:  
   <strong><em>NDVI = (Pnir - Pred)/(Pnir + Pred)</em></strong>, где  
   <strong><em>Pnir</em></strong> - коэффициент отражения в инфракрасной области спектра,  
   <strong><em>Pnir</em></strong> - коэффициент отражения в красной области спектра.  
  
Вычисление NDVI (без трансформации):  
1. Открываем изображения Band 3 и Band 4, они имеют значения со значениями usigned char - целые числа от 0 до 255.
2. Переводим матрицы в тип с плавающей запятой.
3. Получаем сумму этих изображений и их разность и записываем в новые переменные sum и sub соответсвенно.
4. Делим матрицу sub на sum. 
5. Полученные значения в результирующей матрице - значения NDVI для каждого пикселя снимка, принимают нецелочисленные значения от -1 до 1.
  
Вычисление NDVI (с трансформацией):  
Источник: https://www.researchgate.net/publication/258831794_TOA_reflectance_and_NDVI_calculation_for_Landsat_7_ETM_images_of_Sicily  
1. Открываем изображения Band 3 и Band 4 как матрицы.  
2. Для начала получим энергетическую яркость L. Для этого применим к матрицам формулу:\
   <strong><em>L = (M * gain) + bias</em></strong>, где\
   <strong><em>M</em></strong> - матрица со значениями unsigned char,  
   <strong><em>gain и bias</em></strong> - специальные параметры, определенные для спектрометра Landsat 7 еще до его запуска.  
   (для Landsat 7: band 3 gain = 0.621654, bias = -5.62, band4 gain = 0.639764, bias = -5.74)  
3. Читаем из файла MTL значение SUN_EVALUATION - высота солнца над горизонтом в градусах.
4. После этого можем посчитать коэффициент отражения R по формуле:  
   <strong><em>R = (π * L * d^2)/(E * sinθ)</em></strong>, где   
   <strong><em>L</em></strong> - энергетическая яркость,  
   <strong><em>d</em></strong> - расстояние от земли до солнца в астрономических единицах,  
   <strong><em>E</em></strong> - экзоатмосферное солнечное излучение в определенном диапазоне в ватт/кв.мкм.  
   (для Landsat 7: band 3 = 1533, band 4 = 1039),  
   <strong><em>θ</em></strong> - высота солнца над горизонтом в градусах.  
5. Матрицы обрабатываем по алгоритму, описанному ранее (делим разность матриц на их сумму).
6. Полученная матрица - матрица NDVI с значительно более точными результатами, в отличие от использования значений пикселей.

Использование трансформации для перевода значений пикселей в коэффициенты отражения позволяет получить более контрастные изображения NDVI, что в свою очередь упрощает задачу классификации и анализа снимков.
   
Создание цветного изображения:
1. Вычисляем матрицу NDVi.
2. Переводим матрицу в серое изображение, для этого:  
   Прибавляем единицу, чтобы получить диапазон нецелочисленных значений от 0 до 2.  
   Умножаем на 128 и переводим с тип unsigned char, чтобы получить диапазон целых чисел от 0 до 256.  
3. Открываем изображение с цветным градиентом размера 256x1 и применяем функцию LUT библиотеки opencv к серому изображению и к изображению цветного градиента. Каждому значению пикселя задается соответсвие в изобржении градиенте и таким образом создается цветное изображение NDVI.
   

